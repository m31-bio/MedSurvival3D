"""Survival evaluation metrics (torchsurv/lifelines-backed; cluster-only deps)."""
import numpy as _np
import torch


MAX_CANDIDATES = 200


def concordance_index(event_times, predicted_scores, event_observed):
    """Harrell C-index via torchsurv. Higher score = higher risk."""
    from torchsurv.metrics.cindex import ConcordanceIndex
    est = torch.as_tensor(predicted_scores).float().view(-1)
    ev = torch.as_tensor(event_observed).bool().view(-1)
    t = torch.as_tensor(event_times).float().view(-1)
    if ev.sum() == 0:
        return 0.5
    return float(ConcordanceIndex()(est, ev, t))


def time_dependent_auc(
    survival,
    event_times,
    event_observed,
    landmark_years,
    cut_points_years,
):
    """Cumulative/dynamic AUC at each landmark via torchsurv. Returns {landmark: auc}."""
    from torchsurv.metrics.auc import Auc
    survival = torch.as_tensor(survival).float()
    t = torch.as_tensor(event_times).float().view(-1)
    e = torch.as_tensor(event_observed).bool().view(-1)
    landmarks = torch.as_tensor(landmark_years).float().view(-1)
    cut_points = torch.as_tensor(cut_points_years).float().view(-1)
    num_bins = survival.shape[1]

    if survival.numel() == 0 or landmarks.numel() == 0:
        return {}

    # Map each landmark value to its survival-curve column using the same
    # bucketize logic as the old implementation, preserving correctness for
    # non-arange cut_points (e.g. irregular time grids).
    landmark_bins = torch.bucketize(landmarks, cut_points, right=False).clamp(0, num_bins - 1)

    auc = Auc()
    out = {}
    for lm, b in zip(landmarks.tolist(), landmark_bins.tolist()):
        risk = 1.0 - survival[:, int(b)]
        try:
            out[float(lm)] = float(auc(risk, e, t, new_time=torch.tensor(float(lm))))
        except Exception:
            out[float(lm)] = float("nan")
    return out


def _valid_time_grid(event_times, event_observed, num_bins):
    """Return integer time indices suitable for torchsurv BrierScore evaluation."""
    t = torch.as_tensor(event_times).long().view(-1)
    e = torch.as_tensor(event_observed).bool().view(-1)
    lo = max(1, int(t[e].min()) + 1) if e.any() else 1
    hi = min(int(t.max()), num_bins - 1)
    return torch.arange(lo, max(lo + 1, hi))


def _ibs_torchsurv(survival, event_times, event_observed, times, weight=None, weight_new_time=None):
    """Shared helper: integrated Brier score via torchsurv over integer time grid.

    weight=None -> uniform weights (plain, unweighted).
    Pass explicit IPCW weights for the IPCW variant.
    torchsurv BrierScore does NOT apply IPCW automatically; weight=None
    is internally converted to all-ones by torchsurv._update_brier_score_weight.
    """
    from torchsurv.metrics.brier_score import BrierScore
    s = torch.as_tensor(survival).float()
    t = torch.as_tensor(event_times).float().view(-1)
    e = torch.as_tensor(event_observed).bool().view(-1)
    est = s[:, times]                                   # [n, len(times)]
    bs = BrierScore()
    bs(est, e, t, new_time=times.float(), weight=weight, weight_new_time=weight_new_time)
    return float(bs.integral())


def integrated_brier_score(survival, event_times, event_observed):
    """Unweighted integrated Brier score via torchsurv (weight=1 for all samples).

    Bin-index time axis: `event_times` must be discrete bin indices in the same
    units as the survival-matrix columns (`_ibs_torchsurv` uses the time value as
    both the eval grid and the column index `s[:, times]`), NOT continuous years.
    Consequently this score is not on a comparable time grid with year-axis
    metrics such as `time_dependent_auc`.
    """
    survival = torch.as_tensor(survival).float()
    if survival.numel() == 0:
        return 0.0
    times = _valid_time_grid(event_times, event_observed, survival.shape[1])
    try:
        return _ibs_torchsurv(survival, event_times, event_observed, times, weight=None)
    except Exception:
        return float("nan")


def integrated_brier_score_ipcw(survival, event_times, event_observed, eps=1e-7):
    """IPCW integrated Brier score via torchsurv.

    Uses torchsurv.stats.ipcw.get_ipcw to compute the inverse probability of
    censoring weights from the KM censoring distribution, then passes them
    explicitly to BrierScore.  weight=None would give the unweighted (plain)
    score; we pass IPCW weights so this function is genuinely different from
    integrated_brier_score on censored data.

    Bin-index time axis: like integrated_brier_score, `event_times` must be
    discrete bin indices (same units as the survival-matrix columns), NOT
    continuous years; not comparable to year-axis metrics like time_dependent_auc.
    """
    from torchsurv.stats.ipcw import get_ipcw
    survival = torch.as_tensor(survival).float()
    if survival.numel() == 0:
        return 0.0
    t = torch.as_tensor(event_times).float().view(-1)
    e = torch.as_tensor(event_observed).bool().view(-1)
    times = _valid_time_grid(event_times, event_observed, survival.shape[1])
    try:
        ipcw_at_time = get_ipcw(e, t)
        ipcw_at_new_time = get_ipcw(e, t, times.float())
        return _ibs_torchsurv(
            survival, event_times, event_observed, times,
            weight=ipcw_at_time, weight_new_time=ipcw_at_new_time,
        )
    except Exception:
        return float("nan")


def _logrank_chi2(times, events, group_high):
    """Log-rank chi^2 via lifelines. Used internally by max_logrank_cutpoint.

    Returns float('nan') when either group is empty or there are no events,
    so the caller can skip invalid candidates.
    """
    from lifelines.statistics import logrank_test

    times = _np.asarray(times, float)
    events = _np.asarray(events).astype(int)
    g = _np.asarray(group_high).astype(bool)
    if g.sum() == 0 or (~g).sum() == 0 or events.sum() == 0:
        return float("nan")
    return float(logrank_test(times[g], times[~g],
                              event_observed_A=events[g], event_observed_B=events[~g]).test_statistic)


def max_logrank_cutpoint(scores, times, events, q_lo=0.2, q_hi=0.8):
    """Return the cutoff in `scores` that maximizes the log-rank chi^2 among
    candidates drawn from the [quantile(scores, q_lo), quantile(scores, q_hi)]
    bracket. Used for training-time stratification monitoring of NLL/Cox/DeepHit.

    Note: the returned chi^2 is biased upward by the selection, so logged
    chi^2 values are training diagnostics rather than formal significance tests.

    Returns float('nan') when no valid candidate yields two non-empty groups,
    or when every candidate produces zero chi^2 (no separating signal).
    At most 200 candidate cutoffs are evaluated; large input sets are
    sub-sampled to a uniform quantile grid.
    """
    scores = _np.asarray(scores, dtype=float)
    times = _np.asarray(times, dtype=float)
    events = _np.asarray(events, dtype=bool)

    if scores.size == 0 or events.sum() == 0:
        return float("nan")

    lo = float(_np.quantile(scores, q_lo))
    hi = float(_np.quantile(scores, q_hi))
    if lo >= hi:
        return float("nan")

    in_bracket = (scores >= lo) & (scores <= hi)
    unique_candidates = _np.unique(scores[in_bracket])
    if unique_candidates.size == 0:
        return float("nan")

    if unique_candidates.size > MAX_CANDIDATES:
        # Uniform quantile grid across the bracket.
        qs = _np.linspace(q_lo, q_hi, MAX_CANDIDATES)
        candidates = _np.unique(_np.quantile(scores, qs))
    else:
        candidates = unique_candidates

    best_chi2 = -1.0
    best_tied = []
    for c in candidates:
        group_high = scores >= c
        if group_high.sum() == 0 or (~group_high).sum() == 0:
            continue
        chi2 = _logrank_chi2(times, events, group_high)
        if chi2 > best_chi2:
            best_chi2 = chi2
            best_tied = [float(c)]
        elif chi2 == best_chi2:
            best_tied.append(float(c))

    if best_chi2 <= 0 or not best_tied:
        return float("nan")
    # Tie-break: median of tied candidates.
    return float(_np.median(_np.asarray(best_tied)))


def derive_stratification_scores(loss_name, risks, survival_curves, landmark_bin_idx):
    """Return per-sample scalar stratification scores for a survival loss.

    - cox / soft_logrank: returns `risks` unchanged (log-hazard or sigmoid logit).
    - nll / deephit / pmf / mtlr / bcesurv / pchazard / weibull: returns 1 - survival_curves[:, landmark_bin_idx].

    Raises ValueError for unrecognized loss names.
    """
    if loss_name in ("cox", "soft_logrank"):
        if risks is None:
            raise ValueError(f"{loss_name} requires `risks` array")
        return _np.asarray(risks, dtype=float)
    if loss_name in ("nll", "deephit", "pmf", "mtlr", "bcesurv", "pchazard", "weibull"):
        if survival_curves is None:
            raise ValueError(f"{loss_name} requires `survival_curves` array")
        curves = _np.asarray(survival_curves, dtype=float)
        return 1.0 - curves[:, int(landmark_bin_idx)]
    raise ValueError(f"Unknown survival_loss_name: {loss_name!r}")
