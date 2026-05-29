"""
Utilities for discrete-time survival prediction.
"""

import torch
import torch.nn as nn


def concordance_index(event_times, predicted_scores, event_observed):
    """Calculate C-index, where higher predicted scores mean higher risk."""
    if isinstance(event_times, torch.Tensor):
        event_times = event_times.detach().cpu().numpy()
    if isinstance(predicted_scores, torch.Tensor):
        predicted_scores = predicted_scores.detach().cpu().numpy()
    if isinstance(event_observed, torch.Tensor):
        event_observed = event_observed.detach().cpu().numpy()

    n = len(event_times)
    concordant = 0
    permissible = 0

    for i in range(n):
        if event_observed[i] == 0:
            continue
        for j in range(n):
            if i == j:
                continue
            if event_times[i] < event_times[j]:
                permissible += 1
                if predicted_scores[i] > predicted_scores[j]:
                    concordant += 1
                elif predicted_scores[i] == predicted_scores[j]:
                    concordant += 0.5

    if permissible == 0:
        return 0.5
    return concordant / permissible


def _binary_roc_auc(scores: torch.Tensor, target: torch.Tensor) -> float:
    scores = scores.detach().float().view(-1)
    target = target.detach().bool().view(-1)

    num_pos = target.sum()
    num_neg = (~target).sum()
    if scores.numel() == 0 or num_pos == 0 or num_neg == 0:
        return float("nan")

    sorted_scores, order = torch.sort(scores)
    sorted_ranks = torch.arange(
        1,
        scores.numel() + 1,
        device=scores.device,
        dtype=torch.float32,
    )

    # Average ranks for tied prediction scores.
    tied_ranks = sorted_ranks.clone()
    _, counts = torch.unique_consecutive(
        sorted_scores,
        return_counts=True,
    )
    start = 0
    for count in counts.tolist():
        end = start + count
        if count > 1:
            tied_ranks[start:end] = sorted_ranks[start:end].mean()
        start = end

    ranks = torch.empty_like(tied_ranks)
    ranks[order] = tied_ranks
    pos_rank_sum = ranks[target].sum()

    auc = (
        pos_rank_sum - num_pos.float() * (num_pos.float() + 1.0) / 2.0
    ) / (num_pos.float() * num_neg.float())
    return float(auc.cpu().item())


def time_dependent_auc(
    survival,
    event_times,
    event_observed,
    landmark_years,
    cut_points_years,
):
    """
    Calculate cumulative/dynamic AUC at configured landmark times.

    Cases are observed events at or before the landmark. Controls are patients
    known to be event-free after the landmark. Patients censored before or at
    the landmark are excluded for that landmark.
    """
    if not isinstance(survival, torch.Tensor):
        survival = torch.as_tensor(survival)
    if not isinstance(event_times, torch.Tensor):
        event_times = torch.as_tensor(event_times)
    if not isinstance(event_observed, torch.Tensor):
        event_observed = torch.as_tensor(event_observed)
    if not isinstance(landmark_years, torch.Tensor):
        landmark_years = torch.as_tensor(landmark_years)
    if not isinstance(cut_points_years, torch.Tensor):
        cut_points_years = torch.as_tensor(cut_points_years)

    survival = survival.detach().float()
    event_times = event_times.detach().float().view(-1).to(survival.device)
    event_observed = event_observed.detach().bool().view(-1).to(survival.device)
    landmark_years = landmark_years.detach().float().view(-1).to(survival.device)
    cut_points_years = cut_points_years.detach().float().view(-1).to(survival.device)

    if survival.numel() == 0 or landmark_years.numel() == 0:
        return {}

    num_time_bins = survival.shape[1]
    landmark_bins = torch.bucketize(
        landmark_years,
        cut_points_years,
        right=False,
    ).clamp(0, num_time_bins - 1)

    aucs = {}
    for landmark, landmark_bin in zip(landmark_years, landmark_bins):
        observed_by_landmark = event_observed & (event_times <= landmark)
        known_event_free_after_landmark = event_times > landmark
        valid = observed_by_landmark | known_event_free_after_landmark

        risk = 1.0 - survival[:, int(landmark_bin.item())]
        aucs[float(landmark.cpu().item())] = _binary_roc_auc(
            risk[valid],
            observed_by_landmark[valid],
        )

    return aucs


def integrated_brier_score(survival, event_times, event_observed):
    """
    Calculate an unweighted discrete-time integrated Brier score.

    For each time bin, censored samples are used only while their survival
    status is known. Lower values indicate better calibrated survival curves.
    """
    if not isinstance(survival, torch.Tensor):
        survival = torch.as_tensor(survival)
    if not isinstance(event_times, torch.Tensor):
        event_times = torch.as_tensor(event_times)
    if not isinstance(event_observed, torch.Tensor):
        event_observed = torch.as_tensor(event_observed)

    survival = survival.detach().float()
    event_times = event_times.detach().long().view(-1)
    event_observed = event_observed.detach().bool().view(-1)

    if survival.numel() == 0:
        return 0.0

    num_time_bins = survival.shape[1]
    brier_scores = []

    for time_idx in range(num_time_bins):
        known_alive = event_times > time_idx
        known_event = event_observed & (event_times <= time_idx)
        known = known_alive | known_event

        if not torch.any(known):
            continue

        target = known_alive[known].to(dtype=survival.dtype, device=survival.device)
        pred = survival[known, time_idx]
        brier_scores.append(torch.mean((target - pred) ** 2))

    if not brier_scores:
        return 0.0

    return torch.stack(brier_scores).mean().item()


def _censoring_survival_km(event_times, event_observed, num_time_bins, eps=1e-7):
    """
    Estimate the censoring survival curve G(t) with Kaplan-Meier.

    In this curve, censored observations are treated as the event of interest.
    """
    event_times = event_times.long().view(-1)
    event_observed = event_observed.bool().view(-1)
    censored = ~event_observed

    censoring_survival = torch.empty(
        num_time_bins,
        device=event_times.device,
        dtype=torch.float32,
    )
    survival_prob = torch.tensor(1.0, device=event_times.device)

    for time_idx in range(num_time_bins):
        at_risk = event_times >= time_idx
        num_at_risk = at_risk.sum().float()

        if num_at_risk > 0:
            num_censored = (at_risk & censored & (event_times == time_idx)).sum().float()
            survival_prob = survival_prob * (1.0 - num_censored / num_at_risk)

        censoring_survival[time_idx] = survival_prob.clamp_min(eps)

    return censoring_survival


def integrated_brier_score_ipcw(survival, event_times, event_observed, eps=1e-7):
    """
    Calculate an IPCW weighted discrete-time integrated Brier score.

    IPCW uses a Kaplan-Meier estimate of the censoring distribution so censored
    samples that become unknown after censoring are represented by the remaining
    comparable observations. Lower values indicate better calibrated curves.
    """
    if not isinstance(survival, torch.Tensor):
        survival = torch.as_tensor(survival)
    if not isinstance(event_times, torch.Tensor):
        event_times = torch.as_tensor(event_times)
    if not isinstance(event_observed, torch.Tensor):
        event_observed = torch.as_tensor(event_observed)

    survival = survival.detach().float()
    event_times = event_times.detach().long().view(-1).to(survival.device)
    event_observed = event_observed.detach().bool().view(-1).to(survival.device)

    if survival.numel() == 0:
        return 0.0

    num_time_bins = survival.shape[1]
    censoring_survival = _censoring_survival_km(
        event_times,
        event_observed,
        num_time_bins,
        eps=eps,
    )
    event_times_clamped = torch.clamp(event_times, 0, num_time_bins - 1)
    brier_scores = []

    for time_idx in range(num_time_bins):
        observed_event = event_observed & (event_times <= time_idx)
        known_alive = event_times > time_idx

        weights = torch.zeros(
            survival.shape[0],
            device=survival.device,
            dtype=survival.dtype,
        )
        weights[observed_event] = (
            1.0 / censoring_survival[event_times_clamped[observed_event]]
        )
        weights[known_alive] = 1.0 / censoring_survival[time_idx]

        target = known_alive.to(dtype=survival.dtype, device=survival.device)
        pred = survival[:, time_idx]
        brier_scores.append(torch.mean(weights * (target - pred) ** 2))

    if not brier_scores:
        return 0.0

    return torch.stack(brier_scores).mean().item()


def logits_to_hazard(logits: torch.Tensor) -> torch.Tensor:
    """Convert hazard logits to hazard probabilities."""
    return torch.sigmoid(logits)


def hazard_to_survival(hazard: torch.Tensor) -> torch.Tensor:
    """Convert hazard probabilities to a survival function."""
    return torch.cumprod(1.0 - hazard, dim=1)


def survival_to_time(survival: torch.Tensor) -> torch.Tensor:
    """Convert a survival function to expected discrete survival time."""
    return torch.sum(survival, dim=1)


def soft_logrank_loss(
    p_high: torch.Tensor,
    time: torch.Tensor,
    event: torch.Tensor,
) -> torch.Tensor:
    """Directional Mantel-Cox log-rank signal on soft high-risk membership.

    Encourages high-risk patients to have more observed events than expected.
    Returns ``-mean_over_event_times(o_high - e_high)`` so lower = better.
    """
    p_high = p_high.float().view(-1)
    time = time.to(device=p_high.device, dtype=torch.float32).view(-1)
    event = event.to(device=p_high.device, dtype=torch.float32).view(-1)

    # Grad-preserving zero used as the no-op / no-event fallback.
    zero = p_high.sum() * 0.0

    event_mask = event > 0.5
    if not torch.any(event_mask):
        return zero

    unique_event_times = torch.unique(time[event_mask])
    if unique_event_times.numel() == 0:
        return zero

    signal = zero
    for t in unique_event_times:
        at_risk = time >= t
        n_total = at_risk.sum()
        if n_total <= 0:
            continue
        events_at_t = at_risk & (time == t) & event_mask
        d_total = events_at_t.sum()
        if d_total <= 0:
            continue
        n_high = p_high[at_risk].sum()
        o_high = p_high[events_at_t].sum()
        e_high = d_total.float() * n_high / n_total.float()
        signal = signal + (o_high - e_high)

    return -signal / float(unique_event_times.numel())


def group_balance_penalty(
    p_high: torch.Tensor,
    min_frac: float = 0.20,
    max_frac: float = 0.80,
) -> torch.Tensor:
    """Weak range-clamp penalty that fires only when ``p_high.mean()`` exits
    ``[min_frac, max_frac]``. Returns a non-negative scalar."""
    p_high = p_high.float().view(-1)
    mean_high = p_high.mean()
    too_small = torch.relu(torch.tensor(min_frac, device=p_high.device) - mean_high)
    too_large = torch.relu(mean_high - torch.tensor(max_frac, device=p_high.device))
    return too_small.pow(2) + too_large.pow(2)


class NLLSurvLoss(nn.Module):
    """pycox logistic-hazard NLL. Input: raw hazard logits, int time bins, event."""

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        from pycox.models.loss import NLLLogistiHazardLoss
        self._loss = NLLLogistiHazardLoss(reduction=reduction)

    def forward(self, logits, time, event):
        idx = time.to(torch.int64).view(-1)
        ev = event.to(torch.float32).view(-1)
        return self._loss(logits, idx, ev)


class CoxPHLoss(nn.Module):
    """Cox partial likelihood via torchsurv (Efron ties). Input: risk, time, event."""

    def __init__(self, reduction: str = "mean", ties_method: str = "efron"):
        super().__init__()
        from torchsurv.loss import cox
        self._fn = cox.neg_partial_log_likelihood
        self.reduction = reduction
        self.ties_method = ties_method

    def forward(self, risk, time, event):
        log_hz = risk.float().view(-1)
        ev = event.to(torch.bool).view(-1)
        if ev.sum() == 0:
            return log_hz.sum() * 0.0
        t = time.to(torch.float32).view(-1)
        return self._fn(log_hz, ev, t, ties_method=self.ties_method,
                        reduction=self.reduction, checks=False)


class DeepHitLoss(nn.Module):
    """Single-event DeepHit via pycox. Input: raw pmf logits, int time bins, event.

    total = alpha * NLL + (1 - alpha) * ranking. No calibration term (pycox).
    """
    def __init__(self, alpha: float = 0.2, sigma: float = 0.1):
        super().__init__()
        from pycox.models.loss import DeepHitSingleLoss
        self._loss = DeepHitSingleLoss(alpha=float(alpha), sigma=float(sigma))

    def forward(self, pmf_logits, time, event):
        from pycox.models.data import pair_rank_mat
        idx = time.to(torch.int64).view(-1)
        ev = event.to(torch.int64).view(-1)
        rank_mat = torch.as_tensor(
            pair_rank_mat(idx.detach().cpu().numpy(), ev.detach().cpu().numpy()),
            dtype=pmf_logits.dtype, device=pmf_logits.device,
        )
        return self._loss(pmf_logits, idx, ev.to(pmf_logits.dtype), rank_mat)


class PMFLoss(nn.Module):
    """pycox PMF NLL. Input: raw pmf logits, int time bins, event."""
    def __init__(self):
        super().__init__()
        from pycox.models.loss import NLLPMFLoss
        self._loss = NLLPMFLoss()

    def forward(self, pmf_logits, time, event):
        return self._loss(pmf_logits, time.to(torch.int64).view(-1), event.to(torch.float32).view(-1))


class MTLRLoss(nn.Module):
    """pycox MTLR NLL. Input: raw logits [B,K], int time bins, event."""
    def __init__(self):
        super().__init__()
        from pycox.models.loss import NLLMTLRLoss
        self._loss = NLLMTLRLoss()

    def forward(self, logits, time, event):
        return self._loss(logits, time.to(torch.int64).view(-1), event.to(torch.float32).view(-1))


class BCESurvLoss(nn.Module):
    """pycox BCESurv loss. Input: raw logits [B,K], int time bins, event."""
    def __init__(self):
        super().__init__()
        from pycox.models.loss import BCESurvLoss as _BCE
        self._loss = _BCE()

    def forward(self, logits, time, event):
        return self._loss(logits, time.to(torch.int64).view(-1), event.to(torch.float32).view(-1))


class PCHazardLoss(nn.Module):
    """pycox piecewise-constant hazard NLL. Input: logits, time bins, event, interval_frac."""
    def __init__(self):
        super().__init__()
        from pycox.models.loss import NLLPCHazardLoss
        self._loss = NLLPCHazardLoss()

    def forward(self, logits, time, event, interval_frac):
        return self._loss(logits, time.to(torch.int64).view(-1),
                          event.to(torch.float32).view(-1),
                          interval_frac.to(logits.dtype).view(-1))


class WeibullLoss(nn.Module):
    """Parametric Weibull AFT via torchsurv. Input: log_params [B,2], time, event."""
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        from torchsurv.loss import weibull
        self._fn = weibull.neg_log_likelihood_weibull
        self.reduction = reduction

    def forward(self, log_params, time, event):
        ev = event.to(torch.bool).view(-1)
        if ev.sum() == 0:
            return log_params.sum() * 0.0
        return self._fn(log_params, ev, time.float().view(-1),
                        reduction=self.reduction, checks=False)


class SoftLogRankLoss(nn.Module):
    """Differentiable Mantel-Cox log-rank loss + weak group-balance penalty.

    ``forward`` takes ``p_high`` in [0, 1], continuous ``time``, and binary
    ``event`` and returns ``(total, components)`` where ``components`` exposes
    the individual terms and diagnostic statistics for logging.
    """

    def __init__(
        self,
        lambda_balance: float = 0.01,
        min_frac: float = 0.20,
        max_frac: float = 0.80,
    ):
        super().__init__()
        if lambda_balance < 0.0:
            raise ValueError("lambda_balance must be non-negative.")
        if not 0.0 <= min_frac <= max_frac <= 1.0:
            raise ValueError("require 0 <= min_frac <= max_frac <= 1.")
        self.lambda_balance = float(lambda_balance)
        self.min_frac = float(min_frac)
        self.max_frac = float(max_frac)

    def forward(
        self,
        p_high: torch.Tensor,
        time: torch.Tensor,
        event: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        logrank = soft_logrank_loss(p_high, time, event)
        balance = group_balance_penalty(
            p_high,
            min_frac=self.min_frac,
            max_frac=self.max_frac,
        )
        total = logrank + self.lambda_balance * balance

        with torch.no_grad():
            p_h = p_high.detach().float().view(-1)
            components = {
                "logrank": logrank.detach(),
                "balance": balance.detach(),
                "p_high_mean": p_h.mean(),
                "p_high_min": p_h.min() if p_h.numel() > 0 else torch.tensor(0.0, device=p_h.device),
                "p_high_max": p_h.max() if p_h.numel() > 0 else torch.tensor(0.0, device=p_h.device),
                "fraction_high_hard": (p_h >= 0.5).float().mean(),
            }
        return total, components


def _reject_legacy_cox_loss_lambda(kwargs):
    if "cox_loss_lambda" in kwargs and kwargs["cox_loss_lambda"] is not None:
        raise ValueError(
            "`cox_loss_lambda` is no longer supported. Use "
            "`model.survival_loss: {name: cox}` to train Cox alone or "
            "`{name: nll}` for plain NLL."
        )


def build_survival_criterion(cfg, num_time_bins: int):
    """Return (name, criterion) for a survival_loss config block.

    ``cfg`` is the OmegaConf/dict block under ``model.survival_loss``, or
    ``None`` (in which case the default ``{name: 'nll'}`` is used).
    """
    if cfg is None:
        cfg = {"name": "nll"}
    # OmegaConf nodes behave dict-like enough for our purposes.
    if not hasattr(cfg, "get") or not hasattr(cfg, "__contains__"):
        raise ValueError(
            "survival_loss must be a mapping with a 'name' key (e.g. "
            "`survival_loss: {name: deephit}`); got " + repr(cfg) + "."
        )
    if "name" not in cfg:
        raise ValueError(
            "survival_loss block must specify 'name' (one of: nll, cox, deephit)."
        )
    name = str(cfg["name"]).lower()

    if name == "nll":
        return name, NLLSurvLoss(reduction=cfg.get("reduction", "mean"))
    if name == "cox":
        return name, CoxPHLoss(reduction=cfg.get("reduction", "mean"))
    if name == "deephit":
        return name, DeepHitLoss(
            alpha=cfg.get("alpha", 0.2),
            sigma=cfg.get("sigma", 0.1),
        )
    if name == "soft_logrank":
        return name, SoftLogRankLoss(
            lambda_balance=float(cfg.get("lambda_balance", 0.01)),
            min_frac=float(cfg.get("min_frac", 0.20)),
            max_frac=float(cfg.get("max_frac", 0.80)),
        )
    if name == "pmf":
        return name, PMFLoss()
    if name == "mtlr":
        return name, MTLRLoss()
    if name == "bcesurv":
        return name, BCESurvLoss()
    if name == "weibull":
        return name, WeibullLoss(reduction=cfg.get("reduction", "mean"))
    if name == "pchazard":
        return name, PCHazardLoss()
    raise ValueError(
        f"Unknown survival_loss.name: {name!r}. Expected one of: "
        "nll, cox, deephit, soft_logrank, pmf, mtlr, bcesurv, weibull, pchazard."
    )


import numpy as _np


#: Maximum number of candidate cutoffs evaluated by `max_logrank_cutpoint`
#: when the bracket has more candidates. Sub-samples to a uniform quantile grid.
MAX_CANDIDATES = 200


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
