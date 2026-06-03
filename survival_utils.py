"""
Utilities for discrete-time survival prediction.
"""

import torch
import torch.nn as nn


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
    """Unweighted integrated Brier score via torchsurv (weight=1 for all samples)."""
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


class CompositeSurvivalLoss(nn.Module):
    """Container for a weighted sum of survival losses.

    Holds the member criteria (so their parameters register), their names, the
    parallel weights, and the ``primary`` member name. The weighted summation
    and the ``y_hat -> input`` mapping live in ``base_model._survival_loss``;
    this class is only the data holder.
    """

    def __init__(self, members, names, weights, primary):
        super().__init__()
        self.members = nn.ModuleList(members)
        self.names = list(names)
        self.weights = list(weights)
        self.primary = primary


def _reject_legacy_cox_loss_lambda(kwargs):
    if "cox_loss_lambda" in kwargs and kwargs["cox_loss_lambda"] is not None:
        raise ValueError(
            "`cox_loss_lambda` is no longer supported. Use "
            "`model.survival_loss: {name: cox}` to train Cox alone or "
            "`{name: nll}` for plain NLL."
        )


_SINGLE_LOSS_NAMES = (
    "nll", "cox", "deephit", "soft_logrank",
    "pmf", "mtlr", "bcesurv", "weibull", "pchazard",
)


def _parse_composite(cfg):
    """Validate a composite ``survival_loss`` block and normalise it.

    Returns ``(components, primary)`` where ``components`` is a list of
    ``{"name", "cfg", "weight"}`` dicts in config order (``cfg`` is the original
    per-component mapping, carrying that loss's own options) and ``primary`` is
    the lowercased name of the component that drives metrics/inference.

    Validates without constructing any loss objects, so it is dependency-free.
    Raises ``ValueError`` on any malformed block.
    """
    if "components" not in cfg or not cfg["components"]:
        raise ValueError(
            "composite survival_loss requires a non-empty 'components' list."
        )
    if "primary" not in cfg or cfg["primary"] is None:
        raise ValueError(
            "composite survival_loss requires a 'primary' naming one component."
        )

    components = []
    seen = set()
    for entry in cfg["components"]:
        if "name" not in entry:
            raise ValueError("each composite component needs a 'name'.")
        nm = str(entry["name"]).lower()
        if nm == "composite":
            raise ValueError(
                "composite components cannot themselves be 'composite' (no nesting)."
            )
        if nm not in _SINGLE_LOSS_NAMES:
            raise ValueError(
                f"unknown composite component name: {nm!r}. Expected one of: "
                + ", ".join(_SINGLE_LOSS_NAMES) + "."
            )
        if nm in seen:
            raise ValueError(f"duplicate composite component name: {nm!r}.")
        seen.add(nm)
        weight = entry.get("weight", 1.0)
        try:
            weight = float(weight)
        except (TypeError, ValueError):
            raise ValueError(
                f"composite component {nm!r} has non-numeric weight: {weight!r}."
            )
        if weight < 0:
            raise ValueError(
                f"composite component {nm!r} has negative weight: {weight}."
            )
        components.append({"name": nm, "cfg": entry, "weight": weight})

    primary = str(cfg["primary"]).lower()
    if primary not in seen:
        raise ValueError(
            f"composite 'primary' {primary!r} is not among components: {sorted(seen)}."
        )
    return components, primary


def _build_single_criterion(name, cfg):
    """Construct one survival-loss module from its (lowercased) name and config.

    ``cfg`` is the per-loss mapping carrying that loss's options. Shared by the
    single-loss path and each member of a composite.
    """
    if name == "nll":
        return NLLSurvLoss(reduction=cfg.get("reduction", "mean"))
    if name == "cox":
        return CoxPHLoss(reduction=cfg.get("reduction", "mean"))
    if name == "deephit":
        return DeepHitLoss(
            alpha=cfg.get("alpha", 0.2),
            sigma=cfg.get("sigma", 0.1),
        )
    if name == "soft_logrank":
        return SoftLogRankLoss(
            lambda_balance=float(cfg.get("lambda_balance", 0.01)),
            min_frac=float(cfg.get("min_frac", 0.20)),
            max_frac=float(cfg.get("max_frac", 0.80)),
        )
    if name == "pmf":
        return PMFLoss()
    if name == "mtlr":
        return MTLRLoss()
    if name == "bcesurv":
        return BCESurvLoss()
    if name == "weibull":
        return WeibullLoss(reduction=cfg.get("reduction", "mean"))
    if name == "pchazard":
        return PCHazardLoss()
    raise ValueError(
        f"Unknown survival_loss.name: {name!r}. Expected one of: "
        + ", ".join(_SINGLE_LOSS_NAMES) + "."
    )


def build_survival_criterion(cfg, num_time_bins: int):
    """Return (name, criterion) for a survival_loss config block.

    ``cfg`` is the OmegaConf/dict block under ``model.survival_loss``, or
    ``None`` (in which case the default ``{name: 'nll'}`` is used).

    Supported names and their hyperparameters:
      - ``nll``          — pycox logistic-hazard NLL. Opts: reduction (default "mean").
      - ``cox``          — Cox partial likelihood (Efron ties). Opts: reduction (default "mean").
      - ``deephit``      — Single-event DeepHit. Opts: alpha (default 0.2), sigma (default 0.1).
      - ``soft_logrank`` — Differentiable log-rank + balance penalty.
                           Opts: lambda_balance (0.01), min_frac (0.20), max_frac (0.80).
      - ``pmf``          — pycox PMF NLL. No extra opts.
      - ``mtlr``         — pycox MTLR NLL. No extra opts.
      - ``bcesurv``      — pycox BCESurv loss. No extra opts.
      - ``weibull``      — Parametric Weibull AFT. Opts: reduction (default "mean").
      - ``pchazard``     — pycox piecewise-constant hazard NLL. No extra opts.
      - ``composite``    — weighted sum of any of the above. Requires a
                           ``components`` list (each a single-loss block plus an
                           optional ``weight``, default 1.0) and a ``primary``
                           naming the member that drives metrics/inference.
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

    if name == "composite":
        components, primary = _parse_composite(cfg)
        members = [_build_single_criterion(c["name"], c["cfg"]) for c in components]
        names = [c["name"] for c in components]
        weights = [c["weight"] for c in components]
        return name, CompositeSurvivalLoss(members, names, weights, primary)

    return name, _build_single_criterion(name, cfg)


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
