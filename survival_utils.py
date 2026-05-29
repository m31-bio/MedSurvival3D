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
    """
    Negative log-likelihood loss for discrete-time survival prediction.

    ``time`` is expected to contain 0-indexed discrete time-bin indices, and
    ``event`` should be 1 for observed events and 0 for censored observations.
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError("reduction must be one of: 'mean', 'sum', 'none'")
        self.reduction = reduction

    def forward(
        self,
        logits: torch.Tensor,
        time: torch.Tensor,
        event: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, num_time_bins = logits.shape

        hazard = logits_to_hazard(logits).clamp(min=1e-7, max=1.0 - 1e-7)
        survival = hazard_to_survival(hazard)
        survival_padded = torch.cat(
            [
                torch.ones(
                    batch_size,
                    1,
                    device=logits.device,
                    dtype=logits.dtype,
                ),
                survival,
            ],
            dim=1,
        )

        y_time = time.to(device=logits.device, dtype=torch.int64).view(-1, 1)
        y_event = event.to(device=logits.device, dtype=logits.dtype).view(-1, 1)
        y_time = torch.clamp(y_time, 0, num_time_bins - 1)

        s_prev = torch.gather(survival_padded, 1, y_time).clamp(min=1e-7)
        h_this = torch.gather(hazard, 1, y_time).clamp(min=1e-7)
        log_lik_uncensored = torch.log(s_prev) + torch.log(h_this)

        y_time_next = torch.clamp(y_time + 1, 0, num_time_bins)
        s_this = torch.gather(survival_padded, 1, y_time_next).clamp(min=1e-7)
        log_lik_censored = torch.log(s_this)

        neg_log_lik = -(
            y_event * log_lik_uncensored + (1.0 - y_event) * log_lik_censored
        )

        if self.reduction == "mean":
            return neg_log_lik.mean()
        if self.reduction == "sum":
            return neg_log_lik.sum()
        return neg_log_lik.squeeze(1)


class CoxPHLoss(nn.Module):
    """
    Cox proportional hazards negative partial log-likelihood.

    ``risk`` should be a scalar per sample where larger values indicate higher
    risk. ``time`` can be continuous follow-up/event time, and ``event`` should
    be 1 for observed events and 0 for censored observations.
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError("reduction must be one of: 'mean', 'sum', 'none'")
        self.reduction = reduction

    def forward(
        self,
        risk: torch.Tensor,
        time: torch.Tensor,
        event: torch.Tensor,
    ) -> torch.Tensor:
        risk = risk.float().view(-1)
        time = time.to(device=risk.device, dtype=torch.float32).view(-1)
        event = event.to(device=risk.device, dtype=torch.bool).view(-1)

        if risk.numel() == 0 or not torch.any(event):
            return risk.sum() * 0.0

        risk_set = time.view(1, -1) >= time.view(-1, 1)
        masked_risk = risk.view(1, -1).masked_fill(~risk_set, float("-inf"))
        log_risk_set = torch.logsumexp(masked_risk, dim=1)

        neg_log_lik = -(risk - log_risk_set)[event]

        if self.reduction == "mean":
            return neg_log_lik.mean()
        if self.reduction == "sum":
            return neg_log_lik.sum()
        return neg_log_lik


class DeepHitLoss(nn.Module):
    """
    Single-event DeepHit loss (Lee et al. 2018) as a PyTorch ``nn.Module``.

    Inputs to ``forward``:
        - ``pmf``: ``[B, num_time_bins]`` tensor, softmax over time bins.
        - ``time_bin``: ``[B]`` int64 tensor, 0-indexed bin of event/censor time.
        - ``event``: ``[B]`` tensor in {0, 1}; 1 if the event was observed.

    Output: scalar = ``alpha * LL + beta * Ranking + gamma * Calibration``.

    Note: ``gamma`` multiplies the calibration term, which sums across the
    batch (matching the TensorFlow reference). Its effective scale therefore
    grows with batch size — re-tune ``gamma`` if you change batch size.
    """

    _EPSILON = 1e-7

    def __init__(
        self,
        num_time_bins: int,
        alpha: float = 1.0,
        beta: float = 0.5,
        gamma: float = 0.0,
        sigma: float = 0.1,
    ):
        super().__init__()
        if num_time_bins < 2:
            raise ValueError("num_time_bins must be >= 2.")
        if sigma <= 0.0:
            raise ValueError("sigma must be > 0.")
        self.num_time_bins = num_time_bins
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.sigma = float(sigma)

    def _masks(self, time_bin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Build (mask1, mask2) from the time-bin index.

        mask1[i, t] = 1 iff t == time_bin[i]            (event point)
        mask2[i, t] = 1 iff t >= time_bin[i]            (survival region)
        """
        idx = torch.arange(self.num_time_bins, device=time_bin.device).view(1, -1)
        t = time_bin.view(-1, 1)
        mask1 = (idx == t).to(dtype=torch.float32)
        mask2 = (idx >= t).to(dtype=torch.float32)
        return mask1, mask2

    def _loss_log_likelihood(
        self,
        pmf: torch.Tensor,
        mask1: torch.Tensor,
        mask2: torch.Tensor,
        event: torch.Tensor,
    ) -> torch.Tensor:
        eps = self._EPSILON
        observed = (mask1 * pmf).sum(dim=1).clamp_min(eps)
        censored = (mask2 * pmf).sum(dim=1).clamp_min(eps)
        log_obs = torch.log(observed)
        log_cen = torch.log(censored)
        return -(event * log_obs + (1.0 - event) * log_cen).mean()

    def _loss_ranking(
        self,
        pmf: torch.Tensor,
        time_bin: torch.Tensor,
        mask2: torch.Tensor,
        event: torch.Tensor,
    ) -> torch.Tensor:
        """Pairwise ranking penalty at each uncensored event time."""
        if event.sum() == 0:
            return pmf.sum() * 0.0

        surv_at_t = (pmf.unsqueeze(1) * mask2.unsqueeze(0)).sum(dim=2)
        diag = torch.diagonal(surv_at_t)
        diff = diag.view(-1, 1) - surv_at_t

        t_i = time_bin.view(-1, 1).float()
        t_j = time_bin.view(1, -1).float()
        ordered = (t_i < t_j).to(dtype=torch.float32)
        event_i = event.view(-1, 1)
        weight = ordered * event_i

        denom = weight.sum().clamp_min(1.0)
        return (weight * torch.exp(-diff / self.sigma)).sum() / denom

    def _loss_calibration(
        self,
        pmf: torch.Tensor,
        mask2: torch.Tensor,
        event: torch.Tensor,
    ) -> torch.Tensor:
        """DeepHit's calibration term (Lee et al. reference impl, single-event).

        Matches class_DeepHit.py loss_Calibration: r[t] = Σ_i pmf[i, t] * mask2[i, t]
        (sum over batch), then per-sample term = mean_t((r[t] - event[i])^2),
        summed across the batch.
        """
        r = (pmf * mask2).sum(dim=0)                       # [K], summed over batch
        diff = r.unsqueeze(0) - event.unsqueeze(1)          # [B, K]
        per_subject = diff.pow(2).mean(dim=1)               # [B]
        return per_subject.sum()

    def forward(
        self,
        pmf: torch.Tensor,
        time_bin: torch.Tensor,
        event: torch.Tensor,
    ) -> torch.Tensor:
        pmf = pmf.float()
        time_bin = time_bin.to(device=pmf.device, dtype=torch.long).view(-1)
        event = event.to(device=pmf.device, dtype=torch.float32).view(-1)
        time_bin = time_bin.clamp(0, self.num_time_bins - 1)

        mask1, mask2 = self._masks(time_bin)

        ll = self._loss_log_likelihood(pmf, mask1, mask2, event)
        rank = self._loss_ranking(pmf, time_bin, mask2, event)
        cal = self._loss_calibration(pmf, mask2, event)
        return self.alpha * ll + self.beta * rank + self.gamma * cal


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
            num_time_bins=num_time_bins,
            alpha=float(cfg.get("alpha", 1.0)),
            beta=float(cfg.get("beta", 0.5)),
            gamma=float(cfg.get("gamma", 0.0)),
            sigma=float(cfg.get("sigma", 0.1)),
        )
    if name == "soft_logrank":
        return name, SoftLogRankLoss(
            lambda_balance=float(cfg.get("lambda_balance", 0.01)),
            min_frac=float(cfg.get("min_frac", 0.20)),
            max_frac=float(cfg.get("max_frac", 0.80)),
        )
    raise ValueError(
        f"Unknown survival_loss.name: {name!r}. Expected one of: "
        "nll, cox, deephit, soft_logrank."
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
    - nll / deephit: returns 1 - survival_curves[:, landmark_bin_idx].

    Raises ValueError for unrecognized loss names.
    """
    if loss_name in ("cox", "soft_logrank"):
        if risks is None:
            raise ValueError(f"{loss_name} requires `risks` array")
        return _np.asarray(risks, dtype=float)
    if loss_name in ("nll", "deephit"):
        if survival_curves is None:
            raise ValueError(f"{loss_name} requires `survival_curves` array")
        curves = _np.asarray(survival_curves, dtype=float)
        return 1.0 - curves[:, int(landmark_bin_idx)]
    raise ValueError(f"Unknown survival_loss_name: {loss_name!r}")
