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
