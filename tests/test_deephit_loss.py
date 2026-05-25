"""Sanity tests for survival_utils.DeepHitLoss."""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

# Make `SSL3D_survival/` importable when running the test directly.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from survival_utils import DeepHitLoss  # noqa: E402


def test_log_likelihood_collapses_to_nll_when_only_alpha():
    """With beta=gamma=0 and all events observed, DeepHit LL ≈ -mean(log pmf[i, t_i])."""
    torch.manual_seed(0)
    batch, bins = 8, 5
    logits = torch.randn(batch, bins)
    pmf = F.softmax(logits, dim=1)
    time_bin = torch.tensor([0, 1, 2, 3, 4, 0, 2, 4], dtype=torch.long)
    event = torch.ones(batch, dtype=torch.float32)

    loss_fn = DeepHitLoss(num_time_bins=bins, alpha=1.0, beta=0.0, gamma=0.0, sigma=0.1)
    deephit_loss = loss_fn(pmf, time_bin, event)

    nll = -torch.log(pmf.gather(1, time_bin.view(-1, 1)).clamp_min(1e-7)).mean()
    assert torch.allclose(deephit_loss, nll, atol=1e-5), (deephit_loss.item(), nll.item())


def test_finite_and_nonnegative_with_mixed_censoring():
    torch.manual_seed(1)
    batch, bins = 16, 5
    pmf = F.softmax(torch.randn(batch, bins), dim=1)
    time_bin = torch.randint(0, bins, (batch,), dtype=torch.long)
    event = torch.randint(0, 2, (batch,), dtype=torch.float32)

    loss_fn = DeepHitLoss(num_time_bins=bins, alpha=1.0, beta=0.5, gamma=0.5, sigma=0.1)
    value = loss_fn(pmf, time_bin, event)

    assert torch.isfinite(value), value
    assert value.item() >= 0.0, value.item()


def test_zero_loss_on_perfect_predictions():
    """If pmf places ~1.0 mass at the true bin for all uncensored, LL→0; ranking/cal→small."""
    batch, bins = 4, 5
    time_bin = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    event = torch.ones(batch, dtype=torch.float32)

    onehot = F.one_hot(time_bin, num_classes=bins).float()
    pmf = onehot * (1.0 - 1e-6) + (1e-6 / bins)

    loss_fn = DeepHitLoss(num_time_bins=bins, alpha=1.0, beta=0.0, gamma=0.0, sigma=0.1)
    assert loss_fn(pmf, time_bin, event).item() < 1e-3


def test_calibration_nonzero_gamma_changes_loss():
    torch.manual_seed(2)
    bins = 5
    pmf = torch.nn.functional.softmax(torch.randn(8, bins), dim=1)
    time_bin = torch.tensor([0, 1, 2, 3, 4, 0, 2, 4], dtype=torch.long)
    event = torch.tensor([1, 0, 1, 1, 0, 1, 0, 1], dtype=torch.float32)
    no_cal = DeepHitLoss(num_time_bins=bins, alpha=1.0, beta=0.0, gamma=0.0, sigma=0.1)(pmf, time_bin, event)
    with_cal = DeepHitLoss(num_time_bins=bins, alpha=1.0, beta=0.0, gamma=0.5, sigma=0.1)(pmf, time_bin, event)
    assert with_cal != no_cal, (with_cal.item(), no_cal.item())
    assert torch.isfinite(with_cal)


if __name__ == "__main__":
    test_log_likelihood_collapses_to_nll_when_only_alpha()
    test_finite_and_nonnegative_with_mixed_censoring()
    test_zero_loss_on_perfect_predictions()
    test_calibration_nonzero_gamma_changes_loss()
    print("OK")
