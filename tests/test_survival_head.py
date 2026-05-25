"""Output-shape checks for the universal SurvivalHead."""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.survival_head import SurvivalHead  # noqa: E402


REQUIRED_KEYS = {"logits", "hazard", "pmf", "risk", "survival", "survival_time"}


def _features(batch=4, dim=32):
    return torch.randn(batch, dim)


def _check_shapes(out, batch, bins):
    assert REQUIRED_KEYS.issubset(out.keys()), out.keys()
    assert out["logits"].shape == (batch, bins)
    assert out["hazard"].shape == (batch, bins)
    assert out["pmf"].shape == (batch, bins)
    assert out["risk"].shape == (batch,)
    assert out["survival"].shape == (batch, bins)
    assert out["survival_time"].shape == (batch,)


def test_nll_mode_shapes():
    head = SurvivalHead(input_dim=32, num_time_bins=5, survival_loss_name="nll")
    head.eval()
    _check_shapes(head(_features()), batch=4, bins=5)


def test_deephit_mode_shapes_and_pmf_sums_to_one():
    head = SurvivalHead(input_dim=32, num_time_bins=5, survival_loss_name="deephit")
    head.eval()
    out = head(_features())
    _check_shapes(out, batch=4, bins=5)
    assert torch.allclose(out["pmf"].sum(dim=1), torch.ones(4), atol=1e-5)


def test_cox_mode_shapes():
    head = SurvivalHead(input_dim=32, num_time_bins=5, survival_loss_name="cox")
    head.eval()
    _check_shapes(head(_features()), batch=4, bins=5)


def test_survival_is_loss_appropriate():
    feats = _features()
    nll_out = SurvivalHead(input_dim=32, num_time_bins=5, survival_loss_name="nll").eval()(feats)
    deephit_out = SurvivalHead(input_dim=32, num_time_bins=5, survival_loss_name="deephit").eval()(feats)

    # NLL: survival = cumprod(1 - hazard) — monotonically non-increasing.
    diffs = nll_out["survival"][:, 1:] - nll_out["survival"][:, :-1]
    assert (diffs <= 1e-6).all()

    # DeepHit: survival = 1 - cumsum(pmf) — also non-increasing, ends ~0.
    diffs = deephit_out["survival"][:, 1:] - deephit_out["survival"][:, :-1]
    assert (diffs <= 1e-6).all()


if __name__ == "__main__":
    test_nll_mode_shapes()
    test_deephit_mode_shapes_and_pmf_sums_to_one()
    test_cox_mode_shapes()
    test_survival_is_loss_appropriate()
    print("OK")
