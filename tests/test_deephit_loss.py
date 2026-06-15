"""Sanity tests for survival_utils.DeepHitLoss (pycox adapter)."""

import sys
from pathlib import Path

import torch
from pycox.models.loss import DeepHitSingleLoss
from pycox.models.data import pair_rank_mat

# Make `SSL3D_survival/` importable when running the test directly.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from medsurvival3d.models.losses import DeepHitLoss  # noqa: E402
from tests._characterization_data import make_cohort, as_torch


def _rank_mat(idx, ev, dtype, device):
    return torch.as_tensor(pair_rank_mat(idx.cpu().numpy(), ev.cpu().numpy()),
                           dtype=dtype, device=device)


def test_deephit_matches_pycox():
    _, time, event, _ = make_cohort(num_bins=12, seed=10)
    phi = torch.randn(len(time), 12, requires_grad=True)
    t, e = as_torch(time, event)
    idx, ev = t.long(), e.float()
    ours = DeepHitLoss(alpha=0.2, sigma=0.1)(phi, idx, ev)
    ref = DeepHitSingleLoss(alpha=0.2, sigma=0.1)(phi, idx, ev, _rank_mat(idx, ev, phi.dtype, phi.device))
    assert torch.allclose(ours, ref, atol=1e-6)
    ours.backward(); assert phi.grad is not None


def test_deephit_finite_mixed_censoring():
    _, time, event, _ = make_cohort(num_bins=8, seed=11)
    phi = torch.randn(len(time), 8)
    t, e = as_torch(time, event)
    out = DeepHitLoss()(phi, t.long(), e.float())
    assert torch.isfinite(out)
