"""Tests for PCHazardLoss adapter and survival curve (Task 3.4)."""

import sys
from pathlib import Path

import torch
from pycox.models.loss import NLLPCHazardLoss

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from survival_utils import PCHazardLoss, build_survival_criterion  # noqa: E402
from tests._characterization_data import make_cohort, as_torch  # noqa: E402


def test_pchazard_matches_pycox():
    _, time, event, _ = make_cohort(num_bins=10, seed=14)
    phi = torch.randn(len(time), 10, requires_grad=True)
    t, e = as_torch(time, event)
    idx = t.long()
    ev = e.float()
    interval_frac = torch.rand(len(time))
    ours = PCHazardLoss()(phi, idx, ev, interval_frac)
    ref = NLLPCHazardLoss()(phi, idx, ev, interval_frac)
    assert torch.allclose(ours, ref, atol=1e-6)
    ours.backward()
    assert phi.grad is not None


def test_pchazard_factory_builds():
    nm, crit = build_survival_criterion({"name": "pchazard"}, num_time_bins=10)
    assert nm == "pchazard"
    assert isinstance(crit, PCHazardLoss)


def test_pchazard_loss_finite_mixed_censoring():
    _, time, event, _ = make_cohort(num_bins=8, seed=15)
    phi = torch.randn(len(time), 8)
    t, e = as_torch(time, event)
    interval_frac = torch.rand(len(time))
    out = PCHazardLoss()(phi, t.long(), e.float(), interval_frac)
    assert torch.isfinite(out)


def test_pchazard_surv_shape_and_range():
    from models.survival_head import logits_to_survival
    phi = torch.randn(6, 10)
    s = logits_to_survival("pchazard", phi)
    assert s.shape == (6, 10), s.shape
    assert (s >= -1e-5).all()
    assert (s <= 1.0 + 1e-5).all()


def test_pchazard_surv_monotone_nonincreasing():
    from models.survival_head import logits_to_survival
    phi = torch.randn(6, 10)
    s = logits_to_survival("pchazard", phi)
    diffs = s[:, 1:] - s[:, :-1]
    assert (diffs <= 1e-5).all(), f"max non-monotone diff: {diffs.max()}"


def test_pchazard_surv_oracle_pycox():
    """Pin _pchazard_surv to pycox's real PCHazard.predict_surv (sub=1).

    We instantiate a real pycox PCHazard with an identity Linear net so that
    model.predict(phi) == phi, then call model.predict_surv which runs pycox's
    own predict_hazard -> cumsum -> exp pipeline.  This is a true oracle: a
    future divergence from pycox's transform will fail here.
    """
    import numpy as np
    import torch.nn as nn
    from pycox.models import PCHazard
    from models.survival_head import logits_to_survival

    B, K = 5, 8
    phi = torch.randn(B, K)

    # Identity net: model.predict(phi.numpy()) == phi
    net = nn.Linear(K, K, bias=False)
    with torch.no_grad():
        net.weight.copy_(torch.eye(K))
    # duration_index has K+1 entries (bin boundaries) for K bins
    model = PCHazard(net, duration_index=np.arange(K + 1))
    model.sub = 1

    # predict_surv returns [B, K+1] with S(0)=1 at column 0; drop it -> [B, K]
    ref_full = model.predict_surv(phi.numpy(), numpy=False)
    ref = torch.as_tensor(np.asarray(ref_full))[:, 1:].float()

    ours = logits_to_survival("pchazard", phi)
    assert ours.shape == ref.shape, (ours.shape, ref.shape)
    assert torch.allclose(ours.float(), ref, atol=1e-4), \
        f"max abs diff: {(ours.float() - ref).abs().max()}"
