"""Tests pinning per-loss survival curve derivations."""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_nll_curve_is_cumprod_hazard():
    phi = torch.randn(5, 8)
    from medsurvival3d.models.heads.survival_head import logits_to_survival
    s = logits_to_survival("nll", phi)
    assert torch.allclose(s, torch.cumprod(1 - torch.sigmoid(phi), dim=1), atol=1e-6)


def test_pmf_curve_is_one_minus_cumsum_softmax():
    phi = torch.randn(5, 8)
    from medsurvival3d.models.heads.survival_head import logits_to_survival
    s = logits_to_survival("pmf", phi)
    assert torch.allclose(s, 1 - torch.softmax(phi, 1).cumsum(1), atol=1e-6)


def test_bcesurv_curve_is_sigmoid():
    phi = torch.randn(5, 8)
    from medsurvival3d.models.heads.survival_head import logits_to_survival
    s = logits_to_survival("bcesurv", phi)
    assert torch.allclose(s, torch.sigmoid(phi), atol=1e-6)


def test_curves_are_in_unit_range():
    from medsurvival3d.models.heads.survival_head import logits_to_survival
    for name in ("nll", "pmf", "deephit", "bcesurv", "mtlr", "pchazard"):
        phi = torch.randn(6, 10)
        s = logits_to_survival(name, phi)
        assert s.shape[0] == 6, name
        assert (s >= -1e-5).all() and (s <= 1 + 1e-5).all(), name


def test_curves_are_monotone_nonincreasing():
    from medsurvival3d.models.heads.survival_head import logits_to_survival
    # bcesurv applies sigmoid independently per time-bin — not constrained to be
    # monotone by design (matches pycox BceSurv behaviour).
    for name in ("nll", "pmf", "deephit", "mtlr", "pchazard"):
        phi = torch.randn(6, 10)
        s = logits_to_survival(name, phi)
        assert (s[:, 1:] - s[:, :-1] <= 1e-5).all(), name


def test_mtlr_curve_matches_pycox_oracle():
    """Pin our MTLR transform to the pycox reference implementation.

    The reference is pycox's own MTLR.predict_surv_df code path (not a
    hand-copied formula).  We build a pycox MTLR whose net is an identity
    Linear(8, 8) so self.predict(phi) == phi, then let pycox run its full
    MTLR.predict_pmf -> PMFBase.predict_surv -> predict_surv_df chain.
    A future change to _mtlr_surv that diverges from pycox will fail here.
    """
    import numpy as np
    import torch.nn as nn
    from pycox.models import MTLR
    from medsurvival3d.models.heads.survival_head import logits_to_survival

    phi = torch.randn(5, 8)

    # Identity net (Linear with eye weights) so pycox's net forward returns phi unchanged.
    net = nn.Linear(8, 8, bias=False)
    with torch.no_grad():
        net.weight.copy_(torch.eye(8))
    model = MTLR(net, duration_index=np.arange(8))

    # predict_surv_df routes through: MTLR.predict_pmf (cumsum_reverse + pad_col
    # + softmax) -> PMFBase.predict_surv (1 - pmf.cumsum) -> DataFrame transpose.
    surv_df = model.predict_surv_df(phi.numpy())
    # surv_df is indexed by duration (rows=times, cols=subjects); transpose -> [n, t]
    ref = torch.as_tensor(surv_df.values.T).float()

    ours = logits_to_survival("mtlr", phi)
    assert ours.shape == ref.shape
    assert torch.allclose(ours.float(), ref, atol=1e-4)


def test_pchazard_curve_matches_pycox_oracle():
    """Pin _pchazard_surv to pycox's real PCHazard.predict_surv (sub=1).

    We instantiate a real pycox PCHazard with an identity Linear net so that
    model.predict(phi) == phi, then call model.predict_surv which runs pycox's
    own predict_hazard -> cumsum -> exp pipeline.  This is a true oracle: a
    future divergence from pycox's transform will fail here.
    """
    import numpy as np
    import torch.nn as nn
    from pycox.models import PCHazard
    from medsurvival3d.models.heads.survival_head import logits_to_survival

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
