import torch
from pycox.models.loss import NLLLogistiHazardLoss
from survival_utils import NLLSurvLoss
from tests._characterization_data import make_cohort, as_torch


def test_nll_adapter_matches_pycox_directly():
    _, time, event, _ = make_cohort(num_bins=12, seed=8)
    logits = torch.randn(len(time), 12, requires_grad=True)
    t, e = as_torch(time, event)
    ours = NLLSurvLoss()(logits, t, e)
    ref = NLLLogistiHazardLoss()(logits, t.long(), e.float())
    assert torch.allclose(ours, ref, atol=1e-6)
    ours.backward()
    assert logits.grad is not None
