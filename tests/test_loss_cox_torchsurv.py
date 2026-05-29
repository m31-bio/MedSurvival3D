import torch
from torchsurv.loss import cox
from survival_utils import CoxPHLoss
from tests._characterization_data import make_cohort, as_torch


def test_cox_adapter_matches_torchsurv():
    _, time, event, _ = make_cohort(distinct=True, seed=9)
    risk = torch.randn(len(time), requires_grad=True)
    t, e = as_torch(time, event)
    ours = CoxPHLoss()(risk, t, e)
    ref = cox.neg_partial_log_likelihood(risk.view(-1), e.bool(), t.float(),
                                         ties_method="efron", reduction="mean")
    assert torch.allclose(ours, ref, atol=1e-6)
    ours.backward(); assert risk.grad is not None


def test_cox_zero_when_no_events():
    risk = torch.randn(10, requires_grad=True)
    out = CoxPHLoss()(risk, torch.arange(10).float(), torch.zeros(10))
    assert float(out) == 0.0
