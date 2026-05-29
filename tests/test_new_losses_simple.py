import torch
from survival_utils import build_survival_criterion
from tests._characterization_data import make_cohort, as_torch


def _run(name):
    _, time, event, _ = make_cohort(num_bins=10, seed=12)
    t, e = as_torch(time, event)
    nm, crit = build_survival_criterion({"name": name}, num_time_bins=10)
    phi = torch.randn(len(time), 10, requires_grad=True)
    out = crit(phi, t.long(), e.float())
    assert torch.isfinite(out)
    out.backward(); assert phi.grad is not None
    return nm


def test_pmf():     assert _run("pmf") == "pmf"
def test_mtlr():    assert _run("mtlr") == "mtlr"
def test_bcesurv(): assert _run("bcesurv") == "bcesurv"
