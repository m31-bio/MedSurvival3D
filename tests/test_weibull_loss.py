import torch
from torchsurv.loss import weibull
from survival_utils import build_survival_criterion
from tests._characterization_data import make_cohort, as_torch


def test_weibull_matches_torchsurv():
    _, time, event, _ = make_cohort(distinct=True, seed=13)
    log_params = torch.randn(len(time), 2, requires_grad=True)
    t, e = as_torch(time, event)
    nm, crit = build_survival_criterion({"name": "weibull"}, num_time_bins=10)
    ours = crit(log_params, t, e)
    ref = weibull.neg_log_likelihood_weibull(log_params, e.bool(), t.float(), reduction="mean")
    assert torch.allclose(ours, ref, atol=1e-6)
    ours.backward(); assert log_params.grad is not None


def test_weibull_head_curve_monotone_unit_range():
    import torch
    from models.survival_head import SurvivalHead
    head = SurvivalHead(input_dim=16, num_time_bins=10, survival_loss_name="weibull")
    out = head(torch.randn(6, 16))
    s = out["survival"]
    assert s.shape == (6, 10)
    assert (s >= -1e-5).all() and (s <= 1 + 1e-5).all()
    assert (s[:, 1:] - s[:, :-1] <= 1e-5).all()  # nonincreasing
    assert "weibull_params" in out and out["weibull_params"].shape == (6, 2)
