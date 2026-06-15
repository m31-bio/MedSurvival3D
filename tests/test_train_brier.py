import torch
from torchsurv.metrics.brier_score import BrierScore
from medsurvival3d.evaluation.metrics import integrated_brier_score, integrated_brier_score_ipcw
from tests._characterization_data import make_cohort, as_torch


def test_train_ibs_finite_and_in_unit_range():
    _, time, event, survival = make_cohort(seed=17)
    event = event.copy(); event[time >= time.max() - 1] = 0
    s, t, e = as_torch(survival, time, event)
    v = integrated_brier_score(s, t, e)
    w = integrated_brier_score_ipcw(s, t, e)
    assert 0.0 <= v <= 1.0 and 0.0 <= w <= 1.0


def test_train_ibs_matches_torchsurv_brier_at_grid():
    _, time, event, survival = make_cohort(seed=18)
    event = event.copy(); event[time >= time.max() - 1] = 0
    s, t, e = as_torch(survival, time, event)
    lo = max(1, int(t[e.bool()].min()) + 1); hi = int(t.max())
    times = torch.arange(lo, hi)
    est = s[:, times]
    bs = BrierScore()
    bs(est, e.bool(), t.float(), new_time=times.float())
    want = float(bs.integral())
    from medsurvival3d.evaluation.metrics import _ibs_torchsurv  # helper added in step 3
    got = _ibs_torchsurv(s, t, e, times, weight=None)
    assert abs(got - want) < 1e-5
