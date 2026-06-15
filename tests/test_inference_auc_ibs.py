import numpy as np
from sksurv.util import Surv
from sksurv.metrics import integrated_brier_score
from tests._characterization_data import make_cohort
from medsurvival3d.inference.survival import sksurv_ibs


def _prep():
    _, time, event, survival = make_cohort(seed=6)
    event = event.copy()
    event[time >= time.max() - 1] = 0
    y = Surv.from_arrays(event.astype(bool), time.astype(float))
    lo = max(1, int(time[event == 1].min()) + 1)
    hi = int(time.max())
    return time, event, survival, y, lo, hi


def test_ibs_matches_sksurv():
    time, event, survival, y, lo, hi = _prep()
    times = np.arange(lo, hi)
    got = sksurv_ibs(y, y, survival[:, times], times)
    want = integrated_brier_score(y, y, survival[:, times], times)
    assert abs(got - want) < 1e-9


