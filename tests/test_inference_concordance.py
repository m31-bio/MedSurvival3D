import numpy as np
from sksurv.metrics import concordance_index_censored
from tests._characterization_data import make_cohort
from medsurvival3d.inference.survival import sksurv_cindex


def test_cindex_matches_sksurv_reference():
    _, time, event, _ = make_cohort(distinct=True)
    risk = -time + np.random.default_rng(1).normal(scale=0.1, size=time.shape)
    got = sksurv_cindex(time, event, risk)
    want = concordance_index_censored(event.astype(bool), time.astype(float), risk)[0]
    assert abs(got - want) < 1e-9
