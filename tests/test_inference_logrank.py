import numpy as np
from lifelines.statistics import logrank_test
from tests._characterization_data import make_cohort
from medsurvival3d.inference.survival import compute_logrank_stat


def test_logrank_matches_lifelines():
    _, time, event, _ = make_cohort(distinct=True)
    rng = np.random.default_rng(2)
    group_high = (rng.random(time.shape) > 0.5)
    stat, p = compute_logrank_stat(time, event, group_high)
    lr = logrank_test(
        time[group_high], time[~group_high],
        event_observed_A=event[group_high], event_observed_B=event[~group_high],
    )
    assert abs(stat - lr.test_statistic) < 1e-6
    assert abs(p - lr.p_value) < 1e-9
