import numpy as np
from tests._characterization_data import make_cohort
from inference_survival import compute_hazard_ratio


def test_hr_greater_than_one_when_high_group_dies_first():
    _, time, event, _ = make_cohort(distinct=True, seed=4)
    group_high = time < np.median(time)
    hr = compute_hazard_ratio(time, event, group_high)
    assert hr > 1.0
