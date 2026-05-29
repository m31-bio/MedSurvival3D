import numpy as np
from lifelines import KaplanMeierFitter
from tests._characterization_data import make_cohort
from inference_survival import km_survival_at


def test_km_at_horizon_matches_lifelines():
    _, time, event, _ = make_cohort(distinct=True, seed=5)
    horizon = float(np.median(time))
    got = km_survival_at(time, event, horizon)
    kmf = KaplanMeierFitter().fit(time, event)
    want = float(kmf.predict(horizon))
    assert abs(got - want) < 1e-6


def test_km_step_curve_matches_lifelines():
    from inference_survival import km_step_curve
    _, time, event, _ = make_cohort(distinct=True, seed=5)
    t, s = km_step_curve(time, event)
    kmf = KaplanMeierFitter().fit(time, event)
    sf = kmf.survival_function_
    assert np.allclose(t, sf.index.to_numpy())
    assert np.allclose(s, sf.iloc[:, 0].to_numpy())
