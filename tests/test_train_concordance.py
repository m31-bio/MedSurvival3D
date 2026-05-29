import torch
from torchsurv.metrics.cindex import ConcordanceIndex
from survival_utils import concordance_index
from tests._characterization_data import make_cohort, as_torch


def test_train_cindex_matches_torchsurv():
    _, time, event, _ = make_cohort(distinct=True, seed=15)
    risk = torch.randn(len(time))
    t, e = as_torch(time, event)
    got = concordance_index(t, risk, e)              # (event_times, scores, event_observed)
    want = float(ConcordanceIndex()(risk, e.bool(), t.float()))
    assert abs(float(got) - want) < 1e-6
