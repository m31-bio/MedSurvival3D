import torch
from torchsurv.metrics.auc import Auc
from survival_utils import time_dependent_auc
from tests._characterization_data import make_cohort, as_torch


def test_train_auc_matches_torchsurv_per_landmark():
    _, time, event, survival = make_cohort(seed=16)
    event = event.copy(); event[time >= time.max() - 1] = 0
    s, t, e = as_torch(survival, time, event)
    landmarks = torch.tensor([4.0, 7.0, 10.0])
    cuts = torch.arange(survival.shape[1]).float()
    got = time_dependent_auc(s, t, e, landmarks, cuts)   # returns {landmark: auc}
    auc = Auc()
    for lm in landmarks.tolist():
        risk = 1.0 - s[:, int(lm)]
        want = float(auc(risk, e.bool(), t.float(), new_time=torch.tensor(float(lm))))
        assert abs(got[float(lm)] - want) < 1e-5
