import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import base_model
from survival_utils import derive_stratification_scores


def test_all_curve_losses_have_tags():
    tags = base_model._SURVIVAL_LOSS_TAGS
    for name in ("nll", "deephit", "pmf", "mtlr", "bcesurv", "weibull", "cox", "soft_logrank"):
        assert name in tags, name


def test_derive_stratification_scores_curve_losses():
    rng = np.random.default_rng(0)
    risks = rng.random(8)
    # monotone-ish survival curves (descending along time axis)
    curves = np.sort(rng.random((8, 10)), axis=1)[:, ::-1]
    for name in ("nll", "deephit", "pmf", "mtlr", "bcesurv", "weibull"):
        s = derive_stratification_scores(name, risks, curves, landmark_bin_idx=3)
        assert s.shape[0] == 8, f"{name}: expected shape (8,), got {s.shape}"
