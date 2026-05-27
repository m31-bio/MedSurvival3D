"""Tests for derive_stratification_scores and _compute_stratification_metrics."""

import math
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# -------- Score derivation tests --------

def test_derive_scores_cox_returns_risks_directly():
    from survival_utils import derive_stratification_scores

    risks = np.array([0.1, 0.5, -0.3, 1.2])
    out = derive_stratification_scores("cox", risks=risks, survival_curves=None, landmark_bin_idx=2)
    np.testing.assert_array_equal(out, risks)


def test_derive_scores_soft_logrank_returns_risks_directly():
    from survival_utils import derive_stratification_scores

    risks = np.array([-2.0, 0.0, 0.5, 3.0])
    out = derive_stratification_scores("soft_logrank", risks=risks, survival_curves=None, landmark_bin_idx=0)
    np.testing.assert_array_equal(out, risks)


def test_derive_scores_nll_returns_one_minus_survival_at_landmark():
    from survival_utils import derive_stratification_scores

    survival_curves = np.array([
        [0.9, 0.7, 0.5, 0.2],  # patient 0
        [0.95, 0.85, 0.75, 0.65],  # patient 1
    ])
    out = derive_stratification_scores(
        "nll", risks=None, survival_curves=survival_curves, landmark_bin_idx=3,
    )
    np.testing.assert_allclose(out, np.array([0.8, 0.35]))


def test_derive_scores_deephit_returns_one_minus_survival_at_landmark():
    from survival_utils import derive_stratification_scores

    survival_curves = np.array([
        [0.8, 0.6, 0.4, 0.2],
        [1.0, 0.9, 0.8, 0.7],
    ])
    out = derive_stratification_scores(
        "deephit", risks=None, survival_curves=survival_curves, landmark_bin_idx=1,
    )
    np.testing.assert_allclose(out, np.array([0.4, 0.1]))


def test_derive_scores_unknown_loss_raises():
    from survival_utils import derive_stratification_scores

    with pytest.raises(ValueError, match="Unknown survival_loss_name"):
        derive_stratification_scores(
            "mystery", risks=np.array([0.1]), survival_curves=None, landmark_bin_idx=0,
        )
