"""Unit tests for log-rank statistic and HR helpers used in soft_logrank inference."""

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from inference_survival import (  # noqa: E402
    _checkpoint_stratification_cutpoint,
    compute_hazard_ratio,
    compute_logrank_stat,
)


def test_logrank_stat_zero_when_groups_identical():
    time = np.array([1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0])
    event = np.array([1, 1, 0, 1, 1, 1, 0, 1])
    group_high = np.array([True, True, True, True, False, False, False, False])
    chi2, _ = compute_logrank_stat(time, event, group_high)
    assert chi2 < 1e-6


def test_logrank_stat_positive_when_groups_differ():
    # High-risk events all at t=1, low-risk events all at t=5.
    time = np.array([1.0, 1.0, 1.0, 5.0, 5.0, 5.0])
    event = np.array([1, 1, 1, 1, 1, 1])
    group_high = np.array([True, True, True, False, False, False])
    chi2, p = compute_logrank_stat(time, event, group_high)
    assert chi2 >= 5.0
    assert 0.0 <= p <= 1.0


def test_logrank_stat_handles_empty_group():
    time = np.array([1.0, 2.0, 3.0])
    event = np.array([1, 1, 0])
    group_high = np.zeros(3, dtype=bool)
    chi2, p = compute_logrank_stat(time, event, group_high)
    assert math.isnan(chi2) or chi2 == 0.0


def test_hazard_ratio_greater_than_one_when_high_risk_dies_first():
    time = np.array([1.0, 1.0, 5.0, 5.0])
    event = np.array([1, 1, 1, 1])
    group_high = np.array([True, True, False, False])
    hr = compute_hazard_ratio(time, event, group_high)
    assert hr > 1.0


def test_hazard_ratio_nan_when_no_events():
    time = np.array([1.0, 2.0, 3.0])
    event = np.array([0, 0, 0])
    group_high = np.array([True, False, True])
    hr = compute_hazard_ratio(time, event, group_high)
    assert math.isnan(hr)


def test_checkpoint_cutpoint_returns_float_when_present():
    assert _checkpoint_stratification_cutpoint({"stratification_cutpoint": 0.37}) == 0.37


def test_checkpoint_cutpoint_returns_none_when_key_absent():
    assert _checkpoint_stratification_cutpoint({"state_dict": {}}) is None


def test_checkpoint_cutpoint_returns_none_when_value_is_nan():
    assert _checkpoint_stratification_cutpoint(
        {"stratification_cutpoint": float("nan")}
    ) is None


def test_checkpoint_cutpoint_returns_none_when_value_is_none():
    assert _checkpoint_stratification_cutpoint({"stratification_cutpoint": None}) is None


if __name__ == "__main__":
    import subprocess
    subprocess.run(["pytest", "-x", "-v", __file__])
