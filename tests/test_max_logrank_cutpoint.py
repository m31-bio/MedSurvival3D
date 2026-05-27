"""Unit tests for max_logrank_cutpoint (training-time best-split helper)."""

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from survival_utils import max_logrank_cutpoint  # noqa: E402


def _separable_dataset(n_per_group=40, seed=0):
    """Two clearly separated groups; optimal cutoff lies near the middle."""
    rng = np.random.default_rng(seed)
    low_scores = rng.uniform(0.0, 0.4, size=n_per_group)
    high_scores = rng.uniform(0.6, 1.0, size=n_per_group)
    scores = np.concatenate([low_scores, high_scores])
    # Low group: long survival (5-10 years); high group: short (0.5-2 years).
    low_times = rng.uniform(5.0, 10.0, size=n_per_group)
    high_times = rng.uniform(0.5, 2.0, size=n_per_group)
    times = np.concatenate([low_times, high_times])
    events = np.ones_like(times)
    return scores, times, events


def test_recovers_cutoff_on_separable_groups():
    scores, times, events = _separable_dataset()
    cutoff = max_logrank_cutpoint(scores, times, events, q_lo=0.2, q_hi=0.8)
    # Optimal cutoff lands just above the 0.4-0.6 gap, at min(high_scores) ≈ 0.62
    # (the threshold that cleanly separates the two groups maximizes chi²).
    assert 0.4 <= cutoff <= 0.65


def test_returned_cutoff_inside_quantile_bracket():
    scores = np.arange(100, dtype=float)  # 0..99
    times = np.linspace(1.0, 10.0, 100)
    events = np.ones(100)
    cutoff = max_logrank_cutpoint(scores, times, events, q_lo=0.2, q_hi=0.8)
    # 20th percentile is ~19, 80th percentile is ~79 (numpy default linear interp).
    assert 19.0 <= cutoff <= 80.0


def test_nan_when_all_scores_equal():
    scores = np.ones(50)
    times = np.linspace(1.0, 5.0, 50)
    events = np.ones(50)
    cutoff = max_logrank_cutpoint(scores, times, events)
    assert math.isnan(cutoff)


def test_nan_when_zero_events():
    scores = np.linspace(0.0, 1.0, 30)
    times = np.linspace(1.0, 5.0, 30)
    events = np.zeros(30)
    cutoff = max_logrank_cutpoint(scores, times, events)
    assert math.isnan(cutoff)


def test_candidate_cap_limits_evaluations(monkeypatch):
    """Helper must cap chi^2 evaluations even for large unique-score inputs."""
    import survival_utils

    call_count = {"n": 0}
    original = survival_utils._logrank_chi2  # internal name used by max_logrank_cutpoint

    def counted(*args, **kwargs):
        call_count["n"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(survival_utils, "_logrank_chi2", counted)

    scores = np.linspace(0.0, 1.0, 5000)
    times = np.random.default_rng(0).uniform(0.5, 10.0, size=5000)
    events = np.ones(5000)
    max_logrank_cutpoint(scores, times, events, q_lo=0.2, q_hi=0.8)
    assert call_count["n"] <= survival_utils.MAX_CANDIDATES


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
