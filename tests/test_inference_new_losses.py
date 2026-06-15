"""Tests that pmf/mtlr/bcesurv/weibull work through the inference path.

Run with:
  /Users/bw/Documents/Safwat/survival/survival_env/bin/python -m pytest tests/test_inference_new_losses.py -v
"""
import numpy as np
import pytest

from medsurvival3d.inference.survival import CURVE_LOSSES, compute_metrics


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_outputs(n=30, num_bins=24, seed=0):
    """Build a minimal outputs dict matching run_split_inference's return value."""
    rng = np.random.default_rng(seed)
    # Monotone survival curves
    hazard = np.clip(rng.uniform(0.04, 0.12, size=(n, num_bins)), 1e-4, 0.6)
    survival = np.cumprod(1.0 - hazard, axis=1)
    risk = -survival.sum(axis=1)
    time = rng.integers(1, num_bins, size=n).astype(float)
    event = (rng.random(n) > 0.35).astype(int)
    # Ensure at least 2 events so IBS doesn't hit the degenerate branch.
    event[:4] = 1
    pmf = np.diff(np.concatenate([np.ones((n, 1)), survival], axis=1), axis=1) * -1
    pmf = np.clip(pmf, 0, None)
    return {
        "patient_id": [f"p{i}" for i in range(n)],
        "split": "val",
        "fold": 0,
        "time_bin": time.astype(int),
        "time": time,
        "event": event,
        "logits": rng.standard_normal((n, num_bins)),
        "hazards": hazard,
        "pmf": pmf,
        "survival": survival,
        "predicted_survival_time": rng.uniform(0, num_bins, size=n),
        "risk": risk,
        "p_high": rng.random(n),
    }


# ---------------------------------------------------------------------------
# 1. CURVE_LOSSES constant
# ---------------------------------------------------------------------------

def test_curve_losses_constant_has_new_losses():
    for name in ("nll", "deephit", "pmf", "mtlr", "bcesurv", "weibull"):
        assert name in CURVE_LOSSES, f"{name!r} missing from CURVE_LOSSES"


def test_curve_losses_does_not_include_cox():
    assert "cox" not in CURVE_LOSSES


def test_curve_losses_does_not_include_soft_logrank():
    assert "soft_logrank" not in CURVE_LOSSES


# ---------------------------------------------------------------------------
# 2. compute_metrics: new losses must not raise and must return c_index + ibs
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("loss_name", ["pmf", "mtlr", "bcesurv", "weibull"])
def test_compute_metrics_returns_expected_keys_for_new_losses(loss_name):
    outputs = _make_outputs()
    metrics = compute_metrics(outputs, loss_name)
    assert "c_index" in metrics
    assert "ibs" in metrics
    assert "n_patients" in metrics
    assert "n_events" in metrics


@pytest.mark.parametrize("loss_name", ["pmf", "mtlr", "bcesurv", "weibull"])
def test_compute_metrics_does_not_raise_for_new_losses(loss_name):
    outputs = _make_outputs()
    # Must complete without exception.
    result = compute_metrics(outputs, loss_name)
    assert isinstance(result, dict)


@pytest.mark.parametrize("loss_name", ["pmf", "mtlr", "bcesurv", "weibull"])
def test_compute_metrics_ibs_is_float_for_new_losses(loss_name):
    outputs = _make_outputs()
    metrics = compute_metrics(outputs, loss_name)
    # ibs may be nan on degenerate data, but must be a float (not raise).
    assert isinstance(metrics["ibs"], float)


# ---------------------------------------------------------------------------
# 3. compute_metrics: existing losses still pass (regression guard)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("loss_name", ["nll", "deephit"])
def test_compute_metrics_regression_existing_losses(loss_name):
    outputs = _make_outputs()
    metrics = compute_metrics(outputs, loss_name)
    assert "c_index" in metrics
    assert "ibs" in metrics
