"""Unit tests for the survival-loss factory used inside BaseModel."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from survival_utils import (  # noqa: E402
    CoxPHLoss,
    DeepHitLoss,
    NLLSurvLoss,
    build_survival_criterion,
)


def test_nll():
    name, fn = build_survival_criterion({"name": "nll"}, num_time_bins=5)
    assert name == "nll"
    assert isinstance(fn, NLLSurvLoss)


def test_cox():
    name, fn = build_survival_criterion({"name": "cox"}, num_time_bins=5)
    assert name == "cox"
    assert isinstance(fn, CoxPHLoss)


def test_deephit_reads_hyperparameters():
    cfg = {"name": "deephit", "alpha": 2.0, "beta": 0.3, "gamma": 0.1, "sigma": 0.2}
    name, fn = build_survival_criterion(cfg, num_time_bins=5)
    assert name == "deephit"
    assert isinstance(fn, DeepHitLoss)
    assert fn.alpha == 2.0
    assert fn.beta == 0.3
    assert fn.gamma == 0.1
    assert fn.sigma == 0.2
    assert fn.num_time_bins == 5


def test_deephit_defaults():
    name, fn = build_survival_criterion({"name": "deephit"}, num_time_bins=5)
    assert fn.alpha == 1.0
    assert fn.beta == 0.5
    assert fn.gamma == 0.0
    assert fn.sigma == 0.1


def test_missing_block_defaults_to_nll():
    name, fn = build_survival_criterion(None, num_time_bins=5)
    assert name == "nll"
    assert isinstance(fn, NLLSurvLoss)


def test_unknown_name_raises():
    try:
        build_survival_criterion({"name": "wibble"}, num_time_bins=5)
    except ValueError as exc:
        assert "wibble" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_legacy_cox_loss_lambda_rejected():
    """Configs still carrying cox_loss_lambda must raise with a migration hint."""
    from base_model import _reject_legacy_cox_loss_lambda

    try:
        _reject_legacy_cox_loss_lambda({"cox_loss_lambda": 0.09})
    except ValueError as exc:
        assert "cox_loss_lambda" in str(exc)
        assert "survival_loss" in str(exc)
    else:
        raise AssertionError("expected ValueError")


if __name__ == "__main__":
    test_nll()
    test_cox()
    test_deephit_reads_hyperparameters()
    test_deephit_defaults()
    test_missing_block_defaults_to_nll()
    test_unknown_name_raises()
    test_legacy_cox_loss_lambda_rejected()
    print("OK")
