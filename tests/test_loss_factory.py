"""Unit tests for the survival-loss factory used inside BaseModel."""

import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from medsurvival3d.models.losses import (  # noqa: E402
    CoxPHLoss,
    DeepHitLoss,
    NLLSurvLoss,
    _reject_legacy_cox_loss_lambda,
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
    cfg = {"name": "deephit", "alpha": 0.3, "sigma": 0.2}
    name, fn = build_survival_criterion(cfg, num_time_bins=5)
    assert name == "deephit"
    assert isinstance(fn, DeepHitLoss)
    assert fn._loss.alpha == 0.3
    assert fn._loss.sigma == 0.2


def test_deephit_defaults():
    name, fn = build_survival_criterion({"name": "deephit"}, num_time_bins=5)
    assert fn._loss.alpha == 0.2
    assert fn._loss.sigma == 0.1


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
    try:
        _reject_legacy_cox_loss_lambda({"cox_loss_lambda": 0.09})
    except ValueError as exc:
        assert "cox_loss_lambda" in str(exc)
        assert "survival_loss" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_soft_logrank_defaults():
    from medsurvival3d.models.losses import SoftLogRankLoss  # noqa: WPS433
    name, fn = build_survival_criterion({"name": "soft_logrank"}, num_time_bins=5)
    assert name == "soft_logrank"
    assert isinstance(fn, SoftLogRankLoss)
    assert fn.lambda_balance == 0.01
    assert fn.min_frac == 0.20
    assert fn.max_frac == 0.80


def test_soft_logrank_reads_hyperparameters():
    from medsurvival3d.models.losses import SoftLogRankLoss  # noqa: WPS433
    cfg = {
        "name": "soft_logrank",
        "lambda_balance": 0.02,
        "min_frac": 0.25,
        "max_frac": 0.75,
    }
    name, fn = build_survival_criterion(cfg, num_time_bins=5)
    assert name == "soft_logrank"
    assert isinstance(fn, SoftLogRankLoss)
    assert fn.lambda_balance == 0.02
    assert fn.min_frac == 0.25
    assert fn.max_frac == 0.75


@pytest.mark.parametrize("name", [
    "nll", "cox", "deephit", "soft_logrank", "pmf", "mtlr", "bcesurv", "weibull", "pchazard",
])
def test_all_names_build(name):
    nm, crit = build_survival_criterion({"name": name}, num_time_bins=10)
    assert nm == name and crit is not None


# --- composite-loss config parsing/validation (dependency-free) ---------------
# These exercise _parse_composite, which validates and normalises a composite
# block WITHOUT constructing any pycox/torchsurv loss objects.

def test_parse_composite_basic():
    from medsurvival3d.models.losses import _parse_composite  # noqa: WPS433
    cfg = {
        "name": "composite",
        "primary": "nll",
        "components": [
            {"name": "nll", "weight": 1.0},
            {"name": "cox", "weight": 0.5, "reduction": "mean"},
        ],
    }
    components, primary = _parse_composite(cfg)
    assert primary == "nll"
    assert [c["name"] for c in components] == ["nll", "cox"]
    assert [c["weight"] for c in components] == [1.0, 0.5]
    # the original component mapping (with loss opts) is carried through
    assert components[1]["cfg"]["reduction"] == "mean"


def test_parse_composite_defaults_weight_to_one():
    from medsurvival3d.models.losses import _parse_composite  # noqa: WPS433
    cfg = {
        "name": "composite",
        "primary": "cox",
        "components": [{"name": "nll"}, {"name": "cox"}],
    }
    components, _ = _parse_composite(cfg)
    assert [c["weight"] for c in components] == [1.0, 1.0]


def _expect_value_error(cfg, needle):
    from medsurvival3d.models.losses import _parse_composite  # noqa: WPS433
    try:
        _parse_composite(cfg)
    except ValueError as exc:
        assert needle in str(exc), f"{needle!r} not in {str(exc)!r}"
    else:
        raise AssertionError(f"expected ValueError mentioning {needle!r}")


def test_parse_composite_empty_components_raises():
    _expect_value_error(
        {"name": "composite", "primary": "nll", "components": []}, "components"
    )


def test_parse_composite_missing_components_raises():
    _expect_value_error({"name": "composite", "primary": "nll"}, "components")


def test_parse_composite_unknown_member_raises():
    _expect_value_error(
        {"name": "composite", "primary": "nll",
         "components": [{"name": "nll"}, {"name": "wibble"}]},
        "wibble",
    )


def test_parse_composite_nested_composite_raises():
    _expect_value_error(
        {"name": "composite", "primary": "nll",
         "components": [{"name": "nll"}, {"name": "composite"}]},
        "composite",
    )


def test_parse_composite_duplicate_names_raises():
    _expect_value_error(
        {"name": "composite", "primary": "nll",
         "components": [{"name": "nll"}, {"name": "nll"}]},
        "duplicate",
    )


def test_parse_composite_primary_not_in_components_raises():
    _expect_value_error(
        {"name": "composite", "primary": "cox",
         "components": [{"name": "nll"}, {"name": "pmf"}]},
        "primary",
    )


def test_parse_composite_missing_primary_raises():
    _expect_value_error(
        {"name": "composite",
         "components": [{"name": "nll"}, {"name": "cox"}]},
        "primary",
    )


def test_parse_composite_negative_weight_raises():
    _expect_value_error(
        {"name": "composite", "primary": "nll",
         "components": [{"name": "nll", "weight": -0.5}, {"name": "cox"}]},
        "weight",
    )


def test_build_composite_validates_before_constructing():
    """A malformed composite must raise ValueError (validation) rather than
    fail later constructing pycox members. Exercises the factory branch without
    needing the heavy deps installed."""
    cfg = {"name": "composite", "primary": "nll",
           "components": [{"name": "nll"}, {"name": "wibble"}]}
    try:
        build_survival_criterion(cfg, num_time_bins=5)
    except ValueError as exc:
        assert "wibble" in str(exc)
    else:
        raise AssertionError("expected ValueError for unknown composite member")


if __name__ == "__main__":
    test_nll()
    test_cox()
    test_deephit_reads_hyperparameters()
    test_deephit_defaults()
    test_missing_block_defaults_to_nll()
    test_unknown_name_raises()
    test_legacy_cox_loss_lambda_rejected()
    test_soft_logrank_defaults()
    test_soft_logrank_reads_hyperparameters()
    print("OK")
