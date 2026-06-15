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
    from medsurvival3d.evaluation.metrics import derive_stratification_scores

    risks = np.array([0.1, 0.5, -0.3, 1.2])
    out = derive_stratification_scores("cox", risks=risks, survival_curves=None, landmark_bin_idx=2)
    np.testing.assert_array_equal(out, risks)


def test_derive_scores_soft_logrank_returns_risks_directly():
    from medsurvival3d.evaluation.metrics import derive_stratification_scores

    risks = np.array([-2.0, 0.0, 0.5, 3.0])
    out = derive_stratification_scores("soft_logrank", risks=risks, survival_curves=None, landmark_bin_idx=0)
    np.testing.assert_array_equal(out, risks)


def test_derive_scores_nll_returns_one_minus_survival_at_landmark():
    from medsurvival3d.evaluation.metrics import derive_stratification_scores

    survival_curves = np.array([
        [0.9, 0.7, 0.5, 0.2],  # patient 0
        [0.95, 0.85, 0.75, 0.65],  # patient 1
    ])
    out = derive_stratification_scores(
        "nll", risks=None, survival_curves=survival_curves, landmark_bin_idx=3,
    )
    np.testing.assert_allclose(out, np.array([0.8, 0.35]))


def test_derive_scores_deephit_returns_one_minus_survival_at_landmark():
    from medsurvival3d.evaluation.metrics import derive_stratification_scores

    survival_curves = np.array([
        [0.8, 0.6, 0.4, 0.2],
        [1.0, 0.9, 0.8, 0.7],
    ])
    out = derive_stratification_scores(
        "deephit", risks=None, survival_curves=survival_curves, landmark_bin_idx=1,
    )
    np.testing.assert_allclose(out, np.array([0.4, 0.1]))


def test_derive_scores_unknown_loss_raises():
    from medsurvival3d.evaluation.metrics import derive_stratification_scores

    with pytest.raises(ValueError, match="Unknown survival_loss_name"):
        derive_stratification_scores(
            "mystery", risks=np.array([0.1]), survival_curves=None, landmark_bin_idx=0,
        )


# -------- Config plumbing tests --------

def _make_kwargs(**overrides):
    """Minimal kwargs needed to instantiate BaseModel for survival task."""
    base = dict(
        metric_computation_mode="epochwise",
        result_plot=False,
        metrics=[],
        name="test",
        lr=1e-3,
        weight_decay=0.0,
        optimizer="adam",
        nesterov=False,
        scheduler=None,
        T_max=1,
        warmstart=0,
        epochs=1,
        stochastic_depth=0.0,
        resnet_dropout=0.0,
        squeeze_excitation=False,
        undecay_norm=False,
        zero_init_residual=False,
        input_dim=3,
        input_channels=1,
        pretrained=False,
        # kwargs-accessed fields
        warmstart2=0,
        save_preds=False,
        finetune_method=None,
        num_time_bins=4,
        survival_loss={"name": "cox"},
        survival_cut_points_years=[1.0, 2.0, 3.0],
        survival_landmark_years=[1.0, 2.0, 3.0],
    )
    base.update(overrides)
    return base


def test_stratification_config_defaults():
    from medsurvival3d.training.trainer import BaseModel

    model = BaseModel(**_make_kwargs())
    assert model.survival_stratification_landmark_year == 5.0
    assert model.survival_stratification_quantile_range == (0.2, 0.8)


def test_stratification_config_explicit_values():
    from medsurvival3d.training.trainer import BaseModel

    model = BaseModel(**_make_kwargs(
        survival_stratification_landmark_year=3.0,
        survival_stratification_quantile_range=[0.25, 0.75],
    ))
    assert model.survival_stratification_landmark_year == 3.0
    assert model.survival_stratification_quantile_range == (0.25, 0.75)


def test_stratification_quantile_range_validation():
    from medsurvival3d.training.trainer import BaseModel

    with pytest.raises(ValueError, match="quantile_range"):
        BaseModel(**_make_kwargs(
            survival_stratification_quantile_range=[0.8, 0.2],
        ))


def test_soft_logrank_use_max_logrank_cutpoint_default_false():
    from medsurvival3d.training.trainer import BaseModel

    model = BaseModel(**_make_kwargs())
    assert model.soft_logrank_use_max_logrank_cutpoint is False


def test_soft_logrank_use_max_logrank_cutpoint_explicit_true():
    from medsurvival3d.training.trainer import BaseModel

    model = BaseModel(**_make_kwargs(soft_logrank_use_max_logrank_cutpoint=True))
    assert model.soft_logrank_use_max_logrank_cutpoint is True


def test_soft_logrank_use_max_logrank_cutpoint_coerces_truthy():
    from medsurvival3d.training.trainer import BaseModel

    model = BaseModel(**_make_kwargs(soft_logrank_use_max_logrank_cutpoint=1))
    assert model.soft_logrank_use_max_logrank_cutpoint is True


# -------- _compute_stratification_metrics integration tests --------

class _StubLogger:
    """Captures `model.log` calls for inspection."""

    def __init__(self):
        self.calls = {}

    def __call__(self, key, value, **_kwargs):
        # Convert torch tensors / numpy scalars to plain Python floats.
        if hasattr(value, "item"):
            value = value.item()
        self.calls[key] = float(value)


def _make_stub_model(loss_name, train_risks, val_risks, train_times, val_times,
                     train_events, val_events, train_curves=None, val_curves=None,
                     cut_points=(1.0, 2.0, 3.0), landmark_year=2.0,
                     soft_logrank_use_max_logrank_cutpoint=False):
    """Build a minimal object with the attrs _compute_stratification_metrics reads."""
    stub = types.SimpleNamespace()
    stub.survival_loss_name = loss_name
    stub.train_survival_risks = [torch.as_tensor(train_risks)] if train_risks is not None else []
    stub.val_survival_risks = [torch.as_tensor(val_risks)] if val_risks is not None else []
    stub.train_survival_curves = [torch.as_tensor(train_curves)] if train_curves is not None else []
    stub.val_survival_curves = [torch.as_tensor(val_curves)] if val_curves is not None else []
    stub.train_survival_continuous_times = [torch.as_tensor(train_times)] if train_times is not None else []
    stub.val_survival_continuous_times = [torch.as_tensor(val_times)] if val_times is not None else []
    stub.train_survival_events = [torch.as_tensor(train_events)] if train_events is not None else []
    stub.val_survival_events = [torch.as_tensor(val_events)] if val_events is not None else []
    stub.survival_cut_points_years = torch.tensor(cut_points, dtype=torch.float32)
    stub.survival_stratification_landmark_year = landmark_year
    stub.survival_stratification_quantile_range = (0.2, 0.8)
    stub.soft_logrank_use_max_logrank_cutpoint = soft_logrank_use_max_logrank_cutpoint
    stub._stratification_landmark_bin_warned = False
    # Bind the unbound method so `self._resolve_stratification_landmark_bin()`
    # in _compute_stratification_metrics resolves correctly on the stub.
    from medsurvival3d.training.trainer import BaseModel
    stub._resolve_stratification_landmark_bin = lambda: BaseModel._resolve_stratification_landmark_bin(stub)
    stub.trainer = types.SimpleNamespace(sanity_checking=False)
    stub.log = _StubLogger()
    return stub


def _invoke(stub):
    from medsurvival3d.training.trainer import BaseModel

    BaseModel._compute_stratification_metrics(stub)


def test_strat_metrics_soft_logrank_uses_zero_cutoff():
    rng = np.random.default_rng(0)
    train_risks = np.concatenate([rng.uniform(-3, -0.5, 20), rng.uniform(0.5, 3, 20)])
    train_times = np.concatenate([rng.uniform(5, 10, 20), rng.uniform(0.5, 2, 20)])
    train_events = np.ones(40)
    stub = _make_stub_model(
        "soft_logrank",
        train_risks=train_risks, val_risks=train_risks,
        train_times=train_times, val_times=train_times,
        train_events=train_events, val_events=train_events,
    )
    _invoke(stub)
    assert "Train/logrank_chi2" in stub.log.calls
    assert "Val/logrank_chi2" in stub.log.calls
    assert "Train/logrank_p" in stub.log.calls
    assert "Val/logrank_p" in stub.log.calls
    assert "Train/hazard_ratio" in stub.log.calls
    assert "Val/hazard_ratio" in stub.log.calls
    # With clearly separable data and cutoff at 0, chi^2 should be large.
    assert stub.log.calls["Train/logrank_chi2"] > 10.0


def test_strat_metrics_cox_uses_max_logrank_cutoff():
    rng = np.random.default_rng(1)
    train_risks = np.concatenate([rng.uniform(0.0, 0.4, 30), rng.uniform(0.6, 1.0, 30)])
    train_times = np.concatenate([rng.uniform(5, 10, 30), rng.uniform(0.5, 2, 30)])
    train_events = np.ones(60)
    stub = _make_stub_model(
        "cox",
        train_risks=train_risks, val_risks=train_risks,
        train_times=train_times, val_times=train_times,
        train_events=train_events, val_events=train_events,
    )
    _invoke(stub)
    assert stub.log.calls["Train/logrank_chi2"] > 10.0
    assert stub.log.calls["Train/hazard_ratio"] > 1.5


def test_strat_metrics_nll_uses_survival_at_landmark():
    # Two clearly separated groups by survival at year 2 (bin index 1 when
    # cut_points=[1,2,3]).
    train_curves = np.concatenate([
        np.tile([0.95, 0.9, 0.85, 0.8], (20, 1)),    # low risk
        np.tile([0.6, 0.3, 0.2, 0.1], (20, 1)),       # high risk
    ])
    train_times = np.concatenate([np.full(20, 8.0), np.full(20, 1.5)])
    train_events = np.ones(40)
    # For nll, the per-sample `risks` buffer (negative mean survival time) is
    # populated alongside the curves by _update_survival_metric_buffers, so we
    # populate it here too with placeholder zeros (its values are ignored by
    # derive_stratification_scores for nll/deephit).
    placeholder_risks = np.zeros(40)
    stub = _make_stub_model(
        "nll",
        train_risks=placeholder_risks, val_risks=placeholder_risks,
        train_times=train_times, val_times=train_times,
        train_events=train_events, val_events=train_events,
        train_curves=train_curves, val_curves=train_curves,
        cut_points=(1.0, 2.0, 3.0),
        landmark_year=2.0,
    )
    _invoke(stub)
    assert stub.log.calls["Train/logrank_chi2"] > 10.0


def test_strat_metrics_skipped_during_sanity_check():
    stub = _make_stub_model(
        "cox",
        train_risks=np.array([0.1, 0.5]),
        val_risks=np.array([0.1, 0.5]),
        train_times=np.array([1.0, 2.0]),
        val_times=np.array([1.0, 2.0]),
        train_events=np.array([1, 1]),
        val_events=np.array([1, 1]),
    )
    stub.trainer = types.SimpleNamespace(sanity_checking=True)
    _invoke(stub)
    assert stub.log.calls == {}


def test_strat_metrics_skipped_when_train_buffers_empty():
    """validate-only runs: no train data -> no metric logged."""
    stub = _make_stub_model(
        "cox",
        train_risks=None, val_risks=np.array([0.1, 0.5]),
        train_times=None, val_times=np.array([1.0, 2.0]),
        train_events=None, val_events=np.array([1, 1]),
    )
    _invoke(stub)
    assert stub.log.calls == {}


def test_strat_metrics_soft_logrank_flag_off_matches_zero_cutoff():
    """Flag off (default) preserves the existing fixed p_high>0.5 behavior."""
    rng = np.random.default_rng(0)
    train_risks = np.concatenate([rng.uniform(-3, -0.5, 20), rng.uniform(0.5, 3, 20)])
    train_times = np.concatenate([rng.uniform(5, 10, 20), rng.uniform(0.5, 2, 20)])
    train_events = np.ones(40)

    stub_off = _make_stub_model(
        "soft_logrank",
        train_risks=train_risks, val_risks=train_risks,
        train_times=train_times, val_times=train_times,
        train_events=train_events, val_events=train_events,
        soft_logrank_use_max_logrank_cutpoint=False,
    )
    _invoke(stub_off)

    # Baseline: compute chi2 directly at cutoff=0 (matches current branch).
    from medsurvival3d.inference.survival import compute_logrank_stat
    group_high = train_risks > 0.0
    expected_chi2, _ = compute_logrank_stat(train_times, train_events, group_high)
    assert math.isclose(
        stub_off.log.calls["Train/logrank_chi2"], expected_chi2, rel_tol=1e-6
    )


def test_strat_metrics_soft_logrank_flag_on_uses_scanned_cutoff():
    """Flag on switches soft_logrank to max_logrank_cutpoint on the logit."""
    # Build a fixture where the best cutpoint is clearly NOT at 0:
    # all logits are positive, so the head's 0.5 boundary would put everyone
    # in the high group (chi2 ~ 0). The data-driven scan should find a
    # cutpoint inside (0, max) that separates the two outcome groups.
    rng = np.random.default_rng(2)
    low_risk_logits = rng.uniform(0.1, 0.4, 30)   # smaller logits -> longer survival
    high_risk_logits = rng.uniform(0.6, 1.0, 30)  # larger logits -> shorter survival
    train_risks = np.concatenate([low_risk_logits, high_risk_logits])
    train_times = np.concatenate([rng.uniform(5, 10, 30), rng.uniform(0.5, 2, 30)])
    train_events = np.ones(60)

    stub_off = _make_stub_model(
        "soft_logrank",
        train_risks=train_risks, val_risks=train_risks,
        train_times=train_times, val_times=train_times,
        train_events=train_events, val_events=train_events,
        soft_logrank_use_max_logrank_cutpoint=False,
    )
    _invoke(stub_off)

    stub_on = _make_stub_model(
        "soft_logrank",
        train_risks=train_risks, val_risks=train_risks,
        train_times=train_times, val_times=train_times,
        train_events=train_events, val_events=train_events,
        soft_logrank_use_max_logrank_cutpoint=True,
    )
    _invoke(stub_on)

    # Flag-off: everyone above cutoff=0 -> degenerate split -> chi2 is NaN.
    assert math.isnan(stub_off.log.calls["Train/logrank_chi2"])
    # Flag-on: scanned cutpoint inside the data range -> large chi2.
    assert stub_on.log.calls["Train/logrank_chi2"] > 10.0


def test_strat_metrics_soft_logrank_flag_on_nan_cutpoint_yields_nan_metrics():
    """When max_logrank_cutpoint can't find a valid split (e.g. all-same
    scores or all-censored), the existing cutoff_is_nan guard should still
    produce NaN metrics."""
    # All-same logits -> no candidate cutpoint inside (q_lo, q_hi) yields a
    # non-degenerate split, so max_logrank_cutpoint returns NaN.
    train_risks = np.zeros(40)
    train_times = np.linspace(1.0, 10.0, 40)
    train_events = np.ones(40)
    stub = _make_stub_model(
        "soft_logrank",
        train_risks=train_risks, val_risks=train_risks,
        train_times=train_times, val_times=train_times,
        train_events=train_events, val_events=train_events,
        soft_logrank_use_max_logrank_cutpoint=True,
    )
    _invoke(stub)
    assert math.isnan(stub.log.calls["Train/logrank_chi2"])
    assert math.isnan(stub.log.calls["Val/logrank_chi2"])
    assert math.isnan(stub.log.calls["Train/hazard_ratio"])


# -------- on_save_checkpoint bundling tests --------

def test_on_save_checkpoint_bundles_finite_cutpoint():
    from medsurvival3d.training.trainer import BaseModel

    stub = types.SimpleNamespace()
    stub._stratification_cutpoint = 0.42
    checkpoint = {"state_dict": {}}
    BaseModel.on_save_checkpoint(stub, checkpoint)
    assert checkpoint["stratification_cutpoint"] == 0.42


def test_on_save_checkpoint_omits_when_cutpoint_is_none():
    from medsurvival3d.training.trainer import BaseModel

    stub = types.SimpleNamespace()
    stub._stratification_cutpoint = None
    checkpoint = {"state_dict": {}}
    BaseModel.on_save_checkpoint(stub, checkpoint)
    assert "stratification_cutpoint" not in checkpoint


def test_on_save_checkpoint_omits_when_attribute_missing():
    """If validation never ran, the attribute is absent and no key is added."""
    from medsurvival3d.training.trainer import BaseModel

    stub = types.SimpleNamespace()
    checkpoint = {"state_dict": {}}
    BaseModel.on_save_checkpoint(stub, checkpoint)
    assert "stratification_cutpoint" not in checkpoint


def test_strat_metrics_stashes_cutpoint_for_soft_logrank_flag_off():
    """Flag off uses cutoff=0.0; that value must be stashed on the module."""
    rng = np.random.default_rng(0)
    train_risks = np.concatenate([rng.uniform(-3, -0.5, 20), rng.uniform(0.5, 3, 20)])
    train_times = np.concatenate([rng.uniform(5, 10, 20), rng.uniform(0.5, 2, 20)])
    train_events = np.ones(40)
    stub = _make_stub_model(
        "soft_logrank",
        train_risks=train_risks, val_risks=train_risks,
        train_times=train_times, val_times=train_times,
        train_events=train_events, val_events=train_events,
        soft_logrank_use_max_logrank_cutpoint=False,
    )
    _invoke(stub)
    assert stub._stratification_cutpoint == 0.0


def test_strat_metrics_stashes_cutpoint_for_soft_logrank_flag_on():
    """Flag on scans for max-logrank cutpoint; that scanned value must be stashed."""
    rng = np.random.default_rng(2)
    low = rng.uniform(0.1, 0.4, 30)
    high = rng.uniform(0.6, 1.0, 30)
    train_risks = np.concatenate([low, high])
    train_times = np.concatenate([rng.uniform(5, 10, 30), rng.uniform(0.5, 2, 30)])
    train_events = np.ones(60)
    stub = _make_stub_model(
        "soft_logrank",
        train_risks=train_risks, val_risks=train_risks,
        train_times=train_times, val_times=train_times,
        train_events=train_events, val_events=train_events,
        soft_logrank_use_max_logrank_cutpoint=True,
    )
    _invoke(stub)
    # Scanned cutpoint lives strictly between the two clusters (0.4, 0.6).
    assert isinstance(stub._stratification_cutpoint, float)
    assert 0.4 < stub._stratification_cutpoint < 0.7


def test_strat_metrics_stashes_none_when_cutpoint_is_nan():
    """When max_logrank_cutpoint returns NaN, stash None (skip-in-checkpoint signal)."""
    train_risks = np.zeros(40)
    train_times = np.linspace(1.0, 10.0, 40)
    train_events = np.ones(40)
    stub = _make_stub_model(
        "soft_logrank",
        train_risks=train_risks, val_risks=train_risks,
        train_times=train_times, val_times=train_times,
        train_events=train_events, val_events=train_events,
        soft_logrank_use_max_logrank_cutpoint=True,
    )
    _invoke(stub)
    assert stub._stratification_cutpoint is None


def test_strat_metrics_stashes_cutpoint_for_cox():
    """Non-soft_logrank losses already scan; the scanned cutoff must be stashed."""
    rng = np.random.default_rng(1)
    train_risks = np.concatenate([rng.uniform(0.0, 0.4, 30), rng.uniform(0.6, 1.0, 30)])
    train_times = np.concatenate([rng.uniform(5, 10, 30), rng.uniform(0.5, 2, 30)])
    train_events = np.ones(60)
    stub = _make_stub_model(
        "cox",
        train_risks=train_risks, val_risks=train_risks,
        train_times=train_times, val_times=train_times,
        train_events=train_events, val_events=train_events,
    )
    _invoke(stub)
    assert isinstance(stub._stratification_cutpoint, float)
    assert 0.4 < stub._stratification_cutpoint < 0.7


def test_strat_metrics_flag_ignored_for_cox():
    """Setting the soft_logrank flag has no effect when the loss is cox."""
    rng = np.random.default_rng(1)
    train_risks = np.concatenate([rng.uniform(0.0, 0.4, 30), rng.uniform(0.6, 1.0, 30)])
    train_times = np.concatenate([rng.uniform(5, 10, 30), rng.uniform(0.5, 2, 30)])
    train_events = np.ones(60)

    stub_off = _make_stub_model(
        "cox",
        train_risks=train_risks, val_risks=train_risks,
        train_times=train_times, val_times=train_times,
        train_events=train_events, val_events=train_events,
        soft_logrank_use_max_logrank_cutpoint=False,
    )
    stub_on = _make_stub_model(
        "cox",
        train_risks=train_risks, val_risks=train_risks,
        train_times=train_times, val_times=train_times,
        train_events=train_events, val_events=train_events,
        soft_logrank_use_max_logrank_cutpoint=True,
    )
    _invoke(stub_off)
    _invoke(stub_on)

    assert math.isclose(
        stub_off.log.calls["Train/logrank_chi2"],
        stub_on.log.calls["Train/logrank_chi2"],
        rel_tol=1e-9,
    )
    assert math.isclose(
        stub_off.log.calls["Train/hazard_ratio"],
        stub_on.log.calls["Train/hazard_ratio"],
        rel_tol=1e-9,
    )
