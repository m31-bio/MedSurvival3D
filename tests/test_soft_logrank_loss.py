"""Unit tests for the differentiable soft log-rank loss."""

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from medsurvival3d.models.losses import (  # noqa: E402
    group_balance_penalty,
    soft_logrank_loss,
)


def _example_batch():
    # 6 patients, 3 events at times {1.0, 2.0, 3.0}, 3 censored at later times.
    time = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    event = torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    return time, event


def test_loss_is_scalar_with_grad():
    time, event = _example_batch()
    p_high = torch.sigmoid(torch.randn(6, requires_grad=True))
    loss = soft_logrank_loss(p_high, time, event)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    loss.backward()
    # gradient flows back through p_high's source.


def test_loss_is_zero_grad_safe_when_no_events():
    p_high = torch.sigmoid(torch.randn(4, requires_grad=True))
    time = torch.tensor([1.0, 2.0, 3.0, 4.0])
    event = torch.zeros(4)
    loss = soft_logrank_loss(p_high, time, event)
    assert loss.ndim == 0
    assert float(loss) == 0.0
    # backward must not error
    loss.backward()


def test_loss_handles_ties_at_event_time():
    # Two simultaneous events at t=1.0
    time = torch.tensor([1.0, 1.0, 2.0, 3.0])
    event = torch.tensor([1.0, 1.0, 1.0, 0.0])
    p_high = torch.tensor([0.9, 0.8, 0.2, 0.1])
    loss = soft_logrank_loss(p_high, time, event)
    assert torch.isfinite(loss)


def test_loss_decreases_when_high_risk_predicts_early_events():
    # Patients 0-2 event early; if p_high is high for them, loss should be
    # lower (more negative signal -> -signal smaller) than the reversed case.
    time, event = _example_batch()
    good = torch.tensor([0.9, 0.9, 0.9, 0.1, 0.1, 0.1])
    bad = torch.tensor([0.1, 0.1, 0.1, 0.9, 0.9, 0.9])
    assert soft_logrank_loss(good, time, event) < soft_logrank_loss(bad, time, event)


def test_loss_divides_by_unique_event_count():
    # With one unique event time, the loss == -(o_high - e_high)/1.
    time = torch.tensor([1.0, 1.0, 2.0, 3.0])
    event = torch.tensor([1.0, 0.0, 0.0, 0.0])
    p_high = torch.tensor([0.8, 0.2, 0.5, 0.5])
    # At t=1.0: at_risk={0,1,2,3} n=4; events={0} d=1; n_high=2.0; o_high=0.8;
    # e_high = 1 * 2.0 / 4 = 0.5; signal = 0.8 - 0.5 = 0.3; loss = -0.3 / 1
    expected = torch.tensor(-0.3)
    assert torch.allclose(soft_logrank_loss(p_high, time, event), expected, atol=1e-6)


def test_loss_device_consistency_cpu():
    time, event = _example_batch()
    p_high = torch.sigmoid(torch.randn(6))
    loss = soft_logrank_loss(p_high, time, event)
    assert loss.device == p_high.device


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
def test_loss_device_consistency_cuda():
    time, event = _example_batch()
    time = time.cuda()
    event = event.cuda()
    p_high = torch.sigmoid(torch.randn(6)).cuda()
    loss = soft_logrank_loss(p_high, time, event)
    assert loss.device.type == "cuda"


def test_balance_zero_inside_range():
    assert float(group_balance_penalty(torch.tensor([0.5] * 10))) == 0.0
    assert float(group_balance_penalty(torch.tensor([0.2] * 10))) == 0.0
    assert float(group_balance_penalty(torch.tensor([0.8] * 10))) == 0.0


def test_balance_quadratic_below_min():
    # mean = 0.10, min_frac = 0.20 -> penalty = 0.01
    out = float(group_balance_penalty(torch.tensor([0.1] * 10)))
    assert abs(out - 0.01) < 1e-6


def test_balance_quadratic_above_max():
    out = float(group_balance_penalty(torch.tensor([0.9] * 10)))
    assert abs(out - 0.01) < 1e-6


def test_balance_uses_custom_bounds():
    out = float(group_balance_penalty(
        torch.tensor([0.5] * 10),
        min_frac=0.6,
        max_frac=0.7,
    ))
    assert abs(out - 0.01) < 1e-6  # (0.6 - 0.5)^2


if __name__ == "__main__":
    import subprocess
    subprocess.run(["pytest", "-x", "-v", __file__])
