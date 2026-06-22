"""Regression: pycox-based survival losses must accept float16 logits.

Under ``precision: 16-mixed`` autocast, ``SurvivalHead`` emits float16 logits.
The pycox loss wrappers (nll/pmf/mtlr/bcesurv/pchazard) hard-cast ``event`` to
float32 while leaving logits at autocast dtype, so pycox's internal
``zeros_like(phi).scatter(1, idx, events)`` raised
``RuntimeError: scatter(): Expected self.dtype to be equal to src.dtype``.
Surfaced 2026-06-17 while verifying the composite loss (nll primary) on GPU.

deephit is included for coverage; it already casts events to the logit dtype
and so was never affected.
"""

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from medsurvival3d.models.losses import (  # noqa: E402
    build_survival_criterion,
    call_one_loss,
)

PYCOX_LOSSES = ["nll", "pmf", "mtlr", "bcesurv", "deephit", "pchazard"]


def _half_y_hat(batch, num_bins):
    """Float16 logit-like tensors, mimicking SurvivalHead output under autocast."""
    g = torch.Generator().manual_seed(0)

    def h(*shape):
        return torch.randn(*shape, generator=g).half()

    return {
        "logits": h(batch, num_bins),
        "pmf_logits": h(batch, num_bins),
        "risk": h(batch),
        "weibull_params": h(batch, 2),
        "p_high": torch.sigmoid(h(batch)),
    }


@pytest.mark.parametrize("name", PYCOX_LOSSES)
def test_pycox_loss_accepts_float16_logits(name):
    num_bins = 5
    batch = 8
    cfg = {"name": name}
    if name == "deephit":
        cfg = {"name": "deephit", "alpha": 0.2, "sigma": 0.1}
    _, criterion = build_survival_criterion(cfg, num_time_bins=num_bins)

    y_hat = _half_y_hat(batch, num_bins)
    time_bin = torch.tensor([0, 1, 2, 3, 4, 0, 2, 4], dtype=torch.long)
    event = torch.tensor([1, 0, 1, 1, 0, 1, 0, 1], dtype=torch.float32)
    continuous_time = time_bin.float() + 0.5
    bin_edges = torch.tensor([0.0, 1.0, 2.0, 3.0, 5.0])

    loss, _ = call_one_loss(
        name, criterion, y_hat, time_bin, event, continuous_time, bin_edges,
    )
    assert torch.isfinite(loss), (name, loss)
