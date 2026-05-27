"""End-to-end-ish: build the survival head with each survival_loss.name and run one
forward + backward pass on synthetic data, then check only the active terminal
received gradient."""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.survival_head import SurvivalHead  # noqa: E402
from survival_utils import build_survival_criterion  # noqa: E402


def _backward_for(name, cfg=None):
    torch.manual_seed(0)
    num_bins = 5
    head = SurvivalHead(input_dim=32, num_time_bins=num_bins, survival_loss_name=name)
    head.train()

    feats = torch.randn(8, 32, requires_grad=False)
    out = head(feats)
    time_bin = torch.tensor([0, 1, 2, 3, 4, 0, 2, 4], dtype=torch.long)
    continuous_time = time_bin.float()
    event = torch.tensor([1, 0, 1, 1, 0, 1, 0, 1], dtype=torch.float32)

    _, criterion = build_survival_criterion(cfg or {"name": name}, num_time_bins=num_bins)
    if name == "nll":
        loss = criterion(out["logits"], time_bin, event)
    elif name == "cox":
        loss = criterion(out["risk"], continuous_time, event)
    elif name == "deephit":
        loss = criterion(out["pmf"], time_bin, event)

    loss.backward()
    assert torch.isfinite(loss), loss

    # Only the active terminal's gradient should be non-zero.
    active = {"nll": "fc_hazard", "cox": "fc_risk", "deephit": "fc_pmf"}[name]
    for terminal in ("fc_hazard", "fc_risk", "fc_pmf"):
        grad = getattr(head, terminal).weight.grad
        if terminal == active:
            assert grad is not None and grad.abs().sum() > 0, terminal
        else:
            assert grad is None or grad.abs().sum() == 0, terminal


def test_nll():
    _backward_for("nll")


def test_cox():
    _backward_for("cox")


def test_deephit():
    _backward_for("deephit", cfg={
        "name": "deephit",
        "alpha": 1.0,
        "beta": 0.5,
        "gamma": 0.1,
        "sigma": 0.1,
    })


def test_soft_logrank():
    """Under soft_logrank, gradient flows through fc_risk (same terminal as cox)."""
    torch.manual_seed(0)
    num_bins = 5
    head = SurvivalHead(
        input_dim=32,
        num_time_bins=num_bins,
        survival_loss_name="soft_logrank",
    )
    head.train()

    feats = torch.randn(8, 32, requires_grad=False)
    out = head(feats)
    continuous_time = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 3.0, 5.0])
    event = torch.tensor([1, 0, 1, 1, 0, 1, 0, 1], dtype=torch.float32)

    _, criterion = build_survival_criterion(
        {"name": "soft_logrank", "lambda_balance": 0.01},
        num_time_bins=num_bins,
    )
    total, components = criterion(out["p_high"], continuous_time, event)
    assert torch.isfinite(total)
    assert "logrank" in components and "balance" in components

    total.backward()

    # fc_risk is the scalar terminal — same one cox uses.
    assert head.fc_risk.weight.grad is not None
    assert head.fc_risk.weight.grad.abs().sum() > 0
    for terminal in ("fc_hazard", "fc_pmf"):
        grad = getattr(head, terminal).weight.grad
        assert grad is None or grad.abs().sum() == 0, terminal


if __name__ == "__main__":
    test_nll()
    test_cox()
    test_deephit()
    test_soft_logrank()
    print("OK")
