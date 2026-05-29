"""Deterministic synthetic survival data for old-vs-new characterization tests."""
import numpy as np
import torch


def make_cohort(n=400, num_bins=24, seed=0, censor_frac=0.35, distinct=False):
    """Return (risk[n], time[n], event[n], survival[n,num_bins]).

    Higher risk -> earlier events and a faster-dropping survival curve.
    distinct=True yields tie-free continuous times (for exact-match tests).
    """
    r = np.random.default_rng(seed)
    risk = r.normal(size=n)
    base = r.exponential(scale=num_bins / 2, size=n)
    raw = base * np.exp(-0.6 * risk)
    if distinct:
        time = np.clip(raw + r.normal(scale=1e-3, size=n), 1e-3, None)
    else:
        time = np.clip(np.floor(raw), 0, num_bins - 1).astype(int)
    event = (r.random(n) > censor_frac).astype(int)
    # monotone survival curves
    bins = np.arange(num_bins)[None, :]
    hazard = np.clip((0.08 + 0.02 * (risk[:, None] - risk.min())) * (1 + 0.1 * bins), 1e-4, 0.6)
    survival = np.cumprod(1.0 - hazard, axis=1)
    return (
        risk,
        time,
        event,
        survival,
    )


def as_torch(*arrs):
    return tuple(torch.as_tensor(a) for a in arrs)
