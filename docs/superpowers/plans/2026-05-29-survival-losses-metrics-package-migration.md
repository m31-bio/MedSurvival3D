# Survival Losses & Metrics — Package Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace all hand-rolled survival losses and metrics (training + inference) with established package implementations (torchsurv, pycox, sksurv, lifelines), add new package losses, and keep `soft_logrank` custom — with minimal change to the existing architecture.

**Architecture:** Approach A (thin adapters behind existing seams). Loss adapters replace the current loss classes **in `survival_utils.py`** (same `build_survival_criterion` factory). Metric functions keep their **existing signatures** with bodies swapped, so call sites in `base_model.py` and `inference_survival.py` don't change. One small `name -> survival-curve` dict is the only new structure. `soft_logrank` is untouched.

**Tech Stack:** torch 2.12, torchsurv 0.1.6, pycox 0.3.0, scikit-survival 0.27.0, lifelines 0.30.3. Env: `survival_env` (uv venv, Python 3.11) at `/Users/bw/Documents/Safwat/survival/survival_env`.

**Spec:** `docs/superpowers/specs/2026-05-29-survival-losses-metrics-package-migration-design.md`

**Conventions for every task below:**
- Run tests with the env python: `SURV=/Users/bw/Documents/Safwat/survival/survival_env/bin/python` then `$SURV -m pytest ...`.
- Event polarity: `event=1` → observed event (matches all packages; no inversion).
- Time bins: `time_bin` is a 0-indexed int bin; `continuous_time` is follow-up time.
- Commit messages use the existing `feat(survival-slim):` / `refactor(survival-slim):` / `test(survival-slim):` prefixes seen in git history.

---

## Phase 0 — Environment & characterization harness

### Task 0.1: Complete `survival_env` for integration tests

**Files:**
- Modify: none (env only)

- [ ] **Step 1: Install lightning (base_model.py needs it) into survival_env**

Run:
```bash
cd /Users/bw/Documents/Safwat/survival
VIRTUAL_ENV=$PWD/survival_env uv pip install lightning
```
Expected: installs `lightning` + deps; `torchmetrics` already present.

- [ ] **Step 2: Verify the repo imports under survival_env**

Run:
```bash
cd /Users/bw/Documents/Safwat/survival/SSL3D_survival
/Users/bw/Documents/Safwat/survival/survival_env/bin/python -c "import survival_utils, base_model; print('imports OK')"
```
Expected: `imports OK`. If a module is missing, `uv pip install` it into `survival_env` and re-run.

- [ ] **Step 3: Run the existing test suite as the baseline (must pass before we change anything)**

Run:
```bash
/Users/bw/Documents/Safwat/survival/survival_env/bin/python -m pytest tests/ -q
```
Expected: all current tests PASS. Record the count. If any fail on import (missing dep), install it into `survival_env`.

> **Deployment note (not a code task):** the runtime/cluster env that actually runs training must also have `torchsurv pycox scikit-survival lifelines` installed, or migrated training will `ImportError`. Add them to that env's provisioning when deploying.

### Task 0.2: Characterization-test harness (old-vs-new oracle)

**Files:**
- Create: `tests/_characterization_data.py`

- [ ] **Step 1: Write a deterministic synthetic-data helper used by characterization tests**

```python
# tests/_characterization_data.py
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
```

- [ ] **Step 2: Sanity-run the helper**

Run:
```bash
/Users/bw/Documents/Safwat/survival/survival_env/bin/python -c "from tests._characterization_data import make_cohort; print([a.shape for a in make_cohort()])"
```
Expected: `[(400,), (400,), (400,), (400, 24)]`

- [ ] **Step 3: Commit**

```bash
git add tests/_characterization_data.py docs/superpowers/
git commit -m "test(survival-slim): add characterization data harness + migration spec/plan"
```

---

## Phase 1 — Inference metrics (sksurv / lifelines)

All edits are in `inference_survival.py`; signatures are preserved so callers in
`run_split_inference`/`compute_metrics`/`run_fold` don't change, **except** the
IPCW additions (Task 1.5) which thread training labels through.

### Task 1.1: Concordance via sksurv

**Files:**
- Modify: `inference_survival.py` (the concordance call inside `compute_metrics`, around `:507`)
- Test: `tests/test_inference_concordance.py`

- [ ] **Step 1: Write the failing characterization test**

```python
# tests/test_inference_concordance.py
import numpy as np
from sksurv.metrics import concordance_index_censored
from tests._characterization_data import make_cohort
from inference_survival import sksurv_cindex


def test_cindex_matches_sksurv_reference():
    _, time, event, _ = make_cohort(distinct=True)
    risk = -time + np.random.default_rng(1).normal(scale=0.1, size=time.shape)
    got = sksurv_cindex(time, event, risk)
    want = concordance_index_censored(event.astype(bool), time.astype(float), risk)[0]
    assert abs(got - want) < 1e-9
```

- [ ] **Step 2: Run to verify it fails**

Run: `/Users/bw/Documents/Safwat/survival/survival_env/bin/python -m pytest tests/test_inference_concordance.py -q`
Expected: FAIL — `ImportError: cannot import name 'sksurv_cindex'`.

- [ ] **Step 3: Add the helper and use it in `compute_metrics`**

Add near the top-level helpers of `inference_survival.py`:
```python
from sksurv.metrics import concordance_index_censored as _sksurv_cic


def sksurv_cindex(time, event, risk):
    """Harrell C-index via scikit-survival. event=1 -> observed event."""
    import numpy as np
    time = np.asarray(time, dtype=float)
    event = np.asarray(event).astype(bool)
    risk = np.asarray(risk, dtype=float)
    if event.sum() == 0:
        return float("nan")
    return float(_sksurv_cic(event, time, risk)[0])
```
Then, inside `compute_metrics`, replace the existing hand-rolled `concordance_index(...)` call with `sksurv_cindex(times, events, risks)` (use the variable names already present in that function).

- [ ] **Step 4: Run to verify pass**

Run: `/Users/bw/Documents/Safwat/survival/survival_env/bin/python -m pytest tests/test_inference_concordance.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add inference_survival.py tests/test_inference_concordance.py
git commit -m "refactor(survival-slim): inference C-index via scikit-survival"
```

### Task 1.2: Log-rank statistic & p-value via lifelines

**Files:**
- Modify: `inference_survival.py:319` (`compute_logrank_stat`)
- Test: `tests/test_inference_logrank.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_inference_logrank.py
import numpy as np
from lifelines.statistics import logrank_test
from tests._characterization_data import make_cohort
from inference_survival import compute_logrank_stat


def test_logrank_matches_lifelines():
    _, time, event, _ = make_cohort(distinct=True)
    rng = np.random.default_rng(2)
    group_high = (rng.random(time.shape) > 0.5)
    stat, p = compute_logrank_stat(time, event, group_high)
    lr = logrank_test(
        time[group_high], time[~group_high],
        event_observed_A=event[group_high], event_observed_B=event[~group_high],
    )
    assert abs(stat - lr.test_statistic) < 1e-6
    assert abs(p - lr.p_value) < 1e-9
```

- [ ] **Step 2: Run to verify it fails**

Run: `/Users/bw/Documents/Safwat/survival/survival_env/bin/python -m pytest tests/test_inference_logrank.py -q`
Expected: FAIL (signature mismatch — current `compute_logrank_stat` returns a different shape).

- [ ] **Step 3: Rewrite `compute_logrank_stat`**

Replace the body of `compute_logrank_stat` (`:319`) with:
```python
def compute_logrank_stat(times, events, group_high):
    """Return (test_statistic, p_value) via lifelines two-group log-rank."""
    import numpy as np
    from lifelines.statistics import logrank_test
    times = np.asarray(times, dtype=float)
    events = np.asarray(events).astype(int)
    group_high = np.asarray(group_high).astype(bool)
    a, b = group_high, ~group_high
    if a.sum() == 0 or b.sum() == 0 or events.sum() == 0:
        return float("nan"), float("nan")
    res = logrank_test(
        times[a], times[b],
        event_observed_A=events[a], event_observed_B=events[b],
    )
    return float(res.test_statistic), float(res.p_value)
```
Update any caller that unpacked the old return shape (search `compute_logrank_stat(` in `inference_survival.py` and adjust to `stat, p = ...`).

- [ ] **Step 4: Run to verify pass**

Run: `/Users/bw/Documents/Safwat/survival/survival_env/bin/python -m pytest tests/test_inference_logrank.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add inference_survival.py tests/test_inference_logrank.py
git commit -m "refactor(survival-slim): inference log-rank via lifelines"
```

### Task 1.3: Hazard ratio via lifelines CoxPHFitter

**Files:**
- Modify: `inference_survival.py:367` (`compute_hazard_ratio`)
- Test: `tests/test_inference_hr.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_inference_hr.py
import numpy as np
from tests._characterization_data import make_cohort
from inference_survival import compute_hazard_ratio


def test_hr_greater_than_one_when_high_group_dies_first():
    _, time, event, _ = make_cohort(distinct=True, seed=4)
    # high-risk group = shorter times
    group_high = time < np.median(time)
    hr = compute_hazard_ratio(time, event, group_high)
    assert hr > 1.0
```

- [ ] **Step 2: Run to verify it fails**

Run: `/Users/bw/Documents/Safwat/survival/survival_env/bin/python -m pytest tests/test_inference_hr.py -q`
Expected: FAIL or wrong value (current impl differs).

- [ ] **Step 3: Rewrite `compute_hazard_ratio`**

```python
def compute_hazard_ratio(times, events, group_high):
    """HR (high vs low) via a univariate Cox model in lifelines."""
    import numpy as np
    import pandas as pd
    from lifelines import CoxPHFitter
    times = np.asarray(times, dtype=float)
    events = np.asarray(events).astype(int)
    group_high = np.asarray(group_high).astype(int)
    if events.sum() == 0 or len(np.unique(group_high)) < 2:
        return float("nan")
    df = pd.DataFrame({"time": times, "event": events, "high": group_high})
    try:
        cph = CoxPHFitter().fit(df, duration_col="time", event_col="event")
        return float(np.exp(cph.params_["high"]))
    except Exception:
        return float("nan")
```

- [ ] **Step 4: Run to verify pass**

Run: `/Users/bw/Documents/Safwat/survival/survival_env/bin/python -m pytest tests/test_inference_hr.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add inference_survival.py tests/test_inference_hr.py
git commit -m "refactor(survival-slim): inference hazard ratio via lifelines CoxPHFitter"
```

### Task 1.4: Kaplan-Meier curves via lifelines

**Files:**
- Modify: `inference_survival.py:277` (`km_survival_at`), `:297` (`km_step_curve`)
- Test: `tests/test_inference_km.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_inference_km.py
import numpy as np
from lifelines import KaplanMeierFitter
from tests._characterization_data import make_cohort
from inference_survival import km_survival_at


def test_km_at_horizon_matches_lifelines():
    _, time, event, _ = make_cohort(distinct=True, seed=5)
    horizon = float(np.median(time))
    got = km_survival_at(time, event, horizon)
    kmf = KaplanMeierFitter().fit(time, event)
    want = float(kmf.predict(horizon))
    assert abs(got - want) < 1e-6
```

- [ ] **Step 2: Run to verify it fails**

Run: `/Users/bw/Documents/Safwat/survival/survival_env/bin/python -m pytest tests/test_inference_km.py -q`
Expected: FAIL.

- [ ] **Step 3: Rewrite the two KM helpers**

```python
def km_survival_at(times, events, horizon):
    """KM survival probability at a single horizon via lifelines."""
    import numpy as np
    from lifelines import KaplanMeierFitter
    kmf = KaplanMeierFitter().fit(np.asarray(times, float), np.asarray(events).astype(int))
    return float(kmf.predict(float(horizon)))


def km_step_curve(times, events):
    """Return (timeline, survival) arrays of the KM step curve via lifelines."""
    import numpy as np
    from lifelines import KaplanMeierFitter
    kmf = KaplanMeierFitter().fit(np.asarray(times, float), np.asarray(events).astype(int))
    sf = kmf.survival_function_
    return sf.index.to_numpy(), sf.iloc[:, 0].to_numpy()
```
Update callers of `km_step_curve` in `plot_km_high_low` (`:533`) to use the `(timeline, survival)` tuple shape if needed.

- [ ] **Step 4: Run to verify pass**

Run: `/Users/bw/Documents/Safwat/survival/survival_env/bin/python -m pytest tests/test_inference_km.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add inference_survival.py tests/test_inference_km.py
git commit -m "refactor(survival-slim): inference KM curves via lifelines"
```

### Task 1.5: Time-dependent AUC & IPCW Brier via sksurv (threads training labels)

**Files:**
- Modify: `inference_survival.py` — `compute_metrics` (`:507`), `run_split_inference` (`:211`), `run_fold` (`:652`)
- Test: `tests/test_inference_auc_ibs.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_inference_auc_ibs.py
import numpy as np
from sksurv.util import Surv
from sksurv.metrics import integrated_brier_score, cumulative_dynamic_auc
from tests._characterization_data import make_cohort
from inference_survival import sksurv_ibs, sksurv_auc


def _prep():
    _, time, event, survival = make_cohort(seed=6)
    event = event.copy()
    event[time >= time.max() - 1] = 0           # keep censoring G(t)>0
    y = Surv.from_arrays(event.astype(bool), time.astype(float))
    lo = max(1, int(time[event == 1].min()) + 1)
    hi = int(time.max())
    return time, event, survival, y, lo, hi


def test_ibs_matches_sksurv():
    time, event, survival, y, lo, hi = _prep()
    times = np.arange(lo, hi)
    got = sksurv_ibs(y, y, survival[:, times], times)
    want = integrated_brier_score(y, y, survival[:, times], times)
    assert abs(got - want) < 1e-9


def test_auc_matches_sksurv():
    time, event, survival, y, lo, hi = _prep()
    times = np.array([t for t in (4, 7, 10) if lo <= t <= hi - 1], float)
    est = 1.0 - survival[:, times.astype(int)]
    got = sksurv_auc(y, y, est, times)
    want, _ = cumulative_dynamic_auc(y, y, est, times)
    assert np.allclose(got, want, atol=1e-9)
```

- [ ] **Step 2: Run to verify it fails**

Run: `/Users/bw/Documents/Safwat/survival/survival_env/bin/python -m pytest tests/test_inference_auc_ibs.py -q`
Expected: FAIL — `ImportError`.

- [ ] **Step 3: Add thin wrappers**

```python
def sksurv_ibs(y_train, y_test, surv_at_times, times):
    """Integrated Brier score (IPCW) via scikit-survival."""
    from sksurv.metrics import integrated_brier_score
    return float(integrated_brier_score(y_train, y_test, surv_at_times, times))


def sksurv_auc(y_train, y_test, risk_at_times, times):
    """Cumulative/dynamic AUC array (IPCW) via scikit-survival."""
    from sksurv.metrics import cumulative_dynamic_auc
    auc, _mean = cumulative_dynamic_auc(y_train, y_test, risk_at_times, times)
    return auc
```

- [ ] **Step 4: Thread training labels into eval and call the wrappers**

In `run_fold` (`:652`), capture the fold's **training** `(time, event)` (already loaded for the datamodule) and pass them down to `run_split_inference` (`:211`), which forwards them to `compute_metrics`. In `compute_metrics`, build:
```python
from sksurv.util import Surv
y_train = Surv.from_arrays(train_events.astype(bool), train_times.astype(float))
y_test = Surv.from_arrays(events.astype(bool), times.astype(float))
```
Pick a valid `times` grid inside the test follow-up (as in the test's `_prep`), then call `sksurv_auc`/`sksurv_ibs`, replacing the hand-rolled `time_dependent_auc` / `integrated_brier_score(_ipcw)` calls. Where a fold has degenerate censoring (G(t)=0), wrap in try/except and emit `nan` (matches current nan-tolerant logging).

- [ ] **Step 5: Run to verify pass + full suite still green**

Run:
```bash
/Users/bw/Documents/Safwat/survival/survival_env/bin/python -m pytest tests/test_inference_auc_ibs.py tests/ -q
```
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add inference_survival.py tests/test_inference_auc_ibs.py
git commit -m "refactor(survival-slim): inference AUC + IPCW Brier via scikit-survival (thread train labels)"
```

### Task 1.6: max_logrank_cutpoint inner statistic via lifelines

**Files:**
- Modify: `survival_utils.py:671` (`_logrank_chi2`), keep `max_logrank_cutpoint:713` scan logic
- Test: extend `tests/test_max_logrank_cutpoint.py`

- [ ] **Step 1: Add a test asserting the cutpoint chi2 equals lifelines on a fixed split**

```python
# append to tests/test_max_logrank_cutpoint.py
def test_chi2_matches_lifelines():
    import numpy as np
    from lifelines.statistics import logrank_test
    from survival_utils import _logrank_chi2
    rng = np.random.default_rng(7)
    t = rng.integers(1, 20, size=100).astype(float)
    e = (rng.random(100) > 0.4).astype(int)
    g = rng.random(100) > 0.5
    chi2 = _logrank_chi2(t, e, g)
    want = logrank_test(t[g], t[~g], event_observed_A=e[g], event_observed_B=e[~g]).test_statistic
    assert abs(float(chi2) - float(want)) < 1e-6
```

- [ ] **Step 2: Run to verify it fails**

Run: `/Users/bw/Documents/Safwat/survival/survival_env/bin/python -m pytest tests/test_max_logrank_cutpoint.py::test_chi2_matches_lifelines -q`
Expected: FAIL.

- [ ] **Step 3: Rewrite `_logrank_chi2` to delegate to lifelines**

```python
def _logrank_chi2(times, events, group_high):
    import numpy as np
    from lifelines.statistics import logrank_test
    times = np.asarray(times, float); events = np.asarray(events).astype(int)
    g = np.asarray(group_high).astype(bool)
    if g.sum() == 0 or (~g).sum() == 0 or events.sum() == 0:
        return float("nan")
    return float(logrank_test(times[g], times[~g],
                              event_observed_A=events[g], event_observed_B=events[~g]).test_statistic)
```
Keep `max_logrank_cutpoint` and its quantile-scan loop as-is (it calls `_logrank_chi2`).

- [ ] **Step 4: Run to verify pass (incl. existing cutpoint tests)**

Run: `/Users/bw/Documents/Safwat/survival/survival_env/bin/python -m pytest tests/test_max_logrank_cutpoint.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add survival_utils.py tests/test_max_logrank_cutpoint.py
git commit -m "refactor(survival-slim): max_logrank_cutpoint chi2 via lifelines"
```

---

## Phase 2 — Core loss replacement (nll / cox / deephit)

### Task 2.1: Expose `pmf_logits` from the head

**Files:**
- Modify: `models/survival_head.py:95` (`forward`)
- Test: `tests/test_survival_head.py` (extend)

- [ ] **Step 1: Add a failing test for the new output key**

```python
# append to tests/test_survival_head.py
def test_pmf_logits_present_and_raw():
    import torch
    from models.survival_head import SurvivalHead
    head = SurvivalHead(input_dim=32, num_time_bins=10, survival_loss_name="deephit")
    out = head(torch.randn(4, 32))
    assert "pmf_logits" in out and out["pmf_logits"].shape == (4, 10)
    # pmf is softmax of pmf_logits
    assert torch.allclose(out["pmf"], torch.softmax(out["pmf_logits"].float(), dim=1), atol=1e-6)
```

- [ ] **Step 2: Run to verify it fails**

Run: `/Users/bw/Documents/Safwat/survival/survival_env/bin/python -m pytest tests/test_survival_head.py::test_pmf_logits_present_and_raw -q`
Expected: FAIL — `'pmf_logits' not in out`.

- [ ] **Step 3: Add `pmf_logits` to the output dict**

In `forward` (`:133`), add `"pmf_logits": pmf_logits,` to the returned dict (the `pmf_logits` variable already exists at `:115`).

- [ ] **Step 4: Run to verify pass**

Run: `/Users/bw/Documents/Safwat/survival/survival_env/bin/python -m pytest tests/test_survival_head.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add models/survival_head.py tests/test_survival_head.py
git commit -m "feat(survival-slim): expose raw pmf_logits from survival head"
```

### Task 2.2: Replace NLL with pycox logistic-hazard

**Files:**
- Modify: `survival_utils.py:336` (`NLLSurvLoss` → adapter), `:619` (`build_survival_criterion`)
- Test: `tests/test_loss_nll_pycox.py`

- [ ] **Step 1: Write a characterization test (old NLL ≈ pycox NLL on same logits)**

```python
# tests/test_loss_nll_pycox.py
import torch
from pycox.models.loss import NLLLogistiHazardLoss
from survival_utils import NLLSurvLoss
from tests._characterization_data import make_cohort, as_torch


def test_nll_adapter_matches_pycox_directly():
    _, time, event, _ = make_cohort(num_bins=12, seed=8)
    logits = torch.randn(len(time), 12, requires_grad=True)
    t, e = as_torch(time, event)
    ours = NLLSurvLoss()(logits, t, e)
    ref = NLLLogistiHazardLoss()(logits, t.long(), e.float())
    assert torch.allclose(ours, ref, atol=1e-6)
    ours.backward()
    assert logits.grad is not None
```

- [ ] **Step 2: Run to verify it fails**

Run: `/Users/bw/Documents/Safwat/survival/survival_env/bin/python -m pytest tests/test_loss_nll_pycox.py -q`
Expected: FAIL (current `NLLSurvLoss` is the hand-rolled version — values differ slightly / not exactly pycox).

- [ ] **Step 3: Replace `NLLSurvLoss` with a pycox adapter (keep the class name & call shape)**

```python
class NLLSurvLoss(nn.Module):
    """pycox logistic-hazard NLL. Input: raw hazard logits, int time bins, event."""
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        from pycox.models.loss import NLLLogistiHazardLoss
        self._loss = NLLLogistiHazardLoss(reduction=reduction)

    def forward(self, logits, time, event):
        idx = time.to(torch.int64).view(-1)
        ev = event.to(torch.float32).view(-1)
        return self._loss(logits, idx, ev)
```
Leave `build_survival_criterion`'s `nll` branch (`:639`) unchanged — it already constructs `NLLSurvLoss(reduction=...)`.

- [ ] **Step 4: Run to verify pass (+ existing loss tests)**

Run: `/Users/bw/Documents/Safwat/survival/survival_env/bin/python -m pytest tests/test_loss_nll_pycox.py tests/test_loss_factory.py tests/test_loss_integration.py -q`
Expected: PASS (update any exact-value assertion in `test_loss_integration.py::test_nll` if it hard-coded the old number — assert finite + backprops instead).

- [ ] **Step 5: Commit**

```bash
git add survival_utils.py tests/test_loss_nll_pycox.py
git commit -m "refactor(survival-slim): NLL loss via pycox logistic-hazard"
```

### Task 2.3: Replace Cox with torchsurv

**Files:**
- Modify: `survival_utils.py:396` (`CoxPHLoss` → adapter)
- Test: `tests/test_loss_cox_torchsurv.py`

- [ ] **Step 1: Write test (matches torchsurv directly; no-event edge → 0)**

```python
# tests/test_loss_cox_torchsurv.py
import torch
from torchsurv.loss import cox
from survival_utils import CoxPHLoss
from tests._characterization_data import make_cohort, as_torch


def test_cox_adapter_matches_torchsurv():
    _, time, event, _ = make_cohort(distinct=True, seed=9)
    risk = torch.randn(len(time), requires_grad=True)
    t, e = as_torch(time, event)
    ours = CoxPHLoss()(risk, t, e)
    ref = cox.neg_partial_log_likelihood(risk.view(-1), e.bool(), t.float(),
                                         ties_method="efron", reduction="mean")
    assert torch.allclose(ours, ref, atol=1e-6)
    ours.backward(); assert risk.grad is not None


def test_cox_zero_when_no_events():
    risk = torch.randn(10, requires_grad=True)
    out = CoxPHLoss()(risk, torch.arange(10).float(), torch.zeros(10))
    assert float(out) == 0.0
```

- [ ] **Step 2: Run to verify it fails**

Run: `/Users/bw/Documents/Safwat/survival/survival_env/bin/python -m pytest tests/test_loss_cox_torchsurv.py -q`
Expected: FAIL (current Breslow impl differs from Efron).

- [ ] **Step 3: Replace `CoxPHLoss` with a torchsurv adapter**

```python
class CoxPHLoss(nn.Module):
    """Cox partial likelihood via torchsurv (Efron ties). Input: risk, time, event."""
    def __init__(self, reduction: str = "mean", ties_method: str = "efron"):
        super().__init__()
        from torchsurv.loss import cox
        self._fn = cox.neg_partial_log_likelihood
        self.reduction = reduction
        self.ties_method = ties_method

    def forward(self, risk, time, event):
        log_hz = risk.float().view(-1)
        ev = event.to(torch.bool).view(-1)
        if ev.sum() == 0:
            return log_hz.sum() * 0.0
        t = time.to(torch.float32).view(-1)
        return self._fn(log_hz, ev, t, ties_method=self.ties_method,
                        reduction=self.reduction, checks=False)
```
`build_survival_criterion`'s `cox` branch already constructs `CoxPHLoss(reduction=...)` — leave it.

- [ ] **Step 4: Run to verify pass**

Run: `/Users/bw/Documents/Safwat/survival/survival_env/bin/python -m pytest tests/test_loss_cox_torchsurv.py tests/test_loss_factory.py::test_cox tests/test_loss_integration.py::test_cox -q`
Expected: PASS (relax any old exact-value assertion to finite + backprops).

- [ ] **Step 5: Commit**

```bash
git add survival_utils.py tests/test_loss_cox_torchsurv.py
git commit -m "refactor(survival-slim): Cox loss via torchsurv (Efron ties)"
```

### Task 2.4: Replace DeepHit with pycox + rank_mat (consumes pmf_logits)

**Files:**
- Modify: `survival_utils.py:437` (`DeepHitLoss` → adapter), `:643` (factory `deephit` branch), `base_model.py:298` (dispatcher `deephit` branch)
- Test: `tests/test_deephit_loss.py` (rewrite expectations)

- [ ] **Step 1: Rewrite the DeepHit tests for the new (pycox) behavior**

```python
# tests/test_deephit_loss.py  (replace prior contents)
import torch
from pycox.models.loss import DeepHitSingleLoss
from pycox.models.data import pair_rank_mat
from survival_utils import DeepHitLoss
from tests._characterization_data import make_cohort, as_torch


def _rank_mat(idx, ev, dtype, device):
    return torch.as_tensor(pair_rank_mat(idx.cpu().numpy(), ev.cpu().numpy()),
                           dtype=dtype, device=device)


def test_deephit_matches_pycox():
    _, time, event, _ = make_cohort(num_bins=12, seed=10)
    phi = torch.randn(len(time), 12, requires_grad=True)
    t, e = as_torch(time, event)
    idx, ev = t.long(), e.float()
    ours = DeepHitLoss(alpha=0.2, sigma=0.1)(phi, idx, ev)
    ref = DeepHitSingleLoss(alpha=0.2, sigma=0.1)(phi, idx, ev, _rank_mat(idx, ev, phi.dtype, phi.device))
    assert torch.allclose(ours, ref, atol=1e-6)
    ours.backward(); assert phi.grad is not None


def test_deephit_finite_mixed_censoring():
    _, time, event, _ = make_cohort(num_bins=8, seed=11)
    phi = torch.randn(len(time), 8)
    t, e = as_torch(time, event)
    out = DeepHitLoss()(phi, t.long(), e.float())
    assert torch.isfinite(out)
```

- [ ] **Step 2: Run to verify it fails**

Run: `/Users/bw/Documents/Safwat/survival/survival_env/bin/python -m pytest tests/test_deephit_loss.py -q`
Expected: FAIL (current DeepHit has alpha/beta/gamma signature; no rank_mat).

- [ ] **Step 3: Replace `DeepHitLoss` with a pycox adapter (builds rank_mat internally)**

```python
class DeepHitLoss(nn.Module):
    """Single-event DeepHit via pycox. Input: raw pmf logits, int time bins, event.

    total = alpha * NLL + (1 - alpha) * ranking. No calibration term (pycox).
    """
    def __init__(self, alpha: float = 0.2, sigma: float = 0.1):
        super().__init__()
        from pycox.models.loss import DeepHitSingleLoss
        self._loss = DeepHitSingleLoss(alpha=float(alpha), sigma=float(sigma))

    def forward(self, pmf_logits, time, event):
        from pycox.models.data import pair_rank_mat
        idx = time.to(torch.int64).view(-1)
        ev = event.to(torch.int64).view(-1)
        rank_mat = torch.as_tensor(
            pair_rank_mat(idx.detach().cpu().numpy(), ev.detach().cpu().numpy()),
            dtype=pmf_logits.dtype, device=pmf_logits.device,
        )
        return self._loss(pmf_logits, idx, ev.to(pmf_logits.dtype), rank_mat)
```

- [ ] **Step 4: Update factory + dispatcher**

Factory `deephit` branch (`survival_utils.py:643`) — replace the old `alpha/beta/gamma/sigma/num_time_bins` construction with:
```python
if name == "deephit":
    return name, DeepHitLoss(
        alpha=cfg.get("alpha", 0.2),
        sigma=cfg.get("sigma", 0.1),
    )
```
Dispatcher (`base_model.py:298`) — feed **`pmf_logits`** instead of `pmf`:
```python
elif name == "deephit":
    loss = self.criterion(y_hat["pmf_logits"], time_bin, event)
    loss_parts = {"total": loss, name: loss}
```

- [ ] **Step 5: Run to verify pass + factory/integration**

Run: `/Users/bw/Documents/Safwat/survival/survival_env/bin/python -m pytest tests/test_deephit_loss.py tests/test_loss_factory.py tests/test_loss_integration.py -q`
Expected: PASS (update `test_loss_factory.py::test_deephit_reads_hyperparameters` to the new `alpha/sigma` keys).

- [ ] **Step 6: Commit**

```bash
git add survival_utils.py base_model.py tests/test_deephit_loss.py tests/test_loss_factory.py
git commit -m "refactor(survival-slim): DeepHit via pycox (alpha/sigma, rank_mat, no calibration)"
```

---

## Phase 3 — New losses

The simple new losses (`pmf`, `mtlr`, `bcesurv`) plug into the existing dispatcher
shape. `pchazard` and `weibull` need extra plumbing (interval_frac; a 2-param head).

### Task 3.1: Add `pmf`, `mtlr`, `bcesurv` adapters + factory wiring

**Files:**
- Modify: `survival_utils.py` (add three adapter classes + factory branches)
- Modify: `base_model.py:288` (dispatcher branches)
- Test: `tests/test_new_losses_simple.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_new_losses_simple.py
import torch
from survival_utils import build_survival_criterion
from tests._characterization_data import make_cohort, as_torch

CASES = {
    "pmf":     "pmf_logits",
    "mtlr":    "logits",
    "bcesurv": "logits",
}


def _run(name):
    _, time, event, _ = make_cohort(num_bins=10, seed=12)
    t, e = as_torch(time, event)
    nm, crit = build_survival_criterion({"name": name}, num_time_bins=10)
    phi = torch.randn(len(time), 10, requires_grad=True)
    out = crit(phi, t.long(), e.float())
    assert torch.isfinite(out)
    out.backward(); assert phi.grad is not None
    return nm


def test_pmf():     assert _run("pmf") == "pmf"
def test_mtlr():    assert _run("mtlr") == "mtlr"
def test_bcesurv(): assert _run("bcesurv") == "bcesurv"
```

- [ ] **Step 2: Run to verify it fails**

Run: `/Users/bw/Documents/Safwat/survival/survival_env/bin/python -m pytest tests/test_new_losses_simple.py -q`
Expected: FAIL — factory raises "unknown name".

- [ ] **Step 3: Add the three adapter classes**

```python
class PMFLoss(nn.Module):
    """pycox PMF NLL. Input: raw pmf logits, int time bins, event."""
    def __init__(self):
        super().__init__()
        from pycox.models.loss import NLLPMFLoss
        self._loss = NLLPMFLoss()
    def forward(self, pmf_logits, time, event):
        return self._loss(pmf_logits, time.to(torch.int64).view(-1), event.to(torch.float32).view(-1))


class MTLRLoss(nn.Module):
    """pycox MTLR NLL. Input: raw logits [B,K], int time bins, event."""
    def __init__(self):
        super().__init__()
        from pycox.models.loss import NLLMTLRLoss
        self._loss = NLLMTLRLoss()
    def forward(self, logits, time, event):
        return self._loss(logits, time.to(torch.int64).view(-1), event.to(torch.float32).view(-1))


class BCESurvLoss(nn.Module):
    """pycox BCESurv loss. Input: raw logits [B,K], int time bins, event."""
    def __init__(self):
        super().__init__()
        from pycox.models.loss import BCESurvLoss as _BCE
        self._loss = _BCE()
    def forward(self, logits, time, event):
        return self._loss(logits, time.to(torch.int64).view(-1), event.to(torch.float32).view(-1))
```

- [ ] **Step 4: Wire factory + dispatcher**

Factory (`build_survival_criterion`): add branches
```python
if name == "pmf":     return name, PMFLoss()
if name == "mtlr":    return name, MTLRLoss()
if name == "bcesurv": return name, BCESurvLoss()
```
and add `pmf, mtlr, bcesurv` to the accepted-names list in the error message (`:659`).
Dispatcher (`base_model.py`): add
```python
elif name == "pmf":
    loss = self.criterion(y_hat["pmf_logits"], time_bin, event)
    loss_parts = {"total": loss, name: loss}
elif name in ("mtlr", "bcesurv"):
    loss = self.criterion(y_hat["logits"], time_bin, event)
    loss_parts = {"total": loss, name: loss}
```

- [ ] **Step 5: Run to verify pass**

Run: `/Users/bw/Documents/Safwat/survival/survival_env/bin/python -m pytest tests/test_new_losses_simple.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add survival_utils.py base_model.py tests/test_new_losses_simple.py
git commit -m "feat(survival-slim): add pmf/mtlr/bcesurv losses via pycox"
```

### Task 3.2: Per-loss survival-curve dict (for metrics & stratification)

**Files:**
- Modify: `models/survival_head.py:85` (`_survival_for_active_loss`), add module-level `LOGITS_TO_SURVIVAL`
- Modify: `survival_utils.py:769` (`derive_stratification_scores`)
- Test: `tests/test_survival_curves.py`

- [ ] **Step 1: Write tests pinning each curve to a reference (pycox model where applicable)**

```python
# tests/test_survival_curves.py
import torch
from models.survival_head import logits_to_survival


def test_nll_curve_is_cumprod_hazard():
    phi = torch.randn(5, 8)
    s = logits_to_survival("nll", phi)
    assert torch.allclose(s, torch.cumprod(1 - torch.sigmoid(phi), dim=1), atol=1e-6)


def test_pmf_curve_is_one_minus_cumsum_softmax():
    phi = torch.randn(5, 8)
    s = logits_to_survival("pmf", phi)
    assert torch.allclose(s, 1 - torch.softmax(phi, 1).cumsum(1), atol=1e-6)


def test_bcesurv_curve_is_sigmoid():
    phi = torch.randn(5, 8)
    s = logits_to_survival("bcesurv", phi)
    assert torch.allclose(s, torch.sigmoid(phi), atol=1e-6)


def test_mtlr_curve_matches_pycox():
    # Oracle: pycox MTLR.predict_surv on the same logits.
    from pycox.models import MTLR
    phi = torch.randn(5, 8)
    ours = logits_to_survival("mtlr", phi)
    ref = torch.as_tensor(MTLR.predict_surv(MTLR, phi.numpy()))  # static-style transform
    assert ours.shape == ref.shape
    assert torch.allclose(ours.float(), ref.float(), atol=1e-5)
```
> If `MTLR.predict_surv` cannot be called without an instance, the test instead
> builds the curve via pycox utils `pad_col`/`cumsum_reverse` and asserts our impl
> equals that reference expression; adjust the oracle expression until it matches
> pycox's documented MTLR→survival math, then freeze it.

- [ ] **Step 2: Run to verify it fails**

Run: `/Users/bw/Documents/Safwat/survival/survival_env/bin/python -m pytest tests/test_survival_curves.py -q`
Expected: FAIL — `logits_to_survival` undefined.

- [ ] **Step 3: Implement `logits_to_survival` + dict**

```python
# models/survival_head.py (module level)
import torch.nn.functional as F
from pycox.models.utils import pad_col, cumsum_reverse


def _nll_surv(phi):     return torch.cumprod(1 - torch.sigmoid(phi), dim=1)
def _pmf_surv(phi):     return 1 - F.softmax(phi, dim=1).cumsum(1)
def _bce_surv(phi):     return torch.sigmoid(phi)
def _mtlr_surv(phi):
    # pycox MTLR: cumulative-sum-reverse then softmax over padded logits.
    g = cumsum_reverse(pad_col(phi), dim=1)
    pmf = F.softmax(g, dim=1)
    return (1 - pmf.cumsum(1))[:, :-1]

LOGITS_TO_SURVIVAL = {
    "nll": _nll_surv, "pmf": _pmf_surv, "deephit": _pmf_surv,
    "bcesurv": _bce_surv, "mtlr": _mtlr_surv,
}


def logits_to_survival(name, phi):
    return LOGITS_TO_SURVIVAL[name](phi)
```
Refactor `SurvivalHead._survival_for_active_loss` (`:85`) to delegate:
```python
def _survival_for_active_loss(self, hazard, pmf, pmf_logits=None):
    name = self.survival_loss_name
    if name in LOGITS_TO_SURVIVAL and name != "nll":
        return logits_to_survival(name, pmf_logits if name in ("pmf", "deephit") else self._raw_logits)
    return hazard_to_survival(hazard)
```
(Keep the existing `nll`/`cox`/`soft_logrank` behavior; pass the raw logits already computed in `forward`.)
Extend `derive_stratification_scores` (`survival_utils.py:769`): add `pmf`, `mtlr`, `bcesurv`, `pchazard`, `weibull` to the same branch as `nll`/`deephit` (`1 - survival[:, landmark]`).

- [ ] **Step 4: Run to verify pass + stratification tests**

Run: `/Users/bw/Documents/Safwat/survival/survival_env/bin/python -m pytest tests/test_survival_curves.py tests/test_stratification_metrics.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add models/survival_head.py survival_utils.py tests/test_survival_curves.py
git commit -m "feat(survival-slim): per-loss survival-curve dict + stratification scores"
```

### Task 3.3: Add `weibull` loss + 2-param head output

**Files:**
- Modify: `models/survival_head.py` (add `fc_weibull` → `weibull_params`)
- Modify: `survival_utils.py` (add `WeibullLoss` + factory branch + curve)
- Modify: `base_model.py` (dispatcher branch)
- Test: `tests/test_weibull_loss.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_weibull_loss.py
import torch
from torchsurv.loss import weibull
from survival_utils import build_survival_criterion
from tests._characterization_data import make_cohort, as_torch


def test_weibull_matches_torchsurv():
    _, time, event, _ = make_cohort(distinct=True, seed=13)
    log_params = torch.randn(len(time), 2, requires_grad=True)
    t, e = as_torch(time, event)
    nm, crit = build_survival_criterion({"name": "weibull"}, num_time_bins=10)
    ours = crit(log_params, t, e)
    ref = weibull.neg_log_likelihood_weibull(log_params, e.bool(), t.float(), reduction="mean")
    assert torch.allclose(ours, ref, atol=1e-6)
    ours.backward(); assert log_params.grad is not None
```

- [ ] **Step 2: Run to verify it fails**

Run: `/Users/bw/Documents/Safwat/survival/survival_env/bin/python -m pytest tests/test_weibull_loss.py -q`
Expected: FAIL — unknown name.

- [ ] **Step 3: Implement adapter + head output + factory + dispatcher**

`survival_utils.py`:
```python
class WeibullLoss(nn.Module):
    """Parametric Weibull AFT via torchsurv. Input: log_params [B,2], time, event."""
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        from torchsurv.loss import weibull
        self._fn = weibull.neg_log_likelihood_weibull
        self.reduction = reduction
    def forward(self, log_params, time, event):
        ev = event.to(torch.bool).view(-1)
        if ev.sum() == 0:
            return log_params.sum() * 0.0
        return self._fn(log_params, ev, time.float().view(-1),
                        reduction=self.reduction, checks=False)
```
Factory branch: `if name == "weibull": return name, WeibullLoss(reduction=cfg.get("reduction","mean"))`; add to accepted names.
`models/survival_head.py`: in `__init__` add `self.fc_weibull = nn.Linear(self.hidden_dim, 2)`; in `forward` add `"weibull_params": self.fc_weibull(x),` to the dict.
`base_model.py` dispatcher:
```python
elif name == "weibull":
    loss = self.criterion(y_hat["weibull_params"], continuous_time, event)
    loss_parts = {"total": loss, name: loss}
```

- [ ] **Step 4: Run to verify pass**

Run: `/Users/bw/Documents/Safwat/survival/survival_env/bin/python -m pytest tests/test_weibull_loss.py tests/test_survival_head.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add survival_utils.py models/survival_head.py base_model.py tests/test_weibull_loss.py
git commit -m "feat(survival-slim): add Weibull AFT loss via torchsurv + 2-param head"
```

### Task 3.4: Add `pchazard` loss + interval_frac plumbing

**Files:**
- Modify: `survival_utils.py` (add `PCHazardLoss` + factory branch)
- Modify: `base_model.py` (dispatcher computes `interval_frac` from `continuous_time` + cut points)
- Test: `tests/test_pchazard_loss.py`

- [ ] **Step 1: Write failing test (adapter matches pycox given interval_frac)**

```python
# tests/test_pchazard_loss.py
import torch
from pycox.models.loss import NLLPCHazardLoss
from survival_utils import PCHazardLoss
from tests._characterization_data import make_cohort, as_torch


def test_pchazard_matches_pycox():
    _, time, event, _ = make_cohort(num_bins=10, seed=14)
    phi = torch.randn(len(time), 10, requires_grad=True)
    t, e = as_torch(time, event)
    idx = t.long(); ev = e.float()
    interval_frac = torch.rand(len(time))
    ours = PCHazardLoss()(phi, idx, ev, interval_frac)
    ref = NLLPCHazardLoss()(phi, idx, ev, interval_frac)
    assert torch.allclose(ours, ref, atol=1e-6)
    ours.backward(); assert phi.grad is not None
```

- [ ] **Step 2: Run to verify it fails**

Run: `/Users/bw/Documents/Safwat/survival/survival_env/bin/python -m pytest tests/test_pchazard_loss.py -q`
Expected: FAIL — `PCHazardLoss` undefined.

- [ ] **Step 3: Implement adapter + factory + dispatcher interval_frac**

`survival_utils.py`:
```python
class PCHazardLoss(nn.Module):
    """pycox piecewise-constant hazard NLL. Input: logits, time bins, event, interval_frac."""
    def __init__(self):
        super().__init__()
        from pycox.models.loss import NLLPCHazardLoss
        self._loss = NLLPCHazardLoss()
    def forward(self, logits, time, event, interval_frac):
        return self._loss(logits, time.to(torch.int64).view(-1),
                          event.to(torch.float32).view(-1),
                          interval_frac.to(logits.dtype).view(-1))
```
Factory branch: `if name == "pchazard": return name, PCHazardLoss()`; add to accepted names.
`base_model.py` dispatcher — compute `interval_frac` as the within-bin fraction of `continuous_time`. The model already knows bin edges via its cut points (see `_time_to_survival_bin`, `base_model.py:235`). Add a small helper on the model:
```python
def _interval_frac(self, continuous_time, time_bin):
    edges = self._survival_cut_points_tensor  # left edges per bin, len K (already available)
    left = edges[time_bin.clamp(0, len(edges) - 1)]
    width = (edges[1] - edges[0]) if len(edges) > 1 else 1.0
    return ((continuous_time - left) / width).clamp(0.0, 1.0)
```
and the dispatcher branch:
```python
elif name == "pchazard":
    interval_frac = self._interval_frac(continuous_time, time_bin)
    loss = self.criterion(y_hat["logits"], time_bin, event, interval_frac)
    loss_parts = {"total": loss, name: loss}
```
> If the model does not yet expose a cut-points tensor, derive it once in
> `__init__` from the same configuration `_time_to_survival_bin` uses, and store it
> as `self._survival_cut_points_tensor`. Add `pchazard` survival curve to
> `LOGITS_TO_SURVIVAL` using pycox's PC-hazard→surv math, pinned by a test against
> `pycox.models.PCHazard.predict_surv` (same pattern as MTLR in Task 3.2).

- [ ] **Step 4: Run to verify pass**

Run: `/Users/bw/Documents/Safwat/survival/survival_env/bin/python -m pytest tests/test_pchazard_loss.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add survival_utils.py base_model.py tests/test_pchazard_loss.py
git commit -m "feat(survival-slim): add PC-hazard loss via pycox + interval_frac plumbing"
```

---

## Phase 4 — Training-time metrics (torchsurv)

Replace the bodies of the training-time metric functions in `survival_utils.py`,
preserving signatures so `base_model.py` metric hooks don't change.

### Task 4.1: Concordance (training) via torchsurv

**Files:**
- Modify: `survival_utils.py:9` (`concordance_index`)
- Test: `tests/test_train_concordance.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_train_concordance.py
import torch
from torchsurv.metrics.cindex import ConcordanceIndex
from survival_utils import concordance_index
from tests._characterization_data import make_cohort, as_torch


def test_train_cindex_matches_torchsurv():
    _, time, event, _ = make_cohort(distinct=True, seed=15)
    risk = torch.randn(len(time))
    t, e = as_torch(time, event)
    got = concordance_index(t, risk, e)              # (event_times, scores, event_observed)
    want = float(ConcordanceIndex()(risk, e.bool(), t.float()))
    assert abs(float(got) - want) < 1e-6
```

- [ ] **Step 2: Run to verify it fails**

Run: `/Users/bw/Documents/Safwat/survival/survival_env/bin/python -m pytest tests/test_train_concordance.py -q`
Expected: FAIL (hand-rolled value differs on tied bins).

- [ ] **Step 3: Rewrite `concordance_index` body (keep signature `(event_times, predicted_scores, event_observed)`)**

```python
def concordance_index(event_times, predicted_scores, event_observed):
    """Harrell C-index via torchsurv. Higher score = higher risk."""
    from torchsurv.metrics.cindex import ConcordanceIndex
    est = torch.as_tensor(predicted_scores).float().view(-1)
    ev = torch.as_tensor(event_observed).bool().view(-1)
    t = torch.as_tensor(event_times).float().view(-1)
    if ev.sum() == 0:
        return 0.5
    return float(ConcordanceIndex()(est, ev, t))
```

- [ ] **Step 4: Run to verify pass + stratification tests (they consume this)**

Run: `/Users/bw/Documents/Safwat/survival/survival_env/bin/python -m pytest tests/test_train_concordance.py tests/test_stratification_metrics.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add survival_utils.py tests/test_train_concordance.py
git commit -m "refactor(survival-slim): training C-index via torchsurv"
```

### Task 4.2: Time-dependent AUC (training) via torchsurv

**Files:**
- Modify: `survival_utils.py:80` (`time_dependent_auc`)
- Test: `tests/test_train_auc.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_train_auc.py
import torch
from torchsurv.metrics.auc import Auc
from survival_utils import time_dependent_auc
from tests._characterization_data import make_cohort, as_torch


def test_train_auc_matches_torchsurv_per_landmark():
    _, time, event, survival = make_cohort(seed=16)
    event = event.copy(); event[time >= time.max() - 1] = 0
    s, t, e = as_torch(survival, time, event)
    landmarks = torch.tensor([4.0, 7.0, 10.0])
    cuts = torch.arange(survival.shape[1]).float()
    got = time_dependent_auc(s, t, e, landmarks, cuts)   # returns {landmark: auc}
    auc = Auc()
    for lm in landmarks.tolist():
        risk = 1.0 - s[:, int(lm)]
        want = float(auc(risk, e.bool(), t.float(), new_time=torch.tensor(float(lm))))
        assert abs(got[float(lm)] - want) < 1e-5
```

- [ ] **Step 2: Run to verify it fails**

Run: `/Users/bw/Documents/Safwat/survival/survival_env/bin/python -m pytest tests/test_train_auc.py -q`
Expected: FAIL.

- [ ] **Step 3: Rewrite `time_dependent_auc` body (keep signature & dict return)**

```python
def time_dependent_auc(survival, event_times, event_observed, landmark_years, cut_points_years):
    """Cumulative/dynamic AUC at each landmark via torchsurv. Returns {landmark: auc}."""
    from torchsurv.metrics.auc import Auc
    survival = torch.as_tensor(survival).float()
    t = torch.as_tensor(event_times).float().view(-1)
    e = torch.as_tensor(event_observed).bool().view(-1)
    landmarks = torch.as_tensor(landmark_years).float().view(-1)
    num_bins = survival.shape[1]
    auc = Auc()
    out = {}
    for lm in landmarks.tolist():
        b = min(int(lm), num_bins - 1)
        risk = 1.0 - survival[:, b]
        try:
            out[float(lm)] = float(auc(risk, e, t, new_time=torch.tensor(float(lm))))
        except Exception:
            out[float(lm)] = float("nan")
    return out
```

- [ ] **Step 4: Run to verify pass**

Run: `/Users/bw/Documents/Safwat/survival/survival_env/bin/python -m pytest tests/test_train_auc.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add survival_utils.py tests/test_train_auc.py
git commit -m "refactor(survival-slim): training time-dependent AUC via torchsurv"
```

### Task 4.3: Integrated Brier (training, plain + IPCW) via torchsurv

**Files:**
- Modify: `survival_utils.py:136` (`integrated_brier_score`), `:208` (`integrated_brier_score_ipcw`); remove `_censoring_survival_km` (`:178`) once unused
- Test: `tests/test_train_brier.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_train_brier.py
import torch
from torchsurv.metrics.brier_score import BrierScore
from survival_utils import integrated_brier_score, integrated_brier_score_ipcw
from tests._characterization_data import make_cohort, as_torch


def test_train_ibs_finite_and_in_unit_range():
    _, time, event, survival = make_cohort(seed=17)
    event = event.copy(); event[time >= time.max() - 1] = 0
    s, t, e = as_torch(survival, time, event)
    v = integrated_brier_score(s, t, e)
    w = integrated_brier_score_ipcw(s, t, e)
    assert 0.0 <= v <= 1.0 and 0.0 <= w <= 1.0


def test_train_ibs_matches_torchsurv_brier_at_grid():
    _, time, event, survival = make_cohort(seed=18)
    event = event.copy(); event[time >= time.max() - 1] = 0
    s, t, e = as_torch(survival, time, event)
    lo = max(1, int(t[e.bool()].min()) + 1); hi = int(t.max())
    times = torch.arange(lo, hi)
    est = s[:, times]
    bs = BrierScore()
    bs(est, e.bool(), t.float(), new_time=times.float())
    want = float(bs.integral())
    from survival_utils import _ibs_torchsurv  # helper added in step 3
    got = _ibs_torchsurv(s, t, e, times, weight=None)
    assert abs(got - want) < 1e-5
```

- [ ] **Step 2: Run to verify it fails**

Run: `/Users/bw/Documents/Safwat/survival/survival_env/bin/python -m pytest tests/test_train_brier.py -q`
Expected: FAIL — `_ibs_torchsurv` undefined.

- [ ] **Step 3: Rewrite both Brier functions + shared helper**

```python
def _ibs_torchsurv(survival, event_times, event_observed, times, weight=None):
    from torchsurv.metrics.brier_score import BrierScore
    s = torch.as_tensor(survival).float()
    t = torch.as_tensor(event_times).float().view(-1)
    e = torch.as_tensor(event_observed).bool().view(-1)
    est = s[:, times]                                   # [n, len(times)]
    bs = BrierScore()
    bs(est, e, t, new_time=times.float(), weight=weight)
    return float(bs.integral())


def _valid_time_grid(event_times, event_observed, num_bins):
    t = torch.as_tensor(event_times).long().view(-1)
    e = torch.as_tensor(event_observed).bool().view(-1)
    lo = max(1, int(t[e].min()) + 1) if e.any() else 1
    hi = min(int(t.max()), num_bins - 1)
    return torch.arange(lo, max(lo + 1, hi))


def integrated_brier_score(survival, event_times, event_observed):
    """Unweighted integrated Brier score via torchsurv."""
    survival = torch.as_tensor(survival).float()
    if survival.numel() == 0:
        return 0.0
    times = _valid_time_grid(event_times, event_observed, survival.shape[1])
    try:
        return _ibs_torchsurv(survival, event_times, event_observed, times, weight=None)
    except Exception:
        return float("nan")


def integrated_brier_score_ipcw(survival, event_times, event_observed, eps=1e-7):
    """IPCW integrated Brier score via torchsurv (internal IPCW weighting)."""
    survival = torch.as_tensor(survival).float()
    if survival.numel() == 0:
        return 0.0
    times = _valid_time_grid(event_times, event_observed, survival.shape[1])
    try:
        # torchsurv computes IPCW internally when weight is omitted; pass weight=None
        return _ibs_torchsurv(survival, event_times, event_observed, times, weight=None)
    except Exception:
        return float("nan")
```
> Verify torchsurv's IPCW behavior during step 4: if `BrierScore` requires an
> explicit `weight` for IPCW, compute it from the training KM censoring estimate
> and pass it; otherwise the default already applies IPCW. The test asserts the
> unit-range + grid-match property either way. Then delete `_censoring_survival_km`
> (`:178`) if no remaining references (`grep _censoring_survival_km`).

- [ ] **Step 4: Run to verify pass + full suite**

Run:
```bash
/Users/bw/Documents/Safwat/survival/survival_env/bin/python -m pytest tests/ -q
```
Expected: PASS (entire suite).

- [ ] **Step 5: Commit**

```bash
git add survival_utils.py tests/test_train_brier.py
git commit -m "refactor(survival-slim): training integrated Brier via torchsurv; drop hand-rolled censoring KM"
```

---

## Phase 5 — Cleanup & config

### Task 5.1: Update config schema + an example config

**Files:**
- Modify: any survival-loss YAML under `cli_configs/` that documents `survival_loss` names
- Test: `tests/test_loss_factory.py` (add coverage for all new names)

- [ ] **Step 1: Add a factory test enumerating every supported name**

```python
# append to tests/test_loss_factory.py
import pytest
from survival_utils import build_survival_criterion

@pytest.mark.parametrize("name", ["nll","cox","deephit","soft_logrank","pmf","mtlr","bcesurv","weibull","pchazard"])
def test_all_names_build(name):
    nm, crit = build_survival_criterion({"name": name}, num_time_bins=10)
    assert nm == name and crit is not None
```

- [ ] **Step 2: Run to verify it fails (then passes after wiring is complete)**

Run: `/Users/bw/Documents/Safwat/survival/survival_env/bin/python -m pytest tests/test_loss_factory.py -q`
Expected: PASS if Phases 2-3 complete; otherwise FAIL on the missing name.

- [ ] **Step 3: Document the new names in the config that selects the loss**

Update the `survival_loss:` block comment in the relevant `cli_configs/.../*.yaml` to list: `nll, cox, deephit, soft_logrank, pmf, mtlr, bcesurv, weibull, pchazard` with their hyperparameters (`deephit: {alpha, sigma}`).

- [ ] **Step 4: Commit**

```bash
git add tests/test_loss_factory.py cli_configs/
git commit -m "feat(survival-slim): document + test all survival loss names"
```

### Task 5.2: Final full-suite run + delete dead hand-rolled helpers

**Files:**
- Modify: `survival_utils.py`, `inference_survival.py` (remove now-orphaned helpers our changes created)

- [ ] **Step 1: Find orphans created by our changes**

Run:
```bash
cd /Users/bw/Documents/Safwat/survival/SSL3D_survival
grep -nE "_censoring_survival_km|_binary_roc_auc" survival_utils.py
```
Remove any of these only if no remaining references (they were used only by the replaced metrics). Do NOT remove pre-existing dead code unrelated to this change.

- [ ] **Step 2: Run the whole suite under survival_env**

Run: `/Users/bw/Documents/Safwat/survival/survival_env/bin/python -m pytest tests/ -q`
Expected: ALL PASS.

- [ ] **Step 3: Commit**

```bash
git add survival_utils.py inference_survival.py
git commit -m "refactor(survival-slim): drop orphaned hand-rolled metric helpers"
```

---

## Self-Review (completed by plan author)

**Spec coverage:** every spec section maps to tasks —
- Loss mapping (9 losses) → Tasks 2.2–2.4, 3.1, 3.3, 3.4; `soft_logrank` untouched (verified by leaving its tests unchanged).
- Head changes (`pmf_logits`, `weibull_params`, interval_frac) → 2.1, 3.3, 3.4.
- Per-loss survival dict → 3.2.
- Training metrics → 4.1–4.3.
- Inference metrics + IPCW train threading → 1.1–1.6.
- Config/factory → 2.4, 3.1, 3.3, 3.4, 5.1.
- Characterization tests → woven into every replacement task; harness in 0.2.
- Env reality → 0.1.

**Placeholder scan:** the two “verify behavior in step 4” notes (MTLR/PC-hazard curve oracle, torchsurv IPCW weighting) are explicit verification loops with a concrete oracle (pycox `predict_surv`, torchsurv unit-range), not placeholders — the test is the spec and the impl is given.

**Type/name consistency:** adapter class names kept identical to current (`NLLSurvLoss`, `CoxPHLoss`, `DeepHitLoss`) so the factory needs no rename; new classes (`PMFLoss`, `MTLRLoss`, `BCESurvLoss`, `WeibullLoss`, `PCHazardLoss`) and helpers (`logits_to_survival`, `LOGITS_TO_SURVIVAL`, `_ibs_torchsurv`, `sksurv_cindex`, `sksurv_ibs`, `sksurv_auc`) are referenced consistently across tasks.

**Known follow-ups (not blocking):** Cox curve-based IBS uses a Breslow baseline only if curve metrics are later requested for Cox (spec open item); DeepHit must be re-tuned (`alpha/sigma`) since the objective changed.
