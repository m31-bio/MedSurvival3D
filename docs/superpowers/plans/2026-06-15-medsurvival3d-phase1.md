# MedSurvival3D Phase 1 (Module Split) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restructure the flat `SSL3D_survival` code into the `medsurvival3d`
package (at repo root) following the MedClass3D layout, and decompose the
`base_model.py` monolith — without changing behavior.

**Architecture:** Hybrid decomposition (option C): relocate the already-free
functions of `survival_utils.py` into `evaluation/metrics.py` (metrics) and
`models/losses.py` (losses); lift stateless label/bin helpers out of
`base_model.py` into `utils/survival_labels.py`; move the two LR schedulers into
`training/optim.py`; keep the `LightningModule` lifecycle, logging, and metric
buffers in a slimmed `training/trainer.py`. Hydra and `main.py` stay at repo
root with updated imports. **Packaging (`pyproject.toml`/`uv`/`scripts/`/`src/`
move) is Phase 2 — out of scope here.**

**Tech Stack:** Python, PyTorch, Lightning, Hydra, pycox/torchsurv (cluster
only), pytest.

**Spec:** `docs/superpowers/specs/2026-06-15-medsurvival3d-restructure-design.md`

---

## Transition mechanism: re-export shims

To keep every commit importable while contents move leaf-first, the original
modules (`survival_utils.py`, `base_model.py`, `inference_survival.py`, the old
`models/`/`datasets/`/`augmentation/` files) are turned into **thin re-export
shims** as their contents move into `medsurvival3d/`. The shim simply does
`from medsurvival3d.<new> import *` (plus explicit names that aren't exported by
`*`). Importers are migrated to the new paths in Task 11–12, and the shims are
deleted in Task 13. Net result: imports are rewritten exactly once, to final
locations; shims are temporary scaffolding internal to Phase 1.

## Verification model (per spec)

- **Local (every task):** changed modules byte-compile; the import graph is
  acyclic; the dependency-free test subset passes.
- **Dependency-free test command** (the only suite runnable on macOS):
  ```
  ~/miniconda3/envs/fm_agent/bin/python -m pytest tests/test_loss_factory.py -k composite -q
  ```
- **Byte-compile command** (run after each task on the files it touched, e.g.):
  ```
  ~/miniconda3/envs/fm_agent/bin/python -m py_compile medsurvival3d/utils/survival_labels.py
  ```
- **Acyclic import check** (Task 14): import each leaf module in isolation; a
  circular import raises `ImportError` immediately.
- **Cluster (phase gate, user-run, NOT in these tasks):** full `pytest` matches
  pre-refactor results + one training step on a t1c config.

## Import layering (must stay acyclic)

- Leaves (import nothing internal): `utils/survival_labels.py`, `utils/io.py`,
  `training/optim.py`, `evaluation/metrics.py`
- Mid (may import `utils/`): `models/losses.py`
- Top (import the rest; may lazy-import each other): `training/trainer.py`,
  `inference/survival.py`

---

### Task 1: Scaffold the package skeleton

**Files:**
- Create: `medsurvival3d/__init__.py`
- Create: `medsurvival3d/data/__init__.py`, `medsurvival3d/data/preprocessing/__init__.py`
- Create: `medsurvival3d/models/__init__.py`, `medsurvival3d/models/backbones/__init__.py`, `medsurvival3d/models/heads/__init__.py`
- Create: `medsurvival3d/training/__init__.py`
- Create: `medsurvival3d/evaluation/__init__.py`
- Create: `medsurvival3d/inference/__init__.py`
- Create: `medsurvival3d/utils/__init__.py`

- [ ] **Step 1: Create all package directories with empty `__init__.py` files**

Each `__init__.py` is empty (zero bytes) except `medsurvival3d/__init__.py`:

```python
"""MedSurvival3D: 3D medical-image survival modeling (nnssl fine-tuning)."""
```

- [ ] **Step 2: Verify the package imports**

Run: `~/miniconda3/envs/fm_agent/bin/python -c "import medsurvival3d; import medsurvival3d.models, medsurvival3d.data, medsurvival3d.training, medsurvival3d.evaluation, medsurvival3d.inference, medsurvival3d.utils; print('ok')"`
Expected: prints `ok`

- [ ] **Step 3: Commit**

```bash
git add medsurvival3d/
git commit -m "feat(restructure): scaffold medsurvival3d package skeleton"
```

---

### Task 2: `utils/survival_labels.py` — lift stateless label/bin helpers

These six methods in `base_model.py` (lines 244–331) only need config values, not
`self` — convert each to a free function taking the needed config as arguments.

**Files:**
- Create: `medsurvival3d/utils/survival_labels.py`
- Modify: `base_model.py:244-331` (replace the six method bodies with calls to the new functions; keep the methods as thin wrappers so `self`-callers are untouched this task)

- [ ] **Step 1: Write the new module**

Signatures (bodies are the existing method bodies with `self.X` replaced by the
named parameter):

```python
"""Stateless survival label/bin/target transforms (no LightningModule state)."""
import torch


def survival_year_values(values, month_values=None, default=None):
    # body of BaseModel._survival_year_values (no self refs) verbatim
    ...


def format_survival_landmark_label(value):
    # body of BaseModel._format_survival_landmark_label verbatim
    ...


def time_to_survival_bin(continuous_time, cut_points_years, num_time_bins):
    # body of _time_to_survival_bin; self.survival_cut_points_years -> cut_points_years,
    # self.num_time_bins -> num_time_bins
    ...


def interval_frac(continuous_time, time_bin, bin_edges):
    # body of _interval_frac; self._survival_bin_edges -> bin_edges
    ...


def unpack_survival_targets(y, device, cut_points_years, num_time_bins):
    # body of _unpack_survival_targets; self.device -> device,
    # self._time_to_survival_bin(...) -> time_to_survival_bin(..., cut_points_years, num_time_bins)
    ...


def survival_label_tensor(time_bin, event):
    # body of _survival_label_tensor verbatim
    ...
```

- [ ] **Step 2: Rewrite the `base_model.py` methods as thin wrappers**

Each method delegates so all existing `self._...` callers keep working:

```python
def _survival_year_values(self, values, month_values=None, default=None):
    return survival_year_values(values, month_values, default)

def _format_survival_landmark_label(self, value):
    return format_survival_landmark_label(value)

def _time_to_survival_bin(self, continuous_time):
    return time_to_survival_bin(continuous_time, self.survival_cut_points_years, self.num_time_bins)

def _interval_frac(self, continuous_time, time_bin):
    return interval_frac(continuous_time, time_bin, self._survival_bin_edges)

def _unpack_survival_targets(self, y):
    return unpack_survival_targets(y, self.device, self.survival_cut_points_years, self.num_time_bins)

def _survival_label_tensor(self, time_bin, event):
    return survival_label_tensor(time_bin, event)
```

Add to `base_model.py` imports:
```python
from medsurvival3d.utils.survival_labels import (
    survival_year_values,
    format_survival_landmark_label,
    time_to_survival_bin,
    interval_frac,
    unpack_survival_targets,
    survival_label_tensor,
)
```

- [ ] **Step 3: Byte-compile both files**

Run: `~/miniconda3/envs/fm_agent/bin/python -m py_compile medsurvival3d/utils/survival_labels.py base_model.py`
Expected: no output (success)

- [ ] **Step 4: Dependency-free tests still pass**

Run: `~/miniconda3/envs/fm_agent/bin/python -m pytest tests/test_loss_factory.py -k composite -q`
Expected: PASS (same count as before)

- [ ] **Step 5: Commit**

```bash
git add medsurvival3d/utils/survival_labels.py base_model.py
git commit -m "refactor(utils): extract stateless survival label helpers"
```

---

### Task 3: `utils/io.py` — move blosc2 I/O

**Files:**
- Create: `medsurvival3d/utils/io.py` (full contents of `datasets/blosc2io.py`)
- Modify: `datasets/blosc2io.py` → shim

- [ ] **Step 1: Copy `datasets/blosc2io.py` verbatim to `medsurvival3d/utils/io.py`**

- [ ] **Step 2: Replace `datasets/blosc2io.py` with a shim**

```python
from medsurvival3d.utils.io import *  # noqa: F401,F403
```

- [ ] **Step 3: Byte-compile**

Run: `~/miniconda3/envs/fm_agent/bin/python -m py_compile medsurvival3d/utils/io.py datasets/blosc2io.py`
Expected: no output

- [ ] **Step 4: Commit**

```bash
git add medsurvival3d/utils/io.py datasets/blosc2io.py
git commit -m "refactor(utils): move blosc2 I/O to medsurvival3d.utils.io"
```

---

### Task 4: `training/optim.py` — move the two LR schedulers

**Files:**
- Create: `medsurvival3d/training/optim.py` (the two classes from `base_model.py:1138-1290`)
- Modify: `base_model.py` (remove the two class defs; import them instead)

- [ ] **Step 1: Move `CosineAnnealingLR_Warmstart` and `CosineAnnealingLR_DoubleWarmstart` verbatim into `medsurvival3d/training/optim.py`**

Module header:
```python
"""LR schedulers and optimizer construction for survival training."""
import math
from torch.optim.lr_scheduler import _LRScheduler
```

- [ ] **Step 2: In `base_model.py`, delete the two class definitions and import them**

```python
from medsurvival3d.training.optim import (
    CosineAnnealingLR_Warmstart,
    CosineAnnealingLR_DoubleWarmstart,
)
```

- [ ] **Step 3: Byte-compile**

Run: `~/miniconda3/envs/fm_agent/bin/python -m py_compile medsurvival3d/training/optim.py base_model.py`
Expected: no output

- [ ] **Step 4: Dependency-free tests still pass**

Run: `~/miniconda3/envs/fm_agent/bin/python -m pytest tests/test_loss_factory.py -k composite -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add medsurvival3d/training/optim.py base_model.py
git commit -m "refactor(training): move LR schedulers to training.optim"
```

---

### Task 5: `evaluation/metrics.py` — move metric functions out of `survival_utils.py`

Move these from `survival_utils.py`: `concordance_index` (9), `time_dependent_auc`
(20), `_valid_time_grid` (55), `_ibs_torchsurv` (64), `integrated_brier_score`
(82), `integrated_brier_score_ipcw` (94), `_logrank_chi2` (542),
`max_logrank_cutpoint` (559), `derive_stratification_scores` (615).

**Files:**
- Create: `medsurvival3d/evaluation/metrics.py`
- Modify: `survival_utils.py` (remove these defs; re-export them from the new module so existing importers keep working)

- [ ] **Step 1: Move the nine functions verbatim into `medsurvival3d/evaluation/metrics.py`**

Module header (imports they use):
```python
"""Survival evaluation metrics (torchsurv-backed; cluster-only deps)."""
import torch
```
(The `torchsurv`/`numpy` imports inside these functions are local — keep them as-is.)

- [ ] **Step 2: In `survival_utils.py`, delete the nine defs and re-export**

```python
from medsurvival3d.evaluation.metrics import (  # noqa: F401
    concordance_index,
    time_dependent_auc,
    _valid_time_grid,
    _ibs_torchsurv,
    integrated_brier_score,
    integrated_brier_score_ipcw,
    _logrank_chi2,
    max_logrank_cutpoint,
    derive_stratification_scores,
)
```

- [ ] **Step 3: Byte-compile**

Run: `~/miniconda3/envs/fm_agent/bin/python -m py_compile medsurvival3d/evaluation/metrics.py survival_utils.py`
Expected: no output

- [ ] **Step 4: Dependency-free tests still pass**

Run: `~/miniconda3/envs/fm_agent/bin/python -m pytest tests/test_loss_factory.py -k composite -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add medsurvival3d/evaluation/metrics.py survival_utils.py
git commit -m "refactor(evaluation): move survival metrics to evaluation.metrics"
```

---

### Task 6: `models/losses.py` — move loss functions/classes + `call_one_loss`

Move the loss half of `survival_utils.py`: `logits_to_hazard` (121),
`hazard_to_survival` (126), `survival_to_time` (131), `soft_logrank_loss` (136),
`group_balance_penalty` (179), the loss classes `NLLSurvLoss`…`WeibullLoss`,
`SoftLogRankLoss`, `CompositeSurvivalLoss` (193-376), `_reject_legacy_cox_loss_lambda`
(377), `_SINGLE_LOSS_NAMES` (386), `_parse_composite` (392),
`_build_single_criterion` (451), `build_survival_criterion` (488). Also bring the
`_call_one_loss` dispatch and `_SURVIVAL_LOSS_TAGS` from `base_model.py`.

**Files:**
- Create: `medsurvival3d/models/losses.py`
- Modify: `survival_utils.py` (remove the moved loss defs; re-export from new module — `survival_utils.py` is now purely a shim)
- Modify: `base_model.py` (replace `_call_one_loss` with a wrapper calling the free `call_one_loss`; import `_SURVIVAL_LOSS_TAGS`)

- [ ] **Step 1: Move the loss functions/classes/builders verbatim into `medsurvival3d/models/losses.py`**

Module header:
```python
"""Survival loss functions, criteria, and the loss-dispatch map."""
import torch
import torch.nn as nn
from medsurvival3d.utils.survival_labels import interval_frac
```

- [ ] **Step 2: Add `_SURVIVAL_LOSS_TAGS` and `call_one_loss` to `models/losses.py`**

Copy `_SURVIVAL_LOSS_TAGS` (the dict at `base_model.py:26-37`) verbatim. Add the
pure dispatch (body from `base_model._call_one_loss:339-357`, with the pchazard
branch using the imported `interval_frac` and `bin_edges` passed in):

```python
def call_one_loss(name, criterion, y_hat, time_bin, event, continuous_time, bin_edges):
    """Run one survival criterion -> (loss_tensor, components). Pure; no module state."""
    if name == "nll":
        return criterion(y_hat["logits"], time_bin, event), {}
    if name == "cox":
        return criterion(y_hat["risk"], continuous_time, event), {}
    if name == "deephit":
        return criterion(y_hat["pmf_logits"], time_bin, event), {}
    if name == "soft_logrank":
        total, components = criterion(y_hat["p_high"], continuous_time, event)
        return total, components
    if name == "pmf":
        return criterion(y_hat["pmf_logits"], time_bin, event), {}
    if name in ("mtlr", "bcesurv"):
        return criterion(y_hat["logits"], time_bin, event), {}
    if name == "weibull":
        return criterion(y_hat["weibull_params"], continuous_time, event), {}
    if name == "pchazard":
        frac = interval_frac(continuous_time, time_bin, bin_edges)
        return criterion(y_hat["logits"], time_bin, event, frac), {}
    raise ValueError(f"Unexpected survival_loss_name: {name!r}")
```

- [ ] **Step 3: Turn `survival_utils.py` into a full shim**

After Task 5 + this task, `survival_utils.py` re-exports everything:
```python
from medsurvival3d.evaluation.metrics import (  # noqa: F401
    concordance_index, time_dependent_auc, _valid_time_grid, _ibs_torchsurv,
    integrated_brier_score, integrated_brier_score_ipcw, _logrank_chi2,
    max_logrank_cutpoint, derive_stratification_scores,
)
from medsurvival3d.models.losses import (  # noqa: F401
    logits_to_hazard, hazard_to_survival, survival_to_time, soft_logrank_loss,
    group_balance_penalty, NLLSurvLoss, CoxPHLoss, DeepHitLoss, PMFLoss, MTLRLoss,
    BCESurvLoss, PCHazardLoss, WeibullLoss, SoftLogRankLoss, CompositeSurvivalLoss,
    _reject_legacy_cox_loss_lambda, _SINGLE_LOSS_NAMES, _parse_composite,
    _build_single_criterion, build_survival_criterion,
)
```

- [ ] **Step 4: In `base_model.py`, replace `_call_one_loss` and import the tag map**

```python
from medsurvival3d.models.losses import call_one_loss, _SURVIVAL_LOSS_TAGS
```
Delete the local `_SURVIVAL_LOSS_TAGS` dict (lines 26-37). Replace the method:
```python
def _call_one_loss(self, name, criterion, y_hat, time_bin, event, continuous_time):
    return call_one_loss(
        name, criterion, y_hat, time_bin, event, continuous_time,
        self._survival_bin_edges,
    )
```

- [ ] **Step 5: Byte-compile**

Run: `~/miniconda3/envs/fm_agent/bin/python -m py_compile medsurvival3d/models/losses.py survival_utils.py base_model.py`
Expected: no output

- [ ] **Step 6: Dependency-free tests still pass**

Run: `~/miniconda3/envs/fm_agent/bin/python -m pytest tests/test_loss_factory.py -k composite -q`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add medsurvival3d/models/losses.py survival_utils.py base_model.py
git commit -m "refactor(models): move survival losses + dispatch to models.losses"
```

---

### Task 7: Move backbone and head into `models/`

`models/resenc.py` and `models/survival_head.py` import from `survival_utils`
(curve transforms / functions). They keep working via the shim; update their
imports to the new path in this task.

**Files:**
- Create: `medsurvival3d/models/backbones/resenc.py` (verbatim from `models/resenc.py`)
- Create: `medsurvival3d/models/heads/survival_head.py` (verbatim from `models/survival_head.py`)
- Modify: the two new files' imports (`from survival_utils import …` → `from medsurvival3d.models.losses import …` for the curve transforms; verify which names each uses)
- Modify: `models/resenc.py`, `models/survival_head.py` → shims

- [ ] **Step 1: Copy both files into their new locations verbatim**

- [ ] **Step 2: Update imports in the two new files**

For each, replace `from survival_utils import X, Y` with the new source. The
curve transforms (`logits_to_hazard`, `hazard_to_survival`, `survival_to_time`)
now live in `medsurvival3d.models.losses`. Confirm names per file:
```
grep -n "from survival_utils import\|survival_utils\." medsurvival3d/models/backbones/resenc.py medsurvival3d/models/heads/survival_head.py
```
Update each to `from medsurvival3d.models.losses import <names>`.

- [ ] **Step 3: Replace the originals with shims**

`models/resenc.py`:
```python
from medsurvival3d.models.backbones.resenc import *  # noqa: F401,F403
```
`models/survival_head.py`:
```python
from medsurvival3d.models.heads.survival_head import *  # noqa: F401,F403
```

- [ ] **Step 4: Byte-compile**

Run: `~/miniconda3/envs/fm_agent/bin/python -m py_compile medsurvival3d/models/backbones/resenc.py medsurvival3d/models/heads/survival_head.py models/resenc.py models/survival_head.py`
Expected: no output

- [ ] **Step 5: Commit**

```bash
git add medsurvival3d/models/backbones/ medsurvival3d/models/heads/ models/resenc.py models/survival_head.py
git commit -m "refactor(models): move resenc backbone + survival head into package"
```

---

### Task 8: Move datamodule, dataset, and transforms into `data/`

Datamodule logic is unchanged (per spec) — only file location and import lines
change.

**Files:**
- Create: `medsurvival3d/data/base_datamodule.py` (from `datasets/base_datamodule.py`)
- Create: `medsurvival3d/data/datamodules.py` (from `datasets/coca_t1c_combined_b2nd.py`)
- Create: `medsurvival3d/data/survival.py` (from `datasets/survival.py`)
- Create: `medsurvival3d/data/batchgenerators_transforms.py` (from `augmentation/policies/batchgenerators.py`)
- Create: `medsurvival3d/data/preprocessing/*` (from `datasets/preprocess_3D_data/*`)
- Modify: new files' internal imports (e.g. `from .blosc2io import` → `from medsurvival3d.utils.io import`; `from .base_datamodule import` → `from medsurvival3d.data.base_datamodule import`)
- Modify: originals → shims

- [ ] **Step 1: Copy each file to its new location verbatim**

Preserve the `preprocess_3D_data/` subtree under `data/preprocessing/` (including
its `datasets/` subdir).

- [ ] **Step 2: Update intra-package imports in the new files**

Find them:
```
grep -rn "from \.\|import blosc2io\|from datasets\|from augmentation" medsurvival3d/data/
```
Rewrite each to absolute `medsurvival3d.…` paths. The datamodule's
`_target_` in configs references `datasets.coca_t1c_combined_b2nd.…`; **do not
change configs in Phase 1** — keep the original `datasets/coca_t1c_combined_b2nd.py`
shim so Hydra `_target_` still resolves.

- [ ] **Step 3: Replace originals with shims**

Example `datasets/coca_t1c_combined_b2nd.py`:
```python
from medsurvival3d.data.datamodules import *  # noqa: F401,F403
```
Apply the analogous shim to each moved original. (The `_target_`-referenced
module must still expose the datamodule class via the shim's `*`.)

- [ ] **Step 4: Byte-compile the new `data/` tree**

Run: `~/miniconda3/envs/fm_agent/bin/python -m compileall -q medsurvival3d/data`
Expected: no output

- [ ] **Step 5: Commit**

```bash
git add medsurvival3d/data/ datasets/ augmentation/
git commit -m "refactor(data): move datamodule, dataset, transforms, preprocessing into package"
```

---

### Task 9: Move inference into `inference/survival.py`

**Files:**
- Create: `medsurvival3d/inference/survival.py` (from `inference_survival.py`)
- Modify: new file's imports (`from survival_utils import …` → `from medsurvival3d.evaluation.metrics import …` / `medsurvival3d.models.losses`)
- Modify: `inference_survival.py` → shim

- [ ] **Step 1: Copy `inference_survival.py` verbatim to `medsurvival3d/inference/survival.py`**

- [ ] **Step 2: Update its imports**

```
grep -n "from survival_utils import\|survival_utils\." medsurvival3d/inference/survival.py
```
Route metric names to `medsurvival3d.evaluation.metrics`, loss/curve names to
`medsurvival3d.models.losses`.

- [ ] **Step 3: Replace `inference_survival.py` with a shim**

```python
from medsurvival3d.inference.survival import *  # noqa: F401,F403
# Names used via base_model's lazy import must be re-exported:
from medsurvival3d.inference.survival import (  # noqa: F401
    compute_hazard_ratio,
    compute_logrank_stat,
)
```

- [ ] **Step 4: Byte-compile**

Run: `~/miniconda3/envs/fm_agent/bin/python -m py_compile medsurvival3d/inference/survival.py inference_survival.py`
Expected: no output

- [ ] **Step 5: Commit**

```bash
git add medsurvival3d/inference/survival.py inference_survival.py
git commit -m "refactor(inference): move inference_survival into package"
```

---

### Task 10: Move `BaseModel`/`ModelConstructor` into `training/trainer.py`

The slimmed `LightningModule`: keeps `__init__`, `forward`, step/epoch hooks,
logging, metric buffers, `_survival_loss`, `configure_optimizers`,
`_log_survival_metrics`, `_compute_stratification_metrics`. It now imports the
relocated functions directly (not via the `survival_utils` shim).

**Files:**
- Create: `medsurvival3d/training/trainer.py` (from `base_model.py`, minus the parts already moved in Tasks 2/4/6)
- Modify: `base_model.py` → shim

- [ ] **Step 1: Copy the remaining `base_model.py` (after Tasks 2/4/6 edits) into `medsurvival3d/training/trainer.py`**

It already contains the wrapper methods and imports added in Tasks 2/4/6.

- [ ] **Step 2: Repoint `trainer.py`'s imports to final locations**

Replace the top-of-file `from survival_utils import (...)` block with:
```python
from medsurvival3d.evaluation.metrics import (
    concordance_index, integrated_brier_score, integrated_brier_score_ipcw,
    time_dependent_auc, max_logrank_cutpoint, derive_stratification_scores,
)
from medsurvival3d.models.losses import (
    build_survival_criterion, _reject_legacy_cox_loss_lambda,
    call_one_loss, _SURVIVAL_LOSS_TAGS,
)
```
Update the lazy import in `_compute_stratification_metrics` (currently
`from inference_survival import compute_hazard_ratio, compute_logrank_stat`) to
`from medsurvival3d.inference.survival import compute_hazard_ratio, compute_logrank_stat`.
Keep backbone/head imports pointed at `medsurvival3d.models.backbones.resenc` /
`medsurvival3d.models.heads.survival_head`.

- [ ] **Step 3: Replace `base_model.py` with a shim**

```python
from medsurvival3d.training.trainer import *  # noqa: F401,F403
from medsurvival3d.training.trainer import BaseModel, ModelConstructor  # noqa: F401
```

- [ ] **Step 4: Byte-compile**

Run: `~/miniconda3/envs/fm_agent/bin/python -m py_compile medsurvival3d/training/trainer.py base_model.py`
Expected: no output

- [ ] **Step 5: Commit**

```bash
git add medsurvival3d/training/trainer.py base_model.py
git commit -m "refactor(training): move BaseModel/ModelConstructor into training.trainer"
```

---

### Task 11: Repoint `main.py` to package imports

**Files:**
- Modify: `main.py` (imports only; Hydra wiring and `cli_configs/` unchanged)

- [ ] **Step 1: Update `main.py` imports**

```
grep -n "from base_model import\|import base_model\|from survival_utils\|from inference_survival\|from models\.\|from datasets\.\|from augmentation\." main.py
```
Rewrite each to the `medsurvival3d.*` equivalent (e.g.
`from base_model import ModelConstructor` → `from medsurvival3d.training.trainer import ModelConstructor`).

- [ ] **Step 2: Byte-compile**

Run: `~/miniconda3/envs/fm_agent/bin/python -m py_compile main.py`
Expected: no output

- [ ] **Step 3: Commit**

```bash
git add main.py
git commit -m "refactor: repoint main.py to medsurvival3d imports"
```

---

### Task 12: Repoint all tests to package imports

24 test files import from `survival_utils`, `base_model`, `inference_survival`,
`models.*`. They pass via shims now; repoint them to final paths.

**Files:**
- Modify: every file under `tests/` that imports a moved module

- [ ] **Step 1: List the offending imports**

```
grep -rn "from survival_utils\|import survival_utils\|from base_model\|import base_model\|from inference_survival\|from models\.\|from datasets\.\|from augmentation\." tests/
```

- [ ] **Step 2: Rewrite each import to the `medsurvival3d.*` equivalent**

Mapping:
- `survival_utils` metric names → `medsurvival3d.evaluation.metrics`
- `survival_utils` loss/criterion names → `medsurvival3d.models.losses`
- `base_model` → `medsurvival3d.training.trainer`
- `inference_survival` → `medsurvival3d.inference.survival`
- `models.resenc` → `medsurvival3d.models.backbones.resenc`
- `models.survival_head` → `medsurvival3d.models.heads.survival_head`
- `datasets.*` → `medsurvival3d.data.*`

- [ ] **Step 3: Byte-compile all tests**

Run: `~/miniconda3/envs/fm_agent/bin/python -m compileall -q tests`
Expected: no output

- [ ] **Step 4: Dependency-free tests still pass via final paths**

Run: `~/miniconda3/envs/fm_agent/bin/python -m pytest tests/test_loss_factory.py -k composite -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/
git commit -m "refactor(tests): repoint imports to medsurvival3d package"
```

---

### Task 13: Delete the transition shims and old directories

All importers now use final paths, so the shims are orphans created by this work.

**Files:**
- Delete: `survival_utils.py`, `base_model.py`, `inference_survival.py`
- Delete: `models/survival_head.py` (old top-level shim; NOT `_target_`-referenced)
- Keep: `models/resenc.py`, `models/__init__.py` — `models.resenc.ResEncoder_Survival`
  is referenced by `cli_configs/model/resenc_survival.yaml` `_target_`
- Delete: old `datasets/` and `augmentation/` shims **except** any module a Hydra
  `_target_` still references (see note)

- [ ] **Step 1: Confirm nothing imports the shims**

```
grep -rn "from survival_utils\|import survival_utils\|from base_model\|import base_model\|from inference_survival\|^from models\.\|^from datasets\.\|^from augmentation\." --include=*.py . | grep -v medsurvival3d/ | grep -v __pycache__
```
Expected: only matches inside config `_target_` strings (handled in Phase 2), none in `.py` import statements.

- [ ] **Step 2: Handle Hydra `_target_` references**

`cli_configs/**/*.yaml` reference FOUR moved modules via `_target_` (verified by
`grep -rhoE "_target_: (datasets|augmentation|models)\.[A-Za-z0-9_.]+" cli_configs`):
`datasets.coca_t1c_combined_b2nd.*`, `augmentation.policies.batchgenerators.*`,
`datasets.survival.SurvivalDataModule` (cli_configs/data/survival.yaml), and
`models.resenc.ResEncoder_Survival` (cli_configs/model/resenc_survival.yaml).
Phase 1 does **not** edit configs, so **keep all four shim files**
(`datasets/coca_t1c_combined_b2nd.py`, `datasets/survival.py`, `models/resenc.py`,
`augmentation/policies/batchgenerators.py`, plus the `__init__.py` chain needed to
import them) until Phase 2 rewrites the configs. Delete all other shims.
NOTE: do NOT rely on the Step 1 grep's `| grep -v medsurvival3d/` to find these —
`_target_` strings live in YAML, so scan `cli_configs/` separately as shown above.

- [ ] **Step 3: Delete the confirmed-orphan shims**

```bash
git rm survival_utils.py base_model.py inference_survival.py models/survival_head.py
# Delete only the datasets/augmentation shims NOT referenced by a _target_:
git rm datasets/base_datamodule.py datasets/blosc2io.py
git rm -r datasets/preprocess_3D_data
```
(Leave the four `_target_` shims + their `__init__` chain for Phase 2:
`datasets/coca_t1c_combined_b2nd.py`, `datasets/survival.py`, `datasets/__init__.py`,
`models/resenc.py`, `models/__init__.py`,
`augmentation/policies/batchgenerators.py`, `augmentation/__init__.py`,
`augmentation/policies/__init__.py`.)

- [ ] **Step 4: Byte-compile + dependency-free tests**

Run: `~/miniconda3/envs/fm_agent/bin/python -m compileall -q medsurvival3d main.py tests && ~/miniconda3/envs/fm_agent/bin/python -m pytest tests/test_loss_factory.py -k composite -q`
Expected: compile clean; tests PASS

- [ ] **Step 5: Commit**

```bash
# Stage the deletions explicitly — do NOT `git add -A` (it sweeps in
# untracked .codegraph/.gitignore and any other stray files).
git add -u survival_utils.py base_model.py inference_survival.py models/survival_head.py \
  datasets/base_datamodule.py datasets/blosc2io.py datasets/preprocess_3D_data
git status --porcelain   # verify: only the intended deletions are staged
git commit -m "refactor: remove orphaned transition shims (keep _target_ shims for Phase 2)"
```

---

### Task 14: Acyclic import check + final verification

**Files:** none (verification only)

- [ ] **Step 1: Each leaf imports in isolation (no cycle)**

Run:
```
~/miniconda3/envs/fm_agent/bin/python -c "import medsurvival3d.utils.survival_labels, medsurvival3d.utils.io, medsurvival3d.training.optim; print('leaves ok')"
```
Expected: `leaves ok` (no `ImportError` for circular import). NOTE:
`medsurvival3d.models.losses`, `evaluation.metrics`, `training.trainer`, and
`inference.survival` import pycox/torchsurv and will `ModuleNotFoundError` on
macOS — that is expected and is NOT a circular-import failure. Distinguish: a
circular import raises `ImportError: cannot import name … (most likely due to a
circular import)`; a missing dep raises `ModuleNotFoundError: No module named
'pycox'`.

- [ ] **Step 2: Full byte-compile of the package**

Run: `~/miniconda3/envs/fm_agent/bin/python -m compileall -q medsurvival3d main.py tests`
Expected: no output

- [ ] **Step 3: Dependency-free suite**

Run: `~/miniconda3/envs/fm_agent/bin/python -m pytest tests/test_loss_factory.py -k composite -q`
Expected: PASS

- [ ] **Step 4: Update HANDOFF with the cluster gate**

Add to `HANDOFF.md`: Phase 1 complete and locally structural-verified; the
**cluster behavioral gate is unrun** — next session must run the full `pytest`
suite + one t1c training step on the cluster and confirm parity before Phase 2.

- [ ] **Step 5: Commit**

```bash
git add HANDOFF.md
git commit -m "docs: record Phase 1 done (local), cluster gate pending"
```

---

## Self-review notes

- **Spec coverage:** layout (Tasks 1,3,7,8,9), base_model decomposition (Tasks
  2,4,6,10), survival_utils bisection (Tasks 5,6), import layering check (Task
  14), test repoint (Task 12), Hydra/main unchanged (Task 11 + Task 13 Step 2),
  verification model (every task + Task 14). Packaging is explicitly Phase 2.
- **Cluster gate:** behavioral equivalence is NOT provable locally; Task 14 Step
  4 records this and blocks Phase 2 on it.
- **`_target_` shims:** the one subtlety the engineer must not miss — Task 13
  Step 2 keeps the two config-referenced modules alive until Phase 2.
