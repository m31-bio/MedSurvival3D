# MedSurvival3D Restructure — Design

**Date:** 2026-06-15
**Status:** Approved (pending spec review)
**Scope:** Restructure `SSL3D_survival` to follow the
[MedClass3D](https://github.com/AaronC-BME/MedClass3D) directory layout, keeping
Hydra, and decompose the `base_model.py` monolith into focused modules.

## Goal

Adopt MedClass3D's `src/`-layout package structure (`data/ models/ training/
evaluation/ inference/ utils/`) and split the 1298-line `base_model.py`
`LightningModule` into focused, independently testable modules — **without
changing training/inference behavior**. The 24-file test suite is the regression
gate.

## Decisions (locked)

| Decision | Choice |
|---|---|
| Config system | **Keep Hydra** (composition + groups). Not flattening to self-contained YAML. |
| Decomposition mechanism | **Hybrid (C)** — extract stateless math/mappings as pure functions; keep logging, metric buffers, and the Lightning lifecycle on the module. |
| Packaging scope | **Full MedClass3D parity, phased.** Module split first (Phase 1), packaging/entry-points second (Phase 2). |
| Package name | `medsurvival3d` (project: **MedSurvival3D**) |
| Install | **uv** (`uv pip install -e .`) |
| Verification | **Local proves structure** (byte-compile, acyclic import graph, dependency-free tests). **Cluster proves behavior** (full `pytest` + a real training step). |

### Why Hybrid (C), not pure-function (A) or mixins (B)

`base_model.py` is a single stateful `LightningModule` whose losses/metrics/steps
are methods sharing `self.survival_*` config, metric buffers, and `self.hparams`.
- **B (mixins)** keeps `self` shared → boundaries are cosmetic, no real
  decoupling.
- **A (pure functions everywhere)** rewrites the whole step/epoch lifecycle →
  highest churn/risk against a suite we can only fully run on the cluster.
- **C** lifts the *leaf computations* (loss dispatch, metric math, label
  transforms, schedulers) out from under `self` while leaving the
  buffer/lifecycle orchestration intact. This yields the real win
  (`losses`/`metrics`/`survival_labels` become unit-testable) at low risk.

## Phasing trick: stable imports

Phase 1 builds the package **at repo root** (`./medsurvival3d/`), importable via
cwd exactly like the current flat layout — no install required. Phase 2 does
`git mv medsurvival3d src/medsurvival3d` and adds `pyproject.toml`. Because the
package **name** never changes, every `import medsurvival3d.…` statement written
in Phase 1 remains valid in Phase 2. **Imports are rewritten once, not twice.**

## Target directory layout (end state, after both phases)

```
MedSurvival3D/
├── pyproject.toml                    # Phase 2 (uv, deps incl. pycox/torchsurv, console scripts)
├── configs/                          # Phase 2 rename of cli_configs/ (Hydra groups kept)
│   ├── env/  model/  data/
├── scripts/                          # Phase 2
│   ├── train.py                      # thin Hydra launcher (was main.py)
│   ├── predict_test.py  predict_external.py   # wrap inference/
│   └── preprocess_*.py               # wrap data/preprocessing/
├── src/medsurvival3d/                # Phase 1 builds at repo root; Phase 2 moves under src/
│   ├── __init__.py   cli.py          # cli.py = Phase 2
│   ├── data/                         # datamodule kept AS-IS (logic unchanged, imports updated)
│   │   ├── base_datamodule.py
│   │   ├── datamodules.py            # <- coca_t1c_combined_b2nd.py
│   │   ├── survival.py               # <- datasets/survival.py
│   │   ├── batchgenerators_transforms.py   # <- augmentation/policies/batchgenerators.py
│   │   └── preprocessing/            # <- datasets/preprocess_3D_data/
│   ├── models/
│   │   ├── backbones/resenc.py       # <- models/resenc.py (nnssl checkpoint loader kept)
│   │   ├── heads/survival_head.py    # <- models/survival_head.py
│   │   └── losses.py                 # <- survival_utils.py + extracted call_one_loss()
│   ├── training/
│   │   ├── trainer.py                # slimmed BaseModel + ModelConstructor (orchestration)
│   │   └── optim.py                  # 2 LR schedulers + optimizer/scheduler construction
│   ├── evaluation/
│   │   └── metrics.py                # extracted pure C-index/Brier/AUC/KM/logrank/stratification
│   ├── inference/
│   │   └── survival.py               # <- inference_survival.py
│   └── utils/
│       ├── io.py                     # <- datasets/blosc2io.py
│       └── survival_labels.py        # extracted target/bin/label helpers
└── tests/                            # imports updated to medsurvival3d.*
```

Notes:
- `blosc2io.py` → `utils/io.py` (mirrors MedClass3D), confirmed acceptable.
- "Datamodule as-is" = logic untouched; files still move into `data/` and get
  `import` lines updated (unavoidable when the package moves).

## `base_model.py` decomposition

Guiding rule: **math and mappings move out as pure functions; logging, metric
buffers, and the Lightning lifecycle stay on the module.**

| Current method(s) in `base_model.py` | Destination | Rationale |
|---|---|---|
| `__init__`, `forward`, `training_step`, `validation_step`, `predict_step`, `on_*` hooks | `training/trainer.py` (BaseModel, slimmed) | Lightning lifecycle — stateful, stays |
| `ModelConstructor(BaseModel)` | `training/trainer.py` | subclass of the above |
| `_survival_loss`, `_log_composite_member_losses`, `_log_smoothed_*` | `training/trainer.py` | orchestration + `self.log`; delegate to losses |
| `_update_survival_metric_buffers` | `training/trainer.py` | owns `CatMetric`/`MetricCollection` buffer state |
| `_call_one_loss` | `models/losses.py` → `call_one_loss(...)` | pure y_hat→loss dispatch; no `self` |
| *(all of `survival_utils.py`)* | `models/losses.py` | already pure: `build_survival_criterion`, `_parse_composite`, `CompositeSurvivalLoss`, `_build_single_criterion`, `_SINGLE_LOSS_NAMES` |
| metric **math** in `_log_survival_metrics` (C-index, Brier, AUC) | `evaluation/metrics.py` pure fns | tensors in, floats out; trainer logs results |
| `_compute_stratification_metrics`, `_resolve_stratification_landmark_bin` (logrank/HR/KM-cutpoint math) | `evaluation/metrics.py` pure fns | same pattern |
| `_survival_year_values`, `_format_survival_landmark_label`, `_time_to_survival_bin`, `_interval_frac`, `_unpack_survival_targets`, `_survival_label_tensor` | `utils/survival_labels.py` | bin/label/target transforms — config in, tensors out |
| `CosineAnnealingLR_Warmstart`, `CosineAnnealingLR_DoubleWarmstart` + optimizer/scheduler construction from `configure_optimizers` | `training/optim.py` | builders; `configure_optimizers` stays on trainer and calls them |

### The clean seam: epoch-end boundary

Buffers (`CatMetric`) accumulate per-step **inside** the trainer (stateful,
stays). At `on_validation_epoch_end`, the trainer hands the concatenated
`(preds, times, events)` tensors to pure `evaluation.metrics.compute_*`
functions and logs what comes back. This makes the metric math unit-testable
without instantiating a `LightningModule` or `wandb`.

Slimmed `trainer.py` ends up ~400–500 lines of genuine lifecycle/orchestration;
`losses.py`, `evaluation/metrics.py`, and `utils/survival_labels.py` become
independently importable and testable.

### Test handling

Suites such as `test_stratification_metrics`, `test_train_brier`,
`test_inference_concordance` may be pointed at the new pure functions, but in
Phase 1 their **behavior/assertions are preserved** — only import paths and call
targets change. No rewriting of what they assert.

## Import layering (circular-import avoidance)

Strict layering keeps the graph acyclic:
- **Leaves** (import nothing internal): `utils/`, `training/optim.py`
- **Mid** (may import `utils/`): `models/losses.py`, `evaluation/metrics.py`
- **Top** (import the rest): `training/trainer.py`, `inference/survival.py`

Nothing low may import `trainer`. "Import graph resolves" is a Phase-1 DoD item
to catch any violation immediately.

## Phases & definition of done

### Phase 1 — Module split (repo root, no packaging)

**Does:** create `medsurvival3d/` at repo root; move data/models/inference/
augmentation files with updated imports; split `base_model.py` per the table;
fold `survival_utils.py` into `models/losses.py`; extract metric math, label
helpers, schedulers/builders, and `blosc2io.py` as above. `main.py` stays at root
with updated imports (Hydra unchanged). All 24 test files get import-path updates
only.

**Definition of done:**
- *Local (assistant proves):* every module byte-compiles; import graph is
  acyclic; the dependency-free test subset passes (`test_loss_factory -k
  composite` + any suite not importing pycox/torchsurv).
- *Cluster (user proves — the real gate):* full `pytest` passes with results
  identical to pre-refactor; one training step runs on a t1c config.

### Phase 2 — Packaging & entry points

**Does:** `git mv medsurvival3d src/medsurvival3d`; add `pyproject.toml` (uv,
deps incl. `pycox`/`torchsurv`/`sksurv`/`lifelines`, console scripts);
`main.py` → `scripts/train.py` (thin Hydra launcher) plus `predict_test.py`/
`predict_external.py`/`preprocess_*.py` wrappers; `cli_configs/` → `configs/`
(update Hydra `config_path`). No import statements change.

**Definition of done:**
- *Local:* `uv pip install -e .` resolves; package imports after install;
  dependency-free tests pass via the installed package.
- *Cluster:* full suite + a real training run launched through `scripts/train.py`
  with Hydra.

## Out of scope

- Flattening Hydra to self-contained YAML (explicitly keeping Hydra).
- Changing the on-disk data format (`survival_labels.json` + `splits.json` stay;
  no move to `split_labels.csv`).
- New features present in MedClass3D but absent here (`mixup`,
  `balanced_sampler`, multiple backbones/heads, CT/MRI-specific preprocess
  scripts). Folders are created only where current code maps to them.
- Any datamodule logic changes.
- Fixing the parked Brier/AUC time-axis mismatch (`TODO.md` §1) — unrelated.

## Verification constraint (explicit)

This dev machine (macOS) has no `pycox`/`torchsurv`, so the full suite cannot run
locally. Local DoD is structural only; behavioral equivalence is gated on a
cluster `pytest` run at each phase boundary. Each handoff documents exactly what
remains unverified.
