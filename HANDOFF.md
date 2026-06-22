# HANDOFF

Session handoff for the next Claude working in `SSL3D_survival`. Read this
before starting. Last updated: 2026-06-18.

## Composite loss VERIFIED on cluster + 16-mixed fix (2026-06-18, UNCOMMITTED on `main`)

Verified the composite survival loss end-to-end on the workstation (real S3D SSL
ckpt, 1 epoch, 3 train/3 val batches, CSVLogger capture). Two bugs found & fixed:

1. **Composite head wiring** (`medsurvival3d/models/backbones/resenc.py:82`):
   passed raw `self.survival_loss_name` (`"composite"`) to `SurvivalHead`, whose
   `_VALID_LOSSES` guard rejected it → crash at model build. Fixed → pass
   `self.survival_primary_name` (resolved primary, e.g. `nll`). Composite-specific.
2. **`16-mixed` dtype crash in pycox losses** (`medsurvival3d/models/losses.py`):
   under autocast, float16 logits + float32 events crashed pycox in
   `NLLSurvLoss` (scatter) and `DeepHitLoss` (rank matmul). **Pre-existing** —
   not composite's fault; never caught because the default smoke config is
   `soft_logrank` (no pycox). Confirmed: single `nll` @16-mixed crashes identically.
   Fixed via `logits.float()` / `pmf_logits.float()` in those two `forward`s.
   NOTE pmf/mtlr/bcesurv/pchazard do NOT crash (type promotion) — left untouched.

**Evidence:** `Train/loss == 1.0·NLL + 0.3·Cox` to float precision (~2e-7) in
train & val; `CompositeLoss == loss`; metrics driven by `nll` primary. Composite
now trains at production `precision: 16-mixed`, `fit` exit 0.

**New test:** `tests/test_loss_amp_dtype.py` — 6 pycox losses accept float16
logits (RED on nll+deephit before fix, all GREEN after). Loss suite: 44 pass,
only pre-existing failure is `test_pchazard_surv_oracle_pycox` (CUDA `.numpy()`
without `.cpu()` — test-side bug, one of the known 14, unrelated to this work).

**Parked / next:** (a) commit these 3 files to `main`; (b) the 16-mixed fix
unblocks ALL pycox single-loss configs at default precision — worth a sanity run;
(c) `+trainer.csv_logger_dir` is incompatible with `cv.k>1` (logger popped inside
fold loop → `main.py:64` crash on fold 1); debug-only flag, low priority.

## Phase 1 package restructure — COMPLETE (2026-06-15, all on `main`)

The monolithic top-level modules were split into the `medsurvival3d` package per
`docs/superpowers/plans/2026-06-15-medsurvival3d-phase1.md` (Tasks 1–14 done).
Behavior-preserving: every move was a verbatim relocation + import repoint; no
logic changed. Final layout:

- `medsurvival3d/utils/` — `survival_labels.py`, `io.py` (Blosc2IO)
- `medsurvival3d/models/` — `losses.py`, `backbones/resenc.py`, `heads/survival_head.py`
- `medsurvival3d/evaluation/metrics.py`
- `medsurvival3d/training/` — `optim.py`, `trainer.py` (BaseModel, ModelConstructor)
- `medsurvival3d/inference/survival.py`
- `medsurvival3d/data/` — `base_datamodule.py`, `datamodules.py`, `survival.py`,
  `batchgenerators_transforms.py`, `preprocessing/`

**Four `_target_` shims intentionally KEPT** (Hydra configs still reference old paths
until Phase 2 rewrites them): `datasets/coca_t1c_combined_b2nd.py`,
`datasets/survival.py`, `models/resenc.py`, `augmentation/policies/batchgenerators.py`
(+ their `__init__` chains). All other transition shims were deleted (Task 13).
`main.py` needed no import changes (fully Hydra `instantiate`-driven).

**Verified locally:** acyclic internal import graph (static AST check), full
`compileall` of package + `main.py` + `tests/`, and `pytest tests/test_loss_factory.py
-k composite` (11 passed). Also fixed `.gitignore` (`data/` → `/data/`) so the package
`data/` subdir is trackable.

**CLUSTER GATE — PASSED (2026-06-17, workstation `aihub2.uniseg`). Phase 1 fully verified.**

1. ✅ **Full `pytest tests/` ran** on the workstation `.venv` (torch 2.12.0+cu130,
   py3.12): **131 passed, 14 failed — zero restructure regressions.** All 14 are
   pre-existing or environment artifacts, NOT caused by the restructure:
   - 11× `test_stratification_metrics` — stub sets `survival_loss_name`, code reads
     `survival_primary_name`. Verified pre-existing: `survival_primary_name` entered
     `_compute_stratification_metrics` in `9dc125b` (composite loss, pre-restructure),
     which did NOT touch the test; current `trainer.py:551` is byte-identical to the
     pre-restructure `base_model.py` line.
   - 2× `test_pchazard_*` pycox oracle — `.numpy()` on a CUDA tensor; only fires on a GPU box.
   - 1× `test_balance_zero_inside_range` — `3.5e-15 == 0.0` exact-equality on newer torch/BLAS.
2. ✅ **Training smoke PASSED** (`fast_dev_run`, full pipeline on REAL SSL weights).
   Ran from repo root with `WANDB_MODE=offline`:
   `main.py env=local data=methylome_t1c_combined_high_vs_low data.cv.k=1 data.module.fold=0`
   `exp_dir=/home/jma/scratch/ssl3d_smoke`
   `model.chpt_path=.../SSL3D_classification/checkpoints/S3D/checkpoint_final.pth`
   `model.save_preds=false trainer.enable_progress_bar=false trainer.callbacks.progressbar=null`
   `+trainer.fast_dev_run=true`.
   All four `_target_` shims resolved at runtime: 102M-param model built, SSL checkpoint
   loaded (single→2 input channels), datamodule + augmentation built, one train+val batch
   ran fwd/backward → `Trainer.fit stopped: max_steps=1 reached`. Gotchas confirmed:
   (a) progress bar MUST be disabled over non-interactive SSH (rich `clear_live()` →
   `IndexError: pop from empty list` otherwise); (b) leave `seed` UNSET so
   `deterministic=False`, avoiding the `avg_pool3d` non-deterministic-CUDA backward crash.
   Correction to a prior note: `survival_labels.json` + `splits_balanced_survival.json`
   ARE present in `Dataset019_UHN_Mayo_T1c_mask/` (earlier "mismatch" was a truncated `ls`).

Note: `pytest` was installed into the workstation `.venv` (not in `requirements.txt`).

## What earlier sessions did (all on `main`, committed)

1. **`9ca81f2` — per-loss config suite.** Added 8 ready-to-train t1c configs,
   one per survival loss, under `cli_configs/data/`:
   `methylome_t1c_combined_{nll,cox,deephit,pmf,mtlr,bcesurv,weibull,pchazard}.yaml`.
   Each clones the `methylome_t1c_combined_high_vs_low.yaml` base and varies only
   the `survival_loss` block + `data.module.name` + `trainer.logger.name`;
   monitor stays `Val/C-index`. `soft_logrank` already had its own config.
   Also dropped dead `beta`/`gamma` keys (never read by the factory) from the two
   live deephit configs (t1c + t2w high_vs_low) and documented `alpha`/`sigma`.
   That commit also carried a pre-existing working-tree change (the t1c base
   `split_file` switched to `splits_balanced_survival.json`).

2. **`2cc7b1a` — composite loss spec.**
   `docs/superpowers/specs/2026-06-02-composite-survival-loss-design.md`.

3. **`9dc125b` — composite weighted-sum survival loss.** New `name: composite`
   survival loss: a weighted sum of any subset of the 9 losses with a designated
   `primary` member that drives all metrics/inference. See next section.

## The composite loss — how it works

Config shape:
```yaml
survival_loss:
  name: composite
  primary: nll                       # member whose head output drives metrics/inference
  components:
    - {name: nll, weight: 1.0}
    - {name: cox, weight: 0.3, reduction: mean}
```
`total = sum(weight_i * loss_i)`. Example config:
`cli_configs/data/methylome_t1c_combined_composite.yaml`.

Code map:
- `survival_utils.py`:
  - `_SINGLE_LOSS_NAMES` — the 9 valid single-loss names.
  - `_parse_composite(cfg)` — validates + normalises a composite block,
    dependency-free (no pycox/torchsurv import). Returns `(components, primary)`.
  - `_build_single_criterion(name, cfg)` — extracted per-name construction,
    shared by single-loss and composite paths.
  - `CompositeSurvivalLoss(nn.Module)` — thin container: `members` (ModuleList),
    `names`, `weights`, `primary`. No `forward` — it is only a data holder.
  - `build_survival_criterion` — gains a `composite` branch.
- `base_model.py`:
  - `self.survival_primary_name` — set at construction; equals the composite's
    `primary`, else equals `survival_loss_name`. ALL output-selection branches
    (C-index/KM/inference: the sites around the old lines 370/596/754/793/812)
    read `survival_primary_name`, NOT `survival_loss_name`.
  - `_call_one_loss(name, criterion, y_hat, time_bin, event, continuous_time)`
    — the extracted "which y_hat projection feeds which loss" mapping; returns
    `(loss, components)`. Handles soft_logrank's tuple return + pchazard's
    `interval_frac`.
  - `_survival_loss` — composite path loops members, weight-sums into `total`,
    stores each member's UNWEIGHTED loss in `loss_parts` for logging.
  - `_log_composite_member_losses` — logs `Train|Val/member_<Name>Loss`.
  - `_SURVIVAL_LOSS_TAGS` gained a `"composite": "Composite"` entry.
- `inference_survival.py`:
  - `_resolve_survival_loss_name` resolves composite → `primary` (single choke
    point; every downstream branch flows from it).
- `tests/test_loss_factory.py`: 11 composite parse/validation tests.

## Verification status — IMPORTANT

- VERIFIED locally: the dependency-free parse/validation logic (11 tests pass)
  and byte-compilation of all changed files.
- NOT VERIFIED: actually building `CompositeSurvivalLoss` from real
  pycox/torchsurv members, and the `base_model` weighted-sum + metric/inference
  integration. A real composite training step has NOT been run.
- Reason: this dev machine (macOS) has no `pycox`/`torchsurv`. The `fm_agent`
  conda env has torch 2.11 + pytest but not the survival deps. Training happens
  on a Linux cluster (data paths are `/home/jma/...`).
- TO VERIFY: run a short training with
  `cli_configs/data/methylome_t1c_combined_composite.yaml` on the cluster.
  Confirm it starts, `Train/loss` ≈ `1.0*NLL + 0.3*Cox`, and both
  `Train/member_NLLLoss` and `Train/member_CoxPHLoss` appear.

## How to run tests locally

```
~/miniconda3/envs/fm_agent/bin/python -m pytest tests/test_loss_factory.py -k composite -q
```
The non-composite factory tests in that file need pycox and will error locally;
filter with `-k composite` for the dependency-free ones.

## Parked / open items (NOT done — do not assume these are handled)

- **Brier/AUC time-axis mismatch** (`TODO.md` section 1): in
  `base_model._log_survival_metrics`, `integrated_brier_score*` is fed integer
  bin indices while `time_dependent_auc` is fed continuous years, so
  `Train/Brier` and `Train/AUC@5y` are NOT on a comparable axis. User chose
  "Not now" this session. Fix = either a one-line warning comment or unify the
  axis (investigate which axis is correct first).
- No t2w per-loss suite was created (only the t2w stale-key fix).
- Untracked files intentionally left alone: `.hpo_agent/`, `CLAUDE.md`,
  `TODO.md`.

## Project conventions observed this session

- Work is committed directly to `main` (this repo's pattern).
- Commit message co-author trailer is used.
- Config style: inline `# HPARAM` comments with meaning + tuning range (see
  `methylome_t1c_combined_soft_logrank.yaml` as the template).
- Brainstorm → spec (`docs/superpowers/specs/`) → TDD was the workflow used.
