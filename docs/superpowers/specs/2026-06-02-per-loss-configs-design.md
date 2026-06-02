# Per-loss config suite for t1c — Design

Date: 2026-06-02

## Problem

The survival loss factory (`build_survival_criterion`, `survival_utils.py:369`)
supports 9 losses: `nll`, `cox`, `deephit`, `soft_logrank`, `pmf`, `mtlr`,
`bcesurv`, `weibull`, `pchazard`. The config tree does not reflect this:

- Only 3 configs set `survival_loss` (2× `deephit`, 1× `soft_logrank`).
- The 2 `deephit` configs carry dead keys `beta`/`gamma` that the factory never
  reads (`DeepHitLoss.__init__` accepts only `alpha`, `sigma`), and use
  `alpha: 1.0`, which disables the ranking term entirely.
- There is no ready-to-run config demonstrating the other 6 losses, and no
  single place documenting each loss's available parameters.

## Goal

Give each of the 9 losses a documented, ready-to-train t1c config, and remove
the stale keys from the existing live configs.

## Scope decisions (confirmed with user)

- **One config per loss** (not just fix-in-place, not a central reference file).
- **Clone the full-training `high_vs_low` t1c base** (`cv.k=5`, balanced split,
  `lr 5e-6`), monitor `Val/C-index` for all new configs.
- **Defaults + documented ranges**: use the factory's default hyperparameter
  values, annotated with inline `# HPARAM` comments (meaning + tuning range),
  matching the style of the existing `methylome_t1c_combined_soft_logrank.yaml`.
- **Drop dead `beta`/`gamma`** from the 2 existing live deephit configs; leave
  their `alpha` value untouched.

## Deliverables

### 8 new files under `cli_configs/data/`

| Loss | File | `survival_loss` block (factory defaults) |
|------|------|------------------------------------------|
| `nll` | `methylome_t1c_combined_nll.yaml` | `name: nll` (no tunable hyperparams) |
| `cox` | `methylome_t1c_combined_cox.yaml` | `name: cox`, `reduction: mean` |
| `deephit` | `methylome_t1c_combined_deephit.yaml` | `name: deephit`, `alpha: 0.2`, `sigma: 0.1` |
| `pmf` | `methylome_t1c_combined_pmf.yaml` | `name: pmf` (no tunable hyperparams) |
| `mtlr` | `methylome_t1c_combined_mtlr.yaml` | `name: mtlr` (no tunable hyperparams) |
| `bcesurv` | `methylome_t1c_combined_bcesurv.yaml` | `name: bcesurv` (no tunable hyperparams) |
| `weibull` | `methylome_t1c_combined_weibull.yaml` | `name: weibull`, `reduction: mean` |
| `pchazard` | `methylome_t1c_combined_pchazard.yaml` | `name: pchazard` (no tunable hyperparams) |

`soft_logrank` already has a config (`methylome_t1c_combined_soft_logrank.yaml`)
and serves as the documentation-style template; it is left unchanged.

### Structure of each new file

Clone `methylome_t1c_combined_high_vs_low.yaml` verbatim, changing only:

- `data.module.name` → unique per loss (e.g. `Methylome_T1c_Cox`).
- `trainer.logger.name` → unique per loss (e.g. `t1c_cox`).
- `model.survival_loss` → the loss-appropriate block from the table above,
  with documenting comments.
- Keep `early_stopping.monitor` / checkpoint `monitor` = `Val/C-index`
  (unchanged from the base).

Everything else (data module, paths, `fixed_survival_time_bins`, model
optimization knobs, trainer) is identical to the base so the 8 configs are
directly comparable.

### Edits to 2 existing live configs

- `cli_configs/data/methylome_t1c_combined_high_vs_low.yaml`
- `cli_configs/data/methylome_t2w_combined_high_vs_low.yaml`

Remove `beta` and `gamma` from their `survival_loss` blocks. Add brief
documenting comments for `alpha` and `sigma`. Do **not** change the `alpha`
value (left at the existing `1.0`).

## Per-loss parameter reference (from the factory)

- `nll` — pycox logistic-hazard NLL. Opts: `reduction` (default `mean`). No
  loss-shape hyperparams beyond reduction.
- `cox` — Cox partial likelihood (Efron ties). Opts: `reduction` (default `mean`).
- `deephit` — single-event DeepHit. `alpha` (default 0.2): NLL↔ranking tradeoff
  (`total = alpha*NLL + (1-alpha)*ranking`; `alpha=0` → pure ranking).
  `sigma` (default 0.1): ranking kernel temperature.
- `soft_logrank` — differentiable log-rank + balance penalty. `lambda_balance`
  (0.01), `min_frac` (0.20), `max_frac` (0.80). Documented in existing config.
- `pmf` — pycox PMF NLL. No extra opts.
- `mtlr` — pycox MTLR NLL. No extra opts.
- `bcesurv` — pycox BCESurv. No extra opts.
- `weibull` — parametric Weibull AFT (torchsurv). Opts: `reduction` (default `mean`).
- `pchazard` — pycox piecewise-constant hazard NLL. No extra opts (needs
  `interval_frac` at runtime, supplied by the model, not the config).

## Notes

- The prediction head (`models/survival_head.py`) already emits all four
  terminal projections (`fc_hazard`, `fc_pmf`, `fc_risk`, `fc_weibull`) and
  selects the active one from the loss name. No per-loss head/output-dim config
  is required.
- `reduction` for `cox`/`weibull` is included for documentation/visibility even
  though it equals the factory default; the no-hyperparam losses get an explicit
  comment stating they take no tunable parameters, so an empty block is not
  mistaken for something missing.

## Verification

- `python -c "import yaml, glob; [yaml.safe_load(open(f)) for f in glob.glob('cli_configs/data/methylome_t1c_combined_*.yaml')]"` parses all files.
- For each new file, `build_survival_criterion` accepts its `survival_loss`
  block (name returns the expected criterion; no `ValueError`).
- `grep -L beta` confirms `beta`/`gamma` are gone from the 2 edited configs.

## Out of scope

- No change to loss values/hyperparameters of existing live configs (only key
  removal).
- No new loss implementation (e.g. a standalone pairwise ranking loss).
- No t2w per-loss suite (only the t2w stale-key fix).
