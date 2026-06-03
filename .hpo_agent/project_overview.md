# SSL3D_survival — HPO Repo Profile Notes

## Goal (this onboarding session)

Tune hyperparameters for the **DeepHit** survival loss on the T1c+mask methylome
high-vs-low dataset, single fold, to maximize `Val/C-index` while keeping
training/validation curves stable and preferably achieving `Val/logrank_chi2 > 0`
(non-trivial stratification on the validation fold).

## Entry point & launch

- Script: `main.py` (Hydra entrypoint, `config_path=./cli_configs`, `config_name=train`).
- Hydra `run.dir: .` + `chdir: False` → process keeps its launching cwd; relative paths in `cli_configs/*` resolve only when cwd is the repo root.
- Trainer is constructed by Lightning; logger is `lightning.pytorch.loggers.WandbLogger`. `wandb.finish()` is called explicitly at end of each fold loop iteration.
- Default `enable_checkpointing=False`. We force it `False` again via `forced_overrides` and additionally lock `trainer.callbacks.checkpoint.save_top_k=0` as a belt-and-suspenders no-op.

## Config composition

`cli_configs/train.yaml` defaults:
- `env: local` — exists, but `exp_dir`/`data_dir` point to DKFZ paths (`/home/c306h/...`). On AIHub these directories don't exist → **must override `exp_dir`** to a path writable by `jma`.
- `model: resenc_survival` — the only survival-capable model config.
- `data: methylome_t1c_combined_soft_logrank` — default in the file. We pick `methylome_t1c_combined_high_vs_low` instead (this yaml selects `survival_loss.name: deephit`).

`env=cluster` is NOT viable: it pins `/dkfz/cluster/gpu/...` and also disables the progress bar — both bad for AIHub HPO runs. Use `env=local` plus `exp_dir=` override.

## Wandb integration

- Project: `nnSSL_MethylomeRisk_Survival` (set inside the data yaml).
- Group: a fresh UUID per launch (Hydra resolver `make_group_name`, with caching so it's stable within a run).
- Run name: `t1c_mask` from the data yaml → `main.py` mutates it to `t1c_mask_fold_<fold>`. The orchestrator should override `trainer.logger.name=<trial_id>` so each trial's wandb run is locatable by trial ID; final name will be `<trial_id>_fold_0`.
- Metrics logged that matter here:
  - `Val/C-index` (primary objective; logged epoch-end with smoothed variant `Val/C-index_smoothed`)
  - `Val/logrank_chi2`, `Val/logrank_p`, `Val/hazard_ratio` (epoch-end, computed via `_compute_stratification_metrics` using `max_logrank_cutpoint` on the configured quantile range, default `[0.2, 0.8]`)
  - `Val/loss`, `Val/Brier`, `Val/Brier-IPCW`, `Val/AUC@<landmark>`, `Val/mean_AUC_landmarks`
  - Mirror `Train/*` metrics.
- Early stopping in the deephit yaml is on `Val/C-index, mode=max, patience=30, min_delta=0.002` — aligned with the HPO objective. Soft-logrank's yaml uses `Val/logrank_chi2` instead; we do NOT change this here.

## DeepHit loss specifics

`survival_utils.DeepHitLoss(num_time_bins, alpha, beta, gamma, sigma)`:
- `alpha * log_likelihood + beta * pairwise_ranking + gamma * calibration`
- `sigma > 0` is required (controls ranking-term softness; smaller = sharper).
- Calibration term is summed across the batch → **`gamma` scales with batch size**; re-tune jointly with `data.module.batch_size` / `accumulate_grad_batches`.

## Per-trial mutables (NOT forced)

Trial-phase agent should set these per trial:
- `trainer.logger.name=<trial_id>` so the wandb run is recognizable.
- Any knob from `search_space_declared`.

## Forced overrides rationale

| Override | Why |
| --- | --- |
| `env=local` | Pick *some* env so `exp_dir` is defined; we override the actual path next. `cluster` would clobber the progress bar callback. |
| `exp_dir=/home/jma/hpo_runs` | Writable on AIHub for user `jma`. Replaces the DKFZ-specific path in `env/local.yaml`. |
| `model=resenc_survival` | Only model config compatible with the survival pipeline. |
| `data=methylome_t1c_combined_high_vs_low` | Required by the user goal; selects `survival_loss.name=deephit`. |
| `data.cv.k=1` | Single-fold HPO; the data yaml ships `k=5`. |
| `data.module.fold=0` | Explicit fold index. |
| `trainer.enable_checkpointing=False` | Default already; locked so checkpoint files are never written. |
| `trainer.callbacks.checkpoint.save_top_k=0` | Even if the flag flips, this guarantees no checkpoint files. |
| `seed=42` | Reproducibility across trials; also turns on `deterministic=True` and disables cuDNN benchmarking. |
