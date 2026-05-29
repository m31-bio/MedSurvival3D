# Survival Losses & Metrics — Migration to Established Packages

**Date:** 2026-05-29
**Status:** Design (awaiting review)
**Scope:** Replace hand-rolled survival losses and metrics (training + inference) with
established package implementations. Add new package-provided losses. Keep
`soft_logrank` custom. **No baselines.**

## Goal

Stop maintaining hand-rolled survival math. Route losses and metrics through
validated libraries so reported numbers match the field's de-facto standards,
and gain new loss options for experimentation — with the **smallest possible
change to the existing architecture**.

## Non-goals

- No baseline models (RSF, CoxNet, GBSA, DSM, etc.).
- No change to the 3D CNN backbone, SSL pretraining, datasets, or the Lightning
  training loop beyond the loss/metric seams.
- No new abstraction layers. Prefer in-place body replacement that preserves
  existing function signatures so call sites stay untouched.

## Environment

- `survival_env` (uv venv, Python 3.11) at `/Users/bw/Documents/Safwat/survival/survival_env`.
- Installed & import-verified: `torch 2.12.0`, `torchsurv 0.1.6`, `pycox 0.3.0`,
  `scikit-survival 0.27.0`, `lifelines 0.30.3`.
- auton-survival dropped (PyPI `0.1.0` hard-pins `torch<2.0`; incompatible).
- Isolated from the conda base (which is mid-HPO); base untouched.

## Design principles (per user: "keep abstractions low")

1. **Preserve public signatures.** Replace the *bodies* of existing functions
   (`concordance_index`, `time_dependent_auc`, `integrated_brier_score`,
   `integrated_brier_score_ipcw`, inference metric fns) so their callers in
   `base_model.py` and `inference_survival.py` do not change.
2. **One small dict, not a class hierarchy.** The only genuinely new structure is
   a `name -> survival-curve fn` map for per-loss survival derivation.
3. **Thin adapters only where forced.** Package loss call signatures differ
   (raw logits, `rank_mat`, `interval_frac`). Each adapter is a small
   `nn.Module` whose `forward` matches the existing dispatcher call and
   translates to the package call. No shared base class beyond `nn.Module`.
   Adapters live **in `survival_utils.py`, replacing the current loss classes**
   (same file, same `build_survival_criterion` factory) — no new loss module.
4. **Keep `soft_logrank` exactly as-is.**

## Architecture (seams reused)

Unchanged seams:
- `build_survival_criterion(cfg, num_time_bins)` — factory (`survival_utils.py:619`)
- `BaseModel._survival_loss(y_hat, y)` — dispatcher (`base_model.py:288`)
- `SurvivalHead.forward` — multi-output head (`models/survival_head.py:95`)
- Lightning metric hooks (`_update_survival_metric_buffers`,
  `_log_survival_metrics`, `_compute_stratification_metrics`)

### Loss mapping

| Config `name` | Implementation | Head output consumed | Targets passed | Notes |
|---|---|---|---|---|
| `nll` | pycox `NLLLogistiHazardLoss` | raw hazard logits (`logits`) | `time_bin`, `event` | same input as today |
| `cox` | torchsurv `cox.neg_partial_log_likelihood` | `risk` | `continuous_time`, `event` | **Efron ties** (was Breslow) |
| `deephit` | pycox `DeepHitSingleLoss(alpha, sigma)` | raw **pmf logits** (`pmf_logits`, new) | `time_bin`, `event`, `rank_mat` | **drops γ calibration, single-α weighting, fixes ranking bug** |
| `soft_logrank` | **custom `SoftLogRankLoss` (unchanged)** | `p_high` | `continuous_time`, `event` | no equivalent |
| `pmf` (new) | pycox `NLLPMFLoss` | `pmf_logits` | `time_bin`, `event` | |
| `mtlr` (new) | pycox `NLLMTLRLoss` | raw logits [B,K] (`logits`) | `time_bin`, `event` | |
| `bcesurv` (new) | pycox `BCESurvLoss` | raw logits [B,K] (`logits`) | `time_bin`, `event` | |
| `pchazard` (new) | pycox `NLLPCHazardLoss` | raw logits [B,K] (`logits`) | `time_bin`, `event`, `interval_frac` | needs sub-bin fraction |
| `weibull` (new) | torchsurv `weibull` (`neg_log_likelihood_weibull`) | **new [B,2] param head** (`weibull_params`) | `continuous_time`, `event` | parametric AFT |

### Head changes (`models/survival_head.py`)

- Expose `pmf_logits` (currently only softmaxed `pmf` is returned; `fc_pmf` raw
  logits already computed at `:115`).
- Add `fc_weibull` → `weibull_params` `[B, 2]` (log-scale, log-shape) for the
  `weibull` loss. Present in the output dict in all modes (only trained when
  active), consistent with the existing stable-dict pattern.
- `interval_frac` for `pchazard`: the within-bin position of the event time.
  Computed in the dispatcher from `continuous_time` and the bin edges (cut
  points), not in the head.

### Per-loss survival curve (the one new dict)

`LOGITS_TO_SURVIVAL: dict[str, Callable] -> survival [B, K]`:
- `nll`: `cumprod(1 - sigmoid(logits))`
- `pmf`, `deephit`: `1 - cumsum(softmax(pmf_logits))`
- `mtlr`: pycox MTLR→surv transform
- `pchazard`: pycox PC-hazard→surv transform
- `bcesurv`: `sigmoid(logits)` per-node survival
- `cox`: no native curve → risk only; survival curve via Breslow baseline if a
  curve is required for IBS/AUC (else those metrics use risk for ranking only)
- `weibull`: analytic `survival_function_weibull`

This replaces/extends `SurvivalHead._survival_for_active_loss` (`:85`) and feeds
`derive_stratification_scores` (`survival_utils.py:769`), which gains entries for
the new names (`pmf`/`mtlr`/`pchazard`/`bcesurv` → `1 - survival[:, landmark]`;
`weibull` → same; `cox` → risk).

### Metrics — training-time (`survival_utils.py`, torch-native)

Replace bodies, keep signatures:
- `concordance_index(...)` → torchsurv `ConcordanceIndex`
- `time_dependent_auc(...)` → torchsurv `Auc(auc_type='cumulative', new_time=landmarks)`
- `integrated_brier_score(...)` → torchsurv `BrierScore`
- `integrated_brier_score_ipcw(...)` → torchsurv `BrierScore` with IPCW `weight`
- `_censoring_survival_km` (`:178`) becomes unused → remove (orphan from our change).

### Metrics — inference (`inference_survival.py`, numpy)

Replace bodies, keep signatures:
- `concordance_index` → sksurv `concordance_index_censored`
- `time_dependent_auc` → sksurv `cumulative_dynamic_auc`
- `integrated_brier_score` / `_ipcw` → sksurv `integrated_brier_score`
- `compute_logrank_stat` (`:319`), `_logrank_chi2` (`:671`) → lifelines `logrank_test`
- `compute_hazard_ratio` (`:367`) → lifelines `CoxPHFitter`
- `km_survival_at` (`:277`), `km_step_curve` (`:297`) → lifelines `KaplanMeierFitter`
- `max_logrank_cutpoint` (`:713`): keep the quantile-scan logic; swap the inner
  statistic to lifelines.

### New data flow: IPCW needs training labels

sksurv `concordance_index_ipcw` / `integrated_brier_score` require
`survival_train` (training-set `(event, time)`) to estimate the censoring
distribution. Inference currently does not thread training labels into eval.
Add: per-fold training `(time, event)` passed into `run_split_inference` /
`compute_metrics`, converted via `sksurv.util.Surv.from_arrays`. Small, localized
data-flow addition.

### Convention check

Event polarity is consistent across all packages and the current code
(`event=1` → observed event; sksurv/lifelines/torchsurv all use True/1 = event).
No inversion needed (unlike the ProgPath/CLAM lineage, which uses censorship=1).

## Config / factory

Extend the `survival_loss: {name: ...}` schema and `build_survival_criterion`
validation to accept: `nll, cox, deephit, soft_logrank, pmf, mtlr, bcesurv,
pchazard, weibull`. Each new name reads its hyperparameters (e.g. `deephit:
{alpha, sigma}`, `weibull` none, `pchazard` none) with sensible defaults.
Reject unknown names (existing behavior).

## Testing strategy

- **Characterization (old vs new):** before deleting a hand-rolled function,
  add a test comparing old vs new on fixed synthetic data to quantify the shift
  (extends the existing `/tmp/survxcheck` cross-check; promote to
  `tests/test_metrics_vs_packages.py`). Document expected deltas:
  - C-index: exact w/o ties; new estimator removes the measured ~+0.01 binned-tie bias.
  - Cox: Efron vs Breslow → small value change.
  - DeepHit: different objective (no γ, single α) → not comparable; assert
    finite + backprops, re-tune separately.
- **Per-loss smoke tests:** each loss builds via factory, returns finite scalar,
  `.backward()` produces gradients on the correct head output.
- **Existing tests updated:** `test_loss_factory.py`, `test_loss_integration.py`,
  `test_deephit_loss.py`, `test_survival_head.py`,
  `test_stratification_metrics.py`, `test_soft_logrank_*` (soft_logrank tests
  should pass unchanged).

## Risks & mitigations

| Risk | Mitigation |
|---|---|
| DeepHit behavior change invalidates HPO history (β=1.0 finding) | Flag clearly; treat DeepHit as a fresh baseline; re-tune `alpha/sigma` |
| Reported C-index/IBS values shift | Characterization tests quantify each delta before deletion |
| IPCW train-data threading touches inference flow | Localized; covered by inference tests |
| `weibull`/`pchazard` need head/dispatcher plumbing | Implemented in P3; gated behind their config names; other losses unaffected |
| pycox needs raw logits; head softmaxes pmf | Expose `pmf_logits` (already computed) |

## Phasing

1. **P1 — Inference metrics** (sksurv/lifelines). Lowest risk; fixes reported numbers; no training change.
2. **P2 — Loss replacement** (`nll/cox/deephit` → package; expose `pmf_logits`; rank_mat shim).
3. **P3 — New losses** (`pmf/mtlr/bcesurv`, then `pchazard/weibull` with head/dispatcher plumbing).
4. **P4 — Training-time metrics** (torchsurv).

Characterization tests accompany each phase.

## Open items

None blocking. `cox` survival-curve-for-IBS (Breslow baseline) is the only spot
needing a small decision during P4; default is to report Cox ranking metrics
(C-index, AUC) and skip curve-based IBS for Cox unless a curve is requested.
