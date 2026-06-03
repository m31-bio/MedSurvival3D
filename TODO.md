# Survival losses & metrics — review TODO

Review of the recently replaced/added loss functions and metrics
(`survival_utils.py`, `models/survival_head.py`, `base_model.py`,
`inference_survival.py`). Verdict: **coherent and correct by inspection, no hard
bugs found.** Items below are follow-ups, ranked. None are blockers.

## 0. Run the test suite (prerequisite)
- [ ] Tests could NOT be run during review: `torchsurv` and `pycox` are not
      installed in the local conda env, so 15 test modules fail at *collection*
      (not on logic). Run `pytest tests/` on the cluster env that has
      torchsurv + pycox before trusting any of the below.

## Worth fixing / commenting (1–4)

### 1. Brier and AUC use different time units
- [ ] In `base_model._log_survival_metrics`, `integrated_brier_score` /
      `integrated_brier_score_ipcw` are fed `all_time_bins` (integer bin
      indices) while `time_dependent_auc` is fed `all_continuous_times` (years).
      Each is internally self-consistent, but `Train/Brier` (bin-index space)
      and `Train/AUC@5y` (year space) are NOT on a comparable axis.
- [ ] Action: add a one-line comment so nobody later compares them, OR unify the
      time axis if you want them comparable.

### 2. `_valid_time_grid` silent-nan edge case (`survival_utils.py:55`)
- [ ] If the earliest *event* falls in the last bin, `lo` can exceed the number
      of survival columns → `s[:, times]` raises `IndexError`, swallowed by the
      `try/except` in `integrated_brier_score` → returns `nan` with no hint why.
- [ ] Action: either clamp `lo` to a valid column or log a debug message so a
      `nan` Brier on a small/degenerate val batch is explainable.

### 3. Degenerate-case sentinel mismatch
- [ ] Training `concordance_index` returns **0.5** when there are no events;
      inference `sksurv_cindex` returns **nan** for the same case.
- [ ] Action: pick one convention (harmless either way, but avoids confusion
      when comparing train vs. inference C-index on event-free slices).

### 4. PC-hazard `interval_frac` convention only half-tested
- [ ] `test_pchazard_matches_pycox` passes a *random* `interval_frac`, which
      validates the adapter vs. pycox but NOT `base_model._interval_frac`
      (`(t - left_edge)/width`, last-bin extrapolated-width logic at
      `base_model.py:257`). That derivation is the part most likely to be subtly
      wrong; the last-bin width fallback (extrapolate 2nd-to-last bin width) is a
      reasonable but undocumented modeling choice.
- [ ] Action: confirm `test_new_loss_training_paths.py` exercises `_interval_frac`
      with real times; if not, add a regression test pinning it against pycox's
      `LabTransPCHazard`.

## Informational (5–6)

### 5. `metrics:` config is silently ignored
- [ ] `base_model.py:70` warns and drops any configured torchmetrics — metrics
      are now hand-rolled. Confirm you know any `metrics:` block in YAML is now a
      no-op (the warning is easy to miss in logs).

### 6. (perf, not correctness) `soft_logrank_loss` Python loop
- [ ] Loops over unique event times, building one autograd subgraph per distinct
      time → O(unique-event-times) graph size, slow on large batches with many
      distinct times. Correct as-is; vectorize only if it shows up in profiling.

---

## Suggested next actions (from review)
- (a) Run the suite on the cluster env (which `conda activate`?).
- (b) Add a regression test pinning `base_model._interval_frac` against pycox's
      `LabTransPCHazard` (closes gap #4).
- (c) Otherwise leave as-is and just add comments for #1–#3.

Recommended order: (a) then (b).
