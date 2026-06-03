# Composite (weighted-sum) survival loss — Design

Date: 2026-06-02

## Problem

DeepHit is a fixed two-term loss (NLL + ranking). The factory
(`build_survival_criterion`, `survival_utils.py:369`) otherwise exposes 9
losses that can only be used one at a time. There is no way to train on a
weighted combination of arbitrary losses (e.g. a calibrated-curve loss plus a
discrimination loss), even though the prediction head already emits every
projection needed to compute all of them on each forward pass.

## Goal

Allow a `survival_loss` block to specify a weighted sum of any subset of the 9
losses, while keeping all existing single-loss configs working unchanged.

## Decisions (confirmed with user)

- **General weighted sum** of any losses (not restricted to a calibration +
  discrimination pairing).
- **One designated `primary` loss** drives all metrics and inference (C-index,
  survival curve, stratification, saved predictions). Other members contribute
  only to the training gradient.
- **Mapping stays in `base_model`**: factor the existing per-loss
  "which head output feeds which loss" `if/elif` into a shared helper that both
  the single-loss and composite paths call. Do not move the mapping into the
  loss layer.

## Config schema

New `name: composite` form. Existing single-loss blocks are unchanged.

```yaml
survival_loss:
  name: composite
  primary: nll                              # member whose outputs drive metrics/inference
  components:
    - {name: nll,     weight: 1.0}
    - {name: cox,     weight: 0.5, reduction: mean}
    - {name: deephit, weight: 0.3, alpha: 0.2, sigma: 0.1}
```

`total = sum(weight_i * loss_i)` over components.

### Validation rules

- `components` present and non-empty.
- Each component has a `name` that is one of the 9 valid single-loss names
  (not `composite` — no nesting).
- Component names are unique within the list.
- `primary` is present and equals one of the component names.
- Each `weight` is a number >= 0; defaults to 1.0 if omitted.
- Each component's loss-specific options are validated by that loss's existing
  constructor (reuse `_build_single_criterion`).

Violations raise `ValueError` with a message naming the offending field.

## Components and boundaries

### 1. Factory — `survival_utils.py`

- Extract the current per-name construction (the `if/elif` body of
  `build_survival_criterion`) into `_build_single_criterion(name, cfg)
  -> nn.Module`. The existing dispatch behavior is preserved exactly; the public
  `build_survival_criterion` keeps returning `(name, criterion)` for single
  losses.
- Add a `composite` branch to `build_survival_criterion`: parse and validate the
  block, build each member via `_build_single_criterion`, and return
  `("composite", CompositeSurvivalLoss(...))`.

### 2. `CompositeSurvivalLoss` — `survival_utils.py` (new `nn.Module`)

A thin container so member parameters register correctly. Holds:

- `members: nn.ModuleList` — the per-component criteria.
- `names: list[str]` — parallel component names.
- `weights: list[float]` — parallel weights.
- `primary: str` — the primary component name.

It does **not** perform the `y_hat -> input` mapping (that lives in
`base_model`, per the architecture decision). It exposes `members`, `names`,
`weights`, and `primary` for `base_model` to iterate.

### 3. `base_model.py`

- **`survival_primary_name`**: set alongside `survival_loss_name` at
  construction (`~:127`). Equals `criterion.primary` for composite, else equals
  `survival_loss_name`. For single losses the two are identical.
- **`_call_one_loss(name, criterion, y_hat, time_bin, event, continuous_time)
  -> (loss_tensor, components_dict)`**: the extracted 9-branch mapping. Handles
  `soft_logrank`'s `(total, components)` tuple return and `pchazard`'s
  `interval_frac` internally. `components_dict` is the soft_logrank diagnostics
  (empty for the others).
- **`_survival_loss`**:
  - Single loss: `loss, comps = _call_one_loss(name, self.criterion, ...)`;
    `loss_parts = {"total": loss, name: loss, **comps}`. (Behavior unchanged.)
  - Composite: loop members; `li, ci = _call_one_loss(name_i, member_i, ...)`;
    `total += weight_i * li`. Build `loss_parts = {"total": total,
    "composite": total, <name_i>: li (unweighted), ...}` and merge the primary
    member's `ci` components. Per-member **unweighted** values are logged so
    weights can be tuned.
- **Downstream metric/inference branches** that currently read
  `self.survival_loss_name` to pick a head output switch to
  `self.survival_primary_name`: the output-selection sites at `:370, :596,
  :754, :793, :812`. Logging-tag sites (`:701, :740`) use `survival_loss_name`
  (so composite logs under a `composite` tag); add a `composite` entry to
  `_SURVIVAL_LOSS_TAGS` keyed to `loss_parts["composite"]`.

### 4. `inference_survival.py`

Where inference keys off the training loss name to choose risk vs survival curve
(`:283, :625`), resolve composite to its `primary` before branching. A small
helper that reads `primary` from the training config's composite block (falling
back to the loss `name` for single losses) keeps this in one place.

## Data flow

Forward pass produces the stable `y_hat` dict (all projections). `_survival_loss`
computes either one loss or the weighted sum. `total` flows to backprop. Metric
buffers and inference read `y_hat` through `survival_primary_name`, so they
behave as if the primary loss were the active single loss.

## Error handling

- Config validation errors raise `ValueError` at factory time (fail fast before
  training) naming the offending field.
- No silent fallbacks: an unknown member name or a `primary` not in `components`
  is an error, not a default.

## Testing (TDD)

- Factory builds a composite from a config block: returns `CompositeSurvivalLoss`
  with expected `names`, `weights`, `primary`.
- `total` equals `sum(weight_i * loss_i)` on a small synthetic `y_hat`/targets
  batch (compare composite total to manually summing single-loss outputs).
- Validation: empty/missing `components`; unknown member name; duplicate member
  names; `primary` absent or not in components; negative weight — each raises
  `ValueError`.
- `survival_primary_name` equals the configured `primary` for a composite and
  equals the loss name for a single loss.
- Backward compatibility: every existing single-loss name still builds and
  `_survival_loss` returns the same `loss_parts` shape as before.

## Verification

- New and existing loss-factory tests pass (`tests/test_loss_factory.py`).
- A composite config parses and a short smoke run logs `total` plus each
  member's unweighted value.

## Out of scope

- Automatic / uncertainty-based weighting (weights are user-specified constants).
- Nested composites.
- New individual loss implementations.
- Composite-specific metrics beyond reusing the primary member's.
- New example configs (can follow later once the mechanism lands).
