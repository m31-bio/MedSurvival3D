# CLAUDE.md

> **Start here:** Read [`HANDOFF.md`](./HANDOFF.md) before beginning work — it
> carries the latest session context, what's verified vs. unverified, and parked
> items for `SSL3D_survival`.

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.

---

## Workstation workflow

Edit + reason on the **Mac** (source, git, CodeGraph live here). Execute on the
**workstation** (GPU, venv, data live there). The Mac has NO survival deps and
NO CUDA — it cannot reproduce most failures. Loop is: fix → rsync → test remote.
The workstation is the only source of truth for pass/fail.

- **Target:** `jma@aihub2.uniseg:/home/jma/Documents/projects/safwat/coca_classifier_codebase/SSL3D_survival/`
- **SSH** is key-based/non-interactive. Use `-o BatchMode=yes` to fail fast.
- **`uv` not on non-interactive PATH** — prepend `export PATH=$HOME/.local/bin:$PATH`.
- **Venv is `.venv`** (uv-created, NO pip): install via `uv pip install`, run via `.venv/bin/python`.
- **rsync MUST exclude** `.venv .git .codegraph .hpo_agent __pycache__ .pytest_cache *.pyc`
  — else the Mac's CPU torch overwrites the workstation's CUDA env.
- **Always run from repo root** (no `pyproject.toml` yet; imports resolve via cwd).
- **`main.py` needs overrides:** `model.chpt_path=...`, `model.save_preds=false`,
  omit `seed`, `+trainer.fast_dev_run=true`. Bare run crashes on `torch.load(None)`.

**Failure-classification rule:** before fixing ANY failing test, classify it as
*restructure regression* vs *pre-existing / env / test-bug*. Don't rewrite working
production code to satisfy a fragile test.

## Session hygiene

- **Update `HANDOFF.md` at each completed milestone** — record what is verified vs.
  unverified, and what the next session should pick up. Keep it always-current so any
  session can resume cold. (Transient status here; durable invariants stay in this file.)
- **Stop and recommend `/clear` at milestone boundaries** — when a distinct phase of work
  finishes (or when responses show context strain: repetition, forgetting earlier
  decisions), pause and explicitly recommend starting a fresh session. State what's saved
  in `HANDOFF.md` and what the next session should do first. Do NOT silently roll into a
  new distinct phase without recommending the reset.
- Trigger is *milestone / judgment*, NOT a context-% meter (Claude cannot reliably sense
  its own context size). The actual restart (`/clear`) is always the user's call.
