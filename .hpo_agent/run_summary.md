# Run Summary for SSL3D_survival

## Global Best (across all attempts on this repo)

- **Best Val/C-index:** 0.7865 (mode: max) — Attempt 20260528-183049, Trial #2
  - Cost to find: $0.0000
  - Best hparams: `{"data": "methylome_t1c_combined_high_vs_low", "data.cv.k": 1, "data.module.batch_size": 24, "data.module.fold": 0, "env": "local", "exp_dir": "/home/jma/Documents/projects/safwat/coca_classifier_codebase/hpo_runs", "model": "resenc_survival", "model.chpt_path": "/home/jma/Documents/projects/safwat/coca_classifier_codebase/SSL3D_classification/checkpoints/S3D/checkpoint_final.pth", "model.finetune_method": "full_sawtooth", "model.lr": 5e-05, "model.resnet_dropout": 0.0, "model.save_preds": false, "model.survival_head_dropout": 0.0, "model.survival_head_hidden_dim": 128, "model.survival_head_norm": "layernorm", "model.survival_loss.alpha": 1.0, "model.survival_loss.beta": 0.5, "model.survival_loss.gamma": 0.0, "model.survival_loss.sigma": 0.1, "model.weight_decay": 0.001, "trainer.accumulate_grad_batches": 2, "trainer.callbacks.checkpoint.save_top_k": 0, "trainer.enable_checkpointing": false}`

## Attempt 20260527-183953 (2026-05-27)

- Goal: Use the data config cli_configs/data/methylome_t1c_combined_high_vs_low.yaml
(selects survival_loss=deephit). Single fold (data.cv.k=1, data.module.fold=0).
Identify hyperparameters that produce stable training and validation curves
AND maximize Val/C-index. Prefer settings that also achieve
Val/logrank_chi2 > 0 (stratification).

- Trials completed: 3
- Best trial: #1 (Val/C-index = 0.0)
- Cost: $1.4241 (tokens: input_tokens=65, cache_creation_input_tokens=58623, cache_read_input_tokens=1182407, output_tokens=18647)
- Best hparams:


3 trial(s). Best is trial #1 (Val/C-index=0.0).
### Trial 1 *(status: system_blocked)*

- **hparams:** baseline (empty)

**Analysis:**
Baseline trial 1 with hparams={} (deephit on methylome_t1c_combined_high_vs_low, k=1 fold=0, env=local). Two consecutive launches both failed instantly with the same `nohup: failed to run command 'python'` error from the remote shell. Root cause: the remote uv-managed venv interpreter is only reachable by absolute path under nohup-style spawning, and the MCP server's cached launch_command_template still says `python {script}`. Fixed the template on disk; orchestrator restart required before any trial can produce metrics. No model training occurred, so no comment yet on lr/beta/calibration tradeoffs. Once unblocked, the baseline should reach training under env=local + the uv venv and write to wandb project nnSSL_MethylomeRisk_Survival as in prior local runs (e.g. run o2lux7zf from deephit_fold0.log).

**Notes:**
Trial was launched twice (run_ids 43ed8357712a47e0982468f843b0e59b and cbf1d7896e404ef299545a003398cce1); both failed identically at the shell layer. The .hpo_agent/repo_profile.json edit is the only writable fix available within my permission scope.

**Blocker:**
- _diagnosis:_ Remote host jma@aihub2.uniseg has no `python` on PATH (only /usr/bin/python3 and /usr/bin/python3.10, neither of which is the project env). The project is uv-managed; the correct interpreter is /home/jma/Documents/projects/safwat/coca_classifier_codebase/.venv/resnetmae_coca_classifier/bin/python (Python 3.12.11). The MCP server's in-memory launch_command_template is still the literal `python {script}` from server startup; it does not reread repo_profile.json per-trial.
- _error:_ nohup: failed to run command 'python': No such file or directory
- _fix_on_disk:_ Updated .hpo_agent/repo_profile.json launch_command_template -> `/home/jma/Documents/projects/safwat/coca_classifier_codebase/.venv/resnetmae_coca_classifier/bin/python {script}`. This will take effect after MCP server restart.
- _polls_attempted:_ 1
- _recommended_fix:_ Restart the HPO MCP server / orchestrator. Structurally identical to prior session's `{overrides}` placeholder and stale-DNS-resolver blockers, which also required orchestrator restart.
- _retry_attempted_after_fix:_ True
- _where:_ remote shell process spawned by MCP launch_remote_trial

### Trial 2 *(status: system_blocked)*

- **hparams:** `{"model.lr": 5e-05}`

**Analysis:**
Trial 2 (model.lr=5e-5) never reached training. Same shell-layer failure as trial 1 in this attempt: the MCP server's cached launch_command_template still says `python {script}` even though the on-disk profile was updated to the absolute uv-venv path during trial 1. The hparam choice itself (10x lr increase to address the diagnosed head overfitting at baseline lr=5e-6) is unevaluated. Cannot make a substantive lr / weight_decay / dropout judgement until orchestrator restart enables training. If trial 3 also fires before restart, it will fail identically — the only productive next action is the user restarting the MCP server.

**Notes:**
Hparam rationale preserved here so that, after orchestrator restart, this exact configuration (model.lr=5e-5) can be re-run as the first informed point above baseline.

**Blocker:**
- _diagnosis:_ Identical to trial 1 blocker. MCP HPO server still uses its startup-cached launch_command_template `python {script}`. The remote shell has no `python` on PATH; the project's uv-managed venv interpreter at /home/jma/Documents/projects/safwat/coca_classifier_codebase/.venv/resnetmae_coca_classifier/bin/python (Python 3.12.11) requires the absolute path under nohup spawning.
- _error:_ nohup: failed to run command 'python': No such file or directory
- _fix_on_disk:_ .hpo_agent/repo_profile.json launch_command_template was already updated in trial 1's session to use the absolute venv-python path. No further on-disk changes possible from inside trial 2; only orchestrator restart will activate the fix.
- _polls_attempted:_ 1
- _recommended_fix:_ Restart the HPO MCP server / orchestrator so it re-reads .hpo_agent/repo_profile.json. Same recovery pattern as the prior session's `{overrides}` placeholder blocker and stale DNS resolver blocker.
- _retry_attempted_after_fix:_ False
- _where:_ remote shell process spawned by MCP launch_remote_trial

### Trial 3 *(status: system_blocked)*

- **hparams:** `{"model.lr": 0.0001, "model.survival_head_dropout": 0.3, "model.weight_decay": 0.001}`

**Analysis:**
Trial 3 never reached training. Same shell-layer failure as trials 1 and 2 in this attempt -- the MCP server's cached launch_command_template still says `python {script}` despite the on-disk fix. The Hydra args composed correctly (model.lr=0.0001 model.weight_decay=0.001 model.survival_head_dropout=0.3 lead the effective_config_args), confirming hparam plumbing is healthy. End-of-attempt summary: 0 of 3 trials produced metrics; the sole productive action remains an external MCP server restart. The planned three-point sweep (baseline / lr=5e-5 single-knob / lr=1e-4 + WD + head-dropout combo) is preserved verbatim in the trial records for re-run.

**Notes:**
Final trial in this attempt. After orchestrator restart, the records in runs/20260527-183953/trials/trial-00{1,2,3}.json can be used directly to re-execute the same hparam plan with identical Hydra args.

**Blocker:**
- _diagnosis:_ Identical to trial 1 and trial 2 blockers in this attempt. MCP HPO server still uses its startup-cached launch_command_template `python {script}`. Remote host has no `python` on PATH; project uv-venv interpreter at /home/jma/Documents/projects/safwat/coca_classifier_codebase/.venv/resnetmae_coca_classifier/bin/python (Python 3.12.11) requires absolute path under nohup.
- _error:_ nohup: failed to run command 'python': No such file or directory
- _fix_on_disk:_ .hpo_agent/repo_profile.json launch_command_template already updated to absolute venv-python path during trial 1. No further on-disk action available.
- _polls_attempted:_ 1
- _recommended_fix:_ Restart MCP HPO server / orchestrator. After restart, re-run this attempt: trial 1 = hparams={}, trial 2 = {model.lr: 5e-5}, trial 3 = {model.lr: 1e-4, model.weight_decay: 1e-3, model.survival_head_dropout: 0.3}. The Hydra override composition has been verified working at the MCP layer for all three.
- _retry_attempted_after_fix:_ False
- _where:_ remote shell process spawned by MCP launch_remote_trial

## Attempt 20260527-193105 (2026-05-27)

- Goal: Use the data config cli_configs/data/methylome_t1c_combined_high_vs_low.yaml
(selects survival_loss=deephit). Single fold (data.cv.k=1, data.module.fold=0).
Identify hyperparameters that produce stable training and validation curves
AND maximize Val/C-index. Prefer settings that also achieve
Val/logrank_chi2 > 0 (stratification).

- Trials completed: 3
- Best trial: none (all 3 trial(s) failed or were incomplete)
- Cost: $3.5135 (tokens: input_tokens=67, cache_creation_input_tokens=146910, cache_read_input_tokens=2287114, output_tokens=58056)

3 trial(s); 0 completed. No best trial — see individual trial sections for failure details.
### Trial 1 *(status: failed)*

- **hparams:** baseline (empty)
- **metrics:** `{"Val/C-index": null, "Val/logrank_chi2": null, "note": "No usable metrics: training crashed in epoch-0 first backward pass, before any validation epoch. wandb runs created but empty of epoch history."}`

**Analysis:**
Trial 1 (baseline) FAILED — but this attempt made real forward progress through a 3-blocker cascade, two of which I fixed durably. (1) The prior attempt's `nohup: python not found` shell blocker is RESOLVED: launches now spawn the uv-venv interpreter and Hydra composes the config (the launch_command_template fix from the prior session is live, i.e. the MCP server WAS restarted). (2) NEW blocker found & fixed: model construction crashed with `'NoneType' has no attribute 'seek'` = torch.load(None). Root cause: resenc_survival.yaml sets model.pretrained=True but model.chpt_path=null, and resenc.py:46-50 unconditionally torch.loads chpt_path when pretrained=True. Neither env/local.yaml nor forced_overrides supplied the path. The remote project_overview.md explicitly flagged this ('orchestrator must set model.chpt_path from outside'). I located the SSL-pretrained ResEnc checkpoint at /home/jma/.../SSL3D_classification/checkpoints/S3D/checkpoint_final.pth (852MB; its /S3D/ path matches the special stem-conv handling branch at resenc.py:155) and added it to forced_overrides. Injecting it via hparams let model construction + pretrained-weight loading SUCCEED and training START (wandb run wczze7sh). (3) NEXT blocker, currently immovable on the running server: training crashed in the first backward with `avg_pool3d_backward_cuda does not have a deterministic implementation, but you set torch.use_deterministic_algorithms(True)`. Cause chain: forced seed=42 -> main.py:19-22 `if cfg.seed:` hardcodes cfg.trainer.deterministic=True AFTER Hydra compose -> Lightning enables deterministic algorithms -> the ResEnc encoder's avg_pool3d has no deterministic CUDA backward kernel. I empirically proved hparams CANNOT override forced overrides on this MCP server: passing seed=0 produced effective_config `seed=42` (the tool dedupes with the FORCED value winning). So determinism is un-overridable from inside a trial while seed=42 is forced. DURABLE FIX APPLIED: removed `seed: 42` from forced_overrides in .hpo_agent/repo_profile.json (train.yaml defaults seed=False and trainer.deterministic=False, so with no forced seed, main.py skips the determinism-forcing block and avg_pool3d backward runs fine). No model-quality judgement possible yet (lr/beta/calibration/overfitting all unevaluated) — zero epochs completed. Session memory notes a prior MANUAL local run reached Val/C-index ~0.6225 at lr=5e-6, so once unblocked the pretrained baseline should land near there.

**Notes:**
Two profile edits made this attempt (both within .hpo_agent/ write scope): (a) added forced_overrides['model.chpt_path'] = '/home/jma/Documents/projects/safwat/coca_classifier_codebase/SSL3D_classification/checkpoints/S3D/checkpoint_final.pth'; (b) removed forced_overrides['seed'] (was 42). Both take effect only AFTER an MCP-server restart (the server caches the profile at startup and dedupes hparams against the cached forced_overrides — confirmed: my seed=0 hparam was overridden by cached forced seed=42). On the CURRENT cached server every trial will keep crashing at avg_pool3d backward regardless of hparams. Reproducibility tradeoff: removing seed disables seed_everything for the whole sweep. PREFERRED ALTERNATIVE (needs a code change, outside my write scope): keep seed=42 and patch main.py:22 to `cfg.trainer.deterministic = 'warn'` (or call torch.use_deterministic_algorithms(True, warn_only=True)) so the non-deterministic avg_pool3d op warns instead of erroring — this preserves seeding AND lets training run.

**Blocker:**
- _diagnosis:_ Forced seed=42 -> main.py hardcodes trainer.deterministic=True -> PyTorch has no deterministic avg_pool3d_backward_cuda kernel -> RuntimeError in first backward. Immovable from inside a trial because the MCP server dedupes hparams against cached forced_overrides with forced winning (verified: seed=0 hparam -> effective seed=42).
- _error:_ RuntimeError: avg_pool3d_backward_cuda does not have a deterministic implementation, but you set 'torch.use_deterministic_algorithms(True)'.
- _fixes_applied_on_disk:_ ["forced_overrides['model.chpt_path'] = '.../SSL3D_classification/checkpoints/S3D/checkpoint_final.pth'", "removed forced_overrides['seed'] (was 42)"]
- _polls_attempted:_ 2
- _prior_blockers_resolved_this_attempt:_ ['nohup python-not-found (shell layer) — RESOLVED, launches now run', 'torch.load(None) from null model.chpt_path — FIXED by adding chpt_path to forced_overrides + verified weights load and training starts']
- _recommended_fix:_ Restart the MCP HPO server so it re-reads .hpo_agent/repo_profile.json (chpt_path added, seed removed). After restart, baseline hparams={} should train to completion. OPTIONAL/PREFERRED for reproducibility: instead of removing seed, patch main.py:22 to use deterministic='warn' / warn_only=True and re-add seed=42 to forced_overrides.
- _retry_attempted_after_fix:_ False
- _stage:_ training epoch 0, first loss.backward()

### Trial 2 *(status: failed)*

- **hparams:** `{"model.lr": 5e-05}`
- **metrics:** `{"Val/C-index": null, "Val/logrank_chi2": null, "note": "No usable metrics: training crashed in epoch-0 first backward pass (same as trial 1), before any validation epoch. wandb run v6xwfqa0 created but empty of epoch history."}`

**Analysis:**
Trial 2 (model.lr=5e-5) FAILED with the identical avg_pool3d determinism crash as trial 1's blocker #3. The launch response's applied_forced_overrides STILL shows seed=42 and STILL lacks model.chpt_path — PROOF the MCP server has NOT been restarted since my trial-1 profile edits, so both durable fixes remain inert. My hparam injections plumbed correctly (effective_config leads with `model.lr=5e-05 model.chpt_path=.../S3D/checkpoint_final.pth`), so the chpt_path safety-net again got the run PAST model construction + pretrained-weight loading and INTO training (wandb v6xwfqa0). It then died in the first loss.backward(): forced seed=42 -> main.py:19-22 hardcodes cfg.trainer.deterministic=True after Hydra compose -> Lightning enables deterministic algorithms -> the ResEnc encoder's avg_pool3d has no deterministic CUDA backward kernel -> RuntimeError. As proven in trial 1, hparams CANNOT override forced overrides on this MCP server (it dedupes with the forced value winning), so seed=42 is immovable from inside any trial on the current cached server. The lr=5e-5 choice (10x the 5e-6 baseline; squarely inside the profile's healthy [1e-5,5e-4] range; directly tests the standing 'increase lr to address head-overfitting' recommendation from session memory's manual 0.6225 baseline) is therefore STILL unevaluated. No model-quality signal from this attempt yet (0 epochs completed across trials 1+2).

**Notes:**
SINGLE REQUIRED ACTION: restart the MCP HPO server so it re-reads .hpo_agent/repo_profile.json. My trial-1 edits are already on disk and waiting: (a) forced_overrides['model.chpt_path'] = '.../SSL3D_classification/checkpoints/S3D/checkpoint_final.pth'; (b) forced_overrides['seed'] removed (train.yaml defaults seed=False & trainer.deterministic=False, so main.py will skip the determinism-forcing block and avg_pool3d backward will run). After restart, baseline hparams={} and this lr=5e-5 point should both train to completion. PREFERRED reproducibility-preserving ALTERNATIVE (needs a code edit, outside my write scope): re-add seed=42 and patch main.py:22 to cfg.trainer.deterministic='warn' (or torch.use_deterministic_algorithms(True, warn_only=True)). Planned remaining sweep once unblocked: trial 3 build regularization on top of the best of {baseline 5e-6, lr 5e-5}, e.g. {model.lr:1e-4, model.weight_decay:1e-3, model.survival_head_dropout:0.3} to counter the observed head overfitting.

**Blocker:**
- _diagnosis:_ Identical to trial 1 blocker #3 and confirmed un-fixable from inside a trial: cached forced seed=42 -> main.py hardcodes trainer.deterministic=True -> no deterministic avg_pool3d_backward_cuda kernel -> RuntimeError in first backward. MCP server not restarted (applied_forced_overrides still has seed=42, still no chpt_path).
- _error:_ RuntimeError: avg_pool3d_backward_cuda does not have a deterministic implementation, but you set 'torch.use_deterministic_algorithms(True)'.
- _fixes_pending_on_disk:_ ["forced_overrides['model.chpt_path'] added", "forced_overrides['seed'] removed (was 42)"]
- _polls_attempted:_ 1
- _recommended_fix:_ Restart the MCP HPO server to re-read .hpo_agent/repo_profile.json; then re-run. Optionally instead use deterministic='warn'/warn_only=True in main.py to keep seeding.
- _retry_attempted_after_fix:_ False
- _server_restarted_since_trial1_fix:_ False
- _stage:_ training epoch 0, first loss.backward()

### Trial 3 *(status: failed)*

- **hparams:** `{"model.lr": 0.0001, "model.survival_head_dropout": 0.3, "model.weight_decay": 0.001}`
- **metrics:** `{"Val/C-index": null, "Val/logrank_chi2": null, "note": "No usable metrics: training crashed in epoch-0 first backward pass (same determinism crash as trials 1 & 2), before any validation epoch. wandb run rojpr5qc created but empty of epoch history."}`

**Analysis:**
Trial 3 (model.lr=1e-4, weight_decay=1e-3, survival_head_dropout=0.3 — a regularization-focused point chosen to counter the head overfitting seen in session memory's manual 0.6225 baseline) FAILED with the identical avg_pool3d determinism crash as trials 1 & 2. applied_forced_overrides AGAIN shows seed=42 and lacks model.chpt_path — the MCP server was NEVER restarted across all 3 trials of this attempt, so my trial-1 durable profile fixes stayed inert the whole time. Hparam plumbing is verified perfect: effective_config led with `model.lr=0.0001 model.weight_decay=0.001 model.survival_head_dropout=0.3 model.chpt_path=.../S3D/checkpoint_final.pth`, and the chpt_path safety-net again carried the run past model construction + pretrained-weight loading INTO training (wandb rojpr5qc) before dying in the first loss.backward(). END-OF-ATTEMPT STATUS: 0 of 3 trials produced metrics, but this attempt made decisive diagnostic progress over the prior attempt (which was stuck at the shell layer): (1) shell/python blocker RESOLVED; (2) torch.load(None) blocker ROOT-CAUSED and FIXED on disk (model.chpt_path -> .../SSL3D_classification/checkpoints/S3D/checkpoint_final.pth, an 852MB SSL-pretrained ResEnc ckpt whose /S3D/ path matches resenc.py:155's stem-conv branch); (3) determinism blocker ROOT-CAUSED (forced seed=42 -> main.py:19-22 hardcodes trainer.deterministic=True after compose -> no deterministic avg_pool3d_backward_cuda kernel) and FIXED on disk (removed forced seed). All three trials reached training and created wandb runs, proving the full launch+pretrained-load+train path works — only the determinism flag stops epoch completion. The lr sweep (5e-6 baseline / 5e-5 / 1e-4+reg) is entirely unevaluated for model quality; no overfitting/stability/stratification judgement is possible yet.

**Notes:**
SINGLE REQUIRED EXTERNAL ACTION: restart the MCP HPO server so it re-reads .hpo_agent/repo_profile.json. Both durable fixes are already on disk: forced_overrides now has model.chpt_path=.../SSL3D_classification/checkpoints/S3D/checkpoint_final.pth and no longer has seed (train.yaml defaults seed=False & trainer.deterministic=False, so main.py will skip the determinism-forcing block and avg_pool3d backward will run). After restart, re-run this exact 3-point sweep — trial 1 = hparams={} (expect ~0.6225 Val/C-index per the manual baseline), trial 2 = {model.lr:5e-5}, trial 3 = {model.lr:1e-4, model.weight_decay:1e-3, model.survival_head_dropout:0.3} — all Hydra args are verified composing correctly. PREFERRED reproducibility-preserving ALTERNATIVE (needs a code edit, outside my write scope): keep seed=42 and patch main.py:22 to cfg.trainer.deterministic='warn' (or torch.use_deterministic_algorithms(True, warn_only=True)), then re-add seed=42 to forced_overrides; this keeps seed_everything while letting the non-deterministic avg_pool3d op warn-and-proceed.

**Blocker:**
- _diagnosis:_ Identical to trials 1 & 2; un-fixable from inside a trial. Cached forced seed=42 -> main.py hardcodes trainer.deterministic=True -> PyTorch lacks deterministic avg_pool3d_backward_cuda -> RuntimeError in first backward. MCP server not restarted (applied_forced_overrides still has seed=42 and no chpt_path across all 3 trials).
- _error:_ RuntimeError: avg_pool3d_backward_cuda does not have a deterministic implementation, but you set 'torch.use_deterministic_algorithms(True)'.
- _fixes_pending_on_disk:_ ["forced_overrides['model.chpt_path'] added", "forced_overrides['seed'] removed (was 42)"]
- _polls_attempted:_ 1
- _recommended_fix:_ Restart the MCP HPO server to activate the on-disk profile fixes, then re-run the 3-point sweep. Optionally use deterministic='warn'/warn_only=True in main.py to preserve seeding.
- _retry_attempted_after_fix:_ False
- _server_restarted_during_attempt:_ False
- _stage:_ training epoch 0, first loss.backward()

## Attempt 20260528-091623 (2026-05-28)

- Goal: Use the data config cli_configs/data/methylome_t1c_combined_high_vs_low.yaml
(selects survival_loss=deephit). Single fold (data.cv.k=1, data.module.fold=0).
Identify hyperparameters that produce stable training and validation curves
AND maximize Val/C-index. Prefer settings that also achieve
Val/logrank_chi2 > 0 (stratification).

- Trials completed: 3
- Best trial: #3 (Val/C-index = 0.71831)
- Cost: $2.5440 (tokens: input_tokens=39, cache_creation_input_tokens=148098, cache_read_input_tokens=1254732, output_tokens=39633)
- Best hparams:
- `model.lr` = `5e-05`
- `model.survival_head_dropout` = `0`
- `model.weight_decay` = `0.001`

3 trial(s). Best is trial #3 (Val/C-index=0.71831).
### Trial 1

- **hparams:** baseline (empty)
- **metrics:** `{"Train/C-index_final": 0.77503, "Train/DeepHitLoss_min": 0.62931, "Val/Brier_min": 0.12323, "Val/C-index": 0.6507, "Val/C-index_best_epoch": 34, "Val/C-index_final": 0.53239, "Val/C-index_final_epoch": 64, "Val/logrank_chi2_at_cindex_peak_epoch34": 2.39233, "Val/logrank_chi2_best": 7.59482, "Val/logrank_chi2_best_epoch": 32, "Val/logrank_p_at_best_chi2": 0.00585, "Val/logrank_p_at_cindex_peak_epoch34": 0.12193, "Val/loss_min": 0.68553, "Val/loss_min_epoch": 30, "Val/mean_AUC_landmarks_best": 0.67989, "early_stopped_at_epoch": 64, "max_epochs": 200, "num_val_epochs": 65}`

**Analysis:**
FIRST successful trial across 3 attempts on this repo. Baseline hparams={} (deephit, default lr=5e-6, finetune_method=full_sawtooth, AdamW). The two durable repo_profile fixes are now LIVE: applied_forced_overrides includes model.chpt_path=.../S3D/checkpoint_final.pth and NO LONGER contains seed=42, so main.py skips its determinism-forcing block and the non-deterministic avg_pool3d_backward_cuda op runs — the crash that killed all 6 prior trials is gone. Training completed 64 epochs then EarlyStopping fired (monitor=Val/C-index, max_epochs=200). RESULT: best-of-run Val/C-index=0.651 @ epoch34, beating the manual local baseline (~0.6225) noted in session memory. The good region is coherent (epochs 30-34): Val/loss min 0.686 @30, mean_AUC 0.680 @30, Brier min 0.123 @31, logrank_chi2 peak 7.594/p=0.006 @32, C-index peak @34. STRATIFICATION GOAL MET: Val/logrank_chi2>0 throughout the good region, peaking 7.59 (p=0.006, significant). At the C-index-peak epoch (34) logrank_chi2 was 2.39 (p=0.12) — positive but not significant. OVERFITTING after the peak: Val/C-index decays from 0.651 (ep34) to 0.532 (ep64) while Train/C-index climbs to 0.775 and Train/DeepHitLoss falls to 0.629 — clear train/val divergence in the survival head. Val/hazard_ratio spiked to ~4951 @ ep32 = degenerate near-separation artifact on the small val set, disregard. NEXT-TRIAL DIRECTION: the overfit-after-peak pattern argues for regularization (model.weight_decay and/or model.survival_head_dropout) and possibly a modest lr bump (healthy range [1e-5,5e-4]; 5e-6 is conservative and reaches its peak slowly at ep34) to hit the peak sooner. Raising survival_loss.beta could push C-index further but may loosen calibration/logrank.

**Notes:**
Metrics fetched via remote wandb API (MCP fetch_wandb_run is blocked locally: 'No API key configured' on the MCP server's fetch environment — the remote training host IS logged in and synced run go5x8jd4 fine). Used scan_history() over 260 logged rows / 65 validation epochs on the remote venv python to compute true best-of-run (NOT summary/final, which is misleading here since EarlyStopping let val degrade post-peak). For future trials: wandb fetch must go through the remote host until a WANDB_API_KEY is configured for the MCP server. Run id go5x8jd4, project nnSSL_MethylomeRisk_Survival, entity aaronchoi6-m31.

### Trial 2

- **hparams:** `{"model.lr": 5e-05, "model.survival_head_dropout": 0.2, "model.weight_decay": 0.001}`
- **metrics:** `{"Train/C-index_final": 0.64634, "Train/DeepHitLoss_final": 0.69542, "Val/Brier_min": 0.11164, "Val/C-index": 0.65915, "Val/C-index_best_epoch": 26, "Val/C-index_final": 0.60282, "Val/C-index_final_epoch": 56, "Val/hazard_ratio_best": 3.03875, "Val/logrank_at_cindex_peak_epoch26": "NaN (degenerate/undefined KM split)", "Val/logrank_chi2_best": 2.46024, "Val/logrank_chi2_best_epoch": 54, "Val/logrank_p_at_best_chi2": 0.11676, "Val/loss_min": 0.64366, "Val/loss_min_epoch": 28, "Val/mean_AUC_landmarks_best": 0.65234, "early_stopped_at_epoch": 56, "max_epochs": 200, "num_val_epochs": 57, "val_cindex_min_midrun": 0.4704, "val_cindex_min_midrun_epoch": 50}`

**Analysis:**
Trial 2 = lr=5e-5 (10x baseline, into healthy range) + weight_decay=1e-3 + survival_head_dropout=0.2, a bundled regularization+lr step to attack the head overfitting and slow convergence seen in trial 1. RESULT: best Val/C-index=0.65915 @ ep26 -- a NEW attempt-best but only +0.008 over baseline's 0.6507 (within noise). MIXED outcome: (POSITIVES) regularization closed the overfitting gap -- Train/C-index fell 0.775->0.646 and Train/DeepHitLoss rose 0.629->0.695, so val (best 0.659) now matches/exceeds train (final 0.646) instead of trailing it. Calibration improved: Val/loss min 0.686->0.644, Val/Brier min 0.123->0.112. Validation endpoint far more STABLE: final Val/C-index 0.603 @ ep56 vs trial1's collapse to 0.532 @ ep64 -- the gentle post-peak decay confirms regularization tamed the overfit. Val/hazard_ratio best=3.04 is non-degenerate (vs trial1's absurd 4951). (NEGATIVES) STRATIFICATION CRATERED: Val/logrank_chi2 best fell from 7.59/p=0.006 (significant) to 2.46/p=0.117 (NEVER significant at any epoch), and at the peak-C-index epoch (26) logrank was NaN = degenerate KM split. Mechanism: WD+head_dropout compress the survival head's risk-score spread, which barely affects ranking (C-index) but destroys the group separation logrank measures. Also a mid-run instability: Val/C-index dipped to 0.470 (below random) @ ep50 before recovering to 0.606 by ep54. CONFOUND: lr-bump vs WD vs dropout effects are entangled in this single point. VERDICT vs trial 1: trial 2 wins marginally on peak C-index + clearly on stability/calibration; trial 1 wins decisively on stratification (the goal's secondary objective). NEXT-TRIAL (3, last shot) DIRECTION: recover trial-1's strong logrank while keeping trial-2's stability. Best options: (a) keep lr~5e-5 but DROP survival_head_dropout to 0 and lower weight_decay to ~1e-4 (let risk scores spread again for KM separation), OR (b) raise survival_loss.beta (ranking term) to sharpen risk separation/C-index while keeping light regularization. Isolating the lr=5e-5 effect (no dropout) would also de-confound this point.

**Notes:**
Metrics via remote wandb API (scan_history over 57 val epochs, run heakx1ab) -- MCP fetch_wandb_run still blocked locally ('No API key configured'; remote host IS logged in). Recorded best-of-run NOT summary: trajectory peaks 0.659@ep26 then is noisy (dip to 0.470@ep50, recover 0.606@ep54-56), so final-epoch summary (0.603) understates the peak. The NaN logrank at the C-index-peak epoch is important for trial 3: a high C-index epoch with undefined stratification means ranking and separation are decoupled under heavy head regularization.

### Trial 3

- **hparams:** `{"model.lr": 5e-05, "model.survival_head_dropout": 0, "model.weight_decay": 0.001}`
- **metrics:** `{"Train/C-index_final": 0.84117, "Train/DeepHitLoss_final": 0.46337, "Val/Brier_min": 0.10502, "Val/C-index": 0.71831, "Val/C-index_best_epoch": 43, "Val/C-index_final": 0.53239, "Val/C-index_final_epoch": 73, "Val/hazard_ratio_best": 15.65487, "Val/logrank_chi2_at_cindex_peak_epoch43": 1.36132, "Val/logrank_chi2_best": 8.27291, "Val/logrank_chi2_best_epoch": 20, "Val/logrank_p_at_best_chi2": 0.00402, "Val/logrank_p_at_cindex_peak_epoch43": 0.24331, "Val/loss_min": 0.60059, "Val/loss_min_epoch": 58, "Val/mean_AUC_landmarks_best": 0.7085, "early_stopped_at_epoch": 73, "max_epochs": 200, "num_val_epochs": 74, "val_cindex_start_epoch0": 0.4254}`

**Analysis:**
Trial 3 = {lr=5e-5, weight_decay=1e-3, survival_head_dropout=0} -- a deliberate single-variable ablation against trial 2 (only dropout 0.2->0). RESULT: BEST TRIAL OF THE ATTEMPT on every axis. Val/C-index=0.71831 @ ep43 (vs 0.6507 trial1 / 0.65915 trial2, a clear ~+0.06 jump). Val/logrank_chi2 recovered to 8.27 (p=0.004, SIGNIFICANT) @ ep20 -- even stronger than trial1's 7.59, and far above trial2's collapsed 2.46. Best calibration of all three (Val/loss min 0.601 vs 0.686/0.644; Val/Brier min 0.105 vs 0.123/0.112) and best discrimination (Val/mean_AUC_landmarks 0.708 vs 0.680/0.652). Val/hazard_ratio best=15.65 is healthy/non-degenerate. HYPOTHESIS CONFIRMED: trial-2's stratification collapse was caused by survival_head_dropout compressing per-patient risk-score variance (dropout trains the head to be robust -> lower-variance eval risk scores -> KM high/low split loses separation -> weak/NaN logrank), NOT by lr or weight_decay. Removing dropout (keeping lr=5e-5 + wd=1e-3) restored the spread and the stratification. TRADEOFF: overfitting returned with the stabilizer gone -- Train/C-index ran to 0.841 (large train/val gap vs val peak 0.718), Train/DeepHitLoss fell to 0.463, and Val/C-index decayed from its 0.718 peak (ep43) back to 0.532 by the early-stop epoch (73), the same unstable post-peak decay as trial1. NUANCE: the C-index peak (ep43) and logrank peak (ep20) are MISALIGNED -- at the C-index-peak epoch logrank is only 1.36 (p=0.24, not significant), and at the logrank-peak epoch (ep20) C-index is mid-trajectory (~0.55-0.6). So no single epoch maximizes both jointly under this config. Also the run starts with INVERTED risk (Val/C-index 0.425 @ ep0) then learns to flip and climb. NET ATTEMPT VERDICT: trial 3 wins decisively on the primary objective (Val/C-index 0.718) AND satisfies the stratification sub-goal (logrank 8.27, p=0.004). FUTURE-ATTEMPT GUIDANCE: (1) Best known config = {lr=5e-5, weight_decay=1e-3, survival_head_dropout=0}. (2) KEY LESSON: avoid survival_head_dropout for this dataset (>=0.2 destroys logrank stratification); regularize via weight_decay instead. (3) Open problem = post-peak overfitting/instability without dropout. To stabilize WITHOUT recompressing risk scores, try: tighter EarlyStopping patience to lock the ep43 peak; encoder-side resnet_dropout (~0.1-0.2, less likely to compress head risk variance than head dropout); a shorter cosine schedule / mild lr decay; or a modest weight_decay bump (2e-3-5e-3) with head_dropout still 0. (4) To align the C-index and logrank peaks, consider checkpoint selection on a combined metric, or a small bump to survival_loss.beta (ranking) which may pull the high-separation and high-ranking epochs closer.

**Notes:**
Metrics via remote wandb API (scan_history over 74 val epochs, run s2pixjjc) -- MCP fetch_wandb_run still blocked locally ('No API key configured'; remote host IS logged in and synced). Recorded best-of-run NOT summary (final-epoch Val/C-index 0.532 badly understates the 0.718 peak due to post-peak overfitting decay). This run is the new GLOBAL BEST across all attempts on this repo (prior attempts had 0 completed trials). Reproduce with hparams {model.lr:5e-5, model.weight_decay:1e-3, model.survival_head_dropout:0}; forced_overrides supply model.chpt_path=.../S3D/checkpoint_final.pth and no seed (determinism fix). Infra note for future attempts: a WANDB_API_KEY must be configured for the MCP server's environment to make mcp__hpo__fetch_wandb_run work; until then fetch metrics via the remote venv python + wandb.Api().

## Attempt 20260528-173651 (2026-05-28)

- Goal: Use the data config cli_configs/data/methylome_t1c_combined_high_vs_low.yaml
(selects survival_loss=deephit). Single fold (data.cv.k=1, data.module.fold=0).
Identify hyperparameters that produce stable training and validation curves
AND maximize Val/C-index. Prefer settings that also achieve
Val/logrank_chi2 > 0 (stratification).

- Trials completed: 1
- Best trial: none (all 1 trial(s) failed or were incomplete)
- Cost: $0.0000 (tokens: n/a)

1 trial(s); 0 completed. No best trial — see individual trial sections for failure details.
### Trial 1 *(status: failed)*

- **hparams:** `{"data": "methylome_t1c_combined_high_vs_low", "data.cv.k": 1, "data.module.fold": 0, "env": "local", "exp_dir": "/home/jma/Documents/projects/safwat/coca_classifier_codebase/hpo_runs", "model": "resenc_survival", "model.chpt_path": "/home/jma/Documents/projects/safwat/coca_classifier_codebase/SSL3D_classification/checkpoints/S3D/checkpoint_final.pth", "model.save_preds": false, "trainer.callbacks.checkpoint.save_top_k": 0, "trainer.enable_checkpointing": false}`

**Analysis:**
no curves — failed

## Attempt 20260528-175318 (2026-05-28)

- Goal: Use the data config cli_configs/data/methylome_t1c_combined_high_vs_low.yaml
(selects survival_loss=deephit). Single fold (data.cv.k=1, data.module.fold=0).
Identify hyperparameters that produce stable training and validation curves
AND maximize Val/C-index. Prefer settings that also achieve
Val/logrank_chi2 > 0 (stratification).

- Trials completed: 1
- Best trial: none (all 1 trial(s) failed or were incomplete)
- Cost: $0.0000 (tokens: n/a)

1 trial(s); 0 completed. No best trial — see individual trial sections for failure details.
### Trial 1 *(status: failed)*

- **hparams:** `{"data": "methylome_t1c_combined_high_vs_low", "data.cv.k": 1, "data.module.fold": 0, "env": "local", "exp_dir": "/home/jma/Documents/projects/safwat/coca_classifier_codebase/hpo_runs", "model": "resenc_survival", "model.chpt_path": "/home/jma/Documents/projects/safwat/coca_classifier_codebase/SSL3D_classification/checkpoints/S3D/checkpoint_final.pth", "model.save_preds": false, "trainer.callbacks.checkpoint.save_top_k": 0, "trainer.enable_checkpointing": false}`

**Analysis:**
no curves — failed

## Attempt 20260528-183049 (2026-05-28)

- Goal: Use the data config cli_configs/data/methylome_t1c_combined_high_vs_low.yaml
(selects survival_loss=deephit). Single fold (data.cv.k=1, data.module.fold=0).
Identify hyperparameters that produce stable training and validation curves
AND maximize Val/C-index. Prefer settings that also achieve
Val/logrank_chi2 > 0 (stratification).

- Trials completed: 3
- Best trial: #2 (Val/C-index = 0.7865)
- Cost: $0.0000 (tokens: n/a)
- Best hparams:
- `data` = `methylome_t1c_combined_high_vs_low`
- `data.cv.k` = `1`
- `data.module.batch_size` = `24`
- `data.module.fold` = `0`
- `env` = `local`
- `exp_dir` = `/home/jma/Documents/projects/safwat/coca_classifier_codebase/hpo_runs`
- `model` = `resenc_survival`
- `model.chpt_path` = `/home/jma/Documents/projects/safwat/coca_classifier_codebase/SSL3D_classification/checkpoints/S3D/checkpoint_final.pth`
- `model.finetune_method` = `full_sawtooth`
- `model.lr` = `5e-05`
- `model.resnet_dropout` = `0.0`
- `model.save_preds` = `False`
- `model.survival_head_dropout` = `0.0`
- `model.survival_head_hidden_dim` = `128`
- `model.survival_head_norm` = `layernorm`
- `model.survival_loss.alpha` = `1.0`
- `model.survival_loss.beta` = `0.5`
- `model.survival_loss.gamma` = `0.0`
- `model.survival_loss.sigma` = `0.1`
- `model.weight_decay` = `0.001`
- `trainer.accumulate_grad_batches` = `2`
- `trainer.callbacks.checkpoint.save_top_k` = `0`
- `trainer.enable_checkpointing` = `False`

3 trial(s). Best is trial #2 (Val/C-index=0.7865).
### Trial 1

- **hparams:** `{"data": "methylome_t1c_combined_high_vs_low", "data.cv.k": 1, "data.module.fold": 0, "env": "local", "exp_dir": "/home/jma/Documents/projects/safwat/coca_classifier_codebase/hpo_runs", "model": "resenc_survival", "model.chpt_path": "/home/jma/Documents/projects/safwat/coca_classifier_codebase/SSL3D_classification/checkpoints/S3D/checkpoint_final.pth", "model.save_preds": false, "trainer.callbacks.checkpoint.save_top_k": 0, "trainer.enable_checkpointing": false}`
- **metrics:** `{"Train/C-index": 0.4601, "Train/loss": 0.8012, "Val/C-index": 0.7233, "Val/logrank_chi2": NaN, "Val/loss": 0.7623}`

**Analysis:**
The primary metric (Val/C-index) is extremely noisy, swinging 0.10–0.20 between adjacent epochs (e.g., 0.547→0.475 at e16→17, 0.534→0.627 at e26→27, 0.673→0.564 at e36→37) and spanning a 0.475 trough (e17) to the 0.723 global peak (e34); the wild Val/AUC@1y swings (0.19–0.98) and mostly-NaN validation logrank point to a small/low-event validation fold driving this variance, so single-epoch values are unreliable. In contrast, Val/loss falls smoothly and monotonically across the whole run (1.29→0.76) and tracks Train/loss closely (1.30→0.75), while Train/C-index stays pinned near 0.5 — so there is no loss-based overfitting and no train-vs-val discrimination gap; the model is steadily minimizing the objective without train discrimination running away. The best epoch (34) sits mid-run (~53% through 65 epochs), not at the end, but the final stretch (e56–64) settles into a comparable 0.70–0.71 band and the smoothed C-index keeps drifting up to ~0.669 by e64, so the late phase nearly recovers the peak rather than degrading or diverging. The first ~10 epochs are flat at 0.627 with the encoder LR held at 0, and discrimination only starts moving once the encoder unfreezes at e10 — after which the curve is dominated by high-variance oscillation rather than a clean trend.

### Trial 2

- **hparams:** `{"data": "methylome_t1c_combined_high_vs_low", "data.cv.k": 1, "data.module.batch_size": 24, "data.module.fold": 0, "env": "local", "exp_dir": "/home/jma/Documents/projects/safwat/coca_classifier_codebase/hpo_runs", "model": "resenc_survival", "model.chpt_path": "/home/jma/Documents/projects/safwat/coca_classifier_codebase/SSL3D_classification/checkpoints/S3D/checkpoint_final.pth", "model.finetune_method": "full_sawtooth", "model.lr": 5e-05, "model.resnet_dropout": 0.0, "model.save_preds": false, "model.survival_head_dropout": 0.0, "model.survival_head_hidden_dim": 128, "model.survival_head_norm": "layernorm", "model.survival_loss.alpha": 1.0, "model.survival_loss.beta": 0.5, "model.survival_loss.gamma": 0.0, "model.survival_loss.sigma": 0.1, "model.weight_decay": 0.001, "trainer.accumulate_grad_batches": 2, "trainer.callbacks.checkpoint.save_top_k": 0, "trainer.enable_checkpointing": false}`
- **metrics:** `{"Train/C-index": 0.6469, "Train/loss": 0.7304, "Val/C-index": 0.7865, "Val/logrank_chi2": 52.0, "Val/loss": 0.7361}`

**Analysis:**
The primary metric reads as Val/C-index, and the run is sharply two-phased: while the encoder LR is held at 0 (epochs 0–9) Val/C-index sits below chance at ~0.41–0.45, then climbs steeply once the encoder unfreezes at epoch 10, peaking at epoch 20 (0.7865, Val/loss 0.7361). After that peak it never sets a new high and instead oscillates noisily between ~0.54 and ~0.75 (e.g., 0.606 at e22, 0.538 at e23, 0.749 at e30, 0.612 at e32, 0.745 at e33), while training metrics keep marching the other way — Train/C-index 0.65→0.79, Train/loss 0.73→0.58, Train/hazard_ratio inflating from ~2.5 to ~28 — a clear overfitting/memorization signature over the back half of the run. The best epoch lands at only ~40% of the schedule, so the final ~30 epochs add no validation gain; note also that the epoch-20 peak is somewhat fragile, with a degenerate Val/hazard_ratio (~1e23) and scattered NaN logrank values (e6, e10–11, e15, e17, e31) pointing to unstable validation risk stratification rather than a cleanly converged optimum.

### Trial 3

- **hparams:** `{"data": "methylome_t1c_combined_high_vs_low", "data.cv.k": 1, "data.module.batch_size": 24, "data.module.fold": 0, "env": "local", "exp_dir": "/home/jma/Documents/projects/safwat/coca_classifier_codebase/hpo_runs", "model": "resenc_survival", "model.chpt_path": "/home/jma/Documents/projects/safwat/coca_classifier_codebase/SSL3D_classification/checkpoints/S3D/checkpoint_final.pth", "model.finetune_method": "full_sawtooth", "model.lr": 5e-05, "model.resnet_dropout": 0.1, "model.save_preds": false, "model.survival_head_dropout": 0.0, "model.survival_head_hidden_dim": 128, "model.survival_head_norm": "layernorm", "model.survival_loss.alpha": 1.0, "model.survival_loss.beta": 0.5, "model.survival_loss.gamma": 0.0, "model.survival_loss.sigma": 0.1, "model.weight_decay": 0.002, "trainer.accumulate_grad_batches": 2, "trainer.callbacks.checkpoint.save_top_k": 0, "trainer.enable_checkpointing": false}`
- **metrics:** `{"Train/C-index": 0.5773, "Train/loss": 0.7527, "Val/C-index": 0.7625, "Val/logrank_chi2": 2.939, "Val/loss": 0.7522}`

**Analysis:**
Treating Val/C-index as the tracked metric, the trial climbs steeply from ~0.41 (ep0) to its peak of **0.7625 at epoch 17** — only about a third of the way through the 48-epoch run — with the acceleration starting around epochs 12–13, coinciding with the encoder unfreezing (encoder LR leaves 0 at step 50/ep10). The validation curve is very noisy, swinging by ~0.1 between adjacent epochs (e.g., 0.624@ep16 → 0.7625@ep17 → 0.573@ep19; 0.685@ep23 → 0.734@ep24), and the runner-up epochs (25, 13, 42, 24) are scattered across the run rather than clustered, so the "best" epoch is more a noisy spike than a stable optimum. In the back half there's a mild overfitting signature: Train/loss falls monotonically (~0.93→0.62) and train risk separation explodes (Train/logrank_chi2 ~3→52, HR up to ~10 by ep46–47), while Val/C-index merely plateaus/oscillates in the 0.65–0.74 band and never re-attains the ep17 peak — though Val/loss stays roughly flat (~0.69, best 0.687@ep42) with no outright divergence or collapse. Two fragility cues stand out: Val/C-index sits *above* Train/C-index at the top epochs (0.76 vs 0.58 at ep17), and Val/logrank is intermittently NaN — including at the #2 epoch (25) — pointing to a small or easy validation split whose stratification is unstable.

## Attempt 20260528-214016 (2026-05-29)

- Goal: Use the data config cli_configs/data/methylome_t1c_combined_high_vs_low.yaml
(selects survival_loss=deephit). Single fold 1 (not 0) (data.cv.k=1, data.module.fold=1).
Identify hyperparameters that produce stable training and validation curves
AND maximize Val/C-index. Prefer settings that also achieve
Val/logrank_chi2 > 0 (stratification).

- Trials completed: 5
- Best trial: #3 (Val/C-index = 0.6903)
- Cost: $0.0000 (tokens: n/a)
- Best hparams:
- `data` = `methylome_t1c_combined_high_vs_low`
- `data.cv.k` = `1`
- `data.module.fold` = `1`
- `env` = `local`
- `exp_dir` = `/home/jma/Documents/projects/safwat/coca_classifier_codebase/hpo_runs`
- `model` = `resenc_survival`
- `model.chpt_path` = `/home/jma/Documents/projects/safwat/coca_classifier_codebase/SSL3D_classification/checkpoints/S3D/checkpoint_final.pth`
- `model.lr` = `5e-05`
- `model.save_preds` = `False`
- `model.survival_head_dropout` = `0.0`
- `model.survival_loss.beta` = `1.0`
- `model.weight_decay` = `0.001`
- `trainer.callbacks.checkpoint.save_top_k` = `0`
- `trainer.enable_checkpointing` = `False`

5 trial(s). Best is trial #3 (Val/C-index=0.6903).
### Trial 1

- **hparams:** `{"data": "methylome_t1c_combined_high_vs_low", "data.cv.k": 1, "data.module.fold": 1, "env": "local", "exp_dir": "/home/jma/Documents/projects/safwat/coca_classifier_codebase/hpo_runs", "model": "resenc_survival", "model.chpt_path": "/home/jma/Documents/projects/safwat/coca_classifier_codebase/SSL3D_classification/checkpoints/S3D/checkpoint_final.pth", "model.save_preds": false, "trainer.callbacks.checkpoint.save_top_k": 0, "trainer.enable_checkpointing": false}`
- **metrics:** `{"Train/C-index": 0.8116, "Train/loss": 0.6435, "Val/C-index": 0.6372, "Val/logrank_chi2": 0.8012, "Val/loss": 0.7777}`

**Analysis:**
**Diagnosis.** The run trains stably overall but the validation curves are noisy epoch-to-epoch — Val/C-index swings sharply between adjacent epochs (e.g., 0.542 at ep19 → 0.383 at ep20; Val/AUC@1y bounces from 0.44 to 1.0) — consistent with a small validation set; the cosine LR decay to ~0 freezes the tail into a flat plateau (loss pinned at 0.7738 and C-index at 0.6283 across ep79–82).

There is a strong overfitting signature: Train/C-index climbs to ~0.83–0.86 and the train risk separation blows up (hazard_ratio ~2 → 21 → 756, logrank_chi2 >100–190 with p≈1e-25) while Val/C-index only plateaus at ~0.62–0.64, Val/hazard_ratio stays ~1.1–1.7, and Val/logrank_chi2 stays negligible (<1, non-significant). The split between objective and ranking metric is visible — Val/loss bottoms around ep37 (≈0.7505), then rises to ≈0.79 by ep57–59 even as Train/loss keeps falling, whereas Val/C-index keeps creeping up slightly to its peak of 0.6372 at ep56 before settling back to ~0.628.

The best epoch (56) sits at roughly 65% of the 86-epoch run, with the final ~30 epochs producing no improvement (they merely tie at 0.6283 as the LR anneals), so the trial is saturated/mildly overfit rather than under-trained. No catastrophic divergence or collapse on validation — only a few early NaN logrank values from degenerate risk-group assignment — and the annealed tail is stable.

### Trial 2

- **hparams:** `{"data": "methylome_t1c_combined_high_vs_low", "data.cv.k": 1, "data.module.fold": 1, "env": "local", "exp_dir": "/home/jma/Documents/projects/safwat/coca_classifier_codebase/hpo_runs", "model": "resenc_survival", "model.chpt_path": "/home/jma/Documents/projects/safwat/coca_classifier_codebase/SSL3D_classification/checkpoints/S3D/checkpoint_final.pth", "model.lr": 5e-05, "model.save_preds": false, "model.survival_head_dropout": 0.0, "model.weight_decay": 0.001, "trainer.callbacks.checkpoint.save_top_k": 0, "trainer.enable_checkpointing": false}`
- **metrics:** `{"Train/C-index": 0.7194, "Train/loss": 0.6681, "Val/C-index": 0.6327, "Val/logrank_chi2": 1.667, "Val/loss": 0.7677}`

**Analysis:**
The run is clearly two-phase: with the encoder frozen (epochs 0–9, encoder LR held at 0) Val/C-index sits *below* chance at ~0.43, then once the encoder unfreezes (~epoch 10) it climbs steadily to a peak of 0.633 at epoch 39, surrounded by a tight cluster of strong epochs 38–43 (0.62–0.63). Past that point the trial overfits unmistakably: Train/C-index keeps rising to ~0.83 and Train hazard_ratio inflates from ~5 to 60+ (epoch 64) while Val/C-index decays into the 0.53–0.60 band and Val/loss bottoms near epoch 40 (~0.746) before drifting back up to ~0.85 by epoch 69. The validation signal is noisy throughout — neighboring epochs swing (e.g., 0.597→0.586 at epochs 44–45, a partial rebound to 0.617 at epoch 59) and early Val hazard_ratio/logrank are intermittently NaN, indicating degenerate stratification on a small/unstable val set — so single-epoch peaks should be read cautiously. The best epoch sits ~30 epochs before the end (39 of 69) with no genuine late recovery, so this trial is overtrained relative to its best validation point, not under-trained.

### Trial 3

- **hparams:** `{"data": "methylome_t1c_combined_high_vs_low", "data.cv.k": 1, "data.module.fold": 1, "env": "local", "exp_dir": "/home/jma/Documents/projects/safwat/coca_classifier_codebase/hpo_runs", "model": "resenc_survival", "model.chpt_path": "/home/jma/Documents/projects/safwat/coca_classifier_codebase/SSL3D_classification/checkpoints/S3D/checkpoint_final.pth", "model.lr": 5e-05, "model.save_preds": false, "model.survival_head_dropout": 0.0, "model.survival_loss.beta": 1.0, "model.weight_decay": 0.001, "trainer.callbacks.checkpoint.save_top_k": 0, "trainer.enable_checkpointing": false}`
- **metrics:** `{"Train/C-index": 0.6766, "Train/loss": 0.756, "Val/C-index": 0.6903, "Val/logrank_chi2": 5.105, "Val/loss": 0.7746}`

**Analysis:**
The primary metric reads as Val/C-index, and this trial is a textbook overfit. Val/C-index climbs noisily through the warmup/encoder-unfreeze phase (encoder lr leaves 0 at epoch 10, where val briefly collapses — Val/AUC@1y to 0.15, C-index to 0.50), peaks at **epoch 46 (0.690)** alongside the Val/loss trough (~0.775 across epochs 42–46), then steadily degrades to the 0.50–0.60 range by the end while Val/loss climbs back to ~0.90 (epoch 75–76).

Meanwhile Train keeps improving monotonically the whole way — Train/C-index 0.68→0.86, Train/loss 0.76→0.52, and Train/hazard_ratio exploding from ~1.5 to the hundreds (262 at ep65, 555 at ep67, 397 at ep73) with logrank_chi2 >170 — a clear memorization signature opening a train/val C-index gap of ~0.27 by the end, versus near-parity at the peak. The validation signal is also unstable: epoch-to-epoch Val/AUC@1y swings between ~0.13 and ~0.89 and Val/logrank/hazard_ratio go NaN or absurd (3.8e22 at ep47) at many epochs, indicating degenerate/near-single-group val stratification rather than a clean curve. The best epoch sits at ~60% of the run (46 of 76), so roughly 30 epochs of pure post-peak degradation follow it — no sign of under-training, the opposite.

### Trial 4

- **hparams:** `{"data": "methylome_t1c_combined_high_vs_low", "data.cv.k": 1, "data.module.fold": 1, "env": "local", "exp_dir": "/home/jma/Documents/projects/safwat/coca_classifier_codebase/hpo_runs", "model": "resenc_survival", "model.chpt_path": "/home/jma/Documents/projects/safwat/coca_classifier_codebase/SSL3D_classification/checkpoints/S3D/checkpoint_final.pth", "model.lr": 5e-05, "model.resnet_dropout": 0.1, "model.save_preds": false, "model.survival_head_dropout": 0.0, "model.survival_loss.alpha": 1.0, "model.survival_loss.beta": 1.0, "model.survival_loss.sigma": 0.1, "model.weight_decay": 0.002, "trainer.callbacks.checkpoint.save_top_k": 0, "trainer.enable_checkpointing": false}`
- **metrics:** `{"Train/C-index": 0.4657, "Train/loss": 0.8772, "Val/C-index": 0.6836, "Val/logrank_chi2": NaN, "Val/loss": 0.8737}`

**Analysis:**
The primary metric (Val/C-index) peaks very early at **epoch 7 (0.684)**, and the entire top cluster — epochs 5, 6, 7, 9 (~0.679–0.684) — falls within the head-only warmup phase while the encoder LR is pinned at 0; the encoder unfreezes at epoch 10 (encoder LR 0→5e-6), after which validation discrimination turns noisy and degrades, collapsing to 0.540 (epoch 11) and 0.418 (epoch 20) before only partially recovering to ~0.67 around epoch 33. This is a clear overfitting signature: once the encoder trains, Train/C-index climbs near-monotonically from ~0.50 to ~0.79 and Train/loss falls 0.93→0.63 (Train/logrank_chi2 inflating to ~78 and HR to ~36 by epoch 32), while Val/C-index never reattains its early peak — and Val/loss diverges in direction from the val ranking metric, drifting down to ~0.81 (epochs 15/33) even as C-index worsens. The best epoch sits at only **7 of 38 (~18% through)**, so essentially all of the full-finetuning regime hurts the primary metric; the model is overfit on the val ranking side, not under-trained. The logrank signal is also unreliable near the peak: Val/logrank_chi2 is NaN for epochs 4–8 and shows a degenerate spike at epoch 9 (HR ≈ 9.1e10, chi2 24.3), indicating collapsed/unstable risk stratification rather than genuine separation.

### Trial 5

- **hparams:** `{"data": "methylome_t1c_combined_high_vs_low", "data.cv.k": 1, "data.module.fold": 1, "env": "local", "exp_dir": "/home/jma/Documents/projects/safwat/coca_classifier_codebase/hpo_runs", "model": "resenc_survival", "model.chpt_path": "/home/jma/Documents/projects/safwat/coca_classifier_codebase/SSL3D_classification/checkpoints/S3D/checkpoint_final.pth", "model.lr": 5e-05, "model.save_preds": false, "model.survival_head_dropout": 0.0, "model.survival_loss.beta": 1.0, "model.weight_decay": 0.002, "trainer.callbacks.checkpoint.save_top_k": 0, "trainer.enable_checkpointing": false}`
- **metrics:** `{"Train/C-index": 0.6816, "Train/loss": 0.7723, "Val/C-index": 0.6571, "Val/logrank_chi2": 3.608, "Val/loss": 0.8023}`

**Analysis:**
This trial trains a survival model whose primary metric reads as **Val/C-index**, and the dynamics show smooth training progress over a very noisy, non-monotonic validation signal. Train metrics improve steadily and without pause (Train/C-index ~0.50→0.82, Train/loss 1.47→0.61, with Train/logrank_chi2 and hazard_ratio exploding to ~90 and ~60× by epochs 52–63), but **Val/C-index peaks at epoch 33 (0.6571)** and then degrades and oscillates down to the ~0.55–0.63 range, while Val/loss bottoms early around epoch 21 (~0.808) and drifts back up to ~0.85–0.92 — a clear overfitting signature where the optimum sits roughly mid-run (epoch 33 of 64, ~52%) and all top-5 epochs (28–45) precede the end.

Validation is genuinely unstable, not just slowly declining: Val/C-index swings several points epoch-to-epoch (e.g. 0.635→0.606→0.657→0.599 across epochs 28–34), Val/AUC@1y jumps erratically (e.g. 0.06 at epoch 11, 0.82 at epoch 52), and Val/logrank_chi2 frequently returns NaN (including in the #5 epoch, 28), pointing to small/degenerate validation risk strata. There is also a sharp early transient collapse at epochs 10–11 (Val/C-index drops to 0.43 then 0.34, AUC@1y to 0.06), coinciding with the encoder learning rate turning on (encoder LR 0→5e-6 at step 50), after which validation recovers and climbs to its epoch-33 peak. The run is clearly over-trained past its validation optimum — train and val have decoupled — rather than under-trained, since validation is falling while training continues to sharpen.
