# Rika RMFS Refactor Changelog

This document tracks all changes made during the refactoring process to ensure auditability, ownership alignment, and behavior preservation.

---

## 1. Changelog Purpose
To keep a record of all modifications, directory movements, and dependency updates. This ensures that:
* Collaborating researchers understand which components are being shifted.
* Regression checks are conducted at every stage.
* Behavior neutrality is audited and verified.

---

## 2. Standard Entry Format

For all future refactoring stages, developers must append entries in the following format:

```markdown
### [YYYY-MM-DD] Phase [Phase Number] - [Short Goal Summary]
* **Files Changed/Created/Deleted**:
  * `[NEW] path/to/newfile`
  * `[MODIFY] path/to/modifiedfile`
  * `[DELETE] path/to/deletedfile`
* **Behavior Changes**: [Yes / No] (If Yes, explain why)
* **Validation Run**: [Commands used to verify the run, e.g., python profile_netlogo.py]
* **Residual Risks**: [List any issues or regression targets]
```

---

## 2.5 Versioning Convention

We apply the following versioning convention for updates:
* **`+0.0.1`**: Small compatibility cleanup, docs cleanup, smoke cleanup, metadata cleanup, or small behavior cleanup that removes accidental friction without adding a new research capability.
* **`+0.1`**: New functional research/training capability or meaningful simulator/training behavior extension.

---

## 3. Historic Logs

### 2026-06-08 Phase 1 - Repository Audit & Inventory
* **Files Changed/Created/Deleted**:
  * `[NEW] docs/architecture/file_inventory.md`
  * `[NEW] docs/architecture/current_architecture_map.md`
  * `[MODIFY] requirements.txt` (added `tqdm==4.67.1`)
* **Behavior Changes**: No (Inspection only).
* **Validation Run**: Checked environment Python version and verified that the `tqdm` module was successfully installed in the virtual environment.
* **Residual Risks**: File path variables in the simulation python modules are currently hardcoded as relative strings.

### 2026-06-08 Phase 1.5 - Repository Artifact Hygiene
* **Files Changed/Created/Deleted**:
  * `[MODIFY] .gitignore`
  * `[DELETE - Cached Index Only]` `.DS_Store`, `netlogo.state`, `warehouse.db`, `warehouse_ps_old_8.db`, `assign_order.csv`, `pod_info.csv`, `profile.prof`, `output/`, `PS/`, `robot sa data/`, `.claude/`, `.vscode/`
* **Behavior Changes**: No.
* **Validation Run**: Ran `git status --short` and `git diff --cached --name-status` to verify index removals.
* **Residual Risks**: Runtime states are now unversioned (which is intended), but developers must ensure baseline templates (e.g. `generated_pod.csv`) are kept in sync manually.

### 2026-06-08 Phase 2 - Folder Scaffold & Ownership Documentation
* **Files Changed/Created/Deleted**:
  * `[NEW] docs/current/current_state.md`
  * `[NEW] docs/architecture/module_map.md`
  * `[NEW] docs/architecture/file_map.md`
  * `[NEW] docs/architecture/time_units.md`
  * `[NEW] docs/modules/dewa_rts.md`
  * `[NEW] docs/modules/devan_pps.md`
  * `[NEW] docs/modules/lukman_order_generation.md`
  * `[NEW] docs/modules/salsa_charging.md`
  * `[NEW] docs/changelog/README.md`
  * `[NEW] src/rmfs/**/README.md` (Scaffold placeholder folders)
* **Behavior Changes**: No (Expected to be **fully behavior-neutral**; no code changes were introduced).
* **Validation Run**: Verified status and diffs of the newly introduced documentation files and folders.
* **Residual Risks**: Future package imports must be updated carefully in a later package refactor to avoid circular references.

### 2026-06-08 Phase 3 - Behavior-Neutral Layout Cleanup
* **Files Changed/Created/Deleted**:
  * `[NEW] data/README.md`
  * `[NEW] data/input/README.md`
  * `[NEW] data/runtime/README.md`
  * `[NEW] data/archived/README.md`
  * `[MOVE] model/robot_new.py -> src/rmfs/legacy/robot_new.py`
  * `[MOVE] astar.py -> src/rmfs/legacy/astar.py`
  * `[MOVE] astar_only.py -> src/rmfs/legacy/astar_only.py`
  * `[MOVE] generate_pod.py -> src/rmfs/legacy/generate_pod.py`
  * `[MOVE] stock_out_probability.py -> src/rmfs/legacy/stock_out_probability.py`
  * `[MODIFY] docs/current/current_state.md`
  * `[MODIFY] docs/architecture/file_map.md`
  * `[MODIFY] docs/architecture/module_map.md`
  * `[MODIFY] docs/architecture/current_architecture_map.md`
  * `[MODIFY] src/rmfs/legacy/README.md`
* **Behavior Changes**: No. No active behavior files, baseline CSVs, runtime paths, or NetLogo bridge files were changed.
* **Validation Run**: Lightweight syntax checks were run with `/home/dewan/torch-gpu/bin/python` for `netlogo.py`, tracked `engine/*.py` and active `model/*.py`, and quarantined `src/rmfs/legacy/*.py` files.
* **Residual Risks**: Existing local uncommitted CSV changes remain outside this cleanup. The quarantined `robot_new.py` file preserves pre-existing local edits and should be reviewed before any future deletion.

### 2026-06-08 Phase 4 - NetLogo Bridge Package Boundary
* **Files Changed/Created/Deleted**:
  * `[MODIFY] netlogo.py` (replaced 946-line implementation with ~22-line compatibility shim)
  * `[NEW] src/rmfs/app/netlogo_api.py` (full bridge implementation moved here)
  * `[NEW] src/rmfs/__init__.py` (package init)
  * `[NEW] src/rmfs/app/__init__.py` (package init)
  * `[MODIFY] docs/current/current_state.md`
  * `[MODIFY] docs/architecture/file_map.md`
  * `[MODIFY] docs/architecture/module_map.md`
  * `[MODIFY] docs/changelog/README.md`
* **Behavior Changes**: No. The root `netlogo.py` shim re-exports all public symbols from `src/rmfs/app/netlogo_api.py` via `from src.rmfs.app.netlogo_api import *` constrained by `__all__`. No function signatures, return shapes, side effects, paths, seeds, or timing were altered.
* **Validation Run**:
  * `py_compile` passed for `netlogo.py`, `src/rmfs/app/netlogo_api.py`, `profile_netlogo.py`, all tracked `engine/*.py`, `model/*.py`, and `model/tools/*.py` files.
  * Import compatibility test confirmed `setup`, `tick`, `console_tick`, `setup_py`, `DirectedGraph`, and `ACTIVATE_NEAREST` are accessible via `import netlogo`.
  * Reference grep confirmed only `netlogo.py` imports from `src.rmfs.app.netlogo_api`; external callers still use `import netlogo`.
* **Residual Risks**:
  * Full simulation run (`setup()` → `tick()` loop) was not executed; only import/syntax compatibility was verified.
  * If NetLogo's `py` extension sets an unexpected working directory, the `sys.path` fixup in the shim should handle it, but this should be confirmed during the next interactive simulation run.
  * `from pip._internal import main as pipmain` in `netlogo_api.py` is preserved from the original; it is a fragile import that may break across pip versions.

### 2026-06-08 Phase 4.1 - Post-Bridge Cleanup
* **Files Changed/Created/Deleted**:
  * `[DELETE] generated_pod2.csv` (noncanonical generated-pod variant; no active code references)
  * `[DELETE] generated_pod3.csv` (noncanonical generated-pod variant; no active code references)
  * `[DELETE] generated_pod4.csv` (noncanonical generated-pod variant; no active code references)
  * `[DELETE] generated_pod5n2.csv` (noncanonical generated-pod variant; no active code references)
  * `[DELETE] src/rmfs/legacy/robot_new.py` (quarantined unused duplicate Robot class; no active imports found)
  * `[MODIFY] docs/current/current_state.md`
  * `[MODIFY] docs/architecture/file_map.md`
  * `[MODIFY] docs/architecture/module_map.md`
  * `[MODIFY] docs/architecture/current_architecture_map.md`
  * `[MODIFY] docs/changelog/README.md`
  * `[MODIFY] src/rmfs/legacy/README.md`
* **Behavior Changes**: No. No active behavior files were modified. Only noncanonical CSV variants and a quarantined unused legacy file were removed.
* **Validation Run**:
  * `git grep` confirmed zero active-code references to `generated_pod2`, `generated_pod3`, `generated_pod4`, `generated_pod5n2`, and `robot_new` in `simulation.nlogo`, `netlogo.py`, `src/rmfs/app/`, `engine/`, `model/`, and `profile_netlogo.py`.
  * `py_compile` passed for `netlogo.py`, `src/rmfs/app/netlogo_api.py`, `profile_netlogo.py`, and all tracked `engine/*.py`, `model/*.py`, `model/tools/*.py` files.
  * Import compatibility test confirmed `setup`, `tick`, `console_tick`, `setup_py` are accessible via `import netlogo`.
  * `git grep` confirmed no remaining "planned for Phase 4" stale wording in docs/src.
* **Residual Risks**:
  * Full simulation run has not been executed. Behavior equivalence remains a Phase 5 acceptance-check item.
  * `docs/architecture/file_inventory.md` still references `generated_pod2.csv` etc. and `model/robot_new.py` in historical inspection notes. These are Phase 1 audit records and were intentionally not edited.

### 2026-06-08 Phase 5 - Bridge Static Audit
* **Files Changed/Created/Deleted**:
  * `[NEW] docs/architecture/phase5_acceptance_audit.md` (subsequently removed in Phase 6)
  * `[MODIFY] docs/current/current_state.md`
  * `[MODIFY] docs/changelog/README.md`
  * `[MODIFY] src/rmfs/app/README.md` (corrected stale placeholder wording after Phase 4 bridge split)
* **Behavior Changes**: No. Documentation-only audit; no active behavior code was edited.
* **Validation Run**:
  * `py_compile` passed for `netlogo.py`, `src/rmfs/app/netlogo_api.py`, `profile_netlogo.py`, and tracked active `engine/*.py`, `model/*.py`, `model/tools/*.py` files.
  * Import/signature checks confirmed `setup()`, `tick()`, `console_tick()`, and `setup_py()` through root `netlogo` and `src.rmfs.app.netlogo_api`.
  * Shim export consistency check passed.
  * Historical AST comparison found no missing public functions, public classes, or public assignments between pre-Phase-4 root `netlogo.py` and current `src/rmfs/app/netlogo_api.py`.
  * Reference greps found no active imports from `src/rmfs/legacy/**` and no active-code references to deleted generated-pod variants or deleted `robot_new.py`.
* **Static Audit Result**: Verified with residual risks.
* **Residual Risks**:
  * Full NetLogo GUI run was not performed.
  * `setup()`, `tick()`, `setup_py()`, and `console_tick()` were not executed.
  * Behavior equivalence was not measured by simulation outputs.
  * Runtime paths and CSV/state behavior remain root-relative.
  * Decision modules remain unextracted; future researchers should avoid editing active shared internals until extraction.
  * Import checks emitted a Matplotlib cache warning because `/home/dewan/.config/matplotlib` is not writable.

### 2026-06-08 Phase 6 - Remove Acceptance Audit and Run Smoke Check
* **Files Changed/Created/Deleted**:
  * `[DELETE] docs/architecture/phase5_acceptance_audit.md`
  * `[MODIFY] docs/current/current_state.md`
  * `[MODIFY] docs/changelog/README.md`
* **Behavior Changes**: No.
* **Validation Run**:
  * Local-only smoke check of `setup()` and `tick()` calls executed inside a temporary disposable repository.
* **Residual Risks**:
  * Full GUI simulation, BehaviorSpace, and paper fidelity runs remain to be verified.

### 2026-06-13 Phase 10 - TQDM Progress Polish
* **Files Changed/Created/Deleted**:
  * `[MODIFY] src/rmfs/orchestration/local_executor.py`
  * `[MODIFY] src/rmfs/rl/rts/training/progress.py`
  * `[MODIFY] src/rmfs/rl/rts/training/controller.py`
* **Behavior Changes**: No (only progress display improvements and fixing runtime NameError bug in worker wall-time calculation).
* **Validation Run**:
  * `/home/dewan/torch-gpu/bin/python -m py_compile src/rmfs/rl/rts/training/progress.py src/rmfs/rl/rts/training/controller.py scripts/validation/*.py`
  * `/home/dewan/torch-gpu/bin/python scripts/validation/rts_training_controller_dry_run.py`
* **Residual Risks**: None.

### 2026-06-13 Phase 11 - Precise RTS State-Feature Gap Map
* **Files Changed/Created/Deleted**:
  * `[NEW] docs/architecture/rts_state_feature_gap_map.md`
* **Behavior Changes**: No (documentation-only stage).
* **Validation Run**:
  * `git diff --stat`
* **Residual Risks**: None.

### 2026-06-13 Phase 12 - RTS State-Feature Implementation
* **Files Changed/Created/Deleted**:
  * `[MODIFY] src/rmfs/rl/rts/state.py`
  * `[MODIFY] src/rmfs/rl/rts/zone_features.py`
  * `[MODIFY] scripts/validation/rts_rl_rollout_smoke.py`
* **Behavior Changes**: Yes (implemented dynamic SKU turnover rank/value, replenishment station context, neighborhood counts, robot congestion metrics, and zone distance calculations fully grounded in current simulation objects).
* **Validation Run**:
  * `/home/dewan/torch-gpu/bin/python -m py_compile src/rmfs/rl/rts/*.py src/rmfs/rl/rts/training/*.py scripts/validation/*.py`
  * `/home/dewan/torch-gpu/bin/python scripts/validation/rts_on_policy_actor_smoke.py`
  * `/home/dewan/torch-gpu/bin/python scripts/validation/rts_ppo_update_smoke.py`
  * `/home/dewan/torch-gpu/bin/python scripts/validation/rts_rl_rollout_smoke.py`
* **Residual Risks**: None.

### 2026-06-13 Phase 13 - Reward and Alpha Preservation Guard
* **Files Changed/Created/Deleted**: None (audit and verification only).
* **Behavior Changes**: No (confirmed cycle reference and alpha gating follow specifications exactly; no alpha rederivation or reward redesign was performed).
* **Validation Run**:
  * `/home/dewan/torch-gpu/bin/python -m py_compile src/rmfs/rl/rts/reward.py src/rmfs/rl/rts/cycle_reference.py src/rmfs/experiments/cycle_reference_update.py src/rmfs/rl/rts/training/checkpoint.py scripts/validation/*.py`
  * `/home/dewan/torch-gpu/bin/python scripts/validation/cycle_reference_update_proposal_smoke.py`
  * `/home/dewan/torch-gpu/bin/python scripts/validation/rts_ppo_update_smoke.py`
* **Residual Risks**: None.

### 2026-06-13 Phase 14 - Timebase Naming Cleanup
* **Files Changed/Created/Deleted**:
  * `[MODIFY] src/rmfs/orchestration/run_spec.py`
* **Behavior Changes**: No (added `netlogo_steps_requested` alias/property to `RunSpec` for clean human-facing naming semantics without altering timing or step logic).
* **Validation Run**:
  * `/home/dewan/torch-gpu/bin/python -m py_compile src/rmfs/orchestration/*.py src/rmfs/rl/rts/training/controller.py src/rmfs/experiments/ledger/ingest_phase9.py scripts/validation/*.py`
  * `/home/dewan/torch-gpu/bin/python scripts/validation/rts_training_controller_dry_run.py`
  * `/home/dewan/torch-gpu/bin/python scripts/validation/phase9_ingest_smoke.py`
* **Residual Risks**: None.

### 2026-06-13 Phase 15 - Regret-k Targeted Audit Only
* **Files Changed/Created/Deleted**:
  * `[NEW] docs/architecture/regret_k_audit.md`
* **Behavior Changes**: No (audit-only, regret-k task allocation is classified as deferred).
* **Validation Run**:
  * `git diff --stat`
* **Residual Risks**: None.

### 2026-06-13 Phase 16 - Docs and Current-State Cleanup
* **Files Changed/Created/Deleted**:
  * `[MODIFY] docs/current/current_state.md`
  * `[MODIFY] docs/modules/dewa_rts.md`
  * `[MODIFY] docs/architecture/rts_rl_on_policy_training.md`
  * `[MODIFY] docs/architecture/experiment_ledger.md`
* **Behavior Changes**: No (documentation-only cleanup).
* **Validation Run**:
  * Run compileall checks on `src/` and `scripts/` (completed successfully).
  * Executed all 17 validation smoke tests under `scripts/validation/` (all passed successfully).
  * Verified removal of stale planning/scaffold placeholders in docs using git grep.
* **Residual Risks**: None.


### 2026-06-13 Codex Verification Patch - +0.0.1
* **Files Changed/Created/Deleted**:
  * `[MODIFY] src/rmfs/rl/rts/state.py`
  * `[MODIFY] scripts/validation/rts_rl_rollout_smoke.py`
  * `[MODIFY] docs/current/current_state.md`
  * `[MODIFY] docs/modules/dewa_rts.md`
  * `[MODIFY] docs/architecture/rts_rl_on_policy_training.md`
  * `[MODIFY] docs/architecture/rts_state_feature_gap_map.md`
* **Behavior Changes**: No simulator behavior change. Corrected RTS state-feature fidelity metadata for destination robot pressure and tightened smoke coverage for the Phase 12 destination-pressure feature.
* **Validation Run**:
  * `/home/dewan/torch-gpu/bin/python` recursive `py_compile` sweep over RTS, experiments, orchestration, training, experiment, and validation scripts.
  * Listed safe validation smokes from the Codex verification task.
* **Residual Risks**:
  * Full real multi-worker NetLogo execution remains unvalidated.

### 2026-06-14 Targeted Regret-k Active Queue Scheduler - +0.0.1
* **Files Changed/Created/Deleted**:
  * `[NEW] src/rmfs/decisions/task_allocation/__init__.py`
  * `[NEW] src/rmfs/decisions/task_allocation/regret_k.py`
  * `[NEW] scripts/validation/regret_k_allocator_smoke.py`
  * `[NEW] scripts/validation/regret_k_training_config_smoke.py`
  * `[MODIFY] model/inventory.py`
  * `[MODIFY] scripts/training/rts_train_controller.py`
  * `[MODIFY] src/rmfs/orchestration/local_executor.py`
  * `[MODIFY] src/rmfs/orchestration/run_spec.py`
  * `[MODIFY] src/rmfs/rl/rts/training/controller.py`
  * `[MODIFY] src/rmfs/rl/rts/training/on_policy_config.py`
  * `[MODIFY] docs/architecture/regret_k_audit.md`
  * `[MODIFY] docs/architecture/rts_rl_on_policy_training.md`
  * `[MODIFY] docs/architecture/rts_state_feature_gap_map.md`
  * `[MODIFY] docs/current/current_state.md`
  * `[MODIFY] docs/modules/dewa_rts.md`
  * `[MODIFY] docs/changelog/README.md`
* **Behavior Changes**: Yes. Active job assignment now defaults to active job-queue `regret_k` with `regret_k=2`; the previous first-queue-job nearest-idle-robot behavior remains selectable as `legacy_nearest`. Committed-next reservations and future lookahead remain deferred.
* **Validation Run**:
  * `/home/dewan/torch-gpu/bin/python -m py_compile model/inventory.py src/rmfs/decisions/task_allocation/*.py src/rmfs/orchestration/run_spec.py src/rmfs/orchestration/local_executor.py src/rmfs/rl/rts/training/on_policy_config.py src/rmfs/rl/rts/training/controller.py scripts/training/rts_train_controller.py scripts/validation/regret_k_allocator_smoke.py scripts/validation/regret_k_training_config_smoke.py`
  * `/home/dewan/torch-gpu/bin/python scripts/validation/regret_k_allocator_smoke.py`
  * `/home/dewan/torch-gpu/bin/python scripts/validation/regret_k_training_config_smoke.py`
* **Residual Risks**: Full mature scheduler equivalence, committed-next reservations, future pressure feedback, and performance effects were not validated. No long simulation or training campaign was run.

### 2026-06-15 Reward Cold-Start Cleanup - +0.0.1
* **Files Changed/Created/Deleted**:
  * `[NEW] src/rmfs/rl/rts/training/reward_normalizer.py`
  * `[NEW] scripts/validation/rts_reward_cold_start_smoke.py`
  * `[MODIFY] scripts/training/init_rts_checkpoint.py`
  * `[MODIFY] scripts/training/rts_train_controller.py`
  * `[MODIFY] src/rmfs/rl/rts/reward.py`
  * `[MODIFY] src/rmfs/rl/rts/outcome_tracker.py`
  * `[MODIFY] src/rmfs/rl/rts/training/on_policy_config.py`
  * `[MODIFY] src/rmfs/rl/rts/training/controller.py`
  * `[MODIFY] src/rmfs/rl/rts/training/checkpoint.py`
  * `[MODIFY] src/rmfs/rl/rts/training/on_policy_dataset.py`
  * `[MODIFY] src/rmfs/experiments/ledger/ingest_phase9.py`
  * `[MODIFY] scripts/validation/init_rts_checkpoint_smoke.py`
  * `[MODIFY] scripts/validation/rts_training_controller_dry_run.py`
  * `[MODIFY] docs/architecture/rts_rl_on_policy_training.md`
  * `[MODIFY] docs/architecture/rts_rl_phase9_requirements.md`
  * `[MODIFY] docs/architecture/rts_rl_training.md`
  * `[MODIFY] docs/modules/dewa_rts.md`
  * `[MODIFY] docs/current/current_state.md`
  * `[MODIFY] docs/changelog/README.md`
* **Behavior Changes**: Yes. Active RTS-RL v1 training no longer requires a manual reference run or mandatory `cycle_reference.json`; cold-start reward normalization derives `reward_time_scale` from valid completed cycle rows and stores normalizer metadata in checkpoint/training artifacts.
* **Validation Run**:
  * `/home/dewan/torch-gpu/bin/python` recursive `py_compile` sweep over `src/rmfs/rl/rts`, `src/rmfs/experiments`, `scripts/training`, and `scripts/validation`.
  * `/home/dewan/torch-gpu/bin/python scripts/validation/rts_reward_cold_start_smoke.py`
  * Safe existing validation smokes listed in the task.
* **Residual Risks**: No real simulation, `--execute` training, alpha/reference audit, paper reward validation, or performance validation was run.

### 2026-06-15 RTS Light Operational Cleanup - +0.0.1
* **Files Changed/Created/Deleted**:
  * `[DELETE] scripts/training/rts_train_smoke.py`
  * `[MODIFY] scripts/training/rts_train_controller.py`
  * `[MODIFY] src/rmfs/rl/rts/training/on_policy_config.py`
  * `[MODIFY] src/rmfs/rl/rts/training/controller.py`
  * `[MODIFY] src/rmfs/orchestration/local_executor.py`
  * `[MODIFY] scripts/validation/regret_k_training_config_smoke.py`
  * `[MODIFY] docs/validation/rts_rl_training.md`
  * `[MODIFY] docs/architecture/rts_decision_seam.md`
  * `[MODIFY] docs/changelog/README.md`
* **Behavior Changes**: No. Simplified routine RTS operational tasks: artifact labels are now auto-generated if omitted, latest local artifact pointer resolves automatically for resuming, obsolete synthetic cycle reference requirements in smokes are removed, and CLI descriptions/terminology are normalized.
* **Validation Run**:
  * `/home/dewan/torch-gpu/bin/python` recursive `py_compile` checks.
  * Executed a safe RTS validation subset available at the time (cold-start, dry-run controller, dataset/PPO-related smokes, regret-k allocator, regret-k config).
* **Residual Risks**: No real training run, simulation, or exhaustive active-smoke sweep was performed.

### 2026-06-15 RTS Training Terminal-Output Cleanup Patch - +0.0.1
* **Files Changed/Created/Deleted**:
  * `[MODIFY] src/rmfs/rl/rts/training/on_policy_config.py`
  * `[MODIFY] scripts/training/rts_train_controller.py`
  * `[MODIFY] src/rmfs/rl/rts/training/controller.py`
  * `[MODIFY] model/tools/pod_location.py`
  * `[MODIFY] model/tools/pod_info.py`
  * `[MODIFY] scripts/validation/rts_training_controller_dry_run.py`
  * `[MODIFY] docs/architecture/rts_rl_on_policy_training.md`
  * `[MODIFY] docs/current/current_state.md`
  * `[MODIFY] docs/changelog/README.md`
* **Behavior Changes**: No simulator or training behavior changes. Worker subprocess stdout/stderr streams are now redirected to DEVNULL by default to keep the TQDM controller progress clean. Added `--debug-worker-logs` to persist logs per worker only when explicitly enabled. Gated noisy hot-path pod-location success prints.
* **Validation Run**:
  * `/home/dewan/torch-gpu/bin/python` recursive `py_compile` checks.
  * Executed extended `rts_training_controller_dry_run.py` validation test suite covering log suppression, opt-in log creation, and failure reporting.
  * Executed other safe validation smoke tests (`regret_k_training_config_smoke.py`, `rts_training_checkpoint_loader_smoke.py`).
* **Residual Risks**: None. No real multi-worker training execution or simulation updates were performed.

### 2026-06-15 RTS-RL Semantic Recovery Patch - +0.0.1
* **Files Changed/Created/Deleted**:
  * `[NEW] src/rmfs/rl/rts/zone_registry.py`
  * `[NEW] src/rmfs/rl/rts/graph_distance.py`
  * `[NEW] scripts/validation/rts_semantic_recovery_smoke.py`
  * `[MODIFY] model/robot.py`
  * `[MODIFY] src/rmfs/rl/rts/zone_features.py`
  * `[MODIFY] src/rmfs/rl/rts/state.py`
  * `[MODIFY] src/rmfs/rl/rts/storage_resolver.py`
  * `[MODIFY] src/rmfs/rl/rts/outcome_tracker.py`
  * `[MODIFY] src/rmfs/rl/rts/reward.py`
  * `[MODIFY] src/rmfs/rl/rts/rollout_schema.py`
  * `[MODIFY] src/rmfs/rl/rts/evaluation_policy.py`
  * `[MODIFY] src/rmfs/rl/rts/evaluation_summary.py`
  * `[MODIFY] src/rmfs/rl/rts/runtime_config.py`
  * `[MODIFY] src/rmfs/rl/rts/runtime_install.py`
  * `[MODIFY] src/rmfs/rl/rts/training/*.py`
  * `[MODIFY] scripts/validation/rts_*_smoke.py`
  * `[MODIFY] docs/architecture/rts_rl_on_policy_training.md`
  * `[MODIFY] docs/architecture/rts_state_feature_gap_map.md`
  * `[MODIFY] docs/architecture/time_units.md`
  * `[MODIFY] docs/modules/dewa_rts.md`
  * `[MODIFY] docs/current/current_state.md`
  * `[MODIFY] docs/changelog/README.md`
* **Behavior Changes**: Yes. RTS-RL now uses canonical registry-backed storage zones, directed graph-distance semantics, paper-cycle reward lifecycle, and checkpoint/runtime guards for reward/distance/zone semantics. The active robot RTS path also reserves selected storage slots through `StorageManager` and releases picked slots back to the empty-storage list. Return completion is diagnostic; PPO training accepts only completed paper-cycle outcomes.
* **Validation Run**:
  * `/home/dewan/torch-gpu/bin/python -m py_compile` over touched RTS-RL, robot, training, and validation files.
  * `PYTHONPATH=. /home/dewan/torch-gpu/bin/python scripts/validation/rts_semantic_recovery_smoke.py`
  * `PYTHONPATH=. /home/dewan/torch-gpu/bin/python scripts/validation/rts_rl_port_smoke.py`
  * `PYTHONPATH=. /home/dewan/torch-gpu/bin/python scripts/validation/rts_rl_rollout_smoke.py`
  * `PYTHONPATH=. /home/dewan/torch-gpu/bin/python scripts/validation/rts_on_policy_dataset_smoke.py`
  * `PYTHONPATH=. /home/dewan/torch-gpu/bin/python scripts/validation/rts_reward_cold_start_smoke.py`
  * `PYTHONPATH=. /home/dewan/torch-gpu/bin/python scripts/validation/init_rts_checkpoint_smoke.py`
* **Residual Risks**: No full NetLogo simulation, BehaviorSpace experiment, PPO/RL training run, benchmark, or output-equivalence run was performed. The Rika-host path still does not implement mature `replenish_store(z)` pre-return route equivalence; replenishment next-task arrivals are censored for paper-cycle training eligibility.

### 2026-06-17 Owner Architecture Migration - +0.1.0
* **Files Changed/Created/Deleted**:
  * `[NEW] scripts/validation/owner_architecture_migration_smoke.py`
  * `[NEW] src/rmfs/decisions/charging/__init__.py`
  * `[NEW] src/rmfs/decisions/charging/config.py`
  * `[NEW] src/rmfs/decisions/charging/placement.py`
  * `[NEW] src/rmfs/decisions/charging/policy.py`
  * `[NEW] src/rmfs/decisions/charging/types.py`
  * `[NEW] src/rmfs/decisions/pps/__init__.py`
  * `[NEW] src/rmfs/decisions/pps/heuristic.py`
  * `[NEW] src/rmfs/decisions/pps/model_paths.py`
  * `[NEW] src/rmfs/decisions/pps/modes.py`
  * `[NEW] src/rmfs/decisions/pps/runtime.py`
  * `[NEW] src/rmfs/decisions/pps/types.py`
  * `[NEW] src/rmfs/order_generation/__init__.py`
  * `[NEW] src/rmfs/order_generation/bootstrap.py`
  * `[NEW] src/rmfs/order_generation/pod_sku.py`
  * `[NEW] src/rmfs/order_generation/policy.py`
  * `[MODIFY] model/inventory.py`
  * `[MODIFY] scripts/data/build_charging_solution.py`
  * `[MODIFY] scripts/experiments/run_pps_replications.py`
  * `[MODIFY] scripts/run/run_pps_backend_episode.py`
  * `[MODIFY] scripts/training/train_pps_rl.py`
  * `[MODIFY] scripts/validation/order_generation_policy_smoke.py`
  * `[MODIFY] scripts/validation/seed_reproducibility_smoke.py`
  * `[MODIFY] src/rmfs/app/netlogo_api.py`
  * `[MODIFY] src/rmfs/decisions/charging/README.md`
  * `[MODIFY] src/rmfs/decisions/pps/README.md`
  * `[MODIFY] src/rmfs/order_generation/README.md`
  * `[MODIFY] src/rmfs/rl/pps/__init__.py`
  * `[MODIFY] src/rmfs/rl/pps/env.py`
  * `[DELETE] model/item_pod_generator.py`
  * `[DELETE] model/order_generator.py`
  * `[DELETE] model/pod_generator.py`
  * `[DELETE] src/rmfs/rl/pps/model_paths.py`
  * `[DELETE] src/rmfs/runtime_io/order_generation.py`
  * `[DELETE] You`
  * `[DELETE] [!NOTE]`
* **Behavior Changes**: No. Pure architectural refactoring to relocate owner features (Charging -> Salsa, PPS -> Devan, Order Generation -> Lukman) into modular `src/rmfs/` sub-packages, and remove legacy duplicate/scaffold scripts.
* **Validation Run**:
  - `wsl /home/dewan/torch-gpu/bin/python scripts/validation/owner_architecture_migration_smoke.py`
  - `wsl PYTHONPATH=. /home/dewan/torch-gpu/bin/python -c "import netlogo; print('netlogo import ok')"`
* **Residual Risks**: None. All legacy imports successfully redirected and validated via smoke tests.

### 2026-06-17 v1.10.0 RTS-RL Validation Cleanup - +0.0.1
* **Files Changed/Created/Deleted**:
  * `[RESTORE] scripts/validation/rts_on_policy_actor_smoke.py`
  * `[RESTORE] scripts/validation/rts_on_policy_dataset_smoke.py`
  * `[RESTORE] scripts/validation/rts_ppo_update_smoke.py`
  * `[RESTORE] scripts/validation/rts_rl_port_smoke.py`
  * `[RESTORE] scripts/validation/rts_rl_rollout_smoke.py`
  * `[RESTORE] scripts/validation/rts_timebase_smoke.py`
  * `[RESTORE] scripts/validation/rts_training_checkpoint_loader_smoke.py`
  * `[MODIFY] docs/validation/rts_rl_on_policy_training.md`
  * `[MODIFY] docs/validation/rts_rl_training.md`
  * `[MODIFY] docs/architecture/rts_rl_rollout_integration.md`
  * `[MODIFY] docs/changelog/README.md`
* **Behavior Changes**: No simulator or training behavior changes. Active RTS-RL validation smokes accidentally removed during migration were restored from local history and lightly aligned to current cold-start reward semantics.
* **Validation Run**:
  * `/home/dewan/torch-gpu/bin/python scripts/validation/rts_reward_cold_start_smoke.py`
  * `/home/dewan/torch-gpu/bin/python scripts/validation/rts_on_policy_dataset_smoke.py`
  * `/home/dewan/torch-gpu/bin/python scripts/validation/rts_ppo_update_smoke.py`
  * `/home/dewan/torch-gpu/bin/python scripts/validation/rts_training_checkpoint_loader_smoke.py`
  * `/home/dewan/torch-gpu/bin/python scripts/validation/rts_timebase_smoke.py`
  * `/home/dewan/torch-gpu/bin/python scripts/validation/rts_on_policy_actor_smoke.py`
  * `/home/dewan/torch-gpu/bin/python scripts/validation/rts_rl_port_smoke.py`
  * `/home/dewan/torch-gpu/bin/python scripts/validation/rts_rl_rollout_smoke.py`
* **Residual Risks**: No real simulation, `--execute` training, BehaviorSpace run, benchmark, or long validation campaign was run; validation is limited to static checks and safe smokes.
