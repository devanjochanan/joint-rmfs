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
