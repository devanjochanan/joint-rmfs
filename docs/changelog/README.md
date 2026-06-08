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

