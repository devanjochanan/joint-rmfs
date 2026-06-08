# Rika RMFS Current State

## Purpose
This file records the repository state and refactor constraints at the time it was last updated.

## Source of Truth
- Active behavior source remains `simulation.nlogo`, `engine/**`, and active `model/**`.
- Root `netlogo.py` is a compatibility shim that delegates to `src/rmfs/app/netlogo_api.py`; it remains importable as `import netlogo` for NetLogo and local scripts.
- `src/rmfs/**` is scaffold/planning-only except for the active bridge implementation in `src/rmfs/app/netlogo_api.py`.
- No active behavior files have been modified during the refactoring phases so far. Behavior equivalence remains a Phase 5 acceptance-check item.

## Recorded Work
- Phase 1: repository inventory docs were created.
- Phase 1.5: runtime/generated/local artifacts were removed from tracking and ignored.
- Phase 2: scaffold and ownership docs were created.
- Phase 3: documentation-only `data/` planning skeleton was created, and confirmed-unused legacy/sandbox Python files were quarantined in `src/rmfs/legacy/`.
- Phase 4: the first active package boundary was created by moving the NetLogo Python bridge implementation from root `netlogo.py` into `src/rmfs/app/netlogo_api.py`. Root `netlogo.py` was replaced with a compatibility shim.
- Phase 4.1: post-bridge cleanup — deleted noncanonical generated-pod CSV variants and the quarantined `robot_new.py`. Corrected stale documentation wording.

## Recorded Phase 4 Bridge Split
- Root `netlogo.py` remains the stable NetLogo-facing module; it re-exports all public symbols from `src/rmfs/app/netlogo_api.py`.
- `src/rmfs/app/netlogo_api.py` contains the full bridge implementation (classes, helpers, `setup`, `tick`, `console_tick`, `setup_py`).
- Active model behavior still lives in `engine/**` and active `model/**`; these files were not moved or modified.
- POA/PPS/RTS/charging/order-generation logic has not been extracted.
- No baseline CSVs, DB paths, state paths, or output paths were relocated.
- `profile_netlogo.py` remains at the repository root and was not modified.

## Recorded Phase 4.1 Cleanup
- Deleted noncanonical generated-pod CSV variants (`generated_pod2.csv`, `generated_pod3.csv`, `generated_pod4.csv`, `generated_pod5n2.csv`) — no active code references were found.
- Deleted quarantined `src/rmfs/legacy/robot_new.py` — no active imports or references were found outside documentation.
- Remaining quarantined legacy files (`astar.py`, `astar_only.py`, `generate_pod.py`, `stock_out_probability.py`) are retained in `src/rmfs/legacy/`.
- No active behavior files were modified in this cleanup.

## Current Constraint
No behavior refactor has happened. Behavior equivalence remains a Phase 5 acceptance-check item.
