# Rika RMFS Current State

## Purpose
This file records the repository state and refactor constraints at the time it was last updated.

## Source of Truth
- Active behavior source remains `simulation.nlogo`, `netlogo.py`, `engine/**`, and active `model/**`.
- `src/rmfs/**` is scaffold/planning-only except for quarantined unused files in `src/rmfs/legacy/`; active behavior must not import from it yet.
- Current behavior must be preserved.

## Recorded Work
- Phase 1: repository inventory docs were created.
- Phase 1.5: runtime/generated/local artifacts were removed from tracking and ignored.
- Phase 2: scaffold and ownership docs were created.
- Phase 3: documentation-only `data/` planning skeleton was created, and confirmed-unused legacy/sandbox Python files were quarantined in `src/rmfs/legacy/`.

## Recorded Phase 3 Cleanup
- No active behavior files were moved.
- No baseline CSVs were moved; root-level baseline input CSVs remain the current source of truth for now.
- No runtime/output artifacts were moved; runtime DB/state/output artifacts remain ignored and should not be committed.
- `model/robot_new.py`, `astar.py`, `astar_only.py`, `generate_pod.py`, and `stock_out_probability.py` were quarantined only after inspection found no active imports or references outside docs/self-contained sandbox code.
- `profile_netlogo.py` remains at the repository root because it is documented as a local profiling entry point.

## Current Constraint
No behavior refactor has happened yet.
