# RMFS Legacy Module

This folder is part of the future refactored structure for the Rika RMFS simulation.

* **Status**: Phase 3 quarantine for confirmed-unused legacy/sandbox files.
* **Rules**:
  * Active simulation behavior must not import from this folder.
  * Files here are retained for auditability and possible future deletion, not as active package modules.
  * Moving additional files here requires confirming they have no active imports or runtime references.
* **Quarantined in Phase 3**:
  * `robot_new.py`
  * `astar.py`
  * `astar_only.py`
  * `generate_pod.py`
  * `stock_out_probability.py`
* **Future Purpose**: Holding area for legacy sandbox scripts and unused experimental modules.
* **Future Owner**: Team / Shared
