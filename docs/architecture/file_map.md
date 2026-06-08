# Rika RMFS File Migration Map

This planning table outlines the planned movement of existing repository files in future refactoring phases. 

> [!WARNING]
> **This is a planning document only.** 
> Phase 3 only quarantined confirmed-unused legacy/sandbox files in `src/rmfs/legacy/` and added a documentation-only `data/` skeleton.
> Active behavior files remain in their current locations until a later package refactor.

---

## File Migration Plan

| Current Path | Current Role | Future Destination | Migration Phase | Risk | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `simulation.nlogo` | Frontend UI and setup/go buttons triggers. | `simulation.nlogo` (Keep at Root) | N/A | **High** | Cannot be moved; NetLogo expects the `.nlogo` file at root. Bridges to `netlogo.py`. |
| `netlogo.py` | Python entry point called by NetLogo bridge. | `src/rmfs/app/netlogo_bridge.py` | Later package refactor | **High** | NetLogo calls `import netlogo`. The bridge must maintain exact API bindings. Not moved in Phase 3. |
| `engine/coordinate.py` | Coordinate class. | `src/rmfs/core/coordinate.py` | Later package refactor | Low | Core data class. Not moved in Phase 3. |
| `engine/deep_q_network.py` | RL deep network for intersections. | `src/rmfs/core/deep_q_network.py` | Later package refactor | Medium | Neural net definition. Not moved in Phase 3. |
| `engine/heading.py` | Heading angle tracker. | `src/rmfs/core/heading.py` | Later package refactor | Low | Helper enum/class (provisional destination). Not moved in Phase 3. |
| `engine/landscape.py` | 2D patch matrix operations. | `src/rmfs/core/landscape.py` | Later package refactor | Medium | Proximity search query (provisional destination). Not moved in Phase 3. |
| `engine/movement.py` | Coordinate updates utility. | `src/rmfs/core/movement.py` | Later package refactor | Low | Helper class (provisional destination). Not moved in Phase 3. |
| `engine/netlogo_coordinate.py` | Coordinate math helper. | `src/rmfs/core/netlogo_coordinate.py` | Later package refactor | Low | Grid offsets helper (provisional destination). Not moved in Phase 3. |
| `engine/object.py` | Base agent class. | `src/rmfs/core/object.py` | Later package refactor | Medium | Class hierarchy anchor (provisional destination). Not moved in Phase 3. |
| `engine/universe.py` | Abstract clock runner. | `src/rmfs/simulation/universe.py` | Later package refactor | Medium | Drives object ticking. Not moved in Phase 3. |
| `engine/util.py` | Geometric distance checks. | `src/rmfs/core/util.py` | Later package refactor | Low | Math helper. Not moved in Phase 3. |
| `model/inventory.py` | Active simulation environment loop. | `src/rmfs/simulation/inventory.py` | Later package refactor | **High** | Heavy domain file; core decisions (POA, PPS) will be split off in a later phase. Not moved in Phase 3. |
| `model/robot.py` | Robot kinematics and collision check. | `src/rmfs/core/robot.py` | Later package refactor | **High** | RTS and charging details will be split off in a later phase. Not moved in Phase 3. |
| `model/pod.py` | Pod entity. | `src/rmfs/core/pod.py` | Later package refactor | Medium | Domain class. Not moved in Phase 3. |
| `model/station.py` | Station entity. | `src/rmfs/core/station.py` | Later package refactor | Medium | Domain class. Not moved in Phase 3. |
| `model/storage.py` | Storage slot. | `src/rmfs/core/storage.py` | Later package refactor | Low | Domain class. Not moved in Phase 3. |
| `model/order.py` | Order entity. | `src/rmfs/core/order.py` | Later package refactor | Medium | Domain class. Not moved in Phase 3. |
| `model/robot_job.py` | Job task container. | `src/rmfs/core/robot_job.py` | Later package refactor | Medium | Domain class. Not moved in Phase 3. |
| `model/zone.py` | Zoning manager. | `src/rmfs/core/zone.py` | Later package refactor | Medium | Domain class. Not moved in Phase 3. |
| `model/traffic_policy.py` | Traffic priorities. | `src/rmfs/core/traffic_policy.py` | Later package refactor | Low | Policy container. Not moved in Phase 3. |
| `model/deadlock_prevention_manager.py` | Booking coordinates to prevent deadlock. | `src/rmfs/managers/deadlock_prevention_manager.py` | Later package refactor | Medium | Locks registry. Not moved in Phase 3. |
| `model/intersection.py` | Intersection node. | `src/rmfs/core/intersection.py` | Later package refactor | Medium | Node metrics tracker. Not moved in Phase 3. |
| `model/intersection_manager.py` | Intersection RL coordination. | `src/rmfs/managers/intersection_manager.py` | Later package refactor | **High** | Controls traffic flow. Not moved in Phase 3. |
| `model/order_manager.py` | Order registry. | `src/rmfs/managers/order_manager.py` | Later package refactor | Medium | Order queues. Not moved in Phase 3. |
| `model/pod_manager.py` | Pod registry. | `src/rmfs/managers/pod_manager.py` | Later package refactor | Medium | Pod coordinate maps. Not moved in Phase 3. |
| `model/station_manager.py` | Station registry. | `src/rmfs/managers/station_manager.py` | Later package refactor | Medium | Station indices. Not moved in Phase 3. |
| `model/storage_manager.py` | Storage slot registry. | `src/rmfs/managers/storage_manager.py` | Later package refactor | Medium | Storage slots map. Not moved in Phase 3. |
| `model/order_generator.py` | CSV order stream generator. | `src/rmfs/order_generation/order_generator.py` | Later package refactor | Medium | Order stream builder. Not moved in Phase 3. |
| `model/pod_generator.py` | CSV pod stock generator. | `src/rmfs/order_generation/pod_generator.py` | Later package refactor | Medium | Initial stock builder. Not moved in Phase 3. |
| `model/item_pod_generator.py` | Alternate pod stock generator. | `src/rmfs/order_generation/item_pod_generator.py` | Later package refactor | Medium | Initial stock builder. Not moved in Phase 3. |
| `model/layout.py` | Matrix grid builder. | `src/rmfs/simulation/layout.py` | Later package refactor | Medium | Builds grid layouts. Not moved in Phase 3. |
| `model/live_advanced_table.py` | Experimental TKinter UI. | `src/rmfs/legacy/live_advanced_table.py` | Later cleanup decision | Low | Not moved in Phase 3; not part of the approved candidate list. |
| `src/rmfs/legacy/robot_new.py` | Quarantined unused experimental Robot class. | `src/rmfs/legacy/robot_new.py` | Phase 3 quarantine | Low | Moved from `model/robot_new.py` after `git grep` found no active imports/references outside docs. Pre-existing local edits were preserved. |
| `model/tools/job_task.py` | SQLite tasks updater. | `src/rmfs/runtime_io/job_task.py` | Later package refactor | Medium | Database writer. Not moved in Phase 3. |
| `model/tools/order_history.py` | SQLite order updater. | `src/rmfs/runtime_io/order_history.py` | Later package refactor | Medium | Database writer. Not moved in Phase 3. |
| `model/tools/pod_location.py` | SQLite pod coordinate registry. | `src/rmfs/runtime_io/pod_location.py` | Later package refactor | **High** | Database coordinate sync. Not moved in Phase 3. |
| `model/tools/pod_travel.py` | SQLite travel times updater. | `src/rmfs/runtime_io/pod_travel.py` | Later package refactor | Medium | Database writer. Not moved in Phase 3. |
| `model/tools/pre_assign.py` | SQLite preassign updater. | `src/rmfs/runtime_io/pre_assign.py` | Later package refactor | Medium | Database writer. Not moved in Phase 3. |
| `model/tools/timed.py` | Profiling decorator. | `src/rmfs/runtime_io/timed.py` | Later package refactor | Low | Helper utility. Not moved in Phase 3. |
| `model/tools/write_record.py` | Output metrics writer. | `src/rmfs/metrics/write_record.py` | Later package refactor | Low | CSV writer. Not moved in Phase 3. |
| `src/rmfs/legacy/astar.py` | Quarantined unused A* path planning sandbox. | `src/rmfs/legacy/astar.py` | Phase 3 quarantine | Low | Moved from `astar.py`; no active imports/references found outside docs and itself. |
| `src/rmfs/legacy/astar_only.py` | Quarantined unused A* path planning sandbox. | `src/rmfs/legacy/astar_only.py` | Phase 3 quarantine | Low | Moved from `astar_only.py`; no active imports/references found outside docs and itself. |
| `src/rmfs/legacy/stock_out_probability.py` | Quarantined standalone probability script. | `src/rmfs/legacy/stock_out_probability.py` | Phase 3 quarantine | Low | Moved from `stock_out_probability.py`; no active imports/references found outside docs. |
| `profile_netlogo.py` | Headless execution runner. | `profile_netlogo.py` (Keep at Root for now) | Not moved in Phase 3 | Low | Documented profiling CLI; retained at root to avoid disturbing local profiling workflows. |
| `src/rmfs/legacy/generate_pod.py` | Quarantined matrix generator. | `src/rmfs/legacy/generate_pod.py` | Phase 3 quarantine | Low | Moved from `generate_pod.py`; no active imports/references found outside docs. |
| `requirements.txt` | Package manifest. | `requirements.txt` (Keep at Root) | N/A | Low | Unmodified python configs. |
| `docs/` | Repository documentation. | `docs/` (Keep at Root) | N/A | Low | Baseline project docs. |
| `generated_pod.csv` (and other CSVs) | Root baseline input CSVs. | `data/input/` | Later data relocation | **High** | No baseline CSVs were moved in Phase 3; all code references will need paths updated in a later phase. |
| `warehouse.db` (and outputs) | Dynamic runtime logs. | `data/runtime/` | Later data relocation | **High** | No runtime artifacts were moved in Phase 3; SQLite relative path strings must be updated in a later phase. |
