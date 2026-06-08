# Rika RMFS File Migration Map

This planning table outlines the planned movement of existing repository files in future refactoring phases. 

> [!WARNING]
> **This is a planning document only.** 
> No files are physically moved or refactored in Phase 2.

---

## File Migration Plan

| Current Path | Current Role | Future Destination | Migration Phase | Risk | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `simulation.nlogo` | Frontend UI and setup/go buttons triggers. | `simulation.nlogo` (Keep at Root) | N/A | **High** | Cannot be moved; NetLogo expects the `.nlogo` file at root. Bridges to `netlogo.py`. |
| `netlogo.py` | Python entry point called by NetLogo bridge. | `src/rmfs/app/netlogo_bridge.py` | Phase 3 | **High** | NetLogo calls `import netlogo`. The bridge must maintain exact API bindings. |
| `engine/coordinate.py` | Coordinate class. | `src/rmfs/core/coordinate.py` | Phase 3 | Low | Core data class. |
| `engine/deep_q_network.py` | RL deep network for intersections. | `src/rmfs/core/deep_q_network.py` | Phase 3 | Medium | Neural net definition. |
| `engine/heading.py` | Heading angle tracker. | `src/rmfs/core/heading.py` | Phase 3 | Low | Helper enum/class. |
| `engine/landscape.py` | 2D patch matrix operations. | `src/rmfs/core/landscape.py` | Phase 3 | Medium | Proximity search query. |
| `engine/movement.py` | Coordinate updates utility. | `src/rmfs/core/movement.py` | Phase 3 | Low | Helper class. |
| `engine/netlogo_coordinate.py` | Coordinate math helper. | `src/rmfs/core/netlogo_coordinate.py` | Phase 3 | Low | Grid offsets helper. |
| `engine/object.py` | Base agent class. | `src/rmfs/core/object.py` | Phase 3 | Medium | Class hierarchy anchor. |
| `engine/universe.py` | Abstract clock runner. | `src/rmfs/simulation/universe.py` | Phase 3 | Medium | Drives object ticking. |
| `engine/util.py` | Geometric distance checks. | `src/rmfs/core/util.py` | Phase 3 | Low | Math helper. |
| `model/inventory.py` | Active simulation environment loop. | `src/rmfs/simulation/inventory.py` | Phase 3 / 4 | **High** | Heavy domain file; core decisions (POA, PPS) will be split off in Phase 4. |
| `model/robot.py` | Robot kinematics and collision check. | `src/rmfs/core/robot.py` | Phase 3 / 4 | **High** | RTS and charging details will be split off in Phase 4. |
| `model/pod.py` | Pod entity. | `src/rmfs/core/pod.py` | Phase 3 | Medium | domain class. |
| `model/station.py` | Station entity. | `src/rmfs/core/station.py` | Phase 3 | Medium | domain class. |
| `model/storage.py` | Storage slot. | `src/rmfs/core/storage.py` | Phase 3 | Low | domain class. |
| `model/order.py` | Order entity. | `src/rmfs/core/order.py` | Phase 3 | Medium | domain class. |
| `model/robot_job.py` | Job task container. | `src/rmfs/core/robot_job.py` | Phase 3 | Medium | domain class. |
| `model/zone.py` | Zoning manager. | `src/rmfs/core/zone.py` | Phase 3 | Medium | domain class. |
| `model/traffic_policy.py` | Traffic priorities. | `src/rmfs/core/traffic_policy.py` | Phase 3 | Low | Policy container. |
| `model/deadlock_prevention_manager.py` | Booking coordinates to prevent deadlock. | `src/rmfs/managers/deadlock_prevention_manager.py` | Phase 3 | Medium | Locks registry. |
| `model/intersection.py` | Intersection node. | `src/rmfs/core/intersection.py` | Phase 3 | Medium | Node metrics tracker. |
| `model/intersection_manager.py` | Intersection RL coordination. | `src/rmfs/managers/intersection_manager.py` | Phase 3 | **High** | Controls traffic flow. |
| `model/order_manager.py` | Order registry. | `src/rmfs/managers/order_manager.py` | Phase 3 | Medium | Order queues. |
| `model/pod_manager.py` | Pod registry. | `src/rmfs/managers/pod_manager.py` | Phase 3 | Medium | Pod coordinate maps. |
| `model/station_manager.py` | Station registry. | `src/rmfs/managers/station_manager.py` | Phase 3 | Medium | Station indices. |
| `model/storage_manager.py` | Storage slot registry. | `src/rmfs/managers/storage_manager.py` | Phase 3 | Medium | Storage slots map. |
| `model/order_generator.py` | CSV order stream generator. | `src/rmfs/order_generation/order_generator.py` | Phase 4 | Medium | Order stream builder. |
| `model/pod_generator.py` | CSV pod stock generator. | `src/rmfs/order_generation/pod_generator.py` | Phase 4 | Medium | Initial stock builder. |
| `model/item_pod_generator.py` | Alternate pod stock generator. | `src/rmfs/order_generation/item_pod_generator.py` | Phase 4 | Medium | Initial stock builder. |
| `model/layout.py` | Matrix grid builder. | `src/rmfs/order_generation/layout.py` | Phase 4 | Medium | Builds grid layouts. |
| `model/live_advanced_table.py` | Experimental TKinter UI. | `src/rmfs/legacy/live_advanced_table.py` | Phase 3 | Low | Unused visualization tool. |
| `model/robot_new.py` | Unused experimental Robot class. | `src/rmfs/legacy/robot_new.py` | Phase 3 | Low | Dead code quarantine. |
| `model/tools/job_task.py` | SQLite tasks updater. | `src/rmfs/runtime_io/job_task.py` | Phase 3 | Medium | Database writer. |
| `model/tools/order_history.py` | SQLite order updater. | `src/rmfs/runtime_io/order_history.py` | Phase 3 | Medium | Database writer. |
| `model/tools/pod_location.py` | SQLite pod coordinate registry. | `src/rmfs/runtime_io/pod_location.py` | Phase 3 | **High** | Database coordinate sync. |
| `model/tools/pod_travel.py` | SQLite travel times updater. | `src/rmfs/runtime_io/pod_travel.py` | Phase 3 | Medium | Database writer. |
| `model/tools/pre_assign.py` | SQLite preassign updater. | `src/rmfs/runtime_io/pre_assign.py` | Phase 3 | Medium | Database writer. |
| `model/tools/timed.py` | Profiling decorator. | `src/rmfs/runtime_io/timed.py` | Phase 3 | Low | Helper utility. |
| `model/tools/write_record.py` | Output metrics writer. | `src/rmfs/metrics/write_record.py` | Phase 3 | Low | CSV writer. |
| `astar.py` | Unused A* path planning. | `src/rmfs/legacy/astar.py` | Phase 3 | Low | Sandbox script. |
| `astar_only.py` | Unused A* path planning. | `src/rmfs/legacy/astar_only.py` | Phase 3 | Low | Sandbox script. |
| `stock_out_probability.py` | Standalone probability script. | `src/rmfs/legacy/stock_out_probability.py` | Phase 3 | Low | Sandbox script. |
| `profile_netlogo.py` | Headless execution runner. | `src/rmfs/app/profile_netlogo.py` | Phase 3 | Low | Simulation profiling CLI. |
| `generate_pod.py` | Matrix generator. | `src/rmfs/legacy/generate_pod.py` | Phase 3 | Low | Sandbox helper. |
| `requirements.txt` | Package manifest. | `requirements.txt` (Keep at Root) | N/A | Low | Unmodified python configs. |
| `docs/` | Repository documentation. | `docs/` (Keep at Root) | N/A | Low | Baseline project docs. |
| `generated_pod.csv` (and other CSVs) | Root baseline input CSVs. | `data/input/` | Phase 3 | **High** | All code references will need paths updated. |
| `warehouse.db` (and outputs) | Dynamic runtime logs. | `data/runtime/` | Phase 3 | **High** | SQLite relative path strings must be updated. |
