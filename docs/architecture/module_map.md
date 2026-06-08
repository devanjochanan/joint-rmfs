# Rika RMFS Module Map

This document maps the functional components of the simulation to their current source of truth files, planned future directories, owners, and risk classifications.

---

## Module Mapping Overview

### 1. NetLogo Bridge / App Boundary
* **Description**: Facade interface between the NetLogo execution engine and Python simulation.
* **Current Source File(s)**: `netlogo.py` (root compatibility shim) → `src/rmfs/app/netlogo_api.py` (active implementation)
* **Future Destination**: `src/rmfs/app/` (bridge implementation already moved in Phase 4)
* **Owner**: Team / Shared
* **Migration Status**: Phase 4 bridge split completed. Root `netlogo.py` is now a thin shim re-exporting from `src/rmfs/app/netlogo_api.py`.
* **Behavior Risk Level**: **High**
* **Notes**: Any alteration to function signatures breaks NetLogo extension calls. The shim preserves exact API compatibility.

### 2. Simulation Core / Universe
* **Description**: Coordinates simulation ticks, schedules execution steps, and holds universe collections.
* **Current Source File(s)**: `model/inventory.py` (inherits from `engine/universe.py`)
* **Future Destination**: `src/rmfs/simulation/`
* **Owner**: Team / Shared
* **Migration Status**: Not moved (future package refactor)
* **Behavior Risk Level**: **High**
* **Notes**: Central loop driver; extremely sensitive to timing or sequence edits.

### 3. State Managers
* **Description**: Maintain registries and lookups of objects (pods, stations, storage slots, active jobs).
* **Current Source File(s)**: `model/pod_manager.py`, `model/station_manager.py`, `model/storage_manager.py`, `model/order_manager.py`, `model/intersection_manager.py`
* **Future Destination**: `src/rmfs/managers/`
* **Owner**: Team / Shared
* **Migration Status**: Not moved (future package refactor)
* **Behavior Risk Level**: **High**
* **Notes**: Mutates and retrieves active entity positions and queues.

### 4. Domain & Core Objects
* **Description**: Abstract definitions of physical items, paths, coordinates, and basic agents.
* **Current Source File(s)**: `engine/object.py`, `engine/coordinate.py`, `engine/netlogo_coordinate.py`, `engine/heading.py`, `engine/movement.py`, `engine/landscape.py`, `model/order.py`, `model/pod.py`, `model/station.py`, `model/storage.py`, `model/robot_job.py`
* **Future Destination**: `src/rmfs/core/`
* **Owner**: Team / Shared
* **Migration Status**: Not moved (future package refactor)
* **Behavior Risk Level**: **High**
* **Notes**: Fundamental classes used throughout the entire codebase.

### 5. Decisions: POA (Pick Order Assignment)
* **Description**: Core assignment algorithm mapping order batches to picker slots and matching stations.
* **Current Source File(s)**: `model/inventory.py` (specifically methods `assign_order`, `xxx`, `yyy`)
* **Future Destination**: `src/rmfs/decisions/poa/`
* **Owner**: Team / Shared
* **Migration Status**: Not moved. Future decision-extraction phase.
* **Behavior Risk Level**: **High**
* **Notes**: Critical scheduling component affecting simulation speed and throughput. Currently resides entirely in active source files; the future folder is a scaffold placeholder.

### 6. Decisions: PPS (Pick Pod Selection)
* **Description**: Selects the optimal pod to retrieve from storage to fulfill a station's batch orders (e.g., Pile-On or Demand).
* **Current Source File(s)**: `model/inventory.py` (specifically methods `find_best_pod`, `find_pod_with_the_highest_pile_on`, `find_pod_with_the_highest_demand`, `add_picking_task_after_pps`)
* **Future Destination**: `src/rmfs/decisions/pps/`
* **Owner**: Devan
* **Migration Status**: Not moved. Future decision-extraction phase.
* **Behavior Risk Level**: **High**
* **Notes**: Devan's primary research area. Currently resides entirely in active source files; the future folder is a scaffold placeholder.

### 7. Decisions: RTS (Return-to-Storage)
* **Description**: Selects storage spots for pods returning from pickers (fixed locations vs. nearest empty bin).
* **Current Source File(s)**: `model/robot.py` (methods `return_fix` and `return_nearest` logic inside `advance_state_if_needed`)
* **Future Destination**: `src/rmfs/decisions/rts/`
* **Owner**: Dewa
* **Migration Status**: Not moved. Future decision-extraction phase.
* **Behavior Risk Level**: **High**
* **Notes**: Dewa's primary research area. Closely tied to path planning and collision management.

### 8. Decisions: Charging
* **Description**: Tracks charger grid coordinates and calculates battery energy drainage during transits.
* **Current Source File(s)**: `model/robot.py` (energy consumption fields & `calculateEnergy`), `model/layout.py` (charging slot markers)
* **Future Destination**: `src/rmfs/decisions/charging/`
* **Owner**: Salsa
* **Migration Status**: Not moved. Future decision-extraction phase.
* **Behavior Risk Level**: **Medium**
* **Notes**: Salsa's primary research area. Controls energy consumption profiles.

### 9. Order Generation & Pod-SKU Stocking
* **Description**: Generates customer order catalogs and initial pod inventory distributions based on SKU stats.
* **Current Source File(s)**: `model/order_generator.py`, `model/pod_generator.py`, `model/item_pod_generator.py`
* **Future Destination**: `src/rmfs/order_generation/`
* **Owner**: Lukman
* **Migration Status**: Not moved. Future decision-extraction phase.
* **Behavior Risk Level**: **Medium**
* **Notes**: Lukman's primary research area.

### 10. Runtime I/O
* **Description**: Interacts with the local SQLite database (`warehouse.db`) to record simulation events and states dynamically.
* **Current Source File(s)**: `model/tools/job_task.py`, `model/tools/pod_location.py`, `model/tools/pod_travel.py`, `model/tools/order_history.py`, `model/tools/pre_assign.py`
* **Future Destination**: `src/rmfs/runtime_io/`
* **Owner**: Team / Shared
* **Migration Status**: Not moved (future package refactor)
* **Behavior Risk Level**: **Medium**
* **Notes**: Responsible for telemetry writes. Errors lead to diagnostic logs failing or database lock issues.

### 11. Metrics
* **Description**: Output logger that formats finished order logs and writes results.
* **Current Source File(s)**: `model/tools/write_record.py`
* **Future Destination**: `src/rmfs/metrics/`
* **Owner**: Team / Shared
* **Migration Status**: Not moved (future package refactor)
* **Behavior Risk Level**: **Low**
* **Notes**: Outputs CSV telemetry files.

### 12. Legacy / Quarantine
* **Description**: Holds standalone sandbox scripts and offline probability estimators.
* **Current Source File(s)**: `src/rmfs/legacy/astar.py`, `src/rmfs/legacy/astar_only.py`, `src/rmfs/legacy/stock_out_probability.py`, `src/rmfs/legacy/generate_pod.py`, plus root `profile_netlogo.py`
* **Future Destination**: `src/rmfs/legacy/`
* **Owner**: Team / Shared
* **Migration Status**: Confirmed-unused sandbox files quarantined in Phase 3; `robot_new.py` deleted in Phase 4.1 after no active references were found; `profile_netlogo.py` retained at root
* **Behavior Risk Level**: **Low**
* **Notes**: Quarantined files are not imported by the main NetLogo active run. `profile_netlogo.py` is documented as a local profiling entry point and was not moved.
