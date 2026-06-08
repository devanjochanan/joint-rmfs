# Rika RMFS Current Architecture Map

This document outlines the software architecture, execution path, import dependencies, file flows, and organizational ownership areas of the `joint-rmfs` simulation repository before any refactoring.

---

## 1. Summary of Current Architecture

The simulation operates as a **hybrid NetLogo/Python agent-based model**.
* **NetLogo (`simulation.nlogo`)** acts as the frontend graphical UI, drawing the warehouse grid and animating the robots (turtles) and station layouts.
* **Python (`netlogo.py` / `model/` / `engine/`)** serves as the simulation core. It calculates robot pathing, resolves collision conflicts, schedules batch orders, matches pods to picker stations, and computes energy metrics.
* **Communication Bridge**: NetLogo uses its standard `py` extension to load `netlogo.py`, call `setup()` to initialize the state, and call `tick()` at every step.
* **State Persistability**: The entire Python simulation state is stored in an `Inventory` instance (defined in `model/inventory.py`) and serialized to a local pickle file (`netlogo.state`) at the end of every tick. At the beginning of the next tick, this state is deserialized back into Python memory.

---

## 2. Current Execution Entry Points

There are two primary ways the simulation is executed:

1. **NetLogo UI Interface (Standard)**:
   * **Setup**: Clicking the `Setup` button in the NetLogo GUI invokes `netlogo.setup()`, which triggers layout creation, builds graphs, creates initial databases, and saves `netlogo.state`.
   * **Go**: Clicking the `Go` (or `Go-forever`) button invokes `netlogo.tick()` repeatedly, advancing the simulation time step.
2. **Headless Python Console Runner (Profiling/Benchmarking)**:
   * **Profile**: Executing `profile_netlogo.py` directly from Python invokes `netlogo.console_tick()`. This runs a headless simulation loop by loading and saving `netlogo.state` continuously, outputting performance metrics to `profile.prof`.

---

## 3. Current Dependency & Import Flow

The software components are layered as follows:

```mermaid
graph TD
    nlogo[simulation.nlogo] -->|py extension| netlogo_py[netlogo.py]
    profile_py[profile_netlogo.py] -->|CLI loop| netlogo_py
    
    netlogo_py -->|Coordinates / Grid| netlogo_coord[engine.netlogo_coordinate]
    netlogo_py -->|Base Object| eng_object[engine.object]
    
    netlogo_py -->|Zoning & Division| model_zone[model.zone]
    netlogo_py -->|Dijkstra Graph| nx[networkx]
    
    netlogo_py -->|Main State Manager| model_inv[model.inventory]
    
    model_inv -->|Objects & Layout| model_robot[model.robot]
    model_inv -->|Objects & Layout| model_pod[model.pod]
    model_inv -->|Objects & Layout| model_station[model.station]
    model_inv -->|Objects & Layout| model_storage[model.storage]
    
    model_inv -->|Algorithmic Managers| model_order_mgr[model.order_manager]
    model_inv -->|Algorithmic Managers| model_pod_mgr[model.pod_manager]
    model_inv -->|Algorithmic Managers| model_station_mgr[model.station_manager]
    model_inv -->|Algorithmic Managers| model_storage_mgr[model.storage_manager]
    model_inv -->|Algorithmic Managers| model_intersection_mgr[model.intersection_manager]
    
    model_intersection_mgr -->|RL Scheduling| eng_dqn[engine.deep_q_network]
    model_robot -->|Movement & Grid| eng_landscape[engine.landscape]
    model_robot -->|Collisions| model_traffic[model.traffic_policy]
    
    model_inv -->|SQLite Database Tools| db_tools[model.tools.*]
```

* **Core Engine Dependencies**: Core geometric and layout data structures (defined in `engine/`) are imported by the `model/` domain files.
* **Database Telemetry Utilities**: SQLite queries and operations (defined in `model/tools/`) are imported by `model/inventory.py`, `model/robot.py`, and `netlogo.py` to write/load telemetry during runs.

---

## 4. Current Data & Runtime File Flow

The simulation relies on a mix of structured CSV catalogs, layout templates, runtime SQLite logs, and binary pickle dumps:

```
[Input Layout & Catalogs]
  ├─ generated_pod.csv (Grid structure template)
  ├─ raw_order.csv (Customer orders raw data)
  ├─ items_dictionary.csv (SKU weights and classes)
  └─ items_slots_configuration.csv
         │
         ▼
[Initialization & Setup] (Creates static parameters)
  ├─ generated_order.csv (Batched order stream)
  ├─ pods.csv (Initial pod SKU allocations)
  └─ skus_data.csv / sorted_skus_data.csv
         │
         ▼
[Simulation Execution Loop] (Reads & writes state every tick)
  ├─ netlogo.state (Pickled Python Inventory class dump)
  ├─ assign_order.csv (Dynamic queue status of all orders)
  ├─ pod_info.csv (Quantities picked from pods)
  ├─ warehouse.db (SQLite: preassign, job tasks, pod locations/travel)
  │      │
  │      ▼
  └─ [Outputs / Telemetry]
         ├─ output/order-finished.csv (Milestones of completed orders)
         └─ intersection-energy-consumption.csv (Congestion energy tracking)
```

---

## 5. Current Likely Ownership Areas

Based on current code structures, the research and development areas map to individual contributors:

### Dewa: RTS (Return-to-Storage)
* **Responsibility**: Logic governing how robots return pods back to storage areas after picking or replenishment tasks are completed.
* **Key Implementation Location**:
  * `model/robot.py` (lines 727-775): Governs RTS behavior when `current_state` transitions to `returning_pod`.
  * Options:
    * `self.return_fix = True`: Robots return the pod to its original fixed storage coordinate (`self.job.pod_coordinate`).
    * `self.return_nearest = True`: Robots dynamically query the `StorageManager` to find the nearest empty storage cell relative to their current grid location (`storage_manager.getNearestEmptyStorageToLocation`), optimizing travel distances.

### Devan: PPS (Pod-to-Picker Selection)
* **Responsibility**: Algorithm to select the best pod to retrieve from storage for picker stations with active order queues.
* **Key Implementation Location**:
  * `model/inventory.py` (`process_orders`, `find_best_pod`, `find_pod_with_the_highest_pile_on`, `find_pod_with_the_highest_demand`).
  * Algorithms:
    * **Pile-On (`pps_pileon`)**: Prioritizes pods containing items that can satisfy multiple active orders simultaneously.
    * **Demand Matching (`pps_demand`)**: Scores pod candidates based on overall backlog item demands.

### Lukman: Order Generation & Pod-SKU Allocation
* **Responsibility**: Initial layout stocking and order streaming. Batching order flows based on frequency tables, and assigning item stocks to pods.
* **Key Implementation Location**:
  * `model/order_generator.py` and `model/order.py`: Generates the order catalog streams (`generated_order.csv`) based on A/B/C SKU probabilities.
  * `model/pod_generator.py` and `model/item_pod_generator.py`: Allocates item listings to pod slots (`pods.csv`) based on item demand classifications (A: 10%, B: 30%, C: 60%).

### Salsa: Charging & Energy consumption
* **Responsibility**: Grid distribution of charging stations, and calculations of battery/energy consumption profiles during robot transits.
* **Key Implementation Location**:
  * `model/layout.py`: Distributes charging slots on the matrix (value `2`).
  * `model/robot.py` (`calculateEnergy()` and energy fields): Computes motion energy dynamically based on robot mass, cargo mass, speed, acceleration, turns, and lifts.

---

## 6. Proposed Future Destination Map

To clean up the repository structure and isolate runtime states from source files, the following modular organization is planned:

| Current Path | Proposed Future Path | Rationale | Migration Risk |
| :--- | :--- | :--- | :--- |
| `netlogo.py` | `src/bridge/netlogo_bridge.py` | Separates NetLogo bridge boilerplate from model logic. | **Low**: Requires updating the import statements in `simulation.nlogo`. |
| `engine/` | `src/engine/` | Keeps low-level coordinate/grid structures grouped. | **Low**: Minor import search-and-replace. |
| `model/` | `src/model/` | Standardizes domain entity paths under a parent source folder. | **Medium**: Many cross-imports will need updating. |
| `model/tools/` | `src/telemetry/` | Separates SQLite logging and telemetry tools from domain entities. | **Low**: Standard package refactor. |
| `*.py` (root helpers) | `tools/` or `scratch/` | Removes profiling and matrix generation sandboxes from root. | **Low**: Standalone scripts. |
| `*.csv` (root catalogs) | `data/input/` | Isolates input dataset catalogs. | **High**: File path strings are hardcoded in multiple scripts; requires rigorous regression checks. |
| `assign_order.csv`, `pod_info.csv`, `warehouse.db` | `data/runtime/` | Prevents runtime states from cluttering source directories. | **High**: Real-time relative paths must be updated globally. |
| `netlogo.state` | `data/runtime/netlogo.state` | Groups all transient state files together. | **High**: Must update both Python pickling paths and NetLogo state loading logic. |

---

## 7. "Do Not Change in Phase 1" Section

> [!IMPORTANT]
> **No code modifications or refactoring operations are allowed in Phase 1.**
> The active codebase (including `simulation.nlogo`, `netlogo.py`, and all files in `engine/` and `model/`) must remain exactly in its current state. Only the adding of dependencies to `requirements.txt` is permitted. All architectural refactoring and folder migrations must wait for Phase 2 execution.

---

## 8. Open Questions & Ambiguities Found from Inspection

1. **Tracked Runtime Files**:
   * Why are dynamic CSV files (`assign_order.csv`, `pod_info.csv`) and SQLite databases (`warehouse.db`) tracked by Git instead of ignored? Running simulations creates large merge diffs, which can disrupt multi-person workflows.
2. **Overlap between Pod Generators**:
   * What is the structural difference between `model/pod_generator.py` and `model/item_pod_generator.py`? They appear to share highly overlapping responsibilities for SKU allocation.
3. **Dead Code / Experimental Files**:
   * `model/robot_new.py` contains 2,000+ lines of modifications but is not imported anywhere. We need to clarify if this is a legacy experiment or if it was intended to replace `model/robot.py` in some scenarios.
4. **Teleportation Bug Fixes**:
   * There are inline comments indicating teleportation issues (e.g. `# try to fix teleport` around line 628 of `model/inventory.py`). We need to understand if these telemetry writes directly affect robot coordinates or if they are purely diagnostic.
5. **RL Training Framework**:
   * `deep_q_network.py` is present, but the training script or offline parameters sweep code is not. It is unclear how new model checkpoints are created or updated.
