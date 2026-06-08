# Rika RMFS Repository File Inventory

This document provides a comprehensive inventory and risk classification of all files and folders in the `joint-rmfs` repository before any refactoring is performed.

---

## Git State & Preflight Check (WSL)

* **Current Branch**: `main`
* **Git Status Summary**:
  * There are uncommitted changes on several data files, output CSVs, and `model/robot_new.py`.
  * The Python environment is configured on WSL at `/home/dewan/torch-gpu/bin/python`.
  * `tqdm` is installed in the virtual environment (Version 4.67.1) and has been added to `requirements.txt`.
* **Uncommitted Changes Detected**:
  * `.claude/settings.json`
  * `PS/2/generated_pod.csv`, `PS/4/generated_pod.csv`, `PS/5/generated_pod.csv`, `PS/skus_data.csv`
  * `assign_order.csv`, `generated_pod2.csv`, `generated_pod3.csv`, `generated_pod4.csv`, `generated_pod5n2.csv`
  * `model/robot_new.py` (Alternative/experimental robot implementation)
  * `output/order-finished_*.csv`
  * `pod_info.csv`
  * `robot sa data/generated_pod.csv`, `robot sa data/skus_data.csv`
  * `sorted_skus_data.csv`

---

## File Classification Table

| Path | Type | Apparent Responsibility | Risk Level | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `simulation.nlogo` | NetLogo UI | GUI layout, extension imports, and main bridge setup triggers (`setup`, `go`). | **High** | Critical entry point; tightly bound to `netlogo.py` API. |
| `netlogo.py` | Source / Bridge | Core bridge interface for NetLogo. Handles init, graph creation, ticking, and state serialization. | **High** | Central runtime coordinator; manages `netlogo.state` serialization. |
| `engine/` | Source Folder | Base geometry, coordinates, abstract agent loops, and neural network layout. | **Medium** | Defines foundation classes (`Object`, `Universe`, `Landscape`). |
| `engine/coordinate.py` | Source | Base coordinate representation. | Low | Simple utility class. |
| `engine/deep_q_network.py` | Source | PyTorch Neural Network implementation for RL intersection routing. | Medium | Used by the RL intersection manager. |
| `engine/heading.py` | Source | Manages movement headings (angles). | Low | Basic utility. |
| `engine/landscape.py` | Source | Grid patch tracker; queries neighbors. | Medium | Critical for collision and proximity lookups. |
| `engine/movement.py` | Source | Directional movement definitions. | Low | Utility. |
| `engine/netlogo_coordinate.py` | Source | NetLogo-specific coordinate translations. | Low | Bridge coordinate math. |
| `engine/object.py` | Source | Base class for simulation agents (Robots, Pods, Stations). | Medium | Domain ancestor class. |
| `engine/universe.py` | Source | Coordinates ticking of all simulation objects. | Medium | Abstract simulation loop. |
| `engine/util.py` | Source | Distance and intersection math functions. | Low | Math helper utilities. |
| `model/` | Source Folder | Domain logic (robots, inventory, matching algorithms, orders). | **High** | Contains core business logic and algorithms. |
| `model/deadlock_prevention_manager.py` | Source | Manages coordinate locks to prevent physical deadlocks. | Medium | Coordinates coordinate booking. |
| `model/intersection.py` | Source | Models an intersection node and logs state metrics. | Medium | Evaluates traffic congestion. |
| `model/intersection_manager.py` | Source | Dynamically adjusts directions on intersections (using RL/DQN). | **High** | Controls traffic scheduling. |
| `model/inventory.py` | Source | Main coordinator; manages order batching, queue, POA, and PPS. | **High** | Central dispatcher; mutates simulation state. |
| `model/item_pod_generator.py` | Source | Generates item-to-pod allocations. | Medium | Generates initial pod inventory distributions. |
| `model/layout.py` | Source | Generates grid layout (charging, picker slots, empty cells). | Medium | Builds layout matrix. |
| `model/live_advanced_table.py` | Source | TKinter or web-like visualization dashboard logic (unused). | Low | UI experiments. |
| `model/order.py` | Source | Represents an individual order, remaining SKUs, and stations. | Medium | Order entity tracking. |
| `model/order_generator.py` | Source | Generates random order sets based on class parameters. | Medium | Controls order generation. |
| `model/order_manager.py` | Source | Aggregates all pending, active, and completed orders. | Medium | State manager for orders. |
| `model/pod.py` | Source | Tracks individual Pod inventory levels, SKU volumes, and weights. | Medium | Pod entity. |
| `model/pod_generator.py` | Source | Alternate class to populate items into pods. | Medium | High overlap with `item_pod_generator.py`. |
| `model/pod_manager.py` | Source | Map of coordinates-to-pods, tracks available pods. | Medium | Global pod registry. |
| `model/robot.py` | Source | Active Robot class. Handles movement, collision avoidance, and RTS. | **High** | Core behavior logic; calculates energy. |
| `model/robot_job.py` | Source | Encapsulates a robot task sequence (take pod -> picker -> RTS). | Medium | Task container. |
| `model/robot_new.py` | Source | Inactive Robot class with uncommitted edits. | Medium | Needs validation; not currently imported. |
| `model/station.py` | Source | Picker / Replenishment stations and queue boundaries. | Medium | Station entity. |
| `model/station_manager.py` | Source | Station lookup registry and robot routing hooks. | Medium | Global station registry. |
| `model/storage.py` | Source | Represents physical bin positions on grid. | Low | Grid location entity. |
| `model/storage_manager.py` | Source | Manages association between pods and grid storage coordinates. | Medium | Storage assignment registry. |
| `model/tools/` | Source Folder | Database query and updates utilities. | **High** | Database interfaces. |
| `model/tools/job_task.py` | Source | SQLite job logs. | Medium | Interacts with DB. |
| `model/tools/order_history.py` | Source | SQLite order milestones tracking. | Medium | Interacts with DB. |
| `model/tools/pod_location.py` | Source | SQLite real-time pod coordinates registry. | **High** | Essential for updating physical pod placement. |
| `model/tools/pod_travel.py` | Source | SQLite logs for transit times. | Medium | Interacts with DB. |
| `model/tools/pre_assign.py` | Source | SQLite logs for pre-assigned metrics. | Medium | Interacts with DB. |
| `model/tools/timed.py` | Source | Profiling wrappers. | Low | Utility. |
| `model/tools/write_record.py` | Source | CSV writer helper. | Low | Utility. |
| `model/traffic_policy.py` | Source | Defines collision-avoidance rules. | Low | Policy container. |
| `model/zone.py` | Source | Implements zoning and K-Means area divisions. | Medium | Rerouting logic. |
| `netlogo.state` | Runtime State | Pickled dump of the active `Inventory` object. | **High** | Critical serialization file updated every tick. |
| `warehouse.db` | Runtime State | Primary SQLite database for logs and positions. | **High** | Telemetry and state mutations occur here. |
| `assign_order.csv` | Input Data / State | Active tracking file for order-station-pod statuses. | **High** | Primary input/state file for picker dispatch. |
| `pods.csv` | Input Data | Mapping of SKU levels per pod. | Medium | Initial setup catalog. |
| `generated_order.csv` | Input Data | The generated order stream database. | Medium | Input queue. |
| `generated_pod.csv` | Input Data | The 2D grid matrix representation of the warehouse. | **High** | Physical layout configuration template. |
| `skus_data.csv` | Input Data | Tracks remaining stock levels per SKU globally. | Medium | Global inventory levels. |
| `sorted_skus_data.csv` | Input Data | Sorted inventory list of SKUs. | Low | Helper view. |
| `pod_info.csv` | Output Artifact | Logging of items picked from pods. | Medium | Simulation metrics output. |
| `output/` | Output Folder | Finished simulation performance csv logs. | Medium | Historical run metrics. |
| `PS/` | Legacy / Snapshots | Folder containing snapshot database states and csv inputs. | Low | Sandbox/historical run backups. |
| `robot sa data/` | Legacy / Snapshots | Folder containing snapshot databases and csv files. | Low | Sandbox/historical run backups. |
| `.claude/` | Config | Claude project environment overrides and workspace permissions. | Low | Hidden editor configuration. |
| `.vscode/` | Config | VS Code editor workspaces settings. | Low | Hidden editor configuration. |
| `astar.py` | Legacy-Unknown | Offline A* grid search algorithm sandbox. | Low | Dead code; not imported. |
| `astar_only.py` | Legacy-Unknown | Secondary offline A* sandbox. | Low | Dead code; not imported. |
| `stock_out_probability.py`| Legacy-Unknown | Standalone SKU demand probability calculation. | Low | Standalone analytical script. |
| `profile_netlogo.py` | Source | Runner script to profile `netlogo.py` console executions. | Low | Standalone profiling tool. |
| `generate_pod.py` | Source | Script to write layout matrices to csv. | Low | Matrix generator. |
| `requirements.txt` | Config | Python environment dependency manifest. | Low | Added `tqdm==4.67.1`. |

---

## Risky & Behavior-Critical Files Summary

Below are the files classified as **High Risk** due to their involvement in critical simulation subsystems:

1. **NetLogo Bridge & Orchestration**:
   * `simulation.nlogo`: Controls frontend drawing and tick scheduling. Modifying this risks breaking extension calls.
   * `netlogo.py`: The Python facade. Handles serialization (`netlogo.state`), graph topology generation, and step loops.
2. **POA (Pod-to-Order Allocation)**:
   * `model/inventory.py`: Holds the main assignment logic (`assign_order`, `xxx`, `assign_order_old`) which binds orders to picker stations and pods.
3. **PPS (Pod-to-Picker Selection)**:
   * `model/inventory.py`: Implements pod scoring and ranking methods (`find_best_pod`, `find_pod_with_the_highest_pile_on`, `find_pod_with_the_highest_demand`) for active picker slots.
4. **RTS (Return-To-Storage)**:
   * `model/robot.py`: Governs pod returns. Contains logic for returning to fixed coordinates (`self.return_fix`) vs. dynamic nearest coordinates (`self.return_nearest`).
5. **Order Generation & Pod-SKU Generation**:
   * `model/order_generator.py` and `model/pod_generator.py` (plus `model/item_pod_generator.py`): Control the layout setup and incoming order distributions.
6. **Charging / Energy Behavior**:
   * `model/robot.py`: Implements `calculateEnergy` using mass, load mass, friction, velocity, turning, and lifts.
   * `model/layout.py`: Marks physical grid slots as charging stations (value `2`).
7. **Robot Movement & Path Planning**:
   * `model/robot.py` (and `model/robot_new.py`): Dictates velocity updates, acceleration, collision handling, and route execution.
   * `netlogo.py` (`DirectedGraph` class): Implements Dijkstra and modified Dijkstra with zoning penalties.
8. **Inventory Mutation & Station/Bin State**:
   * `model/inventory.py` (`finish_picking_task` & `finish_replenishment_task`): Updates remaining item quantities in pods, stations, and databases.
9. **Side Effect Operations**:
   * `model/tools/pod_location.py`: SQLite upserts representing physical coordinate changes. Teleportation bugs can occur if database writes fail or lag.
   * Root CSVs (`assign_order.csv`, `pod_info.csv`, `netlogo.state`): Continually overwritten during runs.
