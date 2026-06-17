# Salsa Charging Integration & Pending Status

This document describes the integration status and specifications of Salsa's final charging solution on the `main_future` branch.

## 1. Status Summary
* **Status**: **Configuration artifact integrated / parked; active charging mechanism pending/inactive.**
* **Behavior Impact**: There is **no behavior change** in `main_future` because the underlying charging simulation mechanics (active dispatch, battery state tracking, and charger overlays) are currently inactive/absent in this branch.
* **Safety Warning**: **Do not use this configuration as an active experiment factor** until the charging mechanism is fully implemented, verified, and validated.

---

## 2. Source Branch Details
The configuration and metadata are derived from Salsa's baseline branch:
* **Source Branch**: `salsa_charging_baseline`
* **Latest Commit**: `595bc2fec15be5977cc6ad9512ea00a5b4d8a6cf`
* **Commit Message**: `Add final charging solution (hybrid 6/6, global, policy 18/60/50, |C|=12)`
* **Original Branch Files**:
  * `final_solution/README.md` (adapted and integrated into this document)
  * `final_solution/CODE_TOUCHPOINTS.md` (adapted and integrated into this document)
  * `final_solution/build_charging_solution.py` (adapted and ported to `scripts/data/build_charging_solution.py`)
  * `final_solution/charging_config.json` (adapted and ported to `data/input/charging/salsa_charging_config.json`)

---

## 3. Current Canonical Paths in `main_future`
To fit the restructured repository layout, the files have been integrated into current-path aware locations:
* **Parked Configuration**: [salsa_charging_config.json](file:///wsl.localhost/Ubuntu-22.04/home/dewan/Project%20Ta/Fresh%20Start%20Structure%20V1/Rika's%20Version/data/input/charging/salsa_charging_config.json)
* **Regeneration Script**: [build_charging_solution.py](file:///wsl.localhost/Ubuntu-22.04/home/dewan/Project%20Ta/Fresh%20Start%20Structure%20V1/Rika's%20Version/scripts/data/build_charging_solution.py)
* **Status and Documentation**: This file ([salsa_charging_pending.md](file:///wsl.localhost/Ubuntu-22.04/home/dewan/Project%20Ta/Fresh%20Start%20Structure%20V1/Rika's%20Version/docs/integration/salsa_charging_pending.md))

---

## 4. Configuration Specifications
The final, validated operating recommendation from Salsa's research includes the following values:

| Metric / Parameter | Value | Details |
|--------------------|-------|---------|
| **Placement Split** | Hybrid **6 picker-corridor + 6 storage-depot** cells | Balanced dwell-priority and exemplar coverage |
| **Budget Limit** | **|C| = 12** chargers | Upper bound for fleet size N = 20 |
| **Charging Structure** | **Global** active charging | `disable_active_charging: false` |
| **Battery Thresholds** | `battery_low_pct` = 18%, `battery_charged_pct` = 60%, `battery_interrupt_pct` = 50% | Policy 18 / 60 / 50 (validated Taguchi-robust) |

### Charger Positions (`[row, col]`)
The 12 configured coordinates are:
1. `[1, 2]` (Picker processing spot)
2. `[7, 2]` (Picker processing spot)
3. `[13, 2]` (Picker processing spot)
4. `[19, 2]` (Picker processing spot)
5. `[25, 2]` (Picker processing spot)
6. `[3, 4]` (Picker-1 queue-back)
7. `[17, 19]` (Storage depot exemplar)
8. `[20, 10]` (Storage depot exemplar)
9. `[11, 10]` (Storage depot exemplar)
10. `[2, 12]` (Storage depot exemplar)
11. `[14, 36]` (Storage depot exemplar)
12. `[11, 38]` (Storage depot exemplar)

---

## 5. Code Touchpoints Required for Active Charging
To activate this charging solution in the future, the following simulation engine files and mechanics must be implemented and verified:

### 1. `model/robot.py` (Battery State & Charging Policy)
* **Thresholds**: Apply class attributes `BATTERY_LOW_PCT`, `BATTERY_CHARGED_PCT`, and `BATTERY_INTERRUPT_PCT`.
* **State Trackers**: Read and update `battery_level_j` (Joules) and `battery_pct` (%).
* **Drive-by Charging**: Implement `_apply_drive_by_charging()`, adding charge to a robot whenever its cell is in `universe.charger_cells`.
* **Active Dispatch**: Implement `_start_charging_trip()`, routing low-battery robots to the nearest free charger in `universe.active_charger_cells` or `universe.charger_cells`.
* **Interrupt Heuristic**: Implement `_should_interrupt_charging()`, allowing robots to leave chargers early if `BATTERY_INTERRUPT_PCT` is met, jobs are pending, and no other robot is free.

### 2. `engine/universe.py` (Shared Charger State)
* Expose `universe.charger_cells` (set), `universe.active_charger_cells` (set), and `universe.occupied_chargers` (dict) to track charger occupancy and layout.

### 3. NetLogo Setup/Bridge (`netlogo.py`)
* Update `draw_storage_from_generated_file()` to parse `charging_config.json`, extract positions, and register overlay coordinates as active/inactive chargers on the grid map.

### 4. Launch Path Setup
* Ensure the launch configuration parser maps the `battery_*_pct` parameters from `charging_config.json` directly onto the `Robot` class prior to calling `setup()`.

---

## 6. Merge & Integration Status
* **Final Configuration**: Ported to `data/input/charging/salsa_charging_config.json` (identical layout and policy verified).
* **Source Branch Docs**: Ported and adapted to current paths and layout.
* **Active Mechanism**: Not enabled and not validated. It is explicitly labeled pending.
