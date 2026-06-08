# Charging & Energy Module Ownership Profile

This profile documents the ownership details, current code mappings, and plans for the charging and energy module.

---

## 1. Module Overview
* **Owner**: Salsa
* **Responsibility**: Charger coordinate configurations and calculations of battery/energy consumption kinetics during robot movement (accelerating, turning, lifting).
* **Future Folder Location**: `src/rmfs/decisions/charging/`

---

## 2. Refactoring Phase Status

* **Status**: Scaffold placeholder only.
* **Restrictions**:
  * Do not write execution code in the scaffold directories.
  * Do not add battery swap policies or charger relocation/placement heuristics yet.
  * Do not edit charging variables in the simulation model.
  * Preserving current baseline simulation behavior is paramount.

---

## 3. Behavior Source of Truth
The active behavior logic remains housed in:
* `model/robot.py`: Implements battery calculations (specifically `calculateEnergy`, angular rotational energy coefficients, and lift parameters).
* `model/layout.py`: Dynamically marks physical grid cells as chargers (value `2`).

---

## 4. Migration Risks & Verification Targets
Refactoring these formulas affects:
* **Energy kinetics**: Incorrect calculation of friction, mass, or acceleration coefficients breaks telemetry metrics.
* **Movement synchronization**: If energy evaluations take too long or trigger state changes dynamically, robot step rates can stall.
* **Charger layout definitions**: Grid charger marker changes can confuse pathing algorithms which read coordinate weights.
