# RTS (Return-To-Storage) Module Ownership Profile

This profile documents the ownership details, current code mappings, and plans for the RTS module.

---

## 1. Module Overview
* **Owner**: Dewa
* **Responsibility**: RTS (Return-To-Storage) algorithms that decide which storage slots to place pods back into once pickers or replenishers finish their jobs.
* **Future Folder Location**: `src/rmfs/decisions/rts/`

---

## 2. Refactoring Phase Status

* **Status**: Scaffold placeholder only.
* **Restrictions**:
  * Do not write execution code in the scaffold directories.
  * Do not import regret-k or RTS-RL algorithms yet.
  * Do not modify or edit POA, PPS, charging, or order generation logic.
  * Preserving current baseline simulation behavior is paramount.

---

## 3. Behavior Source of Truth
The active behavior logic remains housed in:
* `model/robot.py` (lines 727–775): Implements the state transition to `returning_pod`, executing the choice between fixed original coordinates (`self.return_fix`) and dynamic closest empty coordinate lookup (`self.return_nearest`).
* `model/storage_manager.py`: Implements coordinates search logic (`getNearestEmptyStorageToLocation`).

---

## 4. Migration Risks & Verification Targets
Refactoring RTS logic in future phases affects:
* **Robot kinematics**: Changes to routing paths affect travel time and battery usage.
* **Storage state corruption**: If storage maps lag or fail to book correctly, multiple pods can "teleport" or try to occupies the same coordinate.
* **Deadlock checks**: Modifying return paths can result in intersection blockages.
