# RTS (Return-To-Storage) Module Ownership Profile

This profile documents the ownership details, current code mappings, and plans for the RTS module.

---

## 1. Module Overview
* **Owner**: Dewa
* **Responsibility**: RTS (Return-To-Storage) algorithms that decide which storage slots to place pods back into once pickers or replenishers finish their jobs.
* **Future Folder Location**: `src/rmfs/decisions/rts/`

---

## 2. Refactoring Phase Status

* **Status**: Phase 6 adds an opt-in RTS-RL decision/model layer under `src/rmfs/rl/rts/`.
* **Default behavior**:
  * `CurrentRTSPolicy` remains the simulator default.
  * RTS-RL remains disabled unless an integration explicitly instantiates `RTSRLPolicy`.
* **Restrictions**:
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

## 5. Phase 6 RTS-RL Port

Phase 6 ports the RTS-RL action space, state/features, stock encoder, masked actor-critic model, inference helpers, reward/cycle-reference helpers, and a validation smoke. The optional `RTSRLPolicy` requires an explicit model and safe zone-to-storage resolver; it does not load checkpoints automatically and is not the default policy.

Raw threshold constants are excluded from model feature names. The model receives derived stock-risk signals such as fill ratios, below-threshold ratios, shortage depth, and replenishment signals.

Deferred work includes rollout collection, training, checkpoint/artifact loading, and richer zone/storage contracts.
