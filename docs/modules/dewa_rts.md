# RTS (Return-To-Storage) Module Ownership Profile

This profile documents the ownership details, current code mappings, and plans for the RTS module.

---

## 1. Module Overview
* **Owner**: Dewa
* **Responsibility**: RTS (Return-To-Storage) algorithms that decide which storage slots to place pods back into once pickers or replenishers finish their jobs.
* **Future Folder Location**: `src/rmfs/decisions/rts/`

---

## 2. Refactoring Phase Status

* **Status**: Phase 8 adds synthetic RTS-RL PPO validation and checkpoint infrastructure under `src/rmfs/rl/rts/training/`.
* **Default behavior**:
  * `CurrentRTSPolicy` remains the simulator default.
  * RTS-RL rollout/evaluation remains disabled unless an integration explicitly selects `current_probe` or `random_valid`.
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

## 6. Phase 7 Rollout/Evaluation Integration

Phase 7 adds a process-local RTS runtime registry, worker-local JSONL rollout writer, outcome tracker, storage resolver, random-valid evaluation policy, rollout summary, and local-executor RTS config propagation.

`current_probe` logs decisions and outcomes while preserving `CurrentRTSPolicy` selection behavior. `random_valid` is explicit opt-in and samples only valid Phase 6 action-mask entries before resolving the selected zone to free storage.

Reward is computed only with a valid cycle reference. Missing references produce `reward_computed=false`; no reward is fabricated.

No PPO/training, checkpoint loading, TensorBoard, DuckDB, NetLogo bridge changes, path-planning changes, POA/PPS/order-generation changes, charging changes, or default-policy changes were added. Rollout files are worker-local, and short three-tick executor smokes may not produce RTS decisions.

## 7. Phase 8 PPO and Checkpoint Validation

Phase 8 provides synthetic PPO math/checkpoint validation under `src/rmfs/rl/rts/training/`. It provides dataset loading, feature reconstruction, PPO update math validation, checkpoint layout helpers, latest/history tracking, and synthetic cycle-reference helpers.

Offline/off-policy PPO training is not supported. `current_probe` and `random_valid` rollout rows are diagnostics/evaluation only and are not PPO-trainable. True PPO training requires `rts_rl_explicit` on-policy rows and is deferred to Phase 9.

No simulator behavior is changed, no checkpoint auto-loading is added to the default simulator, and no real PPO training run is performed beyond synthetic validation smokes. Checkpoints live under ignored `data/runtime/**`.
