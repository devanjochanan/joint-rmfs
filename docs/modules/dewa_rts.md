# RTS (Return-To-Storage) Module Ownership Profile

This profile documents the ownership details, current code mappings, and implemented status for the RTS module.

---

## 1. Module Overview
* **Owner**: Dewa
* **Responsibility**: RTS (Return-To-Storage) algorithms that decide which storage slots to place pods back into once pickers or replenishers finish their jobs.
* **Folder Location**: `src/rmfs/rl/rts/` (Active RL and training infrastructure) and `src/rmfs/decisions/rts/` (Inference/Registry).

---

## 2. Refactoring Phase Status

* **Status**: Recovered RTS-RL training infrastructure is implemented locally for the completed recovery phases (Phases 1-15), with the targeted active job-queue regret-k scheduler patch and RTS-RL semantic recovery patch. Validation is limited to static checks and smoke tests.
* **Default behavior**:
  * `CurrentRTSPolicy` remains the simulator default.
  * RTS-RL rollout/evaluation remains disabled unless an integration explicitly selects `rts_rl_explicit` or `random_valid` via feature flags.
* **Restrictions**:
  * Do not modify or edit POA, PPS, charging, or order generation logic.
  * Preserving current baseline simulation behavior is paramount.

---

## 3. Behavior Source of Truth
The active behavior logic remains housed in:
* `model/robot.py`: Implements the state transition to `returning_pod`, executes RTS destination selection, reserves/release storage slots, logs RTS return completion, and notifies RTS-RL when the same robot next arrives at a station.
* `model/storage_manager.py`: Implements coordinates search logic (`getNearestEmptyStorageToLocation`).

---

## 4. Migration Risks & Verification Targets
Refactoring RTS logic affects:
* **Robot kinematics**: Changes to routing paths affect travel time and battery usage.
* **Storage state corruption**: If storage maps lag or fail to book correctly, multiple pods can "teleport" or try to occupy the same coordinate.
* **Deadlock checks**: Modifying return paths can result in intersection blockages.

---

## 5. Completed RTS-RL Infrastructure Phases

### Phase 6 - RTS-RL Port
* Ported the RTS-RL action space, state/features, stock encoder, masked actor-critic model, inference helpers, reward/cycle-reference helpers, and a validation smoke. The optional `RTSRLPolicy` requires an explicit model and safe zone-to-storage resolver; it does not load checkpoints automatically and is not the default policy.
* Raw threshold constants are excluded from model feature names. The model receives derived stock-risk signals such as fill ratios, below-threshold ratios, shortage depth, and replenishment signals.

### Phase 7 - Rollout/Evaluation Integration
* Added a process-local RTS runtime registry, worker-local JSONL rollout writer, outcome tracker, storage resolver, random-valid evaluation policy, rollout summary, and local-executor RTS config propagation.
* `current_probe` logs decisions and outcomes while preserving `CurrentRTSPolicy` selection behavior. `random_valid` is explicit opt-in and samples only valid Phase 6 action-mask entries before resolving the selected zone to free storage.
* RTS-RL reward logging can operate without a mandatory cycle reference. Return completion is logged as pending diagnostic telemetry, while the trainable reward target is completed `paper_cycle_duration`.

### Phase 8 - PPO and Checkpoint Validation
* Provided synthetic PPO math/checkpoint validation under `src/rmfs/rl/rts/training/`. It provides dataset loading, feature reconstruction, PPO update math validation, checkpoint layout helpers, latest/history tracking, and synthetic cycle-reference helpers.
* Offline/off-policy PPO training is not supported. `current_probe` and `random_valid` rollout rows are diagnostics/evaluation only and are not PPO-trainable.

### Phase 9 - On-Policy Training Spine
* Added a training-facing timebase, explicit policy checkpoint loader, `rts_rl_explicit` actor wrapper, trainable rollout metadata, active-checkpoint dataset filtering, optional TQDM/TensorBoard wrappers, and a controller dry-run spine.
* PPO-trainable rows must be `actor_kind=rts_rl_explicit` and match the active `policy_checkpoint_id`. Rows from `current`, `current_probe`, `random_valid`, `heuristic`, and `synthetic` policies are rejected for PPO training.
* Workers collect rollouts only. The controller owns model state, optimizer state, PPO updates, checkpoints, `latest.json`, and training logs. Training-facing duration uses `netlogo_steps_*` names, with warehouse time converted using runtime `tick_to_second`.
* Active v1 training no longer requires a manual reference run or mandatory `cycle_reference.json`. Batch 1 derives `reward_time_scale` from valid completed cycles, Batch 2+ reads it from latest checkpoint metadata, and optional `cycle_reference.json` remains legacy compatibility only. Alpha/reference update remains deferred.

### Phase 10 - SQLite Experiment Infrastructure
* Added a SQLite-backed experiment ledger at `data/output/rmfs_experiments.sqlite`. DuckDB is not used. The experiment ledger is separate from simulator `warehouse.db`.
* Workers never write the experiment ledger; controller-side or post-processing scripts initialize, ingest, and export ledger data. Phase 9 training outputs can be ingested into SQLite, and CSV summaries can be exported from the ledger.
* SQLite capture supports evaluations, checkpoints, experiments, training batches, and worker rollouts.

### Phase 11-15 - State Features & Audits
* **Phase 11 & 12**: Mapped all 13 RTS state-feature families and implemented selected dynamic features grounded in current simulation objects (replenishment station context, SKU turnover rank/value, robot congestion, neighborhood counts, zone distances). Next retrieval context, committed next task, and arrival-rate cycle context remain deferred/defaulted.
* **Phase 13**: Reward/Alpha Preservation Guard enforces that alpha reference updates are gated and only occur from completed, valid runs. No alpha is rederived.
* **Phase 14**: Added `netlogo_steps_requested` alias/property to `RunSpec`.
* **Phase 15**: Audited regret-k task allocation scheduling and classified full mature scheduling as deferred.
* **Targeted Regret-k Patch**: Added active job-queue `regret_k` task allocation with default `k=2`, plus a `legacy_nearest` fallback. This patch selects among currently visible queue jobs and idle robots only; committed-next reservations, future lookahead, and mature scheduling-pressure features remain deferred.
* **RTS-RL Semantic Recovery Patch**: Added canonical registry-backed RTS zones (`rts_z_r##_c##`), directed graph-distance features/resolution, paper-cycle outcome lifecycle, checkpoint schema metadata for reward/distance/zone semantics, and PPO filtering that rejects pending or censored outcomes. Synthetic smoke fixtures may still use explicit toy zone IDs such as `A`/`B` when their storages are tagged.

## 6. Residual RTS-RL Limits

The Rika-host robot path still does not implement the mature repo's full `replenish_store(z)` pre-return route semantics (picker -> replenishment station -> storage) for RTS-RL. The recovery patch records/censors replenishment next-task station arrivals and exposes replenish actions in the action space, but it does not claim mature replenish-route equivalence or performance impact.
