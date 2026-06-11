# RTS (Return-To-Storage) Module Ownership Profile

This profile documents the ownership details, current code mappings, and plans for the RTS module.

---

## 1. Module Overview
* **Owner**: Dewa
* **Responsibility**: RTS (Return-To-Storage) algorithms that decide which storage slots to place pods back into once pickers or replenishers finish their jobs.
* **Future Folder Location**: `src/rmfs/decisions/rts/`

---

## 2. Refactoring Phase Status

* **Status**: Phase 9 adds the first explicit RTS-RL on-policy PPO training spine under `src/rmfs/rl/rts/training/`.
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

## 8. Phase 9 On-Policy Training Spine

Phase 9 adds a training-facing timebase, explicit policy checkpoint loader, `rts_rl_explicit` actor wrapper, trainable rollout metadata, active-checkpoint dataset filtering, optional TQDM/TensorBoard wrappers, and a controller dry-run spine.

PPO-trainable rows must be `actor_kind=rts_rl_explicit` and match the active `policy_checkpoint_id`. Rows from `current`, `current_probe`, `random_valid`, `heuristic`, and `synthetic` policies are rejected for PPO training.

Workers collect rollouts only. The controller owns model state, optimizer state, PPO updates, checkpoints, `latest.json`, and training logs. Training-facing duration uses `netlogo_steps_*` names, with warehouse time converted using runtime `tick_to_second`.

Default simulator behavior remains unchanged, and no checkpoint auto-loading is added outside explicit mode. DuckDB ledgers, full evaluation, long runs, best-checkpoint ranking, and other Phase 10 items remain deferred.

## 10. Phase 10 SQLite Experiment Infrastructure

Phase 10 adds a SQLite-backed experiment ledger at `data/output/rmfs_experiments.sqlite`. DuckDB is not used. The experiment ledger is separate from simulator `warehouse.db`.

Workers never write the experiment ledger; controller-side or post-processing scripts initialize, ingest, and export ledger data. Phase 9 training outputs can be ingested into SQLite, and CSV summaries can be exported from the ledger.

Phase 10 also adds deterministic experiment/scenario IDs, strict feature flags, evaluation seed packs, dry-run evaluation specs, best-checkpoint metadata pointers, and proposal-only cycle-reference update files. `latest.json` remains a latest-checkpoint pointer only; best-checkpoint selection does not copy or mutate checkpoints.

Long runs, real evaluation, DoE, benchmarks, and cycle-reference apply mode remain deferred.

---

## 9. Phase 9 Cleanup Patch

The Phase 9 cleanup patch resolves critical device, metadata, timebase, CLI, and zone_ids issues:
- **CLI Default Safety**: The controller defaults to a dry run unless `--execute` is explicitly set. `--dry-run` is deprecated.
- **RTS-RL Explicit Behavior**: Under `rts_rl_explicit`, no fallback to nearest or random heuristic is permitted.
- **Explicit zone_ids**: Explicit `--zone-ids` (or checkpoint metadata) is required for real training; no raw fallback.
- **Consistent Device Resolution**: 
  - Controller default: `auto` (cuda if available, else cpu).
  - Worker default: `cpu` (safe multi-worker default).
  - Worker `auto` resolves to `cuda` if available.
  - Worker `cuda` fails clearly if CUDA is unavailable.
- **Timebase Semantics**:
  - `warehouse_time` is `Inventory._tick`.
  - `netlogo_step` is derived dynamically as `round(warehouse_time / tick_to_second)`.
  - Hardcoded scaling constants (e.g. 0.15, 0.25) are strictly forbidden.

---

## 11. Phase 10 Cleanup Patch

The Phase 10 cleanup patch refines and strengthens the SQLite-backed experiment infrastructure:
- **Evaluation Ingestion**: Added `ingest_rts_eval_summary.py` to ingest `eval_summary.json` files directly into the evaluations table in SQLite.
- **Best-Checkpoint Tie-Breaker**: Updated `select_best_checkpoint` to use the later checkpoint (`checkpoint_sort_index`) only as a final tie-breaker.
- **Metric Alias Normalization**: Standardized metric names to support both simple (e.g., `avg_order_cycle_time`) and mean-style (e.g., `avg_order_cycle_time_mean`) names. Stored missing energy metrics as infinity.
- **Cycle-Reference Proposal Validation**: Strengthened `propose_cycle_reference_update.py` to optionally validate evaluation completeness (success/completed) and failed replications limit before proposing a reference update.
- **Path-Independent Experiment ID**: Removed filesystem `run_root` from the `experiment_id` derivation hash to ensure it remains stable if a run is moved.
- **Worker Rollout Field Ingestion**: Prefer `netlogo_steps_completed` over legacy `ticks_completed`, and correctly ingest `warehouse_time_*` and `tick_to_second` fields.
- **True Long-Format Metrics Export**: Updated `eval_metrics_long.csv` to export a true long-format layout (columns: `eval_run_id`, `metric_name`, `metric_value`).

