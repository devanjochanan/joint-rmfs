# Rika RMFS Current State

## Purpose
This file records the repository state and refactor constraints at the time it was last updated.

## Source of Truth
- Active behavior source remains `simulation.nlogo`, `engine/**`, and active `model/**`.
- Root `netlogo.py` is a compatibility shim that delegates to `src/rmfs/app/netlogo_api.py`; it remains importable as `import netlogo` for NetLogo and local scripts.
- `src/rmfs/**` contains the recovered and functional RTS-RL training infrastructure (in `src/rmfs/rl/rts/` and `src/rmfs/orchestration/`).
- The recovery phases (Phases 1-15), targeted active job-queue regret-k scheduler patch, and RTS-RL semantic recovery patch have been implemented locally and verified using static checks and smoke tests. No long simulation, training campaign, benchmark, or output-equivalence run was performed.

## Recorded Work
- **Phase 1**: Repository inventory docs were created.
- **Phase 1.5**: Runtime/generated/local artifacts were removed from tracking and ignored.
- **Phase 2**: Scaffold and ownership docs were created.
- **Phase 3**: Documentation-only `data/` planning skeleton was created, and confirmed-unused legacy/sandbox Python files were quarantined in `src/rmfs/legacy/`.
- **Phase 4**: Active package boundary created by moving the NetLogo Python bridge implementation from root `netlogo.py` into `src/rmfs/app/netlogo_api.py`. Root `netlogo.py` was replaced with a compatibility shim.
- **Phase 4.1**: Post-bridge cleanup — deleted noncanonical generated-pod CSV variants and the quarantined `robot_new.py`. Corrected stale documentation wording.
- **Phase 5**: Static checks and audits were conducted for the bridge split and repository scaffold.
- **Phase 6**: Ported the RTS-RL action space, state/features, stock encoder, masked actor-critic model, inference helpers, reward/cycle-reference helpers, and a validation smoke.
- **Phase 7**: Added a process-local RTS registry, worker-local JSONL rollout writer, outcome tracker, storage resolver, random-valid evaluation policy, rollout summary, and local-executor RTS config propagation.
- **Phase 8**: Provided synthetic PPO math/checkpoint validation, dataset loading, feature reconstruction, PPO update math validation, checkpoint layout helpers, latest/history tracking, and synthetic cycle-reference helpers.
- **Phase 9**: Added training-facing timebase, policy checkpoint loader, `rts_rl_explicit` actor wrapper, trainable rollout metadata, active-checkpoint dataset filtering, optional TQDM/TensorBoard wrappers, and a controller dry-run spine.
- **Phase 10**: Added SQLite-backed experiment ledger at `data/output/rmfs_experiments.sqlite`. Captured evaluation summaries, deterministic scenarios, seed packs, checkpoint metadata, and proposed cycle reference updates.
- **Phase 11**: Created a precise RTS state-feature gap map identifying implementation gaps across 13 feature families.
- **Phase 12**: Implemented selected dynamic state-features grounded in current simulation objects (replenishment station context, SKU turnover rank, robot congestion, neighborhood counts, zone distances). Next retrieval context, committed next task, and arrival-rate cycle context remain deferred/defaulted.
- **Phase 13**: Enforced cycle reference and alpha update gating rules to verify update completeness and validity.
- **Phase 14**: Completed timebase naming cleanup, exposing `netlogo_steps_requested` as an alias of `ticks`.
- **Phase 15**: Audited regret-k task allocation scheduling and classified full mature scheduling as deferred.
- **Targeted Regret-k Patch**: Implemented active job-queue regret-k allocation with `robot_task_allocator=regret_k`, `regret_k=2`, and `task_allocator_scope=active_job_queue`. The previous nearest assignment is retained as `legacy_nearest`. Mature committed-next reservations/lookahead remain deferred.
- **Reward Cold-Start Cleanup**: RTS-RL v1 training no longer requires a manual reference run or mandatory `cycle_reference.json`. Batch 1 can derive `reward_time_scale` from completed `paper_cycle_duration` rows, while later batches use checkpoint-stored normalizer metadata. Alpha/reference update remains deferred.
- **RTS-RL Semantic Recovery Patch**: RTS-RL zones are now registry-backed (`rts_z_r##_c##` for production geometry), directed graph distances are used for distance features/resolution, and PPO training targets completed paper-cycle outcomes rather than return-to-storage duration. Return completion remains logged as pending diagnostic telemetry.
- **RTS Training Terminal-Output Cleanup Patch**: Suppressed worker subprocess stdout/stderr streams to `subprocess.DEVNULL` by default to prevent TQDM display corruption. Added `--debug-worker-logs` opt-in flag to persist stdout/stderr logs inside each worker's runtime directory, avoiding repository storage bloat by default. Worker failures are reported in detail (with stderr tail if debug is enabled), and noisy hot-path pod-location success prints are silenced.


## Key Design Constraints & Decisions
- **Strict Opt-In Policy**: The default policy remains `CurrentRTSPolicy` (heuristic/nearest). RTS-RL policies (`rts_rl_explicit`, `random_valid`) are strictly opt-in via feature flags.
- **SQLite Ledger, No DuckDB**: The experiment ledger is SQLite-based (`data/output/rmfs_experiments.sqlite`). Workers collect rollouts only and never write to SQLite directly; SQLite writes are controller-side or post-processing only. DuckDB is not used.
- **On-Policy Only**: PPO training only accepts rows matching the active `policy_checkpoint_id` generated by `rts_rl_explicit`. Offline or off-policy training is not supported.
- **Alpha & Reward Gating**: Cycle/alpha reference updates only occur under complete and valid runs at scheduled times. Alpha is not rederived or continuously updated during simulation.
- **Cold-Start Reward Normalization**: `cycle_reference.json` is optional legacy compatibility. The active v1 path records `reward_mode=cold_start_paper_cycle_duration`, `reward_horizon=paper_cycle_duration`, `reward_reference_required=false`, and `alpha_enabled=false`.
- **RTS-RL Replenish Route Caveat**: The current host does not implement mature `replenish_store(z)` pre-return route equivalence. Replenishment next-task arrivals are recorded/censored for paper-cycle training eligibility, but no mature replenish-route or performance claim is made.
- **Best Checkpoint Deferred**: The recovery training controller preserves every batch checkpoint plus `latest.json` and `checkpoint_history.jsonl`; no copied best-checkpoint folders or automatic evaluation-based best scoring are part of this recovery work.
- **Scheduler Metadata Captured**: Training/controller/worker artifacts and Phase 9 SQLite ingestion preserve active task-allocation metadata so scheduler context is auditable from dry-run outputs.
- **No Overclaims**: No long training campaigns, no paper-ready results, and no performance improvements are claimed. The infrastructure has been verified via local dry-runs and smoke tests only.
