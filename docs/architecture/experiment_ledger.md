# RMFS Experiment Ledger

Phase 10 implements a SQLite-backed experiment/research ledger for tracking RL runs.

The default path is:

```text
data/output/rmfs_experiments.sqlite
```

This ledger is separate from the simulator/runtime database:

```text
warehouse.db = simulator/runtime internal DB
rmfs_experiments.sqlite = experiment/research ledger
```

Workers must never write the experiment ledger. Controller-side or post-processing scripts initialize, ingest, and export ledger data using the Python standard library `sqlite3`.

DuckDB is not used, not required, and not a dependency.

Primary tables:

- `experiments`: High-level experiment definitions, configurations, and metadata.
- `training_batches`: Records of PPO training batches, metrics, and parameters.
- `checkpoints`: Tracked model checkpoints, paths, and performance statistics.
- `worker_rollouts`: Rollout metrics, steps completed, and worker duration.
- `evaluations`: Evaluation summary results and run metrics.
- `cycle_proposals`: Gated cycle/alpha reference updates proposed from complete, valid runs.

All JSON payloads are stored as SQLite `TEXT` with deterministic `json.dumps` where practical.

---

## Phase 10 Ingestion & ID Updates
- **Evaluation Ingestion**: Evaluation summaries can be ingested into the `evaluations` table using `ingest_rts_eval_summary.py`.
- **Path-Independent Experiment ID**: `experiment_id` derivation does not depend on the filesystem `run_root`, ensuring stable IDs when directories are moved.
- **Worker Rollout Field Ingestion**: Ingestion prefers `netlogo_steps_completed` over legacy `ticks_completed`, and captures `warehouse_time_*` and `tick_to_second` fields.
- **Verification**: Schema initialization, ingestion, and CSV exports have been fully implemented and verified locally via smoke tests.
