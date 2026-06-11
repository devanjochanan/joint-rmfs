# RMFS Experiment Ledger

Phase 10 uses SQLite for the experiment/research ledger.

The default path is:

```text
data/output/rmfs_experiments.sqlite
```

This ledger is separate from simulator/runtime databases:

```text
warehouse.db = simulator/runtime internal DB
rmfs_experiments.sqlite = experiment/research ledger
```

Workers must never write the experiment ledger. Controller-side or post-processing scripts initialize, ingest, and export ledger data using Python stdlib `sqlite3`.

DuckDB is not used, not required, and not a dependency.

Primary tables:

- `experiments`
- `training_batches`
- `checkpoints`
- `worker_rollouts`
- `evaluations`
- `cycle_proposals`

All JSON payloads are stored as SQLite `TEXT` with deterministic `json.dumps` where practical.

---

## Phase 10 Ingestion & ID Updates
- **Evaluation Ingestion**: Evaluation summaries can be ingested into the `evaluations` table using `ingest_rts_eval_summary.py`.
- **Path-Independent Experiment ID**: `experiment_id` derivation does not depend on the filesystem `run_root`, ensuring stable IDs when directories are moved.
- **Worker Rollout Field Ingestion**: Ingestion prefers `netlogo_steps_completed` over legacy `ticks_completed`, and captures `warehouse_time_*` and `tick_to_second` fields.


