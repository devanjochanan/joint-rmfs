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

