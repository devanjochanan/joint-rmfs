# Experiment Ledger Validation

Run:

```bash
/home/dewan/torch-gpu/bin/python scripts/validation/experiment_ledger_smoke.py
/home/dewan/torch-gpu/bin/python scripts/validation/phase9_ingest_smoke.py
/home/dewan/torch-gpu/bin/python scripts/validation/eval_ingest_smoke.py
/home/dewan/torch-gpu/bin/python scripts/validation/export_summary_smoke.py
```

These smokes use synthetic fixtures under `data/runtime/phase10_*` and clean them up. They do not run simulator workers, training, evaluation, DuckDB, or benchmarks.

### Verifications Checked:
- **Evaluation Ingestion**: Ingests `eval_summary.json` and verifies correct database row insertion and metrics parsing/retrieval.
- **Path-Independent Experiment ID**: Confirms that identical training runs under different file system paths yield stable, path-independent `experiment_id` hashes.
- **Worker Rollout Aliases**: Asserts that `netlogo_steps_completed` correctly overrides legacy `ticks_completed` during ingestion.


