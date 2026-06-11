# Experiment Ledger Validation

Run:

```bash
/home/dewan/torch-gpu/bin/python scripts/validation/experiment_ledger_smoke.py
/home/dewan/torch-gpu/bin/python scripts/validation/phase9_ingest_smoke.py
/home/dewan/torch-gpu/bin/python scripts/validation/export_summary_smoke.py
```

These smokes use synthetic fixtures under `data/runtime/phase10_*` and clean them up. They do not run simulator workers, training, evaluation, DuckDB, or benchmarks.

