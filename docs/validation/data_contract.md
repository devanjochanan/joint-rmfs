# Data Contract Validation

Run a manifest and snapshot smoke with:

```bash
/home/dewan/torch-gpu/bin/python scripts/run/local_executor_smoke.py \
  --runs 2 \
  --ticks 3 \
  --max-workers 2 \
  --snapshot-inputs \
  --output-root data/runtime/local_executor_smoke/phase4_validation
```

Expected contract files:

- `run_manifest.json`
- `controller_summary.json`
- `input_snapshot/input_manifest.json`
- copied input CSVs under `input_snapshot/`
- aggregate `worker_summary.json` files under worker runtime roots

Generated outputs under `data/runtime/` are ignored local artifacts and should not be committed.

This validation records reproducibility metadata and input hashes. It does not prove paper fidelity, behavior equivalence, benchmark equivalence, or performance improvement. It does not run training, DoE, BehaviorSpace, checkpointing, TensorBoard, or DuckDB.

Debug trace remains opt-in only with `--debug-trace`; worker summaries remain aggregate-only by default.
