# Local Executor Smoke

The local executor smoke launches multiple short simulator workers with separate `RunContext.isolated(...)` runtime directories. It checks that each worker writes its own state, SQLite DB, and runtime CSV files without mutating root runtime-sensitive files.

Workers run in subprocesses instead of threads because `netlogo_api.py` stores the active run context in module-level state. Fresh worker processes avoid shared interpreter races around that context.

Run the default Phase 3B smoke with:

```bash
/home/dewan/torch-gpu/bin/python scripts/run/local_executor_smoke.py \
  --runs 4 \
  --ticks 3 \
  --max-workers 2 \
  --output-root data/runtime/local_executor_smoke/manual
```

The controller writes `manifest.json` and `controller_summary.json` under the output root. Each worker writes `run_spec.json`, `worker_summary.json`, and worker stdout/stderr logs under its own run directory.

Expected per-worker runtime files are:

- `netlogo.state`
- `warehouse.db`
- `assign_order.csv`
- `pod_info.csv`
- `skus_data.csv`
- `sorted_skus_data.csv`
- `worker_summary.json`

Generated outputs under `data/runtime/` are ignored local artifacts and should not be committed.

This smoke does not prove behavior equivalence, benchmark equivalence, paper fidelity, performance improvement, throughput improvement, or congestion improvement. It is not BehaviorSpace, PPO, DoE, training, checkpointing, TensorBoard, or DuckDB.

For this slice, `generated_order.csv`, `generated_pod.csv`, and `pods.csv` remain root-read-only/deferred. Phase 4 should handle fuller manifests and data contracts.
