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

The controller writes `manifest.json` and `controller_summary.json` under the output root. By default, `worker_summary.json` is **aggregate-only** (containing setup information, boundary/final tick digests, and final scalar metrics) to prevent file bloat on long runs.

### Debug Trace Mode
To record detailed per-tick trace lines, enable debug trace mode:
```bash
/home/dewan/torch-gpu/bin/python scripts/run/local_executor_smoke.py \
  --runs 2 \
  --ticks 5 \
  --max-workers 2 \
  --output-root data/runtime/local_executor_smoke/debug_manual \
  --debug-trace \
  --trace-cadence 2 \
  --trace-first-n 1
```

When `--debug-trace` is enabled, a `debug_trace.jsonl` file is created inside each worker's runtime root. It records:
* The first `trace_first_n` ticks (if > 0)
* Every `trace_cadence` ticks (if > 0)
* The final tick of the simulation

Each worker runtime directory contains:
- `netlogo.state`
- `warehouse.db`
- `assign_order.csv`
- `pod_info.csv`
- `skus_data.csv`
- `sorted_skus_data.csv`
- `worker_summary.json`
- `debug_trace.jsonl` (only if `--debug-trace` is explicitly set)

Generated outputs under `data/runtime/` are ignored local artifacts and should not be committed.

This smoke does not prove behavior equivalence, benchmark equivalence, paper fidelity, performance improvement, throughput improvement, or congestion improvement. It is not BehaviorSpace, PPO, DoE, training, checkpointing, TensorBoard, or DuckDB.

For this slice, `generated_order.csv`, `generated_pod.csv`, and `pods.csv` remain root-read-only/deferred. Phase 4 should handle fuller manifests and data contracts.
