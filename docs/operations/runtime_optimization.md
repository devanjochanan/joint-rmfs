# Runtime Optimization Guide

During Phase 5 and Phase 5B, the RMFS workspace was optimized to reduce hot-path execution overhead, limit unnecessary disk I/O, and provide timing/profiling tools for headless simulation runs.

## 1. Detail Database Opt-in (`detail_db`)

Historically, the simulator wrote granular step-by-tick records (e.g., pod movements, task assignments) to a local SQLite database (`warehouse.db`). In large headless sweeps, this generated significant write overhead.

### Optimization:
- **Disabled by default**: Headless profile-driven runs (under `smoke`, `training`, `ablation`) default to `detail_db: false`.
- **Memory Mirror**: When disabled, the SQLite helper does not open `warehouse.db`. Instead, pod-location writes are directed to a lightweight process-local memory mirror. This ensures that inventory assignment and replenishment logic can read the latest pod locations without the cost of writing to disk.
- **How to enable**: If detailed logs are needed for debugging, use the `debug` or `gui` profiles, or explicitly enable via the CLI flag:
  ```bash
  /home/dewan/torch-gpu/bin/python scripts/run/rmfs.py <command> --detail-db
  ```
  or via environment variables:
  ```bash
  export RMFS_DETAIL_DB=1
  ```

## 2. Bounded Logging and Verbose Print Suppression

Hot-path functions (such as pod-location updates and task mappings) could clutter stdout/stderr streams, degrading execution speed when running multiple parallel workers.

### Optimization:
- Verbose helper logs are suppressed by default.
- Print calls on the hot-path are gated behind environment variables:
  - `RMFS_DEBUG=1`
  - `RMFS_DEBUG_DETAIL_DB=1`
- Status files (`worker_status.json`) are only written at the start, at the final step, and throttled periodically based on `--worker-status-cadence` instead of every tick.

## 3. Explicit Timing Instrumentation (`timing_summary.json`)

To diagnose latency bottlenecks across different phases of execution (e.g. CSV loading, NetLogo bridge communication, inference calls), Phase 5 introduced lightweight timing instrumentation.

### Optimization:
- **Disabled by default**: Gated behind `RMFS_TIMING=1` or the `--timing` CLI option on local executor runs.
- **Output**: When active, timing duration totals are written to `<runtime_root>/timing_summary.json`.
- **Tracks sections**:
  - `setup` (initial model instantiation)
  - `tick` (total tick loop time)
  - `pickle_load` & `pickle_dump` (bridge IPC payload times)
  - `sqlite_connect`, `sqlite_execute`, `sqlite_commit`
  - `csv_read` & `csv_write`
  - `pps_observation` (Pick Pod Selection observation vector construction)

## 4. Validating Optimizations

To ensure performance enhancements do not introduce bugs, compile errors, or violate observation space contracts, execute the following smoke tests:

```bash
# Verify detail DB memory mirror works correctly
/home/dewan/torch-gpu/bin/python scripts/validation/detail_db_smoke.py

# Verify runtime execution performance
/home/dewan/torch-gpu/bin/python scripts/validation/runtime_io_performance_smoke.py

# Verify PPS Gymnasium environment contracts
/home/dewan/torch-gpu/bin/python scripts/validation/pps_observation_contract_smoke.py
```

> [!NOTE]
> These validation tools are structural checks. They do not run full benchmarks or claim target speedups, but assert that optimized execution paths behave identically to the baseline.
