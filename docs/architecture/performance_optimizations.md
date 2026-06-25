# Runtime Performance Optimizations

Phase 5 adds performance infrastructure and safe hot-path reductions without
changing simulation semantics.

## Timing

Timing is disabled by default. Enable it with:

```bash
RMFS_TIMING=1
```

or local executor:

```bash
/home/dewan/torch-gpu/bin/python -m src.rmfs.orchestration.local_executor controller --timing ...
```

When enabled for local-executor workers, timing is written to:

```text
<runtime_root>/timing_summary.json
```

Current sections include `setup`, `tick`, `pickle_load`, `pickle_dump`,
`sqlite_connect`, `sqlite_execute`, `sqlite_commit`, `worker_status_write`,
`csv_read`, `csv_write`, and `pps_observation` where those paths are exercised.

## Detail DB

Detail SQLite tables are disabled by default for headless local-executor workers
and fast PPS/backend paths. Enable them explicitly with `--detail-db` or
`RMFS_DETAIL_DB=1`.

Disabled mode avoids creating `warehouse.db` for detail helpers. Pod-location
upserts still update a process-local memory mirror because inventory reads that
latest location during pod assignment and replenishment flow.

## Logging And Status

Detail DB helper print calls are gated behind `RMFS_DEBUG=1` or
`RMFS_DEBUG_DETAIL_DB=1`. Local executor writes `worker_status.json` at start,
final status, and every `--worker-status-cadence` ticks by default.

## CSV And SQLite Boundaries

The active inventory flow still treats `assign_order.csv` as live state inside a
tick/operation. Phase 5 does not defer those writes because that could hide
updates from immediate same-operation reads.

SQLite table names, schemas, timestamps, and enabled-mode behavior are kept
compatible. Broad batching and no-op detail DB expansion should be handled with
separate behavior checks.

## Validation

Lightweight checks:

```bash
/home/dewan/torch-gpu/bin/python scripts/validation/detail_db_smoke.py
/home/dewan/torch-gpu/bin/python scripts/validation/runtime_io_performance_smoke.py
/home/dewan/torch-gpu/bin/python scripts/validation/pps_observation_contract_smoke.py
```

These are smoke/contract checks only. They do not claim benchmark speedup or
full behavior equivalence.

## Deferred

- Persistent workers.
- NetLogo pickle bridge redesign.
- Broad CSV in-memory mirrors.
- SQLite batching where read-after-write dependencies need more proof.
- PPS observation/action extraction.
