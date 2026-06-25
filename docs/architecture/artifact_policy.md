# Runtime Artifact Policy

Phase 4 defines cleanup ownership without changing simulator behavior. Phase 5
adds explicit timing and detail-DB runtime policy for headless workers.

## Defaults

- Runtime scratch state belongs under `data/runtime/tmp/`, `data/runtime/debug/`, or `data/runtime/latest/`.
- Debug traces are disabled by default.
- Successful worker folders may be marked with `.rmfs_cleanup_eligible`. Depending on profile and operator flags, successful headless workers may be cleanup-eligible.
- Failed workers are preserved by policy so their logs and summaries can be inspected.
- The `debug` profile preserves all runtime artifacts (detail DB, traces, worker logs) for inspection.
- Compact summaries belong under `data/output/summaries/`.
- Detail SQLite tables are disabled by default in local-executor headless workers
  and fast PPS/backend paths unless `--detail-db` or `RMFS_DETAIL_DB=1` enables
  them.
- Worker status files are throttled by `--worker-status-cadence`; start and
  final status are always written.

## Cleanup Tool

The operator cleanup CLI command delegates execution to:

```bash
/home/dewan/torch-gpu/bin/python scripts/runtime/cleanup_runtime_artifacts.py --dry-run
```

The cleanup tool always defaults to dry-run mode. `--apply` is required to perform actual deletions.

The cleanup tool targets only:

- `data/runtime/tmp`
- `data/runtime/debug`
- `data/runtime/latest`

It never targets:

- `data/input`
- `data/models`
- `data/output`
- `data/reference`
- benchmark outputs
- training checkpoints or model binaries

Phase 4 and Phase 5 validation runs use dry-run only.

## Detail DB

The detail DB tables are inspection artifacts, not the canonical simulator
state. When disabled, the detail table helpers avoid opening `warehouse.db`.
Pod-location writes still update a process-local memory mirror so inventory code
that reads the latest pod location can continue to do so without creating the
detail DB.

Enable detail DB writes explicitly with:

```bash
RMFS_DETAIL_DB=1
```

or, for local executor:

```bash
/home/dewan/torch-gpu/bin/python -m src.rmfs.orchestration.local_executor controller --detail-db ...
```

## Pending

Full CSV buffering and broader SQLite batching remain future work. Current
`assign_order.csv` paths still read immediately after writes in active
operations, so deferred flushing is intentionally not enabled by default.
