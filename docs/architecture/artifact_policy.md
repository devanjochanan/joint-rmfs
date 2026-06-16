# Runtime Artifact Policy

Phase 4 defines cleanup ownership without changing simulator behavior.

## Defaults

- Runtime scratch state belongs under `data/runtime/tmp/`, `data/runtime/debug/`, or `data/runtime/latest/`.
- Debug traces are disabled by default.
- Successful worker folders may be marked with `.rmfs_cleanup_eligible`.
- Failed workers are preserved by policy so their logs and summaries can be inspected.
- Compact summaries belong under `data/output/summaries/`.

## Cleanup Tool

Use:

```bash
/home/dewan/torch-gpu/bin/python scripts/runtime/cleanup_runtime_artifacts.py --dry-run
```

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

`--apply` is required to delete anything. Phase 4 validation runs dry-run only.

## Pending

Detail SQLite/CSV hot-path disabling remains Phase 5 work. The Phase 4
`--detail-db` flag records intent in local-executor manifests but does not
replace the DB implementation.
