# Debugging and Workspace Cleanup

This guide covers how to inspect detailed simulation traces, analyze execution performance metrics, and manage temporary workspace files safely.

## Debug Profile and Diagnostic Traces

When troubleshooting simulation anomalies, run under the `debug` profile. The debug profile sets:
- `detail_db: true`: Generates detailed SQLite database tables in `warehouse.db`.
- `debug_trace: true`: Writes stdout/stderr log buffers per worker.
- `keep_runtime_artifacts: true`: Instructs the execution runner to retain all workspace folders even after successful runs.

### Dry-run Debug profile:
```bash
/home/dewan/torch-gpu/bin/python scripts/run/rmfs.py debug --seed 123 --dry-run
```

## Core Diagnostic Artifacts

When debugging runs are executed, key diagnostic outputs are generated under `<runtime_root>/` (e.g. `data/runtime/tmp/<worker>/`):
1. **`warehouse.db`**: A detail SQLite database containing tables for pod movements, inventory changes, and picker tasks. It is disabled by default in headless runs to optimize speed.
2. **`worker_summary.json`**: A JSON summary containing exit statuses, seed parameters, scenario IDs, and total tick durations.
3. **`timing_summary.json`**: A JSON trace of task routing durations and API call latency, written if timing logging is enabled.

## Workspace Cleanup

Isolated runs produce subdirectory folders under `data/runtime/tmp/`. Over time, these folders accumulate.

### Failed Worker Preservation
To assist post-mortem debugging, the cleanup tool respects a safety preservation policy:
- **Eligible for deletion**: Folders of *successful* headless workers that are marked with the `.rmfs_cleanup_eligible` file.
- **Preserved by default**: Folders of *failed* workers (which lack the eligibility file). Their logs, traceback messages, and summaries are preserved to help you diagnose faults.

### Safe Cleanup Workflow

1. **Dry-Run first**: List all folders that are candidates for deletion without removing anything:
   ```bash
   /home/dewan/torch-gpu/bin/python scripts/run/rmfs.py cleanup --dry-run
   ```
2. **Review output**: Inspect the list of eligible folders.
3. **Apply cleanup**: Delete only after confirming the list:
   ```bash
   /home/dewan/torch-gpu/bin/python scripts/run/rmfs.py cleanup --apply
   ```
   > [!IMPORTANT]
   > Do not run `cleanup --apply` unless you have reviewed the dry-run output and are certain no critical debugging folders will be lost.
