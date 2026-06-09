# RMFS Simulator Golden Trace Baseline Harness

This harness captures a reproducible trace of the simulator's output for verification and regression testing.

## Overview

The golden trace baseline harness captures setup output, per-tick return values, and state hashes of root database/CSV files, verifying simulator behavior in a clean sandbox.

### What it Captures
- **Git metadata:** Commit hash and branch name.
- **Python context:** Python executable path and version.
- **Payload digests:** Stable SHA256 hashes of `setup()` and `tick()` return payloads, including positions of objects and order assignments.
- **Telemetry scalars:** `total_energy`, `job_queue_len`, `stop_and_go`, and `total_turning`.
- **Pre/Post file states:** Exist status, sizes, and SHA256 digests for key database, state, and CSV tracking files:
  - `assign_order.csv`
  - `pod_info.csv`
  - `netlogo.state`
  - `warehouse.db`
  - `skus_data.csv`
  - `sorted_skus_data.csv`
  - `generated_backlog.csv`
  - `generated_database_order.csv`
  - `generated_order.csv`
  - `generated_pod.csv`

### What it Does Not Prove
- **Behavior equivalence across settings:** Passing a 3-tick baseline only guarantees that those first 3 ticks did not deviate under the *same* configuration. It does not prove full simulation parity or cover edge cases that trigger in later ticks.
- **No performance claim:** Digest comparisons confirm code correctness, not optimization or throughput improvements.

### Why it Exists
To establish a regression safeguard. By running the trace before and after architectural moves, developers can ensure no existing simulator behaviors are accidentally changed or broken.

---

## Workspace Protection

The script runs the simulation within a **temporary workspace copy** of the repository, excluding the `.git` database and local cache directories. This ensures that:
1. No active checkout CSV files, databases (`warehouse.db`), or state pickles (`netlogo.state`) are modified.
2. The working directory remains pristine.
3. Outputs are written only to the user-specified output folder.

---

## Usage

### 3-Tick Smoke Run (Authorized)
Run the script for a lightweight check:
```bash
/home/dewan/torch-gpu/bin/python scripts/trace/golden_trace.py \
  --ticks 3 \
  --output data/runtime/golden_trace/manual_smoke
```

### Output Files Produced
Inside `data/runtime/golden_trace/manual_smoke/`:
- `manifest.json`: Running metadata and system info.
- `trace.jsonl`: Line-by-line digest and metric records per tick.
- `summary.json`: High-level execution summary, setup digests, and pre/post file states.

> [!WARNING]
> Generated traces under `data/runtime/golden_trace/` represent local runtime telemetry and should **not** be committed to the repository.

> [!IMPORTANT]
> Large traces (e.g. 1000 ticks or full sweeps) require **explicit user authorization** before run. Do not execute long-running tasks locally.
