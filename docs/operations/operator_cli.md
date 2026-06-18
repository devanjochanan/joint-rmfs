# Operator CLI Reference

The operator CLI (`scripts/run/rmfs.py`) provides subcommands to inspect, validate, run, and clean up the simulation workspace.

## Command Overview

---

### `profiles`
- **Purpose**: Lists all available run profiles and prints their default properties.
- **Runs Simulation**: No.
- **Resource Weight**: Extremely lightweight.
- **Safe Example**:
  ```bash
  /home/dewan/torch-gpu/bin/python scripts/run/rmfs.py profiles
  ```

---

### `profile`
- **Purpose**: Resolves the exact parameters of a specific profile, incorporating CLI flag overrides (e.g. `--seed`, `--ticks`).
- **Runs Simulation**: No.
- **Resource Weight**: Extremely lightweight.
- **Safe Example**:
  ```bash
  /home/dewan/torch-gpu/bin/python scripts/run/rmfs.py profile ablation --seed 123
  ```

---

### `validate`
- **Purpose**: Executes sanity tests to verify imports, run context paths, layout randomization logic, scenario loaders, and database helpers.
- **Runs Simulation**: Yes (runs minor bounded setups of 1-10 steps for smoke-tests).
- **Resource Weight**: Moderate (takes 5-10 seconds).
- **Safe Example**:
  ```bash
  /home/dewan/torch-gpu/bin/python scripts/run/rmfs.py validate --fast
  ```

---

### `smoke`
- **Purpose**: Runs a short heuristic simulation run using the `smoke` profile defaults.
- **Runs Simulation**: Yes (100 ticks).
- **Resource Weight**: Moderate (takes ~5-15 seconds).
- **Safe Example**:
  ```bash
  /home/dewan/torch-gpu/bin/python scripts/run/rmfs.py smoke --seed 123
  ```

---

### `ablation`
- **Purpose**: Prepares a long-horizon ablation experiment scenario.
- **Runs Simulation**: Yes, unless `--dry-run` or `--no-yes` is passed.
- **Resource Weight**: Heavy / High (runs 100,000 ticks).
- **Safety Rule**: **Always dry-run first** to review resolved settings. Headless execution is not yet implemented in Phase 6, so non-dry-runs will exit with an instruction.
- **Safe Example**:
  ```bash
  /home/dewan/torch-gpu/bin/python scripts/run/rmfs.py ablation --scenario base --seed 123 --dry-run
  ```

---

### `training`
- **Purpose**: Prepares a reinforcement learning training scenario.
- **Runs Simulation**: Yes (but requires Gymnasium/Stable-Baselines).
- **Resource Weight**: Heavy / High (5,000 ticks per episode, many episodes).
- **Safety Rule**: **Always dry-run first**. The training runner is not automatically launched by the operator wrapper unless training dependencies are configured.
- **Safe Example**:
  ```bash
  /home/dewan/torch-gpu/bin/python scripts/run/rmfs.py training --target pps --seed 123 --dry-run
  ```

---

### `debug`
- **Purpose**: Inspects parameters or dry-runs under the `debug` profile (detail DB and debug traces enabled).
- **Runs Simulation**: Yes (dry-run by default).
- **Resource Weight**: Moderate.
- **Safety Rule**: **Preserves workspace artifacts**. Detail SQLite DB (`warehouse.db`) and debug files will remain in the workspace.
- **Safe Example**:
  ```bash
  /home/dewan/torch-gpu/bin/python scripts/run/rmfs.py debug --seed 123 --dry-run
  ```

---

### `cleanup`
- **Purpose**: Scans and deletes temporary runtime files under `data/runtime/tmp`, `data/runtime/debug`, and `data/runtime/latest`.
- **Runs Simulation**: No.
- **Resource Weight**: Lightweight.
- **Safety Rule**: **Dry-run by default**. You must explicitly add `--apply` to execute deletions. It will never delete inputs, reference data, or saved models.
- **Safe Example**:
  ```bash
  /home/dewan/torch-gpu/bin/python scripts/run/rmfs.py cleanup --dry-run
  ```

---

## Safety Guidelines for Operators

1. **Ablation Dry-run**: Always inspect ablation scenarios before running. A full ablation run takes significant time and compute.
2. **Training Dry-run**: RL training runs are highly resource intensive. Inspect the delegated command output before launching.
3. **Cleanup Apply**: Never run `cleanup --apply` without inspecting the dry-run output first.
4. **Full Raw Replay**: Opt-in only. Headless execution defaults to synthetic/deterministic order generators. Replaying full raw orders from CSV is only enabled when `--full-raw-order-replay` is explicitly set.
