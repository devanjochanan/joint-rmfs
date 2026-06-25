# Ablation Studies Guide

Ablation runs are designed for serious evaluations of alternative policy mechanics or layout configurations over extended horizons.

## Ablation Profile Overview

- **Default Horizon**: 100,000 simulation ticks (profile-defined; can be overridden via `--ticks`).
- **Data Loggers**: Detail DB (`warehouse.db` SQLite) and debug traces are disabled by default (`detail_db: false`) to avoid unneeded disk utilization.
- **Layout Behavior**: Pod locations default to being randomized by seed (`pod_location_mode: randomize_slots`), aligning with the profile-driven headless default rules.

## Recommended Workflows

### 1. Inspect Resolved Profile Settings
Always inspect the resolved parameters for a specific seed first to confirm the exact config values:
```bash
/home/dewan/torch-gpu/bin/python scripts/run/rmfs.py profile ablation --seed 123
```

### 2. Verify Ablation Setup
Always dry-run the ablation scenario to check paths and parameter routing before executing:
```bash
/home/dewan/torch-gpu/bin/python scripts/run/rmfs.py ablation --scenario base --seed 123 --dry-run
```

## Running Ablation Experiments

> [!WARNING]
> - A full ablation run (100,000 ticks) represents a long-running execution. Do not run it without explicit plan approval.
> - The headless ablation execution wrapper is **intentionally not fully implemented** in the Phase 6 CLI wrapper (`scripts/run/rmfs.py`). 
> - If you attempt to execute it without `--dry-run` (or if you pass `--yes`), the command will report that the dedicated experiment runner is pending implementation.
> - Headless execution details must be resolved during a future experiment-runner pass.
