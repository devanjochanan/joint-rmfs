# Quickstart Guide

This guide describes how to run and validate the RMFS simulation workspace using the designated Python environment and operator command-line interface.

## 1. Check Branch & Status
Before executing any operations, ensure you are on the `main_future` branch and that there are no unexpected edits:
```bash
git rev-parse --abbrev-ref HEAD
git status --short --branch
```

## 2. Use the Designated Python Environment
All commands should use the Python executable from the WSL virtualenv:
```bash
/home/dewan/torch-gpu/bin/python
```
Do not run python from the Windows command line directly; execute your commands inside WSL or prefix with `wsl` in Windows shells.

## 3. List Run Profiles
To see all preconfigured run profiles and their settings, use the `profiles` subcommand:
```bash
/home/dewan/torch-gpu/bin/python scripts/run/rmfs.py profiles
```

## 4. Inspect a Resolved Profile
You can view how the operator CLI will resolve default settings for a given profile:
```bash
/home/dewan/torch-gpu/bin/python scripts/run/rmfs.py profile ablation --seed 123
```

## 5. Run Validation
Before running any simulation or training scripts, verify that compilation, path routing, and configuration policies are healthy:
```bash
/home/dewan/torch-gpu/bin/python scripts/run/rmfs.py validate --fast
```

## 6. Run a Smoke Simulation
Run a short, bounded heuristic simulation to verify the simulator is fully functional:
```bash
/home/dewan/torch-gpu/bin/python scripts/run/rmfs.py smoke --seed 123
```

## 7. Run an Ablation Dry-Run
Prepare and validate the settings for an ablation study without running the full simulation:
```bash
/home/dewan/torch-gpu/bin/python scripts/run/rmfs.py ablation --scenario base --seed 123 --dry-run
```

## 8. Output Locations
Outputs are routed based on execution context:
- Common heuristic and evaluation metrics are written under `data/output/`.
- Isolated simulation worker outputs are saved to `data/runtime/tmp/operator_<profile>_<seed>/`.
- SQLite experiment ledger is written at `data/output/rmfs_experiments.sqlite`.

## 9. Cleanup Workspace
Clean up temporary runtime artifacts safely using the `cleanup` command:
```bash
/home/dewan/torch-gpu/bin/python scripts/run/rmfs.py cleanup --dry-run
```
To apply deletions, review the dry-run output and run:
```bash
/home/dewan/torch-gpu/bin/python scripts/run/rmfs.py cleanup --apply
```

## 10. Performance & Optimizations
Headless runs default to optimized modes (`detail_db` disabled, print logging suppressed, Periodic status checks). To understand these performance improvements, refer to the [Runtime Optimization Guide](file:///wsl.localhost/Ubuntu-22.04/home/dewan/Project%20Ta/Fresh%20Start%20Structure%20V1/Rika%27s%20Version/docs/operations/runtime_optimization.md).

