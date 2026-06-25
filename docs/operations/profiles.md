# Run Profiles

The RMFS CLI uses run profiles to govern default simulation settings such as tick horizons, database flags, and layout randomization.

## Profile Summary

| Profile | Use case | Default intent | Pod-location behavior |
| :--- | :--- | :--- | :--- |
| `smoke` | Bounded fast testing | Fast check of simulator loop (100 ticks) | Randomized (slot-shuffled by seed) |
| `training` | Heuristic or RL training | Train models under randomized conditions (5,000 ticks) | Randomized (slot-shuffled by seed) |
| `ablation` | Serious experiment evaluation | Evaluation of policy performance (100,000 ticks) | Randomized (slot-shuffled by seed) |
| `debug` | Simulation debugging | Small run with SQLite/tracing enabled (100 ticks) | Randomized (slot-shuffled by seed) |
| `gui` | Manual / NetLogo GUI execution | Legacy compatibility, manual control | Fixed (default CSV locations) |

## Pod-Location Randomization Rules

Profile-driven smoke/training/ablation/debug runs default to randomized pod locations using the run seed. GUI/manual defaults to fixed pod locations unless explicitly overridden.

### Randomization Invariant
Pod-location randomization only shuffles starting pod IDs across existing storage slots. It does not change:
- Pod contents
- SKU allocations
- Item quantities
- Orders
- Storage-slot geometry

The same seed resolves to the same randomized layout, ensuring experiment reproducibility.

## Inspecting Live Defaults
Do not rely on static documentation for changing values. You can query the profile resolver for live defaults at any time:

* **Show all profile defaults**:
  ```bash
  /home/dewan/torch-gpu/bin/python scripts/run/rmfs.py profiles
  ```
* **Show resolved values for a specific profile and seed**:
  ```bash
  /home/dewan/torch-gpu/bin/python scripts/run/rmfs.py profile ablation --seed 123
  ```
