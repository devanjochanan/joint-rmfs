# RMFS Simulation / RTS-RL Research Repo

Welcome to the Robotic Mobile Fulfillment System (RMFS) simulation and Return-to-Storage Reinforcement Learning (RTS-RL) research repository.

## Current status
The repository is stabilized on the `main_future` branch. Phase 6 introduced a human-operable command-line interface (CLI) to run the simulation, run fast validations, dry-run training, dry-run ablation scenarios, and clean temporary workspace artifacts.

## Quickstart
Ensure you are using the designated Python environment in WSL (`/home/dewan/torch-gpu/bin/python`).
To check that the repository runs, run a fast validation check:
```bash
/home/dewan/torch-gpu/bin/python scripts/run/rmfs.py validate --fast
```
For more information, please see the [Quickstart Guide](file:///wsl.localhost/Ubuntu-22.04/home/dewan/Project%20Ta/Fresh%20Start%20Structure%20V1/Rika%27s%20Version/docs/operations/quickstart.md).

## Operator CLI
The repository exposes a central command-line interface wrapper in `scripts/run/rmfs.py`. All commands are run with the virtualenv python:
```bash
/home/dewan/torch-gpu/bin/python scripts/run/rmfs.py <command> [args]
```
For full options and command descriptions, refer to the [Operator CLI Guide](file:///wsl.localhost/Ubuntu-22.04/home/dewan/Project%20Ta/Fresh%20Start%20Structure%20V1/Rika%27s%20Version/docs/operations/operator_cli.md).

## Common commands
* **List available profiles**:
  ```bash
  /home/dewan/torch-gpu/bin/python scripts/run/rmfs.py profiles
  ```
* **Inspect resolved profile settings**:
  ```bash
  /home/dewan/torch-gpu/bin/python scripts/run/rmfs.py profile ablation --seed 123
  ```
* **Run validation checks**:
  ```bash
  /home/dewan/torch-gpu/bin/python scripts/run/rmfs.py validate --fast
  ```
* **Run a short bounded heuristic simulation**:
  ```bash
  /home/dewan/torch-gpu/bin/python scripts/run/rmfs.py smoke --seed 123
  ```
* **Dry-run a long-horizon ablation experiment**:
  ```bash
  /home/dewan/torch-gpu/bin/python scripts/run/rmfs.py ablation --scenario base --seed 123 --dry-run
  ```
* **Dry-run PPS reinforcement learning training**:
  ```bash
  /home/dewan/torch-gpu/bin/python scripts/run/rmfs.py training --target pps --seed 123 --dry-run
  ```
* **Inspect files that would be cleaned up**:
  ```bash
  /home/dewan/torch-gpu/bin/python scripts/run/rmfs.py cleanup --dry-run
  ```

## Run profiles
The CLI uses profiles to define default parameters (e.g. simulation horizons, order count, debug levels, database flags):
* `smoke`: Fast bounded test (100 ticks, no detail DB, randomized pod locations).
* `training`: Bounded run configured for reinforcement learning training.
* `ablation`: Bounded run configured for serious evaluation and ablation studies (100,000 ticks).
* `debug`: Bounded run with SQLite detail databases and debug traces enabled.
* `gui`: Manual compatibility mode with fixed pod locations and legacy fallback.

For detailed run-profile descriptions, see the [Profiles Guide](file:///wsl.localhost/Ubuntu-22.04/home/dewan/Project%20Ta/Fresh%20Start%20Structure%20V1/Rika%27s%20Version/docs/operations/profiles.md).

## Data, runtime, output, and model paths
* **Input directories**:
  * `data/input/base/`: Master inputs (`items.csv`, `pods.csv`, `generated_pod.csv`, `raw_order.csv`).
  * `data/input/dictionaries/`: Mapping configuration dictionaries.
  * `data/input/scenarios/`: Evaluated simulation scenarios.
* **Runtime directories**:
  * `data/runtime/latest/`: Default and manual context workspace outputs.
  * `data/runtime/tmp/`: Headless worker execution context outputs.
  * `data/runtime/debug/`: Detailed debug traces.
* **Output / Model directories**:
  * `data/output/`: Heuristic and evaluation metrics, SQLite experiment ledger.
  * `data/models/pps/`: Stable-Baselines model files (`pps_rl_best.zip`).
  * `data/models/rts/`: Return-to-storage RL checkpoints.

For the path routing contract, see [Runtime Paths Documentation](file:///wsl.localhost/Ubuntu-22.04/home/dewan/Project%20Ta/Fresh%20Start%20Structure%20V1/Rika%27s%20Version/docs/architecture/runtime_paths.md).

## Validation
Validation checks are defined to prevent regressions on core path routing, order generation policies, layout randomization contracts, compilation errors, and dependencies.
For details, see the [Validation Guide](file:///wsl.localhost/Ubuntu-22.04/home/dewan/Project%20Ta/Fresh%20Start%20Structure%20V1/Rika%27s%20Version/docs/operations/validation.md).

## Safe ablation/training workflow
* **Ablations**: Heavy experiments must be dry-run first. The dedicated ablation runner is unimplemented; actual long-horizon evaluation requires dry-run validation. Learn more in the [Ablation Guide](file:///wsl.localhost/Ubuntu-22.04/home/dewan/Project%20Ta/Fresh%20Start%20Structure%20V1/Rika%27s%20Version/docs/operations/ablation.md).
* **Training**: PPS RL training requires specific dependencies (Gymnasium/Stable-Baselines). Normal simulation and heuristic runs do not load these packages. Learn more in the [Training Guide](file:///wsl.localhost/Ubuntu-22.04/home/dewan/Project%20Ta/Fresh%20Start%20Structure%20V1/Rika%27s%20Version/docs/operations/training.md).

## Legacy / pending components
* **Charging & Energy**: Battery charging layout and scheduling mechanisms (Salsa's area) are pending/inactive. `run_baseline.py` is parked as a legacy charging runner and is disabled by default.
* **Raw Order Replay**: Headless profile-driven runs generate mock orders; replaying raw orders is opt-in only via `--full-raw-order-replay` and is not default.

## Where to read next
* [Quickstart Guide](file:///wsl.localhost/Ubuntu-22.04/home/dewan/Project%20Ta/Fresh%20Start%20Structure%20V1/Rika%27s%20Version/docs/operations/quickstart.md)
* [Profiles Guide](file:///wsl.localhost/Ubuntu-22.04/home/dewan/Project%20Ta/Fresh%20Start%20Structure%20V1/Rika%27s%20Version/docs/operations/profiles.md)
* [Operator CLI Guide](file:///wsl.localhost/Ubuntu-22.04/home/dewan/Project%20Ta/Fresh%20Start%20Structure%20V1/Rika%27s%20Version/docs/operations/operator_cli.md)
* [Runtime Optimization Guide](file:///wsl.localhost/Ubuntu-22.04/home/dewan/Project%20Ta/Fresh%20Start%20Structure%20V1/Rika%27s%20Version/docs/operations/runtime_optimization.md)
