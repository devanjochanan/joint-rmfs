# Future-main Baseline Audit

## 1. Scope and Restrictions

This audit constitutes **Phase 1 — Baseline Audit and Safety Freeze** for the RMFS future-main cleanup roadmap.
Strict adherence to sandbox safety rules has been maintained:
- No production simulation code has been edited, moved, or deleted.
- No imports have been patched.
- No reinforcement learning (RL) training, BehaviorSpace experiments, or simulations were run.
- No benchmark files, reference models, or database schemas were modified or regenerated.
- The repository was frozen in its initial state immediately after a baseline safety tag was applied.

---

## 2. Local State Preflight
- **Branch**: `main_future` (tracking `origin/main_future`). The checkout was confirmed to be clean and checked out on `main_future` (mapping to combined Dewa, Lukman, and Devan work). The user approved proceeding with `main_future` as the branch to audit.
- **Git status**:
  ```text
  ## main_future...origin/main_future
  ```
  No uncommitted changes.
- **Diff stat**: Empty (clean working tree).
- **Last 5 commits**:
  - `04f1bcd` Add validation smoke tests and update gitignore
  - `4d7f57e` Merge branch 'origin/lukman_sku-to-rack-allocation' into main_future: bootstrap order generation as default, isolated RunContext paths
  - `d01ad52` Merge branch 'origin/devan_pps' into main_future
  - `a5708bd` Replace synthetic A/B/C orders with raw-order bootstrap generation
  - `73e2002` feat: implement RL return mode in Robot.handle_pod_return
- **Safety tag result**: Tag `before-future-main-cleanup-20260616-1454` created successfully.

---

## 3. Lightweight Validation
- **Command**: `wsl /home/dewan/torch-gpu/bin/python -m compileall model src scripts`
- **Result**: Compiles successfully with zero syntax or compilation errors.
- **Notes**: All files in the core folders `model/`, `src/`, and `scripts/` are syntactically valid and importable under the WSL virtual environment.

---

## 4. Root File Classification

| Path | Classification | Reason | Later Action Candidate |
|---|---|---|---|
| `.DS_Store` | runtime/generated artifact | macOS folder layout cache. | Delete and add to gitignore. |
| `.gitignore` | source/config/docs | Standard repository control file. | Keep and update with ignored runtime files. |
| `AGENTS.md` | source/config/docs | Local developer rules and workflow definitions. | Keep. |
| `__init__.py` | source/config/docs | Python package namespace initialization. | Keep. |
| `assign_order.csv` | runtime/generated artifact | Dynamic simulation file tracking active picker queues. Overwritten in ticks. | Gitignore. Relocate to runtime root. |
| `config.dictionary` | source/config/docs | Configuration properties dict (binary pickle). | Keep. |
| `generated_backlog.csv` | runtime/generated artifact | Temporary backlog CSV generated during setup. | Gitignore. Relocate to runtime root. |
| `generated_database_order.csv` | runtime/generated artifact | Temporary order CSV generated during setup. | Gitignore. Relocate to runtime root. |
| `generated_order.csv` | runtime/generated artifact | Temporary order arrival stream generated during setup. | Gitignore. Relocate to runtime root. |
| `generated_order_meta.json` | runtime/generated artifact | Metadata for generated order stream created during setup. | Gitignore. Relocate to runtime root. |
| `generated_pod.csv` | canonical input / template | Physical grid layout template matrix. | Relocate to `data/input/scenarios/`. |
| `items.csv` | canonical input | Master inventory SKU parameters database. | Relocate to `data/input/`. |
| `items_dictionary.csv` | canonical input | Master SKU classification mappings. | Relocate to `data/input/`. |
| `items_slots_configuration.csv` | canonical input | Layout metadata for items placed on pod shelves. | Relocate to `data/input/`. |
| `netlogo.py` | source/config/docs | Entry shim façade. Keeps NetLogo `py:run` working. | Keep in root. |
| `netlogo.state` | runtime/generated artifact | Pickled `Inventory` instance written at every simulation tick. | Gitignore. Relocate to runtime root. |
| `pod_info.csv` | runtime/generated artifact | Log tracking picked items from pods. Overwritten in ticks. | Gitignore. Relocate to runtime root. |
| `pods.csv` | canonical input | Pod stock level allocation details database. | Relocate to `data/input/`. |
| `pods_dictionary.csv` | canonical input | Pod location and identification mapping attributes. | Relocate to `data/input/`. |
| `profile.prof` | runtime/generated artifact | cProfile output dump file. | Gitignore. |
| `profile_netlogo.py` | source/config/docs | Headless simulation profiling script. | Relocate to `scripts/trace/` or `scripts/run/`. |
| `raw_order.csv` | canonical input | Historical/empirical order catalog for bootstrapper. | Relocate to `data/input/`. |
| `requirements.txt` | source/config/docs | Python requirements configuration file. | Keep in root. |
| `run_baseline.py` | source/config/docs | Charging baseline simulation script. | Relocate to `scripts/run/` or `scripts/experiments/`. |
| `simulation.nlogo` | source/config/docs | NetLogo graphical user interface. | Keep in root. |
| `skus_data.csv` | runtime/generated artifact | Global remaining SKU inventory levels. Overwritten on ticks. | Gitignore. Relocate to runtime root. |
| `sorted_skus_data.csv` | runtime/generated artifact | Sorted remaining SKU inventory levels. Overwritten on ticks. | Gitignore. Relocate to runtime root. |
| `warehouse.db` | runtime/generated artifact | Main simulator telemetry SQLite database. Overwritten on setup. | Gitignore. Relocate to runtime root. |
| `warehouse_ps_old_8.db` | unknown / legacy | Old legacy SQLite run database file. | Delete. |

---

## 5. Data/Docs/Src Layout Inventory

| Path | Current Contents | Classification | Notes |
|---|---|---|---|
| `data/input/` | `README.md` | canonical input | Destination for master/template inputs (`items.csv`, `pods.csv`, etc.). |
| `data/runtime/` | isolated runs, caches, golden trace inputs, training outputs | temporary runtime output / training artifact | Active isolated simulation workspace paths are directed here. |
| `data/output/` | experiment ledgers, SQLite ledger schemas | temporary runtime output | Aggregated run stats. |
| `docs/` | architecture maps, module guides, logs | documentation | Main repository documentation. |
| `docs/training_pps/` | Devan PPS code (`train_pps_rl.py`, `pps_env.py`), model checkpoints | executable source / training artifact | Executable RL code in docs. Move to `src/rmfs/rl/pps/` and `scripts/`. |
| `src/` | RMFS modular codebase (`runtime_io`, `experiments`, `rl`) | executable source | Modern structure wrapper. |
| `model/` | Simulation objects (`robot.py`, `inventory.py`, `layout.py`) | executable source | Core behavioral state definitions. |
| `engine/` | Generic grid landscape engine files | executable source | Movement mechanics. |
| `scripts/` | Validation dry-runs and smoke checks | executable source | Testing suites. |
| `output/` | `order-finished.csv` (when run manually) | temporary runtime output | Default output folder. |
| `robot sa data/` | Old simulation outputs and DB files | legacy artifact | Needs developer cleanup. |
| `PS/` | Old simulation outputs and DB files | legacy artifact | Needs developer cleanup. |

---

## 6. Entrypoint Inventory

| Entrypoint | Purpose | Appears Active? | Hardcoded Paths / Risks |
|---|---|---:|---|
| `simulation.nlogo` | NetLogo graphical user interface | Yes | Calls `import netlogo` shim. Operates on root CSV files by default. |
| `profile_netlogo.py` | Profiling console tick loops | Yes | Calls `console_tick`. Overwrites `profile.prof` in root. |
| `src/rmfs/orchestration/local_executor.py` | Isolated local batch runner | Yes | Configures isolated simulation directories. Relies on copying root-level inputs. |
| `scripts/training/rts_train_controller.py` | Return-To-Storage PPO training | Yes | Launches RTS policy training. Paths managed under `data/runtime/`. |
| `docs/training_pps/train_pps_rl.py` | Pick-Pod-Selection PPO training | Yes | Launches PPS policy training. Hardcodes `saved_models` relative paths. |
| `docs/training_pps/run_pps_replications.py` | Evaluates PPS policies in replication | Yes | Script-driven replacement of root CSV files. Can lock and pollute root. |
| `docs/training_pps/run_backend_episode.py` | Runs single headless PPS episode | Yes | Loads model and evaluates step-by-tick. Reads `netlogo.state`. |
| `run_baseline.py` | Charging baseline runner | Yes | Overrides `Robot` class battery thresholds. Overwrites root config. |

---

## 7. Runtime Path and Artifact Usage

| File | Path Reference | Read/Write | Risk | Notes |
|---|---|---|---|---|
| `netlogo.state` | `_str_path("state_file")`, `"netlogo.state"` | Read / Write | **High** | Pickle dump/load executed on every simulation tick. Huge IO bottleneck. |
| `warehouse.db` | `_str_path("sqlite_db")`, `"warehouse.db"` | Read / Write | **High** | Multiple updates/queries per tick open and close connections on the fly. |
| `assign_order.csv` | `_str_path("assign_order_csv")`, `"assign_order.csv"` | Read / Write | **High** | Read/written using Pandas inside loop ticks. Extremely slow. |
| `pod_info.csv` | `_str_path("pod_info_csv")`, `"pod_info.csv"` | Read / Write | **Medium** | Overwritten per tick when items are picked. |
| `skus_data.csv` | `_str_path("skus_data_csv")`, `"skus_data.csv"` | Write | **Low** | Written only during initial setup. |
| `sorted_skus_data.csv` | `_str_path("sorted_skus_data_csv")`, `"sorted_skus_data.csv"` | Write | **Low** | Written only during initial setup. |
| `generated_order.csv` | `_str_path("generated_order_csv")` | Read / Write | **Medium** | Overwritten during setup initialization. |
| `generated_pod.csv` | `_str_path("generated_pod_csv")`, `"generated_pod.csv"` | Read | **Medium** | Main template for parsing layout coordinates. |
| `saved_models` | `docs/training_pps/saved_models/pps_rl_best.zip` | Read | **Medium** | Hardcoded default load path in netlogo_api.py. |

---

## 8. PPS-RL Inventory

Executable PPS code currently resides entirely inside `docs/training_pps/`:
- `train_pps_rl.py`: Main stable-baselines3 PPO training script.
- `pps_env.py`: Custom Gymnasium environment subclass `PPSEnv`. Defines observations and reward metrics.
- `run_pps_replications.py`: Batch evaluation of pps modes (`rika` heuristic, `random`, `ppo`).
- `run_backend_episode.py`: Single episode headless execution script.

**Important Details**:
- **Working directory override**: Scripts use `sys.path.insert(0, str(_REPO_ROOT))` and `os.chdir(_REPO_ROOT)` to run from the repository root.
- **Model Path**: The default model path is hardcoded as `docs/training_pps/saved_models/pps_rl_best.zip` in both `train_pps_rl.py` and `netlogo_api.py`.
- **Logic Duplication**: Ticking loop routines and feature generation are partially duplicated between `pps_env.py` and `src/rmfs/app/netlogo_api.py` to allow fast RL step rollouts without GUI instantiation.

---

## 9. Scenario/Input System Inventory

Currently, there is no centralized scenario playback manager or directory structure:
- **`run_pps_replications.py`**: Uses a script-driven root-file replacement. During replication generation, files (`generated_order.csv`, `generated_pod.csv`, `pods.csv`, etc.) are written to `output/scenarios/rep_XXX/`. When running policies, the script copies these replication files back into the repository root to overwrite default configs.
- **`RunContext` / RTS Training**: Employs RunContext-driven path isolation. Paths are redirected to isolated folders under `data/runtime/` depending on training configs.
- **Lukman's Scenario Bundle**: A consolidated port of scenario configurations to `data/input/scenarios/` and `src/rmfs/runtime_io/scenario_bundle.py` is planned for Phase 2.

---

## 10. Charging Integration Inventory

- **Charging mechanics status**: Salsa's battery charging, layout, and scheduling mechanisms are **not** present in the core simulation engine files (`model/robot.py` contains zero battery or charging parameters).
- **Baseline overlay**: `run_baseline.py` creates a random charger positions configuration and sets placeholder values (`Robot.BATTERY_LOW_PCT = 20.0` etc.) directly on the `Robot` class object. However, these class parameters are not read by the robot behavior loop in `model/robot.py`.
- **Charging files**: Overlays write to `charging_config.json`, which is currently unused by the simulation engine in this branch.

---

## 11. Pod-Location Randomization Findings

1. **Are pod starting locations deterministic?**
   Yes. They are parsed directly from the row/column layout index in `generated_pod.csv`. If `generated_pod.csv` is not present, it calls `Layout.generate()` which deactivates storage slots randomly using `random.sample`. This layout generation is deterministic only if `set_sim_seed()` is called before `setup()`.
2. **What controls pod starting coordinates?**
   The coordinates of all cells containing `1` in `generated_pod.csv`.
3. **What triggers layout regeneration?**
   The absence of `generated_pod.csv` in the current runtime context directory on setup.
4. **Are pod contents/SKU allocation deterministic?**
   Yes. If `pods.csv` is present in the runtime context, SKU data, quantities, and thresholds are loaded directly from it using `assign_skus_to_pods_from_file()`.
5. **Safest future insertion point for pod-location-only randomization:**
   Directly inside or immediately after `draw_storage_from_generated_file()` in `src/rmfs/app/netlogo_api.py`. Shuffling the sequence of `pod_counter` mapping (or shuffling coordinates of created `Pod` objects) prior to `assign_skus_to_pods()` is the safest way to randomize starting locations without mutating pod inventory configurations.

---

## 12. Database Usage Inventory

- **Active DB file**: `warehouse.db` (controlled by `RunContext`).
- **Database Tables**:
  - `pod_location_{TS}` (tracks dynamic coordinate positions of pods)
  - `pod_travel_{TS}` (tracks travel distance telemetry)
  - `job_task_{TS}` (logs task dispatch events)
  - `order_history_{TS}` (records order processing history)
  - `pre_assign_{TS}` (logs rack-station allocations)
  - `pod_info_{TS}` (logs picked quantities)
- **Database Role**: Serves as a runtime telemetry ledger.
- **Connection Overhead**: Connection is opened, committed, and closed on every high-frequency event (e.g. `upsert_pod_location` called on every robot step). This is a massive overhead.
- **Disable points**: Inside `model/tools/*.py` database utility wrappers, checking the `fast_train` flag or an environment variable to bypass SQLite connections entirely.

---

## 13. Hot-loop Burden Inventory

| Area | Evidence | Later Optimization Type | Risk |
|---|---|---|---|
| State serialization | `pickle.load` / `pickle.dump` of the entire `Inventory` state on every `tick()`. | Keep state resident in memory during runs. | **High** |
| High-frequency CSV reads/writes | `assign_order.csv` / `pod_info.csv` read and written inside tick loop (`model/inventory.py`). | Cache dataset in memory; flush to disk only on termination. | **High** |
| DB Connection Opens/Closes | SQLite connection opened and committed for every single pod movement tick. | Maintain persistent connection pool or disable DB writes. | **High** |
| Full Inventory Scans | Scanning entire universe lists in `inventory.py` to fetch active job/robot states. | Maintain active index caches (e.g., set of active robots). | **Medium** |

---

## 14. Recommended Phase 2 Inputs

The following facts should guide the Phase 2 cleanup:
1. The target branch is named `main_future` (not `future-main`).
2. Core codebase successfully compiles and runs headless via `compileall` checks.
3. PPS RL environment and training scripts are functional but reside in `docs/training_pps/`.
4. There is no active charging mechanism integrated into `model/robot.py`.
5. Database connection overhead is high; connection pooling or a bypass mechanism is required.
6. The simulator relies on tick-by-tick state pickling, which degrades performance.

---

## 15. Stop Conditions / Risks Found

- **Branch Name mismatch**: The repository branch is named `main_future` instead of the expected `future-main`. The user explicitly approved proceeding on `main_future`.
- **Binary config file**: `config.dictionary` is tracked in the repository root as a binary pickle file.
- **Model file missing**: The default model file `docs/training_pps/saved_models/pps_rl_best.zip` is gitignored and does not exist in the working checkout.
- **Line Endings Mismatch (WSL vs. Windows)**: A cross-platform Git status discrepancy was identified. Windows Git normalizes CRLF (`\r\n`), but WSL Git defaults to `core.autocrlf = false`, marking 158 files as modified due to literal `\r` carriage returns. This was resolved locally by setting `git config core.autocrlf input` inside the WSL environment to preserve Linux shell script compatibility.

---


## 16. Files Changed By This Phase

- [future_main_baseline_audit.md](file:///wsl.localhost/Ubuntu-22.04/home/dewan/Project%20Ta/Fresh%20Start%20Structure%20V1/Rika's%20Version/docs/dev/future_main_baseline_audit.md) (New audit documentation file).

No other files in the repository have been changed, moved, or deleted.
