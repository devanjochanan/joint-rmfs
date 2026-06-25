# Full-postT Scenario Integration Handoff

## Purpose
This note explains the recent `joint-rmfs` changes that were made to run the
shared scenario inputs from `_full_postt_parallel_runs` directly inside
`joint-rmfs`, while keeping the runtime switchable by scenario name.

Use this file as the first handoff reference before touching the integration.

## What Was Added

### 1. Scenario bundles now support more than `items.csv` and `pods.csv`
`joint-rmfs` scenario bundles can now carry:

- `items.csv`
- `pods.csv`
- `generated_pod.csv`
- `raw_order.csv`

This was added so a scenario can bring its own:

- SKU universe
- pod allocation
- physical layout
- cutoff-order source for replay/bootstrap generation

Main file:
- `src/rmfs/runtime_io/scenario_bundle.py`

### 2. A sync script was added for `_full_postt_parallel_runs`
New script:
- `scripts/data/sync_full_postt_scenarios.py`

It copies scenario files from:
- `_full_postt_parallel_runs/four_scenario_1000_shared_latest/<scenario>/netlogo-rmfs/...`

into:
- `joint-rmfs/data/input/scenarios/<scenario>/`

Current intended synced scenarios:

- `scenario4_sij`
- `cindy_s3`
- `my_scenario`

### 3. The headless runner can now activate a scenario by name
Updated runner:
- `scripts/run/run_pps_backend_episode.py`

New support:
- `--scenario <name>`

It also prints an `[INPUT]` banner after setup so runs show which scenario was
actually activated.

### 4. Non-RL heuristic runs were made more portable
`joint-rmfs` previously imported some `torch`-dependent modules too early, which
caused plain heuristic runs to fail on machines without `torch`.

This was softened by deferring RL-only imports in:

- `model/intersection_manager.py`
- `src/rmfs/rl/rts/training/__init__.py`

Important:
- heuristic / non-RL runs should now work without `torch`
- actual RL training/inference still requires the full RL dependency stack

### 5. Layout/runtime compatibility fixes were added
Some old `joint-rmfs` assumptions did not match the synced full-postT scenario
geometry. Fixes were added for:

- station path bounds on shorter layouts
- robot spawning from active layout-compatible nodes
- pickup-route fallback if the first route graph is disconnected
- `assigned_station` CSV dtype so string station ids like `picker-1` do not
  crash pandas

Main files:
- `src/rmfs/app/netlogo_api.py`
- `model/robot.py`

### 6. Partial replenishment parity was ported into `joint-rmfs`
This is not a full one-to-one clone of the current `_full_postt_parallel_runs`
runtime, but these behaviors were added:

- pending replenishment dispatch queue
- `must_replenish_before_pick`
- PPS pod filtering for pods awaiting replenishment
- specific-SKU replenishment completion support
- replenishment aging / mandatory refresh helpers

Main files:
- `model/inventory.py`
- `model/pod.py`
- `model/pod_manager.py`
- `model/robot_job.py`
- `src/rmfs/decisions/pps/heuristic.py`

## Important Caveats

### Scenario activation currently rewrites the active base input
When a scenario is activated, its normalized bundle is copied into:

- `data/input/base/`

So after a run, the active base files reflect the last scenario that was
activated.

This is acceptable for now, but it means:
- `data/input/base` is not a neutral permanent dataset
- repeated manual experiments should assume "last activated scenario wins"

### The runtime is heavier than `_full_postt_parallel_runs`
This integrated `joint-rmfs` path is currently slower than the user's lighter
RMFS runner because it carries more framework/runtime structure, setup logic,
and bookkeeping.

That slowdown is mostly architectural, not just from the integration patch.

### This patch changes behavior, not only infrastructure
The integration is not documentation-only or routing-only. It also changes live
simulation behavior in a few places, especially:

- replenishment handling
- robot spawn selection
- route fallback during pickup

So results should not be assumed identical to older `joint-rmfs` runs.

## Files Most Relevant To Future Edits

If the next AI needs to continue this work, start here:

- `scripts/data/sync_full_postt_scenarios.py`
- `src/rmfs/runtime_io/scenario_bundle.py`
- `scripts/run/run_pps_backend_episode.py`
- `src/rmfs/app/netlogo_api.py`
- `model/inventory.py`
- `model/robot.py`

If the next AI needs scenario data only, check:

- `data/input/scenarios/README.md`

## Typical Workflow

### Sync scenario inputs from `_full_postt_parallel_runs`
From the repository root:

```powershell
& ".\.rmfs\Scripts\python.exe" joint-rmfs\scripts\data\sync_full_postt_scenarios.py
```

### Run one scenario headlessly
From `joint-rmfs`:

```powershell
& "..\.rmfs\Scripts\python.exe" scripts\run\run_pps_backend_episode.py `
  --mode heuristic `
  --profile smoke `
  --max-ticks 1000 `
  --seed 42 `
  --scenario scenario4_sij `
  --full-raw-order-replay `
  --pod-location-mode fixed `
  --progress-seconds 15
```

## Last Verified Runs
Verified after the integration patch:

- `scenario4_sij`, 1000 ticks
- `cindy_s3`, 1000 ticks
- `my_scenario`, 1000 ticks

These were run with:

- `--mode heuristic`
- `--full-raw-order-replay`
- `--pod-location-mode fixed`
- `--seed 42`

Observed outputs:

- `scenario4_sij`: throughput `56`, pod visits `40`, total energy `970745.18`
- `cindy_s3`: throughput `56`, pod visits `53`, total energy `1201749.28`
- `my_scenario`: throughput `53`, pod visits `40`, total energy `888785.91`

Do not treat these as paper-ready benchmark claims. They are only the last
known sanity-check outputs for this integrated state.
