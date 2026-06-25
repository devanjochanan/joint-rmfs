# netlogo_api.py Responsibility Map

Stage 3 note: no broad `src/rmfs/app/netlogo_api.py` refactor should occur.
`setup()` and `tick()` remain the bridge entry points for now.

This map documents current responsibilities so later extraction work can be
planned behind wrappers and smoke tests instead of moving bridge code in place.

| Lines | Current responsibility | Owner area | Eventual target module | Moving risk | Recommendation |
| --- | --- | --- | --- | --- | --- |
| 1-119 | Bridge module imports, public `__all__`, compatibility surface for root `netlogo.py` and NetLogo. | Shared bridge | Keep facade in `netlogo.py`; narrow bridge exports later only after callers are audited. | High | Keep for now. |
| 124-165 | `RunContext` and scenario wrapper functions, runtime path lookup helpers. | Shared runtime I/O | `src/rmfs/runtime_io` already owns the data model; bridge should keep thin wrappers only. | Medium | Extract later behind wrapper, but keep bridge entry points stable. |
| 173-239 | Runtime seed, env parsing, run-profile resolution, task allocator runtime config, PPS mode switch wrapper. | Shared runtime plus Devan PPS for PPS mode | Runtime config helper plus PPS public API. | Medium | Keep for now; extract only after setup/tick smoke coverage is stable. |
| 247-493 | PPS observation construction, candidate selection, action execution, PPO/random PPS policy application. | Devan PPS | `src/rmfs/decisions/pps` public wrapper for bridge-facing helpers. | Medium | Extract later behind a small PPS bridge adapter. Do not expose through broader `netlogo_api.py.__all__` just for validation scripts. |
| 494-523 | Metrics bridge helpers for throughput, order completion time, pod visits, picked quantity, pile-on rate. | Shared metrics, Devan PPS consumers | `src/rmfs/metrics` or `src/rmfs/decisions/pps/metrics.py`. | Low to medium | Extract later behind wrapper. |
| 527-688 | `DirectedGraph` path helper class and modified Dijkstra path functions. | Shared pathing | Shared pathing module under `src/rmfs/pathing` after equivalence checks. | High | Keep for now; do not move without route comparison smokes. |
| 690-770 | Robot initialization, layout draw entry points, order stream bootstrap call during layout setup. | Shared simulation core, Lukman order generation | Robot/layout initialization module plus `src/rmfs/order_generation` wrapper. | High | Keep for now; extraction needs setup return-shape and generated-order checks. |
| 772-920 | Backlog order Jaccard similarity, clustering, assignment into picker stations and active order CSV state. | Lukman order generation plus shared runtime I/O | `src/rmfs/order_generation/backlog.py` later. | High | Keep for now; order semantics and CSV state coupling are sensitive. |
| 922-1195 | Storage/pod/station layout parsing from `generated_pod.csv`, graph construction, pod-location randomization hook. | Shared layout/pathing, Lukman pod inputs | `model/layout.py` or future `src/rmfs/layout` bridge adapter after tests. | High | Keep for now; pathing and storage semantics are tightly coupled. |
| 1197-1237 | Station path construction and graph neighbor helper. | Shared pathing/layout | Future `src/rmfs/pathing` or layout helper module. | Medium to high | Extract later with route/path smoke tests. |
| 1240-1305 | `assign_skus_to_pods` and `assign_skus_to_pods_from_file`, including SKU global data and `pod_info.csv` initialization. | Lukman pod-SKU allocation plus shared runtime I/O | `src/rmfs/order_generation/pod_sku.py` with a bridge-facing wrapper. | High | Keep for now; do not migrate root/runtime CSV behavior in Stage 1. |
| 1307-1382 | `setup()`: runtime dirs, detail DB config, table resets, runtime artifact cleanup, `Inventory`, layout, PPS config, pickle save, setup return. | Shared bridge and simulation core | Keep bridge function; move internals only behind adapters after setup smoke/golden trace coverage. | Very high | Do not move in Stage 1. |
| 1385-1418 | `tick()`: load state, refresh universe references, configure PPS, `Inventory.tick()`, PPS policy hook, pickle save, NetLogo return payload. | Shared bridge and simulation core | Keep bridge function; later split state I/O and tick orchestration behind wrappers. | Very high | Do not move in Stage 1. |
| 1421-1455 | `console_tick()`: headless loop variant for profiling/manual console use. | Shared bridge/profiling | Future run CLI or local executor path. | High | Keep for now; avoid long-run behavior changes. |
| 1458-end | `setup_py()` package installer helper. | Shared legacy utility | Remove or replace later only after caller audit. | Low to medium | Keep for now unless explicitly deprecated. |

## RTS/RTS-RL Hooks

RTS and RTS-RL behavior is not directly implemented in this bridge as a large
section. The active hook point is the `Inventory` object constructed in
`setup()` and advanced in `tick()`, with RTS runtime behavior installed through
the inventory/runtime modules outside this file. Moving setup/tick orchestration
would therefore affect RTS rollout logging even if no RTS code is edited.

## Charging Hook Status

Charging configuration and placement scaffolds exist under
`src/rmfs/decisions/charging` and `data/input/charging`. Stage 3 adds a narrow
runtime bridge at `src/rmfs/app/charging_bridge.py` that can be explicitly
enabled by callers using `charging_enabled=True` feature flags. It loads the
Salsa config, installs a charger registry and robot battery/status accounting
on a provided universe object, detects low-battery robots, assigns available
chargers deterministically, and emits summary metrics for validation.

`src/rmfs/app/netlogo_api.py` does not call this bridge from `setup()` or
`tick()` in Stage 3. Default baseline runs therefore remain unchanged, and
physical charger-trip/pathing behavior remains deferred.

## Stage 3 Boundary

No broad `netlogo_api.py` refactor should occur in Stage 3. The bridge remains
the compatibility layer for NetLogo and local scripts, while extraction targets
stay documented for a later stage with dedicated behavior checks.
