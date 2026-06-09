# Runtime Paths

Phase 2 introduces `src.rmfs.runtime_io.RunContext` as a narrow runtime I/O boundary for the NetLogo bridge.

`RunContext.default()` preserves the current root-relative behavior:

- `netlogo.state`
- `warehouse.db`
- `assign_order.csv`
- `pod_info.csv`
- `skus_data.csv`
- `sorted_skus_data.csv`

`RunContext.isolated(runtime_root=...)` redirects mutable runtime artifacts into `runtime_root` while keeping canonical input CSVs root-compatible for this slice. The simulator still reads root-level generated inputs such as `generated_order.csv`, `generated_pod.csv`, and `pods.csv`.

This phase does not prove full behavior equivalence. It is path isolation only, not a logic refactor, decision-module refactor, order-generation refactor, RL/checkpoint refactor, or parallel executor.

Generated runtime outputs must not be committed.

Deferred higher-risk paths for later slices include order/layout generator outputs, RL `saved_models/`, profile artifacts, `output/`, `result/`, and robot job CSV logs.
