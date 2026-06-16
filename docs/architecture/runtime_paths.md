# Runtime Paths

Phase 3 makes `src.rmfs.runtime_io.RunContext` the runtime/input path boundary.

Canonical inputs:

- `data/input/base/items.csv`
- `data/input/base/pods.csv`
- `data/input/base/generated_pod.csv`
- `data/input/base/raw_order.csv`
- `data/input/dictionaries/items_dictionary.csv`
- `data/input/dictionaries/pods_dictionary.csv`
- `data/input/dictionaries/items_slots_configuration.csv`

Runtime outputs:

- default/manual context: `data/runtime/latest/`
- isolated/headless workers: `data/runtime/tmp/<run-or-worker>/`
- debug scratch: `data/runtime/debug/`
- worker timing summaries, when enabled: `<runtime_root>/timing_summary.json`

Output/model roots:

- `data/output/`
- `data/models/pps/`
- `data/models/rts/`

Legacy root compatibility remains available through `RunContext.legacy_root()` and
the scenario activation `--legacy-root` flag. If canonical inputs are missing,
`RunContext` can warn and fall back to legacy root files, but root is no longer
the canonical input location.

Pod-location randomization is explicit-only. Set
`RMFS_POD_LOCATION_MODE=randomize_slots` plus `RMFS_POD_LOCATION_SEED` or
`RMFS_SIM_SEED` to shuffle which `pod_id` starts at each existing storage slot.
This does not modify `items.csv`, `pods.csv`, or `generated_pod.csv`.

Runtime timing is explicit-only through `RMFS_TIMING=1` or `--timing` on local
executor paths. Detail DB writes are explicit for headless workers through
`RMFS_DETAIL_DB=1` or `--detail-db`.

Not changed yet: broad CSV buffering, behavior equivalence claims, charging
mechanics, and high-risk NetLogo bridge redesign.
