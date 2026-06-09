# Input Snapshot

Phase 4 can copy active generated inputs into `input_snapshot/` when the local executor is run with `--snapshot-inputs`.

Copied inputs:

- `generated_order.csv`
- `generated_pod.csv`
- `pods.csv`
- `items.csv`
- `generated_backlog.csv`
- `generated_database_order.csv`

Hash-only inputs:

- `items_dictionary.csv`
- `items_slots_configuration.csv`
- `pods_dictionary.csv`

Excluded for now:

- `raw_order.csv`, because it is legacy-only through `stock_out_probability.py` and not part of the active executor/setup path.

Runtime/state/output/checkpoint artifacts are never copied as inputs: `netlogo.state`, `warehouse.db`, `assign_order.csv`, `pod_info.csv`, `skus_data.csv`, `sorted_skus_data.csv`, `output/`, `result/`, `saved_models/`, `profile.prof`, and `netlogo_profile_summary.txt`.

When an input snapshot is used, workers call `RunContext.isolated(..., input_root=input_snapshot_root)`. Mutable runtime files still go to each worker runtime directory.

### Residual Root Dependency Note
Phase 4 snapshots active generated inputs for worker reads, but does not refactor legacy generator path ownership. The order/pod generation modules remain root-compatible and should not be modified until a dedicated data-generation phase or owner handoff. This avoids overclaiming "full snapshot independence."
