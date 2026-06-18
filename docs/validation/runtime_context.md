# Runtime Context Validation

Use the isolated smoke to check that bridge-level runtime files can be redirected away from the repository root:

```bash
/home/dewan/torch-gpu/bin/python scripts/trace/run_context_smoke.py \
  --ticks 3 \
  --runtime-root data/runtime/run_context_smoke/manual
```

The smoke configures `netlogo` with `RunContext.isolated(...)`, runs `setup()` and three ticks, checks isolated runtime files, and fails if known root mutable files change.

Expected isolated runtime files:

- `netlogo.state`
- `warehouse.db`
- `assign_order.csv`
- `pod_info.csv`
- `skus_data.csv`
- `sorted_skus_data.csv`

This validation is a lightweight smoke only. It does not prove full behavior equivalence, benchmark equivalence, paper fidelity, throughput improvement, congestion improvement, or performance improvement.

Default root-relative behavior remains available through `RunContext.default()` and `netlogo.reset_run_context()`. Canonical CSV inputs remain root-level for now. Parallel execution is not implemented yet.
