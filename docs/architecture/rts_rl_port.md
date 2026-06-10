# RTS-RL Port

Phase 6 ports the RTS-RL decision/model layer from the advanced `netlogo-rmfs` repo into the current Rika-host repo as opt-in modules under `src/rmfs/rl/rts/`.

Ported/adapted pieces:

- joint RTS action space with `store(zone)` and `replenish_store(zone)`
- action masks and validation
- current-repo-safe RTS state JSON
- zone, stock, and action-row feature matrices
- masked actor-critic model with a stock-row encoder
- masked inference helpers
- reward and cycle-reference helpers
- optional `RTSRLPolicy` adapter

The adapter is not wired into the simulator default. `CurrentRTSPolicy` remains the default and RTS-RL remains disabled unless explicitly instantiated by a caller.

Raw threshold constants from the advanced source are excluded from model feature names. The model sees derived signals such as `pod_below_threshold_ratio`, `replenishment_signal_active`, `zero_global_low_sku_count`, `below_threshold_sku_ratio`, `shortage_depth`, `global_low_depth`, and fill ratios.

Deferred to Phase 7:

- safe simulator rollout collection
- concrete zone/storage data contracts
- calibrated cycle-time references from real ledgers

Deferred to Phase 8:

- PPO training
- checkpoint/artifact resolution
- parallel training orchestration
- TensorBoard/DuckDB or benchmark pipelines

No simulator behavior, POA/PPS, charging, order generation, pod-SKU allocation, path planning, NetLogo bridge shape, or default RTS policy was changed.
