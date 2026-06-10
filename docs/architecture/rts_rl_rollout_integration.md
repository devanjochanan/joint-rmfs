# RTS-RL Rollout Integration

Phase 7 adds opt-in rollout/evaluation integration for RTS decisions. The simulator default remains `CurrentRTSPolicy`, and RTS-RL is not enabled unless a caller selects an RTS runtime mode.

## Modes

- `current`: default mode. `CurrentRTSPolicy` remains active and rollout logging is disabled by default.
- `current_probe`: `CurrentRTSPolicy` still selects the destination, while decision, state, mask, feature shape, and outcome rows are logged.
- `random_valid`: explicit evaluation mode that samples only valid Phase 6 RTS actions, resolves the selected zone to free storage, and logs decisions/outcomes.
- `rts_rl_explicit`: reserved for Python-level explicit model/resolver integration. Phase 7 does not add checkpoint or model-file loading.

## Runtime Shape

`local_executor` configures a process-local RTS runtime registry before `netlogo.setup()`. `Inventory.__init__` installs the runtime from that registry after assigning:

```python
self.rts_policy = CurrentRTSPolicy()
```

`Robot.handle_pod_return` records the decision after policy selection, and `Robot.advance_state` records completion immediately after `upsert_pod_location`.

Rollout files are written only under each worker runtime root:

- `rts_rollout.jsonl`
- `rts_rollout_summary.json`

No NetLogo bridge, POA/PPS, charging, order generation, pod-SKU allocation, path planning, PPO, training, checkpoints, TensorBoard, or DuckDB integration was added.

## Reward

Outcome rows include reward JSON. Reward is computed only when a valid cycle reference is configured and exists. If the reference is missing, Phase 7 records `reward_computed=false` and does not fabricate a reward.

## Replenish-Store Execution Restriction in Evaluation
random_valid is a behavior-changing evaluation mode, but in Phase 7 it executes store-branch actions only. Replenish-store action validity may still be logged in current_probe/state/mask data, but actual replenish-store execution is deferred until a real replenishment execution contract exists.

