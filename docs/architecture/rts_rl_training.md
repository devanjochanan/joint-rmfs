# RTS-RL Training Infrastructure

Phase 8.1/8.2 adds the first RTS-RL training infrastructure layer under `src/rmfs/rl/rts/training/`.

This is synthetic/offline PPO infrastructure only. It is not true on-policy PPO yet because Phase 7 rollout rows do not contain `old_log_prob`, `old_value`, or `policy_checkpoint_id`. The helper `build_offline_ppo_batch` computes old log-probs and values from the current model for synthetic smokes.

## Added Pieces

- training config validation
- rollout JSONL pairing and eligibility filtering
- feature reconstruction from `state_json`, `zone_ids`, and `action_mask`
- padded action/stock tensor batches
- masked categorical PPO update helpers
- GAE/return calculation
- checkpoint, lineage, latest pointer, and checkpoint history helpers
- synthetic cycle-reference helpers

## Checkpoint Layout

Synthetic smoke checkpoints use:

```text
<output_root>/<artifact_label>/
  latest.json
  checkpoint_history.jsonl
  batch_000001/
    checkpoint/
      model.pt
      optimizer.pt
      metadata.json
      feature_schema.json
      cycle_reference.json
    batch_summary.json
```

`latest.json` points to the latest checkpoint directory. `checkpoint_history.jsonl` records batch history. Checkpoints live under ignored `data/runtime/**` paths.

No simulator behavior is changed. No checkpoint auto-loading is added to the default simulator path, and `CurrentRTSPolicy` remains the default RTS behavior.

