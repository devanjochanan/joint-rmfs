# RTS-RL PPO Math & Checkpoint Validation

Phase 8 adds RTS-RL PPO math/checkpoint validation layers under `src/rmfs/rl/rts/training/`.

Offline/off-policy PPO training is not supported. `current_probe` and `random_valid` rollout rows are diagnostics/evaluation only and are not PPO-trainable. True PPO training requires `rts_rl_explicit` on-policy rows and is deferred to Phase 9. The helper `build_synthetic_ppo_smoke_batch` computes old log-probs and values from the current model for synthetic validation smokes.

## Added Validation Pieces

- validation config checks
- rollout JSONL pairing and duplicate filtering
- feature reconstruction from `state_json`, `zone_ids`, and `action_mask`
- padded action/stock tensor batches
- masked categorical PPO update validation helpers
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

