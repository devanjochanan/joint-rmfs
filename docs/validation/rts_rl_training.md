# RTS-RL Training Validation

Phase 8.1/8.2 validation is synthetic only. It does not run the simulator, local executor, BehaviorSpace, benchmarks, or real training.

Run:

```bash
/home/dewan/torch-gpu/bin/python scripts/validation/rts_ppo_update_smoke.py
```

This smoke builds synthetic Phase 7-style decision/outcome rows, reconstructs features, builds an offline PPO batch, runs a short PPO update, checks finite losses, verifies model parameters change, and validates all-invalid/invalid-selected action errors.

Run:

```bash
/home/dewan/torch-gpu/bin/python scripts/training/rts_train_smoke.py --artifact-label phase8_synthetic_smoke --output-root data/runtime/rts_training_smoke
```

This smoke writes a synthetic cycle reference under the smoke output directory, runs one synthetic PPO update, saves a checkpoint, writes `latest.json`, appends `checkpoint_history.jsonl`, writes `batch_summary.json`, loads the checkpoint back, and verifies a loaded-model forward path.

This is not true on-policy PPO yet because Phase 7 rollout rows lack `old_log_prob`, `old_value`, and `policy_checkpoint_id`. No PPO training run was performed beyond the synthetic smoke.

