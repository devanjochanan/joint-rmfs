# RTS-RL PPO Math & Checkpoint Validation

Phase 8 validation is synthetic validation only. It does not run the simulator, local executor, BehaviorSpace, benchmarks, or training.

Run:

```bash
/home/dewan/torch-gpu/bin/python scripts/validation/rts_ppo_update_smoke.py
```

This smoke builds synthetic validation decision/outcome rows, reconstructs features, builds a synthetic PPO smoke batch, runs a short PPO update, checks finite losses, verifies model parameters change, validates all-invalid/invalid-selected action errors, and asserts that strict on-policy eligibility guards reject invalid/heuristic rows.

Run:

```bash
/home/dewan/torch-gpu/bin/python scripts/training/rts_train_smoke.py --artifact-label phase8_synthetic_smoke --output-root data/runtime/rts_training_smoke
```

This smoke writes a synthetic cycle reference under the smoke output directory, runs one synthetic PPO update, saves a checkpoint, writes `latest.json`, appends `checkpoint_history.jsonl`, writes `batch_summary.json`, loads the checkpoint back, and verifies a loaded-model forward path.

Offline/off-policy PPO training is not supported. `current_probe` and `random_valid` rollout rows are diagnostics/evaluation only and are not PPO-trainable. True PPO training requires `rts_rl_explicit` on-policy rows and is deferred to Phase 9. No PPO training run was performed beyond synthetic validation smokes.

