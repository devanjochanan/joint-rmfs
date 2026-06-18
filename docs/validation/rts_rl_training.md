# RTS-RL PPO Math & Checkpoint Validation

Phase 8 PPO math validation is synthetic validation only. It does not run the simulator, local executor, BehaviorSpace, benchmarks, real training, or `training --execute`.

Run:

```bash
/home/dewan/torch-gpu/bin/python scripts/validation/rts_ppo_update_smoke.py
```

This validation smoke builds synthetic decision/outcome rows, reconstructs features, builds a synthetic PPO smoke batch, runs a short in-process PPO update, checks finite losses, verifies model parameters change, validates all-invalid/invalid-selected action errors, and asserts that strict on-policy eligibility guards reject invalid/heuristic rows.

Offline/off-policy PPO training is not supported. `current_probe` and `random_valid` rollout rows are diagnostics/evaluation only and are not PPO-trainable. Active v1 PPO training requires `rts_rl_explicit` on-policy rows and is validated by `docs/validation/rts_rl_on_policy_training.md`. The deleted legacy `scripts/training/rts_train_smoke.py` synthetic checkpoint workflow should not be used as an active training path.

