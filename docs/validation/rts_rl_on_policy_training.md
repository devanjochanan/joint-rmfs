# RTS-RL On-Policy Training Validation

Phase 9 validation is smoke-sized. It does not run the simulator, local executor workers, long training, BehaviorSpace, DoE, benchmarks, or DuckDB writes.

Pure smokes:

```bash
/home/dewan/torch-gpu/bin/python scripts/validation/rts_timebase_smoke.py
/home/dewan/torch-gpu/bin/python scripts/validation/rts_training_checkpoint_loader_smoke.py
/home/dewan/torch-gpu/bin/python scripts/validation/rts_on_policy_actor_smoke.py
/home/dewan/torch-gpu/bin/python scripts/validation/rts_on_policy_dataset_smoke.py
/home/dewan/torch-gpu/bin/python scripts/validation/rts_training_controller_dry_run.py
```

The actor smoke verifies finite `old_log_prob` and `old_value`, nonblank `policy_checkpoint_id`, valid selected action, and `actor_kind=rts_rl_explicit`.

The dataset smoke verifies that active-checkpoint `rts_rl_explicit` rows pass while `current`, `current_probe`, `random_valid`, `heuristic`, `synthetic`, checkpoint-mismatched, and missing old-policy-value rows fail.

The controller dry run writes controller directories, batch input references, worker specs, and JSON summaries only. It does not launch simulator workers or update model checkpoints.

