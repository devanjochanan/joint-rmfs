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

The actor smoke verifies finite `old_log_prob` and `old_value`, nonblank `policy_checkpoint_id`, valid selected action, `actor_kind=rts_rl_explicit`, `policy_mode=sample`, and `decision.mode=rl`.

The dataset smoke verifies that active-checkpoint `rts_rl_explicit` rows pass while `current`, `current_probe`, `random_valid`, `heuristic`, `synthetic`, checkpoint-mismatched, and missing old-policy-value rows fail.

The controller dry run writes controller directories, batch input references, worker specs, and JSON summaries only. It does not launch simulator workers or update model checkpoints.

## Validation & Execution Rules (Phase 9 Cleanup)

- **Safe Defaults**: The controller CLI runs in dry-run mode unless `--execute` is explicitly set. `--dry-run` is deprecated.
- **Explicit zone_ids**: Non-dry runs require explicit `--zone-ids` or checkpoint metadata to determine the valid zone list.
- **RTS-RL Explicit Behavior**: Under `rts_rl_explicit`, the policy behaves strictly with no fallback options to heuristically select zones.
- **Device Resolution rules**:
  - Controller default device: `auto` (cuda if available, else cpu).
  - Worker default device: `cpu` (for safety in multi-worker scenarios).
  - Explicit `--worker-device auto` resolves to `cuda` if available.
  - Explicit `--worker-device cuda` fails clearly if CUDA is unavailable.
- **Timebase conversion semantics**:
  - `warehouse_time` equals `Inventory._tick`.
  - `netlogo_step` equals `round(warehouse_time / tick_to_second)`.
  - Conversion does not hardcode factors (e.g. 0.15 or 0.25).
