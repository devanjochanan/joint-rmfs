# RTS-RL Rollout Integration Validation

Run the pure synthetic smoke:

```bash
/home/dewan/torch-gpu/bin/python scripts/validation/rts_rl_rollout_smoke.py
```

The smoke validates runtime config defaults, registry configure/reset, JSONL writer behavior, rollout summary aggregation, decision/outcome linking, missing-reference reward behavior, valid-reference reward computation, storage resolver mutation safety, random-valid mask safety, and current-probe logging without changing policy selection.

Short local executor smokes:

```bash
/home/dewan/torch-gpu/bin/python scripts/run/local_executor_smoke.py --runs 2 --ticks 3 --max-workers 2 --snapshot-inputs --output-root data/runtime/local_executor_smoke/phase7_default_smoke
```

```bash
/home/dewan/torch-gpu/bin/python scripts/run/local_executor_smoke.py --runs 2 --ticks 3 --max-workers 2 --snapshot-inputs --rts-policy-mode current_probe --rts-rollout --output-root data/runtime/local_executor_smoke/phase7_current_probe_smoke
```

The three-tick executor smokes may produce zero RTS decisions. That is acceptable; simulator-length forcing is intentionally avoided, and event coverage comes from the synthetic smoke.

This validation does not prove behavior equivalence, benchmark equivalence, paper fidelity, throughput improvement, congestion improvement, or performance improvement.

## Replenish-Store Execution Restriction in Evaluation
random_valid is a behavior-changing evaluation mode, but in Phase 7 it executes store-branch actions only. Replenish-store action validity may still be logged in current_probe/state/mask data, but actual replenish-store execution is deferred until a real replenishment execution contract exists.

