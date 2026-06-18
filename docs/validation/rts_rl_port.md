# RTS-RL Port Validation

Run the pure Phase 6 smoke:

```bash
/home/dewan/torch-gpu/bin/python scripts/validation/rts_rl_port_smoke.py
```

The smoke validates:

- action encode/decode round trips
- action mask shape and invalid/all-invalid rejection
- synthetic state JSON serialization
- action and stock feature matrix shape alignment
- absence of raw threshold feature names
- reward computation with a valid reference
- missing-reference reward behavior
- cycle-reference JSON round trip
- CPU model forward/inference
- CUDA model inference when available
- `RTSRLPolicy` import
- `CurrentRTSPolicy` import

This validation does not run a simulator rollout and does not prove paper fidelity, behavior equivalence, throughput improvement, or benchmark equivalence. It adds no PPO/training, checkpoints, TensorBoard, DuckDB, BehaviorSpace, or DoE.

Worker summaries and executor behavior are unaffected by Phase 6. RTS-RL remains disabled by default.

Phase 6 validation result:

- `py_compile` for `src/rmfs/rl/rts/*.py`, `src/rmfs/decisions/rts/*.py`, and `scripts/validation/rts_rl_port_smoke.py`: passed
- `scripts/validation/rts_rl_port_smoke.py`: passed
- `CurrentRTSPolicy` import validation: passed
