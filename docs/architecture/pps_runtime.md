# PPS Runtime Ownership

Default PPS behavior remains the Rika heuristic. PPS-RL is used only when
explicitly requested through a script flag, `PPS_MODE=ppo`, or
`netlogo.set_pps_mode("ppo")`.

Default PPO model path:

```text
data/models/pps/pps_rl_best.zip
```

## Ownership

- `src/rmfs/rl/pps/env.py`: Gymnasium training/evaluation environment.
- `src/rmfs/rl/pps/model_paths.py`: PPS model path resolution.
- `scripts/training/train_pps_rl.py`: training CLI.
- `scripts/validation/run_pps_backend_episode.py`: one-episode backend CLI.
- `scripts/experiments/run_pps_replications.py`: paired policy experiment CLI.

## Intentional Duplication

PPS observation/action construction still exists in both the NetLogo bridge and
the Gym environment. It was not extracted in Phase 4 or Phase 5 because changing
that code could alter observation tensor semantics or action ordering.
Extracting it should be handled with focused comparison tests in a later phase.

Phase 5 adds `scripts/validation/pps_observation_contract_smoke.py` as a static
contract check for observation-space shape, action count/order, and the default
model path. It does not run training and does not prove behavior equivalence.

No performance improvement or behavior equivalence is claimed for Phase 5.
