# Phase 5B Stabilization Notes

This pass keeps simulator semantics intact while removing current execution
blockers before the human-operator documentation phase.

## Run Profiles

Human-facing runs should select one profile first:

- `smoke`: small bounded run, detail DB off, deterministic pod-slot randomization.
- `training`: bounded PPS/RL training horizon, detail DB off, pod slots randomized by seed.
- `ablation`: 100000 tick default horizon, bounded demand, detail DB off, pod slots randomized by seed.
- `debug`: small bounded run with detail DB/debug artifacts enabled.
- `gui`: manual compatibility profile with fixed pod locations and legacy fallback behavior.

Low-level demand and pod-location flags remain available for explicit
experiments, but full raw-order replay is opt-in only.

## Import Contract

`import netlogo` must not import `gymnasium` or Stable-Baselines. PPS training
dependencies are loaded only by PPS training modules or by explicit PPO model
loading.

## Legacy Charging Baseline

`run_baseline.py` is parked as a legacy charging runner. It is discoverable but
inactive by default because `main_future` does not include the required charging
mechanism. Set `RMFS_ALLOW_LEGACY_CHARGING_BASELINE=1` only for explicit legacy
investigation.

## Remaining Phase 6/Follow-Up Notes

- PPS observation/action construction is still duplicated between the NetLogo
  bridge and PPSEnv; it was not extracted here to avoid tensor/action risk.
- Replication orchestration still has older scenario snapshot/restore behavior;
  it now uses explicit profile env defaults, but deeper rewrite should wait for
  a dedicated experiment-runner pass.
- This phase does not claim performance improvement or behavior equivalence.
