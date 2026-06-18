# PPS RL Training

PPS reinforcement-learning code now lives in `src/rmfs/rl/pps/`; executable entrypoints live under `scripts/`.

The existing Rika heuristic remains the default PPS mode. PPO PPS is selected only by passing `--mode ppo` / `--pps-mode ppo`, calling `netlogo.set_pps_mode("ppo")`, or setting `PPS_MODE=ppo`.

Default model path:

```text
data/models/pps/pps_rl_best.zip
```

If that model is missing, normal heuristic runs still work. A missing model only matters when PPO PPS is explicitly requested.

## Commands

```bash
/home/dewan/torch-gpu/bin/python scripts/training/train_pps_rl.py --help
/home/dewan/torch-gpu/bin/python scripts/validation/run_pps_backend_episode.py --mode rika --max-ticks 3000 --seed 20260601
/home/dewan/torch-gpu/bin/python scripts/validation/run_pps_backend_episode.py --mode ppo --max-ticks 3000 --seed 20260601 --model-path data/models/pps/pps_rl_best.zip
/home/dewan/torch-gpu/bin/python scripts/experiments/run_pps_replications.py --replications 30 --max-ticks 3000 --modes rika random ppo --model-path data/models/pps/pps_rl_best.zip
```

Do not run PPO training or replication sweeps as part of lightweight integration checks.
