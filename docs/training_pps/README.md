# PPS RL Training Utilities

This folder contains the PPS reinforcement-learning and backend-evaluation utilities that were ported from `Fresh Start Main RMFS backup`.

Run commands from the repository root so generated CSV files, TensorBoard logs, and models stay in the normal project folders.

## Install Dependencies

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Train PPO PPS

Normal training I/O:

```powershell
$env:RMFS_FAST_TRAIN="0"
.\.venv\Scripts\python.exe docs\training_pps\train_pps_rl.py --episodes 500 --save-path docs/training_pps/saved_models/pps_rl.pt --max-ticks 3000
```

Fast training I/O:

```powershell
$env:RMFS_FAST_TRAIN="1"
.\.venv\Scripts\python.exe docs\training_pps\train_pps_rl.py --episodes 500 --save-path docs/training_pps/saved_models/pps_rl.pt --max-ticks 3000
```

## Evaluate One Backend Episode

```powershell
.\.venv\Scripts\python.exe docs\training_pps\run_backend_episode.py --mode rika --max-ticks 3000 --seed 20260601
.\.venv\Scripts\python.exe docs\training_pps\run_backend_episode.py --mode random --max-ticks 3000 --seed 20260601
.\.venv\Scripts\python.exe docs\training_pps\run_backend_episode.py --mode ppo --max-ticks 3000 --seed 20260601 --model-path docs/training_pps/saved_models/pps_rl_best.zip
```

## Run Replications

```powershell
.\.venv\Scripts\python.exe docs\training_pps\run_pps_replications.py --replications 30 --max-ticks 3000 --modes rika random ppo --model-path docs/training_pps/saved_models/pps_rl_best.zip
```