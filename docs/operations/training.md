# Reinforcement Learning Training Guide

This guide covers RL training operations, focusing on the Pick Pod Selection (PPS) training flow.

## PPS RL Training

PPS training uses reinforcement learning (e.g. Proximal Policy Optimization via Stable-Baselines3) to learn pod selection strategies.

### 1. Training Dependencies
Running PPS training requires a specific Python environment containing RL libraries (`Gymnasium`, `Stable-Baselines3`, `PyTorch`).
- **Isolation Policy**: These dependencies are loaded dynamically. Standard heuristic simulations (`import netlogo`) and validations do **not** require these packages to be installed, preventing import failures on lean test machines.
- **Environment**: Use the designated WSL environment `/home/dewan/torch-gpu/bin/python` where these libraries are installed.

### 2. Training Commands

- **Dry-Run Training CLI**:
  ```bash
  /home/dewan/torch-gpu/bin/python scripts/run/rmfs.py training --target pps --seed 123 --dry-run
  ```
- **Inspect Underlying Trainer Options**:
  ```bash
  /home/dewan/torch-gpu/bin/python scripts/training/train_pps_rl.py --help
  ```

### 3. Safety Rules
- **Do Not Run Training inside Validation**: The `validate` subcommand only executes rapid compile and tick tests. It must never launch training loops.
- **A training run is highly expensive**: Always check the resolved profile settings (`profile training`) and run with `--dry-run` to inspect the command delegation before committing resources.
- **RTS RL Training Status**: Return-to-Storage (RTS) training logic is separate. No automatic RTS training entrypoint is exposed in the Phase 6 operator wrapper CLI. RTS-RL is configured on an operator-specific basis; do not attempt to trigger RTS training from `scripts/run/rmfs.py`.
