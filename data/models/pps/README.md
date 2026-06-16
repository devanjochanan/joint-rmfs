# PPS Model Artifacts

This folder is the default output location for PPS PPO training artifacts.
The Rika heuristic remains the default PPS path; PPO is used only when
explicitly requested.

Expected runtime outputs:
- `pps_rl_best.zip`: best PPO model used when PPO PPS is explicitly requested.
- `pps_rl_final.zip`: final/latest model at the end of a training run.
- `checkpoints/`: periodic PPO checkpoints.
- `runs/`: TensorBoard logs.
- `metrics/`: per-episode CSV metrics.
- `train_state.json`: resume metadata such as training seed/progress.

Large generated model/log files should stay local and should not be committed.
