"""Synthetic RTS-RL PPO training infrastructure."""

from .config import RTSTrainingConfig
from .rollout_dataset import RTSRolloutDataset, RTSTrainingStep, RTSPaddedTrainingBatch

try:
    from .ppo import RTSPPORolloutBatch, PPOUpdateResult
except ModuleNotFoundError:
    RTSPPORolloutBatch = None
    PPOUpdateResult = None

__all__ = [
    "RTSTrainingConfig",
    "RTSRolloutDataset",
    "RTSTrainingStep",
    "RTSPaddedTrainingBatch",
    "RTSPPORolloutBatch",
    "PPOUpdateResult",
]

