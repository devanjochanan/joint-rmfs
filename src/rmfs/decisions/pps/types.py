"""PPS module data types."""

from enum import Enum


class PPSMode(str, Enum):
    """Supported Pick Pod Selection (PPS) modes."""
    PPO = "ppo"
    PPO_CONSTRAINED = "ppo_constrained"
    RANDOM = "random"
    HEURISTIC = "heuristic"
    DEMAND = "demand"
