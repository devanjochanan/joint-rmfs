"""PPS mode normalization helpers."""

from .types import PPSMode


def normalize_pps_mode(mode: str) -> str:
    """Normalize user input or environment variable into a canonical PPSMode string."""
    mode_str = str(mode).strip().lower().replace("-", "_").replace(" ", "_")
    if mode_str in {"ppo", "rl", "pps_rl", "ppo_pps"}:
        return PPSMode.PPO.value
    if mode_str in {"random", "random_pps", "untrained", "untrained_ppo", "untrained_ppo_pps"}:
        return PPSMode.RANDOM.value
    if mode_str in {"rika", "rika_pps", "heuristic", "pile_on", "pileon", "baseline"}:
        return PPSMode.HEURISTIC.value
    if mode_str in {"demand", "demand_pps"}:
        return PPSMode.DEMAND.value
    return PPSMode.HEURISTIC.value
