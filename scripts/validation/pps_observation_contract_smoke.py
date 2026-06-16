"""PPS observation/action contract smoke checks.

This does not reset the environment or run training; it validates static shape
and default-selection contracts that future deduplication must preserve.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.rmfs.rl.pps.env import (
    MAX_PODS_OBS,
    NUM_ACTIONS,
    NUM_STATIONS,
    NUM_TRAFFIC_ZONES,
    PPSEnv,
    TOP_K_SKUS,
)
from src.rmfs.rl.pps.model_paths import get_default_pps_model_path


def main() -> int:
    env = PPSEnv(max_episode_ticks=1)

    expected_pod_feature_dim = TOP_K_SKUS + NUM_STATIONS + NUM_STATIONS + NUM_TRAFFIC_ZONES
    assert env.observation_space["pod_features"].shape == (
        MAX_PODS_OBS,
        expected_pod_feature_dim,
    )
    assert env.observation_space["station_features"].shape == (NUM_STATIONS, TOP_K_SKUS)
    assert env.observation_space["num_candidates"].shape == (1,)
    assert env.observation_space["zone_robot_counts"].shape == (NUM_TRAFFIC_ZONES,)
    assert env.action_space.nvec.tolist() == [NUM_ACTIONS] * MAX_PODS_OBS

    model_path = get_default_pps_model_path()
    expected_suffix = Path("data") / "models" / "pps" / "pps_rl_best.zip"
    assert model_path.parts[-len(expected_suffix.parts):] == expected_suffix.parts

    import netlogo

    assert netlogo.set_pps_mode("heuristic") == "heuristic"
    assert netlogo.set_pps_mode("ppo") == "ppo"
    assert netlogo.set_pps_mode("heuristic") == "heuristic"

    print("pps observation contract smoke ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
