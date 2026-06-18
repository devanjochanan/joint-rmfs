#!/usr/bin/env python3
"""Validate PPS-RL model loader/inference plumbing with an artificial PPO model."""

from __future__ import annotations

import contextlib
import io
import os
from pathlib import Path
import shutil
import sys
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.rmfs.decisions.pps import runtime as pps_runtime


ARTIFACT_ROOT = REPO_ROOT / "data" / "runtime" / "pps_rl_artificial_model_smoke"


class ArtificialPPSEnv:
    def __init__(self):
        try:
            import gymnasium as gym
            from gymnasium import spaces
        except ImportError as exc:
            raise SystemExit(
                "pps_rl_artificial_model_smoke requires gymnasium. "
                "Install project PPS-RL dependencies before running this smoke."
            ) from exc

        class _Env(gym.Env):
            metadata = {"render_modes": []}

            def __init__(self) -> None:
                super().__init__()
                self.observation_space = spaces.Dict(
                    {
                        "pod_features": spaces.Box(
                            low=0.0,
                            high=1.0,
                            shape=(60, 511),
                            dtype=np.float32,
                        ),
                        "station_features": spaces.Box(
                            low=0.0,
                            high=1.0,
                            shape=(3, 500),
                            dtype=np.float32,
                        ),
                        "num_candidates": spaces.Box(
                            low=0,
                            high=60,
                            shape=(1,),
                            dtype=np.int32,
                        ),
                        "zone_robot_counts": spaces.Box(
                            low=0.0,
                            high=100.0,
                            shape=(5,),
                            dtype=np.float32,
                        ),
                    }
                )
                self.action_space = spaces.MultiDiscrete([4] * 60)

            def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
                super().reset(seed=seed)
                return dummy_observation(), {}

            def step(self, action):
                return dummy_observation(), 0.0, True, False, {}

        self.env = _Env()


def dummy_observation() -> dict[str, np.ndarray]:
    return {
        "pod_features": np.zeros((60, 511), dtype=np.float32),
        "station_features": np.zeros((3, 500), dtype=np.float32),
        "num_candidates": np.asarray([0], dtype=np.int32),
        "zone_robot_counts": np.zeros((5,), dtype=np.float32),
    }


def reset_loader_cache() -> None:
    pps_runtime._PPS_RL_MODEL = None
    pps_runtime._PPS_RL_MODEL_SOURCE = None
    pps_runtime._PPS_RL_LOAD_ATTEMPTED = False
    pps_runtime._PPS_RL_LOAD_ATTEMPTED_FOR = None
    pps_runtime._PPS_RL_ACTIVE_LOGGED = False


def main() -> int:
    old_env = {
        key: os.environ.get(key)
        for key in ("PPS_MODE", "PPS_RL_MODEL_PATH", "PPS_RL_ENABLED", "MPLCONFIGDIR")
    }
    shutil.rmtree(ARTIFACT_ROOT, ignore_errors=True)
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(ARTIFACT_ROOT / "matplotlib"))
    try:
        from stable_baselines3 import PPO
    except ImportError as exc:
        raise SystemExit(
            "pps_rl_artificial_model_smoke requires stable-baselines3. "
            "Install project PPS-RL dependencies before running this smoke."
        ) from exc

    try:
        env = ArtificialPPSEnv().env
        model_base = ARTIFACT_ROOT / "artificial_pps_model"
        model = PPO("MultiInputPolicy", env, n_steps=2, batch_size=2, n_epochs=1, seed=7, verbose=0)
        model.save(model_base)

        reset_loader_cache()
        os.environ["PPS_MODE"] = "heuristic"
        assert pps_runtime.get_pps_mode() == "heuristic"

        os.environ["PPS_MODE"] = "ppo"
        os.environ["PPS_RL_ENABLED"] = "1"
        os.environ["PPS_RL_MODEL_PATH"] = str(model_base)
        assert pps_runtime.get_pps_mode() == "ppo"
        loaded = pps_runtime.load_pps_rl_model()
        assert loaded is not None, "artificial PPS-RL model did not load"
        actions, _state = loaded.predict(dummy_observation(), deterministic=True)
        actions = np.asarray(actions)
        assert actions.shape == (60,), f"unexpected PPS-RL action shape: {actions.shape}"
        assert actions.min() >= 0 and actions.max() <= 3, "PPS-RL action values outside MultiDiscrete bounds"

        os.environ["PPS_RL_MODEL_PATH"] = str(ARTIFACT_ROOT / "missing_pps_model")
        with contextlib.redirect_stdout(io.StringIO()) as captured:
            missing = pps_runtime.load_pps_rl_model()
        message = captured.getvalue()
        assert missing is None
        assert "Model not found" in message and "Using heuristic PPS" in message

        print("pps rl artificial model smoke ok")
        print("validated loader/inference plumbing only; trained PPS-RL quality is not validated")
        return 0
    finally:
        reset_loader_cache()
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        shutil.rmtree(ARTIFACT_ROOT, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
