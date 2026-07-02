"""PPS reinforcement learning runtime, model loading, and state management."""

from __future__ import annotations

import hashlib
import os
import pickle
import traceback
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .types import PPSMode
from .modes import normalize_pps_mode
from .model_paths import DEFAULT_PPS_MODEL_PATH, configured_pps_model_path, pps_model_candidates


PPS_RL_NUM_STATIONS = 3
PPS_RL_TOP_K_SKUS = 500
PPS_RL_MAX_PODS = 60
PPS_RL_NUM_TRAFFIC_ZONES = 5
PPS_RL_MAX_ZONE_ROBOT_COUNT = 100.0
PPS_RL_TRAFFIC_ZONES = (
    ((5, 43), (0, 5)),
    ((5, 43), (6, 11)),
    ((5, 43), (12, 18)),
    ((5, 43), (19, 24)),
    ((5, 43), (25, 30)),
)
PPS_RL_POD_FEATURE_DIM = (
    PPS_RL_TOP_K_SKUS
    + PPS_RL_NUM_STATIONS
    + PPS_RL_NUM_STATIONS
    + PPS_RL_NUM_TRAFFIC_ZONES
)
PPS_RL_MODEL_PATH = os.environ.get(
    "PPS_RL_MODEL_PATH",
    str(DEFAULT_PPS_MODEL_PATH),
)

_PPS_RL_MODEL = None
_PPS_RL_MODEL_SOURCE = None
_PPS_RL_LOAD_ATTEMPTED = False
_PPS_RL_LOAD_ATTEMPTED_FOR = None
_PPS_RL_ACTIVE_LOGGED = False
_PPS_MODE = normalize_pps_mode(os.environ.get("PPS_MODE", "heuristic"))


def get_pps_mode() -> str:
    """Return the current PPS mode."""
    env_mode = os.environ.get("PPS_MODE")
    if env_mode is not None and str(env_mode).strip():
        return normalize_pps_mode(env_mode)
    return normalize_pps_mode(_PPS_MODE)


def is_pps_rl_enabled() -> bool:
    """Check if PPS RL (PPO) is enabled based on current mode and env vars."""
    value = os.environ.get("PPS_RL_ENABLED", "1").strip().lower()
    return get_pps_mode() == "ppo" and value not in {"0", "false", "no", "off"}


def _pps_rl_model_candidates() -> list[str]:
    return [str(path) for path in pps_model_candidates(configured_pps_model_path())]


def _ensure_numpy_pickle_compat() -> None:
    """Allow NumPy 2.x SB3 archives to load in older NumPy 1.x environments."""
    try:
        import importlib
        import sys
        import numpy.core as numpy_core
    except ImportError:
        return

    sys.modules.setdefault("numpy._core", numpy_core)
    for submodule in (
        "multiarray",
        "umath",
        "numeric",
        "fromnumeric",
        "_multiarray_umath",
        "_methods",
        "overrides",
    ):
        try:
            module = importlib.import_module(f"numpy.core.{submodule}")
        except Exception:
            continue
        sys.modules.setdefault(f"numpy._core.{submodule}", module)


def _pps_rl_model_matches_current_observation(model) -> bool:
    spaces = getattr(getattr(model, "observation_space", None), "spaces", None)
    if not spaces:
        return False

    required = {
        "pod_features",
        "station_features",
        "num_candidates",
        "zone_robot_counts",
    }
    if not required.issubset(spaces.keys()):
        return False

    return (
        spaces["pod_features"].shape == (PPS_RL_MAX_PODS, PPS_RL_POD_FEATURE_DIM)
        and spaces["station_features"].shape == (
            PPS_RL_NUM_STATIONS,
            PPS_RL_TOP_K_SKUS,
        )
        and spaces["num_candidates"].shape == (1,)
        and spaces["zone_robot_counts"].shape == (PPS_RL_NUM_TRAFFIC_ZONES,)
    )


def load_pps_rl_model():
    """Load the PPO model from the configured path, with NumPy compat fallback."""
    global _PPS_RL_MODEL, _PPS_RL_MODEL_SOURCE, _PPS_RL_LOAD_ATTEMPTED, _PPS_RL_LOAD_ATTEMPTED_FOR

    configured_path = str(configured_pps_model_path())
    candidates = _pps_rl_model_candidates()
    if _PPS_RL_MODEL is not None and _PPS_RL_MODEL_SOURCE in candidates:
        return _PPS_RL_MODEL
    if _PPS_RL_MODEL is not None and _PPS_RL_MODEL_SOURCE not in candidates:
        _PPS_RL_MODEL = None
        _PPS_RL_MODEL_SOURCE = None
    if not is_pps_rl_enabled():
        return None
    if _PPS_RL_LOAD_ATTEMPTED and _PPS_RL_LOAD_ATTEMPTED_FOR == configured_path:
        return None

    _PPS_RL_LOAD_ATTEMPTED = True
    _PPS_RL_LOAD_ATTEMPTED_FOR = configured_path
    model_path = next((path for path in candidates if os.path.exists(path)), None)
    if model_path is None:
        print(f"[PPS_RL] Model not found at {configured_path}. Using heuristic PPS.")
        return None

    try:
        from stable_baselines3 import PPO
        _ensure_numpy_pickle_compat()
        model = PPO.load(model_path, device="cpu")
        if not _pps_rl_model_matches_current_observation(model):
            print(
                "[PPS_RL] Loaded model uses the old observation shape. "
                "Retrain PPO after the traffic-zone feature update."
            )
            return None
        _PPS_RL_MODEL = model
        _PPS_RL_MODEL_SOURCE = str(model_path)
        print(f"[PPS_RL] Loaded PPO model from {model_path}")
        return _PPS_RL_MODEL
    except Exception:
        print("[PPS_RL] Failed to load PPO model. Using heuristic PPS.")
        traceback.print_exc()
        return None


def load_pps_rl_model_strict(
    model_path: str | os.PathLike[str],
    expected_sha256: Optional[str] = None,
) -> object:
    """Load a PPS PPO model with zero silent fallback.

    Raises on missing file, SHA-256 mismatch, observation-shape mismatch,
    or any load error — never falls back to heuristic.
    """
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"PPS model not found: {path}")

    if expected_sha256 is not None:
        sha256 = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(8192), b""):
                sha256.update(chunk)
        actual = sha256.hexdigest()
        if actual != expected_sha256:
            raise ValueError(
                f"SHA-256 mismatch for {path}: "
                f"expected {expected_sha256}, got {actual}"
            )

    from stable_baselines3 import PPO

    _ensure_numpy_pickle_compat()
    model = PPO.load(str(path), device="cpu")

    if not _pps_rl_model_matches_current_observation(model):
        raise ValueError(
            f"Observation space mismatch: model from {path} does not match "
            f"current PPS constants (pods={PPS_RL_MAX_PODS}, "
            f"features={PPS_RL_POD_FEATURE_DIM}, stations={PPS_RL_NUM_STATIONS}, "
            f"zones={PPS_RL_NUM_TRAFFIC_ZONES})"
        )

    return model


def build_pps_rl_sku_index(
    universe,
    items_csv: Optional[str] = None,
    skus_data_csv: Optional[str] = None,
) -> dict:
    """Build a mapping from SKU ID to index for the model observation space."""
    sku_ids = []
    for csv_file, column in ((items_csv, "item_id"), (skus_data_csv, "item_id")):
        if not csv_file or not os.path.exists(csv_file):
            continue
        try:
            df = pd.read_csv(csv_file, usecols=[column])
            sku_ids = sorted(df[column].dropna().astype(int).unique().tolist())
            if sku_ids:
                break
        except Exception:
            sku_ids = []

    if not sku_ids:
        sku_set = set()
        for pod in universe.pod_manager.pods:
            for sku in pod.skus.keys():
                try:
                    sku_set.add(int(sku))
                except (TypeError, ValueError):
                    sku_set.add(sku)
        sku_ids = sorted(sku_set)

    return {sku: i for i, sku in enumerate(sku_ids[:PPS_RL_TOP_K_SKUS])}


def configure_pps_rl_strategy(
    universe,
    items_csv: Optional[str] = None,
    skus_data_csv: Optional[str] = None,
) -> bool:
    """Enable PPO PPS when the model is available; otherwise keep heuristic PPS."""
    global _PPS_RL_ACTIVE_LOGGED
    pps_mode = get_pps_mode()

    if pps_mode == "heuristic":
        universe.pps_rl = False
        universe.pps_rl_random = False
        universe.pps_pileon = True
        universe.pps_demand = False
        return False

    if pps_mode == "demand":
        universe.pps_rl = False
        universe.pps_rl_random = False
        universe.pps_pileon = False
        universe.pps_demand = True
        return False

    if pps_mode == "random":
        universe.pps_pileon = False
        universe.pps_demand = False
        universe.pps_rl = True
        universe.pps_rl_random = True
        if not hasattr(universe, "pps_picked_quantity"):
            universe.pps_picked_quantity = 0
        if not hasattr(universe, "pps_pod_visits"):
            universe.pps_pod_visits = 0
        if not _PPS_RL_ACTIVE_LOGGED:
            print("[PPS_RANDOM] NetLogo simulation is using random PPO-style PPS.")
            _PPS_RL_ACTIVE_LOGGED = True
        return True

    if load_pps_rl_model() is None:
        universe.pps_rl = False
        universe.pps_rl_random = False
        universe.pps_pileon = True
        universe.pps_demand = False
        return False

    universe.pps_pileon = False
    universe.pps_demand = False
    universe.pps_rl = True
    universe.pps_rl_random = False

    if not hasattr(universe, "pps_rl_sku_index"):
        universe.pps_rl_sku_index = build_pps_rl_sku_index(universe, items_csv, skus_data_csv)
    if not hasattr(universe, "pps_picked_quantity"):
        universe.pps_picked_quantity = 0
    if not hasattr(universe, "pps_pod_visits"):
        universe.pps_pod_visits = 0

    if not _PPS_RL_ACTIVE_LOGGED:
        print("[PPS_RL] NetLogo simulation is using PPO for pick-pod selection.")
        _PPS_RL_ACTIVE_LOGGED = True
    return True


def runtime_set_pps_mode(
    mode: str,
    state_file_path: Optional[str] = None,
    items_csv: Optional[str] = None,
    skus_data_csv: Optional[str] = None,
) -> str:
    """Switch PPS mode at runtime and reconfigure strategy if state file exists."""
    global _PPS_MODE, _PPS_RL_LOAD_ATTEMPTED, _PPS_RL_ACTIVE_LOGGED, _PPS_RL_MODEL

    _PPS_MODE = normalize_pps_mode(mode)
    os.environ["PPS_MODE"] = _PPS_MODE
    if _PPS_MODE == "ppo" and _PPS_RL_MODEL is None:
        _PPS_RL_LOAD_ATTEMPTED = False
    _PPS_RL_ACTIVE_LOGGED = False

    if state_file_path and os.path.exists(state_file_path):
        try:
            with open(state_file_path, "rb") as file:
                universe = pickle.load(file)
            for obj in universe._objects:
                obj.setUniverse(universe)
            configure_pps_rl_strategy(universe, items_csv, skus_data_csv)
            with open(state_file_path, "wb") as file:
                pickle.dump(universe, file)
        except Exception:
            traceback.print_exc()

    print(f"[PPS_MODE] Current PPS mode: {_PPS_MODE}")
    return _PPS_MODE
