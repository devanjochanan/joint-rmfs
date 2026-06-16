"""Path helpers for PPS RL model artifacts."""

from __future__ import annotations

import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_PPS_MODEL_PATH = REPO_ROOT / "data" / "models" / "pps" / "pps_rl_best.zip"


def configured_pps_model_path() -> Path:
    """Return the configured PPS model path, defaulting to data/models/pps."""
    return Path(os.environ.get("PPS_RL_MODEL_PATH", str(DEFAULT_PPS_MODEL_PATH)))


def pps_model_candidates(path: str | os.PathLike[str] | None = None) -> list[Path]:
    """Return candidate SB3 archive paths, accepting values with or without .zip."""
    configured = Path(path) if path is not None else configured_pps_model_path()
    candidates = [configured]
    if configured.suffix != ".zip":
        candidates.append(configured.with_suffix(configured.suffix + ".zip"))
    return candidates
