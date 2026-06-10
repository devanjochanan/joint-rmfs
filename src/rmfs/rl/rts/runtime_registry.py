"""Process-local registry for RTS rollout runtime options."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

from .runtime_config import RTSRuntimeConfig, validate_rts_runtime_config


_CONFIG = RTSRuntimeConfig()
_RUNTIME_ROOT: Path | None = None


def configure_rts_runtime(
    config: RTSRuntimeConfig | Mapping | None,
    runtime_root: Path | None = None,
) -> None:
    global _CONFIG, _RUNTIME_ROOT
    if config is None:
        resolved = RTSRuntimeConfig()
    elif isinstance(config, RTSRuntimeConfig):
        resolved = config
        validate_rts_runtime_config(resolved)
    else:
        resolved = RTSRuntimeConfig.from_dict(config)
    _CONFIG = resolved
    _RUNTIME_ROOT = Path(runtime_root) if runtime_root is not None else None


def get_rts_runtime_config() -> RTSRuntimeConfig:
    return _CONFIG


def get_rts_runtime_root() -> Path | None:
    return _RUNTIME_ROOT


def reset_rts_runtime() -> None:
    configure_rts_runtime(None, None)

