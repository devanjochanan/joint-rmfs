"""Small debug-print helper for hot-loop simulator paths."""

from __future__ import annotations

import os
from typing import Any


TRUE_VALUES = {"1", "true", "yes", "on"}


def debug_enabled(*extra_env: str) -> bool:
    names = ("RMFS_DEBUG", *extra_env)
    return any(os.environ.get(name, "").strip().lower() in TRUE_VALUES for name in names)


def debug_print(*args: Any, **kwargs: Any) -> None:
    if debug_enabled():
        print(*args, **kwargs)


def pps_debug_print(*args: Any, **kwargs: Any) -> None:
    if debug_enabled("PPS_ENV_VERBOSE"):
        print(*args, **kwargs)
