"""Runtime policy helpers for optional detail SQLite tables.

The RMFS simulator has several SQLite tables that are useful for debugging and
inspection, but expensive for headless training and ablation. This module keeps
that policy explicit while preserving the semantic pod-location read path with
an in-memory mirror when detail DB writes are disabled.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from threading import Lock
from typing import Any

from .timing import timed


TRUE_VALUES = {"1", "true", "yes", "on"}
FALSE_VALUES = {"0", "false", "no", "off"}

_DETAIL_DB_ENABLED: bool | None = None
_DETAIL_DB_PATH: Path | None = None
_POD_LOCATION_CACHE: dict[tuple[str, str], tuple[int, int]] = {}
_LOCK = Lock()


def _normalize_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    normalized = str(value).strip().lower()
    if normalized in TRUE_VALUES:
        return True
    if normalized in FALSE_VALUES:
        return False
    return default


def env_detail_db_enabled(default: bool = True) -> bool:
    return _normalize_bool(os.environ.get("RMFS_DETAIL_DB"), default)


def configure_detail_db(
    enabled: bool | None = None,
    db_path: str | Path | None = None,
    default: bool = True,
) -> bool:
    """Configure detail DB writes for the current process."""

    global _DETAIL_DB_ENABLED, _DETAIL_DB_PATH
    _DETAIL_DB_ENABLED = env_detail_db_enabled(default) if enabled is None else bool(enabled)
    _DETAIL_DB_PATH = Path(db_path) if db_path is not None else None
    return _DETAIL_DB_ENABLED


def set_detail_db_enabled(enabled: bool) -> bool:
    return configure_detail_db(enabled=enabled, db_path=_DETAIL_DB_PATH)


def is_detail_db_enabled(default: bool = True) -> bool:
    if _DETAIL_DB_ENABLED is not None:
        return _DETAIL_DB_ENABLED
    return env_detail_db_enabled(default)


def is_detail_db_configured() -> bool:
    return _DETAIL_DB_ENABLED is not None


def resolve_db_path(db_path: str | Path = "warehouse.db") -> str:
    if str(db_path) == "warehouse.db" and _DETAIL_DB_PATH is not None:
        return str(_DETAIL_DB_PATH)
    return str(db_path)


def connect(db_path: str | Path = "warehouse.db") -> sqlite3.Connection:
    with timed("sqlite_connect"):
        return sqlite3.connect(resolve_db_path(db_path))


def execute(cursor: sqlite3.Cursor, statement: str, params: tuple[Any, ...] | None = None):
    with timed("sqlite_execute"):
        if params is None:
            return cursor.execute(statement)
        return cursor.execute(statement, params)


def commit(conn: sqlite3.Connection) -> None:
    with timed("sqlite_commit"):
        conn.commit()


def _cache_key(db_path: str | Path, pod_id: str) -> tuple[str, str]:
    return (resolve_db_path(db_path), str(pod_id))


def reset_pod_location_cache(db_path: str | Path = "warehouse.db") -> None:
    target = resolve_db_path(db_path)
    with _LOCK:
        for key in [key for key in _POD_LOCATION_CACHE if key[0] == target]:
            del _POD_LOCATION_CACHE[key]


def record_pod_location(
    pod_id: str,
    x: int | float,
    y: int | float,
    db_path: str | Path = "warehouse.db",
) -> None:
    with _LOCK:
        _POD_LOCATION_CACHE[_cache_key(db_path, str(pod_id))] = (int(x), int(y))


def get_cached_pod_location(
    pod_id: str,
    db_path: str | Path = "warehouse.db",
) -> tuple[int, int] | None:
    with _LOCK:
        return _POD_LOCATION_CACHE.get(_cache_key(db_path, str(pod_id)))


def debug_log(message: str) -> None:
    if (
        os.environ.get("RMFS_DEBUG", "").strip().lower() in TRUE_VALUES
        or os.environ.get("RMFS_DEBUG_DETAIL_DB", "").strip().lower() in TRUE_VALUES
    ):
        print(message)
