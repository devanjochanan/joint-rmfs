"""Read-only storage resolver for RTS evaluation policies."""

from __future__ import annotations

from typing import Any

from .zone_features import infer_zone_id


def find_free_storage_in_zone(context: Any, zone_id: str, branch: str) -> Any | None:
    warehouse = getattr(context, "warehouse", None)
    storage_manager = getattr(warehouse, "storage_manager", None)
    storages = list(getattr(storage_manager, "storages", []) or [])
    candidates = [
        storage
        for storage in storages
        if infer_zone_id(storage) == str(zone_id)
        and bool(getattr(storage, "is_empty", False))
        and getattr(storage, "assigned_pod", None) is None
    ]
    if not candidates:
        return None
    origin = _origin(context)
    return sorted(candidates, key=lambda storage: (_distance(storage, origin), _coord(storage)))[0]


def _origin(context: Any) -> tuple[float, float]:
    station = getattr(context, "station", None)
    if station is not None and _has_coord(station):
        return (float(getattr(station, "pos_x")), float(getattr(station, "pos_y")))
    robot = getattr(context, "robot", None)
    if robot is not None and _has_coord(robot):
        return (float(getattr(robot, "pos_x")), float(getattr(robot, "pos_y")))
    return (0.0, 0.0)


def _has_coord(obj: Any) -> bool:
    return getattr(obj, "pos_x", None) is not None and getattr(obj, "pos_y", None) is not None


def _coord(storage: Any) -> tuple[float, float]:
    return (float(getattr(storage, "pos_x", 0.0)), float(getattr(storage, "pos_y", 0.0)))


def _distance(storage: Any, origin: tuple[float, float]) -> float:
    x, y = _coord(storage)
    return abs(x - origin[0]) + abs(y - origin[1])

