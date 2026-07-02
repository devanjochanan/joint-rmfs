"""Read-only storage resolver for RTS evaluation policies."""

from __future__ import annotations

from typing import Any

from .action_context import select_candidate_storage
from .graph_distance import graph_distance_or_fallback
from .static_runtime_index import get_static_runtime_index
from .zone_registry import build_zone_registry


def find_free_storage_in_zone(context: Any, zone_id: str, branch: str) -> Any | None:
    storage, feasibility = select_candidate_storage(context, str(zone_id))
    if storage is not None and feasibility.available and feasibility.reachable:
        return storage
    warehouse = getattr(context, "warehouse", None)
    registry_obj = getattr(warehouse, "committed_next_registry", None)
    robot = getattr(context, "robot", None)
    if registry_obj is not None and robot is not None:
        proposal = registry_obj.get_action_proposal(robot, branch, zone_id)
        proposed_storage = getattr(proposal, "candidate_storage", None)
        if proposed_storage is not None and _available(proposed_storage):
            return proposed_storage
    storage_manager = getattr(warehouse, "storage_manager", None)
    static_index = get_static_runtime_index(warehouse)
    if static_index is not None:
        candidates = [
            storage for storage in getattr(static_index, "storages_by_zone", {}).get(str(zone_id), ())
            if _available(storage)
        ]
    else:
        storages = list(getattr(storage_manager, "storages", []) or [])
        registry = build_zone_registry(context, (str(zone_id),))
        candidates = [
            storage
            for storage in storages
            if registry.zone_id_for_storage(storage) == str(zone_id)
            and _available(storage)
        ]
    if not candidates:
        return None
    origin = _origin(context)
    return sorted(candidates, key=lambda storage: (_distance(context, storage, origin), _coord(storage)))[0]


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


def _available(storage: Any) -> bool:
    return bool(getattr(storage, "is_empty", False)) and getattr(storage, "assigned_pod", None) is None


def _coord(storage: Any) -> tuple[float, float]:
    return (float(getattr(storage, "pos_x", 0.0)), float(getattr(storage, "pos_y", 0.0)))


def _distance(context: Any, storage: Any, origin: tuple[float, float]) -> float:
    warehouse = getattr(context, "warehouse", None)
    result = graph_distance_or_fallback(warehouse, _Coord(*origin), storage)
    if result.distance is not None:
        return float(result.distance)
    x, y = _coord(storage)
    return abs(x - origin[0]) + abs(y - origin[1])


class _Coord:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y
