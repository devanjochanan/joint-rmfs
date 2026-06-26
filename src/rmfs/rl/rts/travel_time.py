"""RTS travel-time conversion helpers.

The simulator moves robots through two directed topologies: empty robots use
``warehouse.graph`` and robots carrying pods use ``warehouse.graph_pod``.
These helpers keep that distinction explicit for RTS cycle estimates without
changing simulator motion.
"""

from __future__ import annotations

import math
from typing import Any


EMPTY_ROBOT = "empty_robot"
LOADED_ROBOT = "loaded_robot"
ROBOT_TOPOLOGIES = (EMPTY_ROBOT, LOADED_ROBOT)
TRAVEL_TIME_VERSION = "rts_travel_time.v1"
TIME_CONVERSION_VERSION = "warehouse_distance_over_robot_speed_seconds.v1"


def graph_wrapper_for_topology(warehouse: Any, topology: str) -> Any | None:
    normalized = str(topology)
    if normalized == EMPTY_ROBOT:
        return getattr(warehouse, "graph", None)
    if normalized == LOADED_ROBOT:
        return getattr(warehouse, "graph_pod", None)
    raise ValueError(f"unsupported RTS robot topology: {topology!r}")


def nominal_robot_speed(robot: Any) -> float | None:
    speed = getattr(robot, "maximum_speed", None)
    try:
        value = float(speed)
    except Exception:
        return None
    if not math.isfinite(value) or value <= 0.0:
        return None
    return value


def distance_to_seconds(distance: float | int | None, speed: float | int | None) -> float | None:
    if distance is None or speed is None:
        return None
    try:
        dist_value = float(distance)
        speed_value = float(speed)
    except Exception:
        return None
    if not math.isfinite(dist_value) or not math.isfinite(speed_value) or speed_value <= 0.0 or dist_value < 0.0:
        return None
    return float(dist_value / speed_value)


def backend_steps_to_seconds(warehouse: Any, steps: float | int | None) -> float | None:
    if steps is None:
        return None
    tick_to_second = getattr(warehouse, "tick_to_second", None)
    try:
        step_value = float(steps)
        tick_value = float(tick_to_second)
    except Exception:
        return None
    if not math.isfinite(step_value) or not math.isfinite(tick_value) or tick_value <= 0.0:
        return None
    return float(max(0.0, step_value) * tick_value)
