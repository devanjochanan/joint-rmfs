"""Fixed physical macro-region features for RTS-RL."""

from __future__ import annotations

from typing import Any

from .static_state_context import get_or_build_static_state_context


MACRO_REGION_SEMANTIC_VERSION = "rts_macro_region_3x3.v1"
TRAFFIC_PRESSURE_SEMANTIC_VERSION = "rts_bounded_robot_pressure.v2"


def macro_region_id_for_object(context: Any, obj: Any | None) -> str:
    if obj is None:
        return ""
    warehouse = getattr(context, "warehouse", context)
    static_context = get_or_build_static_state_context(warehouse)
    x_norm = static_context.norm_x(getattr(obj, "pos_x", getattr(obj, "x", 0.0)))
    y_norm = static_context.norm_y(getattr(obj, "pos_y", getattr(obj, "y", 0.0)))
    col = _bucket_3(x_norm)
    row = _bucket_3(y_norm)
    return f"mr3x3_r{row}_c{col}"


def macro_region_pressures(context: Any, target: Any | None) -> dict[str, float | str]:
    target_region = macro_region_id_for_object(context, target)
    robots = [
        obj
        for obj in getattr(getattr(context, "warehouse", None), "_objects", []) or []
        if str(getattr(obj, "object_type", "")).lower() == "robot"
    ]
    denominator = max(1, len(robots))
    present = sum(1 for robot in robots if macro_region_id_for_object(context, robot) == target_region)
    destination = sum(
        1
        for robot in robots
        if getattr(robot, "destination", None) is not None
        and macro_region_id_for_object(context, getattr(robot, "destination", None)) == target_region
    )
    return {
        "macro_region_id": target_region,
        "macro_region_present_robot_count": float(present),
        "macro_region_destination_robot_count": float(destination),
        "macro_region_robot_pressure_denominator": float(denominator),
        "macro_region_present_robot_pressure": _pressure(present, denominator),
        "macro_region_destination_robot_pressure": _pressure(destination, denominator),
    }


def macro_region_metadata() -> dict[str, str]:
    return {
        "macro_region_semantic_version": MACRO_REGION_SEMANTIC_VERSION,
        "macro_region_grid": "fixed normalized warehouse coordinates, 3x3 physical grid",
        "traffic_pressure_semantic_version": TRAFFIC_PRESSURE_SEMANTIC_VERSION,
        "traffic_pressure_normalization": "bounded robot count divided by active robot denominator; zone denominator is capped by zone storage capacity",
    }


def _bucket_3(value: Any) -> int:
    try:
        normalized = float(value)
    except Exception:
        normalized = 0.0
    normalized = max(0.0, min(1.0, normalized))
    return min(2, int(normalized * 3.0))


def _pressure(count: int, denominator: int) -> float:
    return max(0.0, min(1.0, float(count) / float(max(1, denominator))))
