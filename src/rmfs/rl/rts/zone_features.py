"""Zone-row feature extraction for RTS-RL."""

from __future__ import annotations

from typing import Any, Sequence


def infer_zone_id(obj: Any) -> str:
    for attr in ("zone_id", "zone", "storage_zone"):
        value = getattr(obj, attr, None)
        if value is not None:
            return str(value)
    x = getattr(obj, "pos_x", 0)
    return f"col_{int(x) // 10 if _is_number(x) else 0}"


def build_zone_rows(
    context: Any,
    zone_ids: Sequence[str],
    *,
    replenishment_signal_active: bool = False,
    replenishment_station_available: bool = False,
) -> tuple[list[dict[str, float | str]], list[str]]:
    warehouse = getattr(context, "warehouse", None)
    storage_manager = getattr(warehouse, "storage_manager", None)
    storages = list(getattr(storage_manager, "storages", []) or [])
    robots = list(getattr(warehouse, "_objects", []) or [])
    warnings: list[str] = []
    rows = []
    zone_ids = tuple(str(zone_id) for zone_id in zone_ids)
    if not storages:
        warnings.append("storage_manager.storages unavailable; zone occupancy defaults to zero")
    for index, zone_id in enumerate(zone_ids):
        zone_storages = [storage for storage in storages if infer_zone_id(storage) == zone_id]
        free = [storage for storage in zone_storages if bool(getattr(storage, "is_empty", False)) and getattr(storage, "assigned_pod", None) is None]
        total = len(zone_storages)
        present_robot_count = sum(1 for robot in robots if infer_zone_id(robot) == zone_id)
        replenish_valid = bool(free) and replenishment_signal_active and replenishment_station_available
        rows.append(
            {
                "zone_id": zone_id,
                "zone_row_index": float(index),
                "zone_col_index": float(index),
                "occupation_level": 1.0 - (float(len(free)) / float(total)) if total else 0.0,
                "free_slot_count": float(len(free)),
                "zone_destination_robot_count": 0.0,
                "neighbor_zone_destination_robot_count": 0.0,
                "superzone_destination_robot_count": 0.0,
                "zone_present_robot_count": float(present_robot_count),
                "neighbor_zone_present_robot_count": 0.0,
                "superzone_present_robot_count": float(max(0, len(robots) - present_robot_count)),
                "storage_cycle_time_estimate": 0.0,
                "replenish_cycle_time_estimate": 0.0,
                "sku_similarity": 0.0,
                "candidate_zone_to_selected_replenishment_station_distance": 0.0,
                "candidate_zone_to_nearest_replenishment_station_distance": 0.0,
                "store_action_valid": 1.0 if free else 0.0,
                "replenish_store_action_valid": 1.0 if replenish_valid else 0.0,
            }
        )
    return rows, warnings


def _is_number(value: Any) -> bool:
    try:
        float(value)
        return True
    except Exception:
        return False
