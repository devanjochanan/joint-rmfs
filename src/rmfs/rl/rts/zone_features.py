"""Zone-row feature extraction for RTS-RL."""

from __future__ import annotations

from typing import Any, Sequence

from .graph_distance import (
    DISTANCE_STATUS_FALLBACK,
    distance_cache_metadata,
    graph_cycle_distance_or_fallback,
    graph_distance_or_fallback,
)
from .zone_registry import RTSZoneRegistry, build_zone_registry

SKU_SIMILARITY_VERSION = "rts_positive_sku_similarity.v1"


class _Coord:
    def __init__(self, x: Any, y: Any):
        self.x = x
        self.y = y


def infer_zone_id(obj: Any) -> str:
    for attr in ("rts_zone_id", "zone_id", "zone", "storage_zone"):
        value = getattr(obj, attr, None)
        if value is not None:
            return str(value)
    return ""


def infer_coordinate_zone_id(
    coord: Any,
    zone_ids: Sequence[str],
    *,
    registry: RTSZoneRegistry | None = None,
) -> str:
    for attr in ("rts_zone_id", "zone_id", "zone", "storage_zone"):
        val = getattr(coord, attr, None)
        if val is not None and str(val) in set(str(zone_id) for zone_id in zone_ids):
            return str(val)
    if registry is not None:
        zone_id = registry.zone_id_for_coordinate(coord)
        if zone_id in set(str(zone_id) for zone_id in zone_ids):
            return zone_id
    return ""


def infer_pressure_zone_id(
    coord: Any,
    zone_ids: Sequence[str],
    *,
    registry: RTSZoneRegistry,
) -> str:
    pair = _coordinate_pair(coord)
    if pair is None:
        return ""
    zone_id_set = set(str(zone_id) for zone_id in zone_ids)
    exact_zone_id = registry.coordinate_to_zone_id.get(pair)
    if exact_zone_id in zone_id_set:
        return str(exact_zone_id)

    x, y = pair
    candidates: list[str] = []
    for zone_id in zone_id_set:
        zone_info = registry.zones_by_id.get(zone_id)
        if zone_info is None:
            continue
        if any(abs(int(storage_x) - x) + abs(int(storage_y) - y) == 1 for storage_x, storage_y in zone_info.coordinates):
            candidates.append(zone_id)

    # A pathway coordinate can touch two zones; lexicographic order keeps the pressure owner deterministic.
    return sorted(candidates)[0] if candidates else ""


def build_zone_rows(
    context: Any,
    zone_ids: Sequence[str],
    *,
    replenishment_signal_active: bool = False,
    replenishment_station_available: bool = False,
) -> tuple[list[dict[str, float | str]], list[str]]:
    warehouse = getattr(context, "warehouse", None)
    pod = getattr(context, "pod", None)
    station = getattr(context, "station", None)
    
    storage_manager = getattr(warehouse, "storage_manager", None)
    storages = list(getattr(storage_manager, "storages", []) or [])
    robots = [obj for obj in getattr(warehouse, "_objects", []) or [] if _is_robot_object(obj)]
    active_robot_denominator = max(1, len(robots))
    warnings: list[str] = []
    rows = []
    registry = build_zone_registry(context, zone_ids)
    zone_ids = registry.zone_ids
    present_pressure_zone_ids = [
        infer_pressure_zone_id(robot, zone_ids, registry=registry)
        for robot in robots
    ]
    destination_pressure_zone_ids = [
        infer_pressure_zone_id(robot.destination, zone_ids, registry=registry)
        for robot in robots
        if getattr(robot, "destination", None) is not None
    ]
    
    if not storages:
        warnings.append("storage_manager.storages unavailable; zone occupancy defaults to zero")
        
    station_manager = getattr(warehouse, "station_manager", None)
    stations = getattr(station_manager, "stations", []) or []
    repl_stations = [s for s in stations if getattr(s, "station_type", "") == "replenishment"]
    station_coord = _coord_for(station)
    fallback_distance_seen = False
    
    for zone_id in zone_ids:
        zone_info = registry.zones_by_id[zone_id]
        zone_storages = [storage for storage in storages if registry.zone_id_for_storage(storage) == zone_id]
        free = [storage for storage in zone_storages if bool(getattr(storage, "is_empty", False)) and getattr(storage, "assigned_pod", None) is None]
        total = len(zone_storages)
        free_slot_ratio = float(len(free)) / float(max(1, total))
        
        present_robot_count = sum(
            1 for pressure_zone_id in present_pressure_zone_ids
            if pressure_zone_id == zone_id
        )
        destination_robot_count = sum(
            1 for pressure_zone_id in destination_pressure_zone_ids
            if pressure_zone_id == zone_id
        )
        
        neighbors = zone_info.neighbor_zone_ids
        neighbor_present_count = sum(
            1 for pressure_zone_id in present_pressure_zone_ids
            if pressure_zone_id in neighbors
        )
        neighbor_dest_count = sum(
            1 for pressure_zone_id in destination_pressure_zone_ids
            if pressure_zone_id in neighbors
        )
        
        superzone_members = [
            other_zone_id
            for other_zone_id in zone_ids
            if registry.zones_by_id[other_zone_id].superzone_id == zone_info.superzone_id
        ]
        superzone_present_count = sum(
            1 for pressure_zone_id in present_pressure_zone_ids
            if pressure_zone_id in superzone_members
        )
        superzone_dest_count = sum(
            1 for pressure_zone_id in destination_pressure_zone_ids
            if pressure_zone_id in superzone_members
        )
        
        zone_skus = set()
        for storage in zone_storages:
            p = getattr(storage, "assigned_pod", None)
            if p is not None:
                zone_skus.update(
                    sku
                    for sku, details in (getattr(p, "skus", {}) or {}).items()
                    if _float(details.get("current_qty", 0.0)) > 0.0
                )
                
        pod_skus = {
            sku
            for sku, details in (getattr(pod, "skus", {}) or {}).items()
            if _float(details.get("current_qty", 0.0)) > 0.0
        } if pod is not None else set()
        sku_similarity_count = len(pod_skus.intersection(zone_skus))
        if pod_skus and zone_skus:
            sku_similarity = sku_similarity_count / max(1, len(pod_skus))
        else:
            sku_similarity = 0.0

        representative = _representative_storage(zone_storages)
        representative_coord = _coord_for(representative)
        storage_cycle = graph_cycle_distance_or_fallback(warehouse, station_coord, representative_coord)
        if storage_cycle.status == DISTANCE_STATUS_FALLBACK:
            fallback_distance_seen = True

        replenish_valid = bool(free) and replenishment_signal_active and replenishment_station_available
        zone_dest_pressure = _pressure(destination_robot_count, active_robot_denominator)
        neighbor_dest_pressure = _pressure(neighbor_dest_count, active_robot_denominator)
        superzone_dest_pressure = _pressure(superzone_dest_count, active_robot_denominator)
        zone_present_pressure = _pressure(present_robot_count, active_robot_denominator)
        neighbor_present_pressure = _pressure(neighbor_present_count, active_robot_denominator)
        superzone_present_pressure = _pressure(superzone_present_count, active_robot_denominator)
        
        rows.append(
            {
                "zone_id": zone_id,
                "zone_row_index": float(zone_info.row_index),
                "zone_col_index": float(zone_info.col_index),
                "selected_superzone_id": zone_info.superzone_id,
                "occupation_level": 1.0 - free_slot_ratio if total else 0.0,
                "free_slot_count": float(len(free)),
                "total_slot_count": float(total),
                "free_slot_ratio": float(free_slot_ratio),
                "zone_destination_robot_count": float(destination_robot_count),
                "neighbor_zone_destination_robot_count": float(neighbor_dest_count),
                "superzone_destination_robot_count": float(superzone_dest_count),
                "zone_present_robot_count": float(present_robot_count),
                "neighbor_zone_present_robot_count": float(neighbor_present_count),
                "superzone_present_robot_count": float(superzone_present_count),
                "robot_pressure_denominator": float(active_robot_denominator),
                "zone_destination_robot_pressure": zone_dest_pressure,
                "neighbor_zone_destination_robot_pressure": neighbor_dest_pressure,
                "superzone_destination_robot_pressure": superzone_dest_pressure,
                "zone_present_robot_pressure": zone_present_pressure,
                "neighbor_zone_present_robot_pressure": neighbor_present_pressure,
                "superzone_present_robot_pressure": superzone_present_pressure,
                "storage_cycle_time_estimate": float(storage_cycle.value_or_zero),
                "replenish_cycle_time_estimate": 0.0,
                "sku_similarity_count": float(sku_similarity_count),
                "sku_similarity_fraction": float(sku_similarity),
                "sku_similarity": float(sku_similarity),
                "distance_status": storage_cycle.status,
                "store_action_valid": 1.0 if free else 0.0,
                "replenish_store_action_valid": 1.0 if replenish_valid else 0.0,
            }
        )
    if fallback_distance_seen:
        warnings.append("directed graph distance unavailable for at least one RTS feature; explicit metric fallback used")
    return rows, warnings


def build_zone_registry_metadata(context: Any, zone_ids: Sequence[str]) -> dict[str, Any]:
    registry = build_zone_registry(context, zone_ids)
    metadata = registry.metadata()
    warehouse = getattr(context, "warehouse", None)
    metadata.update(distance_cache_metadata(warehouse))
    metadata.update(
        {
            "sku_similarity_version": SKU_SIMILARITY_VERSION,
            "robot_pressure_denominator": "total active warehouse robot objects",
        }
    )
    return metadata


def _selected_replenishment_station(station: Any, repl_stations: Sequence[Any]) -> Any | None:
    if str(getattr(station, "station_type", "")) == "replenishment":
        return station
    if not repl_stations:
        return None
    station_coord = _coord_for(station)
    return min(
        repl_stations,
        key=lambda candidate: _metric_distance(_coord_for(candidate), station_coord),
    )


def _nearest_replenishment_distance(warehouse: Any, repl_stations: Sequence[Any], target_coord: Any) -> Any:
    if not repl_stations or target_coord is None:
        return graph_distance_or_fallback(warehouse, None, target_coord, allow_metric_fallback=False)
    return min(
        (
            graph_distance_or_fallback(warehouse, _coord_for(station), target_coord)
            for station in repl_stations
        ),
        key=lambda result: (
            result.distance is None,
            float(result.distance if result.distance is not None else 0.0),
            result.source,
        ),
    )


def _representative_storage(storages: Sequence[Any]) -> Any | None:
    if not storages:
        return None
    return sorted(
        storages,
        key=lambda storage: (
            float(getattr(storage, "pos_x", 0.0)),
            float(getattr(storage, "pos_y", 0.0)),
            int(getattr(storage, "storage_number", getattr(storage, "id", 0)) or 0),
        ),
    )[0]


def _coord_for(obj: Any) -> _Coord | None:
    if obj is None:
        return None
    coordinate = getattr(obj, "coordinate", None)
    if coordinate is not None and getattr(coordinate, "x", None) is not None and getattr(coordinate, "y", None) is not None:
        return _Coord(coordinate.x, coordinate.y)
    x = getattr(obj, "pos_x", getattr(obj, "x", None))
    y = getattr(obj, "pos_y", getattr(obj, "y", None))
    if x is None or y is None:
        return None
    return _Coord(x, y)


def _coordinate_pair(obj: Any) -> tuple[int, int] | None:
    coord = _coord_for(obj)
    if coord is None:
        return None
    try:
        return int(round(float(coord.x))), int(round(float(coord.y)))
    except Exception:
        return None


def _is_robot_object(obj: object) -> bool:
    return str(getattr(obj, "object_type", "")).lower() == "robot"


def _metric_distance(a: Any, b: Any) -> float:
    if a is None or b is None:
        return float("inf")
    return abs(float(getattr(a, "x", 0.0)) - float(getattr(b, "x", 0.0))) + abs(
        float(getattr(a, "y", 0.0)) - float(getattr(b, "y", 0.0))
    )


def _pressure(count: int, denominator: int) -> float:
    return max(0.0, min(1.0, float(count) / float(max(1, denominator))))


def _float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0
