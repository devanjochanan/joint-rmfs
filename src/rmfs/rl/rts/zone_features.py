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
    robots = list(getattr(warehouse, "_objects", []) or [])
    warnings: list[str] = []
    rows = []
    registry = build_zone_registry(context, zone_ids)
    zone_ids = registry.zone_ids
    
    if not storages:
        warnings.append("storage_manager.storages unavailable; zone occupancy defaults to zero")
        
    station_manager = getattr(warehouse, "station_manager", None)
    stations = getattr(station_manager, "stations", []) or []
    repl_stations = [s for s in stations if getattr(s, "station_type", "") == "replenishment"]
    station_coord = _coord_for(station)
    selected_repl_station = _selected_replenishment_station(station, repl_stations)
    selected_repl_coord = _coord_for(selected_repl_station)
    fallback_distance_seen = False
    
    for zone_id in zone_ids:
        zone_info = registry.zones_by_id[zone_id]
        zone_storages = [storage for storage in storages if registry.zone_id_for_storage(storage) == zone_id]
        free = [storage for storage in zone_storages if bool(getattr(storage, "is_empty", False)) and getattr(storage, "assigned_pod", None) is None]
        total = len(zone_storages)
        
        present_robot_count = sum(
            1 for robot in robots
            if infer_coordinate_zone_id(robot, zone_ids, registry=registry) == zone_id
        )
        destination_robot_count = sum(
            1 for robot in robots
            if getattr(robot, "destination", None) is not None
            and infer_coordinate_zone_id(robot.destination, zone_ids, registry=registry) == zone_id
        )
        
        neighbors = zone_info.neighbor_zone_ids
        neighbor_present_count = sum(
            1 for robot in robots
            if infer_coordinate_zone_id(robot, zone_ids, registry=registry) in neighbors
        )
        neighbor_dest_count = sum(
            1 for robot in robots
            if getattr(robot, "destination", None) is not None
            and infer_coordinate_zone_id(robot.destination, zone_ids, registry=registry) in neighbors
        )
        
        superzone_members = [
            other_zone_id
            for other_zone_id in zone_ids
            if registry.zones_by_id[other_zone_id].superzone_id == zone_info.superzone_id
        ]
        superzone_present_count = sum(
            1 for robot in robots
            if infer_coordinate_zone_id(robot, zone_ids, registry=registry) in superzone_members
        )
        superzone_dest_count = sum(
            1 for robot in robots
            if getattr(robot, "destination", None) is not None
            and infer_coordinate_zone_id(robot.destination, zone_ids, registry=registry) in superzone_members
        )
        
        zone_skus = set()
        for storage in zone_storages:
            p = getattr(storage, "assigned_pod", None)
            if p is not None:
                zone_skus.update(getattr(p, "skus", {}).keys())
                
        pod_skus = set(getattr(pod, "skus", {}).keys()) if pod is not None else set()
        if pod_skus and zone_skus:
            sku_similarity = len(pod_skus.intersection(zone_skus)) / len(pod_skus)
        else:
            sku_similarity = 0.0

        representative = _representative_storage(zone_storages)
        representative_coord = _coord_for(representative)
        storage_cycle = graph_cycle_distance_or_fallback(warehouse, station_coord, representative_coord)
        if storage_cycle.status == DISTANCE_STATUS_FALLBACK:
            fallback_distance_seen = True

        if selected_repl_coord is not None and representative_coord is not None:
            selected_repl_distance = graph_distance_or_fallback(
                warehouse,
                selected_repl_coord,
                representative_coord,
            )
        else:
            selected_repl_distance = graph_distance_or_fallback(
                warehouse,
                station_coord,
                representative_coord,
            )
        if selected_repl_distance.status == DISTANCE_STATUS_FALLBACK:
            fallback_distance_seen = True

        nearest_repl_distance = _nearest_replenishment_distance(
            warehouse,
            repl_stations,
            representative_coord,
        )
        if nearest_repl_distance.status == DISTANCE_STATUS_FALLBACK:
            fallback_distance_seen = True

        replenish_cycle = selected_repl_distance
        
        replenish_valid = bool(free) and replenishment_signal_active and replenishment_station_available
        
        rows.append(
            {
                "zone_id": zone_id,
                "zone_row_index": float(zone_info.row_index),
                "zone_col_index": float(zone_info.col_index),
                "selected_superzone_id": zone_info.superzone_id,
                "occupation_level": 1.0 - (float(len(free)) / float(total)) if total else 0.0,
                "free_slot_count": float(len(free)),
                "zone_destination_robot_count": float(destination_robot_count),
                "neighbor_zone_destination_robot_count": float(neighbor_dest_count),
                "superzone_destination_robot_count": float(superzone_dest_count),
                "zone_present_robot_count": float(present_robot_count),
                "neighbor_zone_present_robot_count": float(neighbor_present_count),
                "superzone_present_robot_count": float(superzone_present_count),
                "storage_cycle_time_estimate": float(storage_cycle.value_or_zero),
                "replenish_cycle_time_estimate": float(replenish_cycle.value_or_zero),
                "sku_similarity": float(sku_similarity),
                "candidate_zone_to_selected_replenishment_station_distance": float(selected_repl_distance.value_or_zero),
                "candidate_zone_to_nearest_replenishment_station_distance": float(nearest_repl_distance.value_or_zero),
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


def _metric_distance(a: Any, b: Any) -> float:
    if a is None or b is None:
        return float("inf")
    return abs(float(getattr(a, "x", 0.0)) - float(getattr(b, "x", 0.0))) + abs(
        float(getattr(a, "y", 0.0)) - float(getattr(b, "y", 0.0))
    )
