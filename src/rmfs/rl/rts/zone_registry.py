"""Canonical RTS storage-zone registry derived from layout geometry."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any, Mapping, Sequence


ZONE_GEOMETRY_VERSION = "rts_storage_zone_registry.v1"
ZONE_ID_PATTERN = "rts_z_r##_c##"
_ZONE_ID_RE = re.compile(r"^rts_z_r(?P<row>\d+)_c(?P<col>\d+)$")
_NON_PRODUCTION_PREFIX = "col_"


@dataclass(frozen=True)
class RTSZoneInfo:
    zone_id: str
    row_index: int
    col_index: int
    storage_numbers: tuple[int, ...]
    coordinates: tuple[tuple[int, int], ...]
    centroid_x: float
    centroid_y: float
    superzone_id: str
    neighbor_zone_ids: tuple[str, ...]


@dataclass(frozen=True)
class RTSZoneRegistry:
    zone_ids: tuple[str, ...]
    zones_by_id: Mapping[str, RTSZoneInfo]
    storage_number_to_zone_id: Mapping[int, str]
    coordinate_to_zone_id: Mapping[tuple[int, int], str]
    geometry_hash: str
    geometry_version: str = ZONE_GEOMETRY_VERSION
    zone_id_pattern: str = ZONE_ID_PATTERN

    @property
    def zone_count(self) -> int:
        return len(self.zone_ids)

    def zone_id_for_storage(self, storage: Any) -> str:
        number = _storage_number(storage)
        if number is not None and number in self.storage_number_to_zone_id:
            return str(self.storage_number_to_zone_id[number])
        coord = _coordinate_pair(storage)
        if coord is not None and coord in self.coordinate_to_zone_id:
            return str(self.coordinate_to_zone_id[coord])
        for attr in ("rts_zone_id", "zone_id", "zone", "storage_zone"):
            value = getattr(storage, attr, None)
            if value is not None and str(value) in self.zones_by_id:
                return str(value)
        return ""

    def zone_id_for_coordinate(self, coord: Any) -> str:
        pair = _coordinate_pair(coord)
        if pair is None:
            return ""
        if pair in self.coordinate_to_zone_id:
            return str(self.coordinate_to_zone_id[pair])
        x, y = pair
        containing = []
        for zone_id in self.zone_ids:
            info = self.zones_by_id[zone_id]
            xs = [item[0] for item in info.coordinates]
            ys = [item[1] for item in info.coordinates]
            if xs and ys and min(xs) <= x <= max(xs) and min(ys) <= y <= max(ys):
                containing.append(zone_id)
        if containing:
            return sorted(containing)[0]
        if not self.zone_ids:
            return ""
        return min(
            self.zone_ids,
            key=lambda zone_id: (
                (self.zones_by_id[zone_id].centroid_x - x) ** 2
                + (self.zones_by_id[zone_id].centroid_y - y) ** 2,
                self.zones_by_id[zone_id].row_index,
                self.zones_by_id[zone_id].col_index,
                zone_id,
            ),
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "zone_geometry_version": self.geometry_version,
            "zone_geometry_hash": self.geometry_hash,
            "zone_id_pattern": self.zone_id_pattern,
            "zone_count": self.zone_count,
            "zone_ids": list(self.zone_ids),
            "zone_order": list(self.zone_ids),
        }


def is_non_production_zone_id(zone_id: Any) -> bool:
    return str(zone_id or "").startswith(_NON_PRODUCTION_PREFIX)


def validate_no_col_zone_ids(zone_ids: Sequence[str], *, context: str) -> None:
    bad = [str(zone_id) for zone_id in zone_ids if is_non_production_zone_id(zone_id)]
    if bad:
        raise ValueError(f"{context} cannot use fallback col_* RTS zone ids: {bad}")


def parse_zone_indices(zone_id: str) -> tuple[int, int] | None:
    match = _ZONE_ID_RE.match(str(zone_id or "").strip())
    if match is None:
        return None
    return int(match.group("row")), int(match.group("col"))


def superzone_id_for(row_index: int, col_index: int) -> str:
    return f"sr{int(row_index) // 2}_sc{int(col_index) // 2}"


def build_zone_registry(context_or_storage_manager: Any, zone_ids: Sequence[str] | None = None) -> RTSZoneRegistry:
    storage_manager = _storage_manager(context_or_storage_manager)
    storages = tuple(getattr(storage_manager, "storages", []) or ())
    requested = tuple(str(zone_id) for zone_id in (zone_ids or ()))
    if requested and _can_use_explicit_storage_zones(storages, requested):
        return _build_explicit_registry(storages, requested)
    registry = _build_canonical_registry(storages)
    if requested:
        missing = [zone_id for zone_id in requested if zone_id not in registry.zones_by_id]
        if missing:
            raise ValueError(
                "RTS zone_ids do not match canonical storage geometry: "
                f"missing={missing}, available={list(registry.zone_ids)}"
            )
        if tuple(requested) != tuple(registry.zone_ids):
            registry = _filter_registry(registry, requested)
    return registry


def schema_metadata_for_zone_ids(zone_ids: Sequence[str]) -> dict[str, Any]:
    zones = tuple(str(zone_id) for zone_id in zone_ids)
    validate_no_col_zone_ids(zones, context="RTS checkpoint schema")
    payload = {
        "zone_geometry_version": ZONE_GEOMETRY_VERSION,
        "zone_id_pattern": ZONE_ID_PATTERN if all(parse_zone_indices(z) for z in zones) else "explicit_zone_ids",
        "zone_count": len(zones),
        "zone_ids": list(zones),
        "zone_order": list(zones),
    }
    payload["zone_geometry_hash"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    return payload


def zone_ids_from_action_feature_names(action_feature_names: Sequence[str]) -> tuple[str, ...]:
    prefix = "next_pod_zone_one_hot__"
    legacy_prefix = "next_retrieval_zone_one_hot__"
    return tuple(
        str(name).removeprefix(prefix)
        for name in action_feature_names
        if str(name).startswith(prefix)
    ) or tuple(
        str(name).removeprefix(legacy_prefix)
        for name in action_feature_names
        if str(name).startswith(legacy_prefix)
    )


def _storage_manager(value: Any) -> Any:
    if hasattr(value, "storages"):
        return value
    warehouse = getattr(value, "warehouse", value)
    return getattr(warehouse, "storage_manager", None)


def _build_canonical_registry(storages: Sequence[Any]) -> RTSZoneRegistry:
    coords = [_coordinate_pair(storage) for storage in storages]
    coords = [coord for coord in coords if coord is not None]
    x_groups = _contiguous_groups(sorted({coord[0] for coord in coords}))
    y_groups = _contiguous_groups(sorted({coord[1] for coord in coords}))
    x_group_by_value = {value: index for index, group in enumerate(x_groups) for value in group}
    y_group_by_value = {value: index for index, group in enumerate(y_groups) for value in group}

    storage_rows: dict[str, list[Any]] = {}
    for storage in storages:
        coord = _coordinate_pair(storage)
        if coord is None:
            continue
        col_index = x_group_by_value.get(coord[0])
        row_index = y_group_by_value.get(coord[1])
        if col_index is None or row_index is None:
            continue
        zone_id = f"rts_z_r{row_index:02d}_c{col_index:02d}"
        storage_rows.setdefault(zone_id, []).append(storage)
    return _registry_from_grouped_storages(storage_rows)


def _build_explicit_registry(storages: Sequence[Any], requested: Sequence[str]) -> RTSZoneRegistry:
    requested_set = set(str(zone_id) for zone_id in requested)
    storage_rows: dict[str, list[Any]] = {str(zone_id): [] for zone_id in requested}
    for storage in storages:
        zone_id = _explicit_zone_attr(storage)
        if zone_id in requested_set:
            storage_rows.setdefault(zone_id, []).append(storage)
    return _registry_from_grouped_storages(storage_rows, requested_order=tuple(str(z) for z in requested))


def _registry_from_grouped_storages(
    storage_rows: Mapping[str, Sequence[Any]],
    *,
    requested_order: tuple[str, ...] | None = None,
) -> RTSZoneRegistry:
    raw_infos: dict[str, dict[str, Any]] = {}
    for zone_id, zone_storages in storage_rows.items():
        coordinates = tuple(sorted(coord for coord in (_coordinate_pair(s) for s in zone_storages) if coord is not None))
        storage_numbers = tuple(sorted(number for number in (_storage_number(s) for s in zone_storages) if number is not None))
        parsed = parse_zone_indices(zone_id)
        if parsed is None:
            centroid_x = _mean([coord[0] for coord in coordinates])
            centroid_y = _mean([coord[1] for coord in coordinates])
            raw_infos[str(zone_id)] = {
                "row_index": None,
                "col_index": None,
                "coordinates": coordinates,
                "storage_numbers": storage_numbers,
                "centroid_x": centroid_x,
                "centroid_y": centroid_y,
            }
        else:
            raw_infos[str(zone_id)] = {
                "row_index": parsed[0],
                "col_index": parsed[1],
                "coordinates": coordinates,
                "storage_numbers": storage_numbers,
                "centroid_x": _mean([coord[0] for coord in coordinates]),
                "centroid_y": _mean([coord[1] for coord in coordinates]),
            }

    if any(info["row_index"] is None or info["col_index"] is None for info in raw_infos.values()):
        row_values = sorted({round(float(info["centroid_y"]), 9) for info in raw_infos.values()})
        col_values = sorted({round(float(info["centroid_x"]), 9) for info in raw_infos.values()})
        for info in raw_infos.values():
            if info["row_index"] is None:
                info["row_index"] = row_values.index(round(float(info["centroid_y"]), 9))
            if info["col_index"] is None:
                info["col_index"] = col_values.index(round(float(info["centroid_x"]), 9))

    ordered = requested_order or tuple(
        sorted(raw_infos.keys(), key=lambda zid: (raw_infos[zid]["row_index"], raw_infos[zid]["col_index"], zid))
    )
    zone_id_set = set(ordered)
    neighbors_by_zone: dict[str, tuple[str, ...]] = {}
    for zone_id in ordered:
        info = raw_infos[zone_id]
        neighbor_coords = {
            (int(info["row_index"]), int(info["col_index"])),
            (int(info["row_index"]) - 1, int(info["col_index"])),
            (int(info["row_index"]) + 1, int(info["col_index"])),
            (int(info["row_index"]), int(info["col_index"]) - 1),
            (int(info["row_index"]), int(info["col_index"]) + 1),
        }
        neighbors_by_zone[zone_id] = tuple(
            other
            for other in ordered
            if other in zone_id_set
            and (int(raw_infos[other]["row_index"]), int(raw_infos[other]["col_index"])) in neighbor_coords
        )

    zones_by_id: dict[str, RTSZoneInfo] = {}
    storage_number_to_zone_id: dict[int, str] = {}
    coordinate_to_zone_id: dict[tuple[int, int], str] = {}
    for zone_id in ordered:
        info = raw_infos[zone_id]
        row_index = int(info["row_index"])
        col_index = int(info["col_index"])
        zone_info = RTSZoneInfo(
            zone_id=zone_id,
            row_index=row_index,
            col_index=col_index,
            storage_numbers=tuple(info["storage_numbers"]),
            coordinates=tuple(info["coordinates"]),
            centroid_x=float(info["centroid_x"]),
            centroid_y=float(info["centroid_y"]),
            superzone_id=superzone_id_for(row_index, col_index),
            neighbor_zone_ids=neighbors_by_zone[zone_id],
        )
        zones_by_id[zone_id] = zone_info
        for number in zone_info.storage_numbers:
            storage_number_to_zone_id[int(number)] = zone_id
        for coord in zone_info.coordinates:
            coordinate_to_zone_id[coord] = zone_id

    fingerprint_payload = [
        {
            "zone_id": zone_id,
            "row_index": zones_by_id[zone_id].row_index,
            "col_index": zones_by_id[zone_id].col_index,
            "storage_numbers": list(zones_by_id[zone_id].storage_numbers),
            "coordinates": [list(coord) for coord in zones_by_id[zone_id].coordinates],
        }
        for zone_id in ordered
    ]
    geometry_hash = hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    return RTSZoneRegistry(
        zone_ids=tuple(ordered),
        zones_by_id=zones_by_id,
        storage_number_to_zone_id=storage_number_to_zone_id,
        coordinate_to_zone_id=coordinate_to_zone_id,
        geometry_hash=geometry_hash,
        zone_id_pattern=(
            ZONE_ID_PATTERN if all(parse_zone_indices(zone_id) is not None for zone_id in ordered) else "explicit_zone_ids"
        ),
    )


def _filter_registry(registry: RTSZoneRegistry, requested: Sequence[str]) -> RTSZoneRegistry:
    grouped = {
        zone_id: [
            _SyntheticStorage(number, x, y)
            for number, (x, y) in zip(
                registry.zones_by_id[zone_id].storage_numbers,
                registry.zones_by_id[zone_id].coordinates,
                strict=False,
            )
        ]
        for zone_id in requested
    }
    return _registry_from_grouped_storages(grouped, requested_order=tuple(str(z) for z in requested))


class _SyntheticStorage:
    def __init__(self, storage_number: int, x: int, y: int):
        self.storage_number = int(storage_number)
        self.pos_x = int(x)
        self.pos_y = int(y)


def _can_use_explicit_storage_zones(storages: Sequence[Any], requested: Sequence[str]) -> bool:
    requested_set = set(str(zone_id) for zone_id in requested)
    if not requested_set:
        return False
    explicit = {_explicit_zone_attr(storage) for storage in storages}
    explicit.discard("")
    if not explicit:
        return False
    return requested_set.issubset(explicit)


def _explicit_zone_attr(storage: Any) -> str:
    for attr in ("rts_zone_id", "zone_id", "zone", "storage_zone"):
        value = getattr(storage, attr, None)
        if value is not None:
            return str(value)
    return ""


def _contiguous_groups(values: Sequence[int]) -> list[tuple[int, ...]]:
    groups: list[list[int]] = []
    for value in values:
        if not groups or int(value) != groups[-1][-1] + 1:
            groups.append([int(value)])
        else:
            groups[-1].append(int(value))
    return [tuple(group) for group in groups]


def _coordinate_pair(obj: Any) -> tuple[int, int] | None:
    if obj is None:
        return None
    coordinate = getattr(obj, "coordinate", None)
    if coordinate is not None and getattr(coordinate, "x", None) is not None and getattr(coordinate, "y", None) is not None:
        return int(round(float(coordinate.x))), int(round(float(coordinate.y)))
    x = getattr(obj, "pos_x", getattr(obj, "x", None))
    y = getattr(obj, "pos_y", getattr(obj, "y", None))
    if x is None or y is None:
        return None
    try:
        return int(round(float(x))), int(round(float(y)))
    except Exception:
        return None


def _storage_number(storage: Any) -> int | None:
    for attr in ("storage_number", "id", "_id"):
        value = getattr(storage, attr, None)
        if value is None:
            continue
        try:
            return int(value)
        except Exception:
            continue
    return None


def _mean(values: Sequence[int]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))

