"""Run-local immutable RTS geometry and directed-distance index."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import time
from types import MappingProxyType
import weakref
from typing import Any, Mapping

import numpy as np

from .travel_time import EMPTY_ROBOT, LOADED_ROBOT, graph_wrapper_for_topology
from .zone_registry import RTSZoneRegistry, build_zone_registry, validate_no_col_zone_ids


STATIC_RUNTIME_INDEX_VERSION = "rts_static_runtime_index.v2"
VRSLA_SLOT_SCORE_VERSION = "rts_vrsla_slot_cycle_distance.v1"
_INDEX_BY_WAREHOUSE: "weakref.WeakKeyDictionary[Any, RTSStaticRuntimeIndex]" = weakref.WeakKeyDictionary()


@dataclass
class RTSStaticRuntimeIndexDiagnostics:
    install_count: int = 0
    build_count: int = 0
    rebuild_count: int = 0
    invalidation_count: int = 0
    cached_retrieval_count: int = 0
    missing_retrieval_count: int = 0
    identity_validation_count: int = 0
    layout_hash_count: int = 0
    storage_hash_count: int = 0
    graph_hash_count: int = 0
    graph_matrix_build_count: int = 0
    empty_matrix_build_count: int = 0
    loaded_matrix_build_count: int = 0
    vrsla_slot_score_build_count: int = 0

    def to_json_dict(self) -> dict[str, int]:
        return {
            "install_count": int(self.install_count),
            "build_count": int(self.build_count),
            "rebuild_count": int(self.rebuild_count),
            "invalidation_count": int(self.invalidation_count),
            "cached_retrieval_count": int(self.cached_retrieval_count),
            "missing_retrieval_count": int(self.missing_retrieval_count),
            "identity_validation_count": int(self.identity_validation_count),
            "layout_hash_count": int(self.layout_hash_count),
            "storage_hash_count": int(self.storage_hash_count),
            "graph_hash_count": int(self.graph_hash_count),
            "graph_matrix_build_count": int(self.graph_matrix_build_count),
            "empty_matrix_build_count": int(self.empty_matrix_build_count),
            "loaded_matrix_build_count": int(self.loaded_matrix_build_count),
            "vrsla_slot_score_build_count": int(self.vrsla_slot_score_build_count),
        }


_DIAGNOSTICS = RTSStaticRuntimeIndexDiagnostics()


@dataclass(frozen=True)
class RTSGraphDistanceIndex:
    topology: str
    graph_identity_hash: str
    node_to_index: Mapping[str, int]
    index_to_node: tuple[str, ...]
    distance_matrix: np.ndarray
    node_count: int
    edge_count: int
    matrix_bytes: int

    def distance_between_nodes(self, src_node: str, dst_node: str) -> float | None:
        src_index = self.node_to_index.get(src_node)
        dst_index = self.node_to_index.get(dst_node)
        if src_index is None or dst_index is None:
            return None
        distance = float(self.distance_matrix[src_index, dst_index])
        return distance if math.isfinite(distance) else None


@dataclass(frozen=True)
class RTSRuntimeZoneManifest:
    zone_ids: tuple[str, ...]
    zone_count: int
    zone_construction_version: str
    layout_identity_hash: str
    storage_coordinate_hash: str
    empty_graph_identity_hash: str | None
    loaded_graph_identity_hash: str | None

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "zone_ids": list(self.zone_ids),
            "zone_count": int(self.zone_count),
            "zone_construction_version": self.zone_construction_version,
            "layout_identity_hash": self.layout_identity_hash,
            "storage_coordinate_hash": self.storage_coordinate_hash,
            "empty_graph_identity_hash": self.empty_graph_identity_hash,
            "loaded_graph_identity_hash": self.loaded_graph_identity_hash,
        }


@dataclass(frozen=True)
class RTSStaticRuntimeIndex:
    version: str
    layout_identity_hash: str
    storage_coordinate_hash: str
    empty_graph_identity_hash: str
    loaded_graph_identity_hash: str
    zone_registry: RTSZoneRegistry
    zone_ids: tuple[str, ...]
    storage_id_to_zone_id: Mapping[str, str]
    storage_number_to_zone_id: Mapping[int, str]
    storage_coordinate_to_zone_id: Mapping[tuple[int, int], str]
    storage_by_id: Mapping[str, Any]
    storages_by_zone: Mapping[str, tuple[Any, ...]]
    empty_graph: RTSGraphDistanceIndex
    loaded_graph: RTSGraphDistanceIndex
    layout_normalization_metadata: Mapping[str, Any]
    storage_macro_region_by_id: Mapping[str, str]
    vrsla_slot_score_version: str
    vrsla_slot_cycle_distance: Mapping[str, Mapping[str, float]]
    vrsla_slot_hotness_rank: Mapping[str, Mapping[str, float]]
    vrsla_slots_by_picker: Mapping[str, tuple[str, ...]]
    vrsla_slot_score_identity: str
    build_seconds: float
    total_matrix_bytes: int

    def __getstate__(self):
        raise TypeError("RTSStaticRuntimeIndex is runtime-only and must not be pickled")

    @property
    def manifest(self) -> RTSRuntimeZoneManifest:
        return RTSRuntimeZoneManifest(
            zone_ids=self.zone_ids,
            zone_count=len(self.zone_ids),
            zone_construction_version=self.zone_registry.geometry_version,
            layout_identity_hash=self.layout_identity_hash,
            storage_coordinate_hash=self.storage_coordinate_hash,
            empty_graph_identity_hash=self.empty_graph_identity_hash,
            loaded_graph_identity_hash=self.loaded_graph_identity_hash,
        )

    def graph_index(self, topology: str) -> RTSGraphDistanceIndex:
        if str(topology) == EMPTY_ROBOT:
            return self.empty_graph
        if str(topology) == LOADED_ROBOT:
            return self.loaded_graph
        raise ValueError(f"unsupported RTS topology: {topology!r}")

    def distance_between(self, src: Any, dst: Any, *, topology: str) -> float | None:
        src_node = node_id_for_coordinate(src)
        dst_node = node_id_for_coordinate(dst)
        if src_node is None or dst_node is None:
            return None
        return self.graph_index(topology).distance_between_nodes(src_node, dst_node)

    def vrsla_cycle_distance(self, picker: Any, storage: Any) -> float | None:
        picker_id = stable_station_id(picker)
        storage_id = stable_storage_id(storage)
        value = self.vrsla_slot_cycle_distance.get(picker_id, {}).get(storage_id)
        return float(value) if value is not None and math.isfinite(float(value)) else None

    def vrsla_hotness_rank(self, picker: Any, storage: Any) -> float:
        picker_id = stable_station_id(picker)
        storage_id = stable_storage_id(storage)
        return float(self.vrsla_slot_hotness_rank.get(picker_id, {}).get(storage_id, 0.0))

    def vrsla_sorted_storage_ids(self, picker: Any) -> tuple[str, ...]:
        return tuple(self.vrsla_slots_by_picker.get(stable_station_id(picker), ()))

    def to_summary_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "layout_identity_hash": self.layout_identity_hash,
            "storage_coordinate_hash": self.storage_coordinate_hash,
            "empty_graph_identity_hash": self.empty_graph_identity_hash,
            "loaded_graph_identity_hash": self.loaded_graph_identity_hash,
            "zone_count": len(self.zone_ids),
            "zone_ids": list(self.zone_ids),
            "build_seconds": float(self.build_seconds),
            "empty_graph_nodes": self.empty_graph.node_count,
            "empty_graph_edges": self.empty_graph.edge_count,
            "empty_matrix_bytes": self.empty_graph.matrix_bytes,
            "loaded_graph_nodes": self.loaded_graph.node_count,
            "loaded_graph_edges": self.loaded_graph.edge_count,
            "loaded_matrix_bytes": self.loaded_graph.matrix_bytes,
            "vrsla_slot_score_version": self.vrsla_slot_score_version,
            "vrsla_slot_score_identity": self.vrsla_slot_score_identity,
            "vrsla_picker_count": len(self.vrsla_slots_by_picker),
            "total_matrix_bytes": self.total_matrix_bytes,
            "serialization": "weakkey_runtime_cache_not_attached_to_warehouse_pickle",
        }


def get_or_build_static_runtime_index(warehouse: Any) -> RTSStaticRuntimeIndex:
    """Return the installed run-local index, building it only when absent.

    This is intentionally O(1) after setup: it does not rehash layout, storage,
    or graph identities. Call validate_static_runtime_index_identity() in
    diagnostics/tests when an identity audit is required.
    """
    if warehouse is None:
        raise ValueError("warehouse is required to build RTS static runtime index")
    cached = get_static_runtime_index(warehouse)
    if cached is not None:
        return cached
    return install_static_runtime_index(warehouse)


def get_static_runtime_index(warehouse: Any) -> RTSStaticRuntimeIndex | None:
    if warehouse is None:
        return None
    try:
        index = _INDEX_BY_WAREHOUSE.get(warehouse)
    except TypeError as exc:
        raise TypeError("warehouse object must support weak references for runtime RTS index") from exc
    if index is None:
        _DIAGNOSTICS.missing_retrieval_count += 1
        return None
    _DIAGNOSTICS.cached_retrieval_count += 1
    return index


def install_static_runtime_index(warehouse: Any) -> RTSStaticRuntimeIndex:
    """Install exactly one active static index for this warehouse object."""
    if warehouse is None:
        raise ValueError("warehouse is required to install RTS static runtime index")
    cached = get_static_runtime_index(warehouse)
    if cached is not None:
        return cached
    index = build_static_runtime_index(warehouse)
    _INDEX_BY_WAREHOUSE[warehouse] = index
    _clear_pair_distance_cache(warehouse)
    _DIAGNOSTICS.install_count += 1
    return index


def invalidate_static_runtime_index(warehouse: Any) -> None:
    """Drop the active static index; the next install/rebuild creates fresh matrices."""
    if warehouse is None:
        return
    try:
        _INDEX_BY_WAREHOUSE.pop(warehouse, None)
    except TypeError as exc:
        raise TypeError("warehouse object must support weak references for runtime RTS index") from exc
    _clear_pair_distance_cache(warehouse)
    _DIAGNOSTICS.invalidation_count += 1


def rebuild_static_runtime_index(warehouse: Any) -> RTSStaticRuntimeIndex:
    """Replace the worker-local index for a warehouse; old matrices become collectible."""
    if warehouse is None:
        raise ValueError("warehouse is required to rebuild RTS static runtime index")
    invalidate_static_runtime_index(warehouse)
    index = build_static_runtime_index(warehouse)
    _INDEX_BY_WAREHOUSE[warehouse] = index
    _clear_pair_distance_cache(warehouse)
    _DIAGNOSTICS.rebuild_count += 1
    return index


def validate_or_rebuild_static_runtime_index(warehouse: Any) -> RTSStaticRuntimeIndex:
    """Debug/lifecycle helper: audit identity once and rebuild if it changed."""
    cached = get_static_runtime_index(warehouse)
    if cached is not None and validate_static_runtime_index_identity(warehouse, cached):
        return cached
    if cached is None:
        return install_static_runtime_index(warehouse)
    return rebuild_static_runtime_index(warehouse)


def validate_static_runtime_index_identity(warehouse: Any, index: RTSStaticRuntimeIndex | None = None) -> bool:
    _DIAGNOSTICS.identity_validation_count += 1
    if index is None:
        index = get_static_runtime_index(warehouse)
    if index is None:
        return False
    try:
        storage_manager = getattr(warehouse, "storage_manager", None)
        storages = tuple(getattr(storage_manager, "storages", []) or ())
        return (
            index.layout_identity_hash == _layout_identity_hash(warehouse)
            and index.storage_coordinate_hash == _storage_coordinate_hash(storages)
            and index.empty_graph_identity_hash == _graph_identity_hash(
                getattr(graph_wrapper_for_topology(warehouse, EMPTY_ROBOT), "graph", None),
                EMPTY_ROBOT,
            )
            and index.loaded_graph_identity_hash == _graph_identity_hash(
                getattr(graph_wrapper_for_topology(warehouse, LOADED_ROBOT), "graph", None),
                LOADED_ROBOT,
            )
        )
    except Exception:
        return False


def runtime_index_identity_matches(warehouse: Any, index: RTSStaticRuntimeIndex) -> bool:
    return validate_static_runtime_index_identity(warehouse, index)


def build_static_runtime_index(warehouse: Any) -> RTSStaticRuntimeIndex:
    _DIAGNOSTICS.build_count += 1
    start = time.perf_counter()
    registry = build_zone_registry(warehouse)
    validate_no_col_zone_ids(registry.zone_ids, context="runtime RTS zone manifest")
    storage_manager = getattr(warehouse, "storage_manager", None)
    storages = tuple(getattr(storage_manager, "storages", []) or ())
    storages_by_zone = {
        zone_id: tuple(storage for storage in storages if registry.zone_id_for_storage(storage) == zone_id)
        for zone_id in registry.zone_ids
    }
    storage_id_to_zone = {
        stable_storage_id(storage): registry.zone_id_for_storage(storage)
        for storage in storages
        if registry.zone_id_for_storage(storage)
    }
    storage_by_id = {stable_storage_id(storage): storage for storage in storages}
    normalization = _layout_normalization_metadata(warehouse)
    storage_macro_by_id = {
        stable_storage_id(storage): _macro_region_for_storage(storage, normalization)
        for storage in storages
    }
    layout_hash = _layout_identity_hash(warehouse)
    storage_hash = _storage_coordinate_hash(storages)
    empty_index = _build_graph_distance_index(warehouse, EMPTY_ROBOT)
    loaded_index = _build_graph_distance_index(warehouse, LOADED_ROBOT)
    vrsla_scores = _build_vrsla_slot_scores(
        warehouse=warehouse,
        storages=storages,
        storage_coordinate_hash=storage_hash,
        loaded_index=loaded_index,
    )
    return RTSStaticRuntimeIndex(
        version=STATIC_RUNTIME_INDEX_VERSION,
        layout_identity_hash=layout_hash,
        storage_coordinate_hash=storage_hash,
        empty_graph_identity_hash=empty_index.graph_identity_hash,
        loaded_graph_identity_hash=loaded_index.graph_identity_hash,
        zone_registry=registry,
        zone_ids=registry.zone_ids,
        storage_id_to_zone_id=MappingProxyType(storage_id_to_zone),
        storage_number_to_zone_id=MappingProxyType(dict(registry.storage_number_to_zone_id)),
        storage_coordinate_to_zone_id=MappingProxyType(dict(registry.coordinate_to_zone_id)),
        storage_by_id=MappingProxyType(storage_by_id),
        storages_by_zone=MappingProxyType(storages_by_zone),
        empty_graph=empty_index,
        loaded_graph=loaded_index,
        layout_normalization_metadata=MappingProxyType(normalization),
        storage_macro_region_by_id=MappingProxyType(storage_macro_by_id),
        vrsla_slot_score_version=VRSLA_SLOT_SCORE_VERSION,
        vrsla_slot_cycle_distance=vrsla_scores["cycle_distance"],
        vrsla_slot_hotness_rank=vrsla_scores["hotness_rank"],
        vrsla_slots_by_picker=vrsla_scores["slots_by_picker"],
        vrsla_slot_score_identity=vrsla_scores["identity"],
        build_seconds=max(0.0, time.perf_counter() - start),
        total_matrix_bytes=int(empty_index.matrix_bytes + loaded_index.matrix_bytes),
    )


def runtime_zone_manifest_from_warehouse(warehouse: Any) -> RTSRuntimeZoneManifest:
    return get_or_build_static_runtime_index(warehouse).manifest


def resolve_runtime_zone_ids(warehouse: Any, requested_zone_ids: Any) -> tuple[str, ...]:
    manifest = runtime_zone_manifest_from_warehouse(warehouse)
    requested = tuple(str(zone_id).strip() for zone_id in (requested_zone_ids or ()) if str(zone_id).strip())
    if not requested or requested == ("auto",):
        return manifest.zone_ids
    if "auto" in requested:
        raise ValueError("RTS zone_ids must be exactly 'auto' or an explicit deterministic zone list")
    if len(set(requested)) != len(requested):
        raise ValueError("RTS zone_ids contain duplicates")
    missing = [zone_id for zone_id in requested if zone_id not in set(manifest.zone_ids)]
    if missing:
        raise ValueError(f"RTS zone_ids are unknown for runtime warehouse: {missing}")
    return requested


def node_id_for_coordinate(obj: Any) -> str | None:
    pair = coordinate_pair(obj)
    if pair is None:
        return None
    return f"{pair[0]},{pair[1]}"


def coordinate_pair(obj: Any) -> tuple[int, int] | None:
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


def stable_storage_id(storage: Any) -> str:
    if getattr(storage, "storage_id", None) is not None:
        return str(getattr(storage, "storage_id"))
    number = getattr(storage, "storage_number", getattr(storage, "id", getattr(storage, "_id", None)))
    if number is not None:
        return str(number)
    pair = coordinate_pair(storage)
    return "" if pair is None else f"{pair[0]}:{pair[1]}"


def stable_station_id(station: Any) -> str:
    if getattr(station, "station_id", None) is not None:
        return str(getattr(station, "station_id"))
    number = getattr(station, "id", getattr(station, "_id", None))
    if number is not None:
        return str(number)
    pair = coordinate_pair(station)
    return "" if pair is None else f"{pair[0]}:{pair[1]}"


def _build_vrsla_slot_scores(
    *,
    warehouse: Any,
    storages: tuple[Any, ...],
    storage_coordinate_hash: str,
    loaded_index: RTSGraphDistanceIndex,
) -> dict[str, Any]:
    _DIAGNOSTICS.vrsla_slot_score_build_count += 1
    stations = tuple(getattr(getattr(warehouse, "station_manager", None), "stations", []) or ())
    pickers = tuple(
        station
        for station in stations
        if str(getattr(station, "station_type", "")).strip().lower() in {"picker", "picking"}
    )
    cycle_by_picker: dict[str, Mapping[str, float]] = {}
    rank_by_picker: dict[str, Mapping[str, float]] = {}
    sorted_by_picker: dict[str, tuple[str, ...]] = {}
    identity_payload = {
        "version": VRSLA_SLOT_SCORE_VERSION,
        "loaded_graph_identity_hash": loaded_index.graph_identity_hash,
        "storage_coordinate_hash": storage_coordinate_hash,
        "pickers": [
            [stable_station_id(picker), *(coordinate_pair(picker) or (None, None))]
            for picker in sorted(pickers, key=stable_station_id)
        ],
    }
    for picker in sorted(pickers, key=stable_station_id):
        picker_id = stable_station_id(picker)
        picker_node = node_id_for_coordinate(picker)
        rows: list[tuple[float, tuple[int, str], str]] = []
        distances: dict[str, float] = {}
        for storage in storages:
            storage_id = stable_storage_id(storage)
            storage_node = node_id_for_coordinate(storage)
            out_leg = (
                loaded_index.distance_between_nodes(storage_node, picker_node)
                if storage_node is not None and picker_node is not None
                else None
            )
            return_leg = (
                loaded_index.distance_between_nodes(picker_node, storage_node)
                if storage_node is not None and picker_node is not None
                else None
            )
            distance = (
                float(out_leg) + float(return_leg)
                if out_leg is not None and return_leg is not None
                else math.inf
            )
            distances[storage_id] = distance
            rows.append((distance, _stable_id_key(storage_id), storage_id))
        ordered = [storage_id for _distance, _key, storage_id in sorted(rows, key=lambda item: (item[0], item[1]))]
        if not ordered:
            ranks = {}
        elif len(ordered) == 1:
            ranks = {ordered[0]: 1.0}
        else:
            denom = float(len(ordered) - 1)
            ranks = {storage_id: 1.0 - (index / denom) for index, storage_id in enumerate(ordered)}
        cycle_by_picker[picker_id] = MappingProxyType(distances)
        rank_by_picker[picker_id] = MappingProxyType(ranks)
        sorted_by_picker[picker_id] = tuple(ordered)
    return {
        "identity": _digest(identity_payload),
        "cycle_distance": MappingProxyType(cycle_by_picker),
        "hotness_rank": MappingProxyType(rank_by_picker),
        "slots_by_picker": MappingProxyType(sorted_by_picker),
    }


def _stable_id_key(value: str) -> tuple[int, str]:
    try:
        return int(value), value
    except Exception:
        return 10**12, value


def _build_graph_distance_index(warehouse: Any, topology: str) -> RTSGraphDistanceIndex:
    _DIAGNOSTICS.graph_matrix_build_count += 1
    if str(topology) == EMPTY_ROBOT:
        _DIAGNOSTICS.empty_matrix_build_count += 1
    elif str(topology) == LOADED_ROBOT:
        _DIAGNOSTICS.loaded_matrix_build_count += 1
    graph_wrapper = graph_wrapper_for_topology(warehouse, topology)
    graph = getattr(graph_wrapper, "graph", None)
    if graph is None:
        raise RuntimeError(f"RTS static index requires warehouse graph for {topology}")
    nodes = tuple(sorted(str(node) for node in graph.nodes()))
    node_to_index = {node: index for index, node in enumerate(nodes)}
    graph_hash = _graph_identity_hash(graph, topology)
    matrix = np.full((len(nodes), len(nodes)), np.inf, dtype=np.float32)
    if nodes:
        np.fill_diagonal(matrix, 0.0)
    _validate_nonnegative_finite_weights(graph, topology)
    import networkx as nx

    for source in nodes:
        lengths = nx.single_source_dijkstra_path_length(graph, source, weight="weight")
        src_index = node_to_index[source]
        for target, distance in lengths.items():
            dst_index = node_to_index.get(str(target))
            if dst_index is not None:
                matrix[src_index, dst_index] = np.float32(distance)
    return RTSGraphDistanceIndex(
        topology=str(topology),
        graph_identity_hash=graph_hash,
        node_to_index=MappingProxyType(node_to_index),
        index_to_node=nodes,
        distance_matrix=matrix,
        node_count=len(nodes),
        edge_count=int(graph.number_of_edges()),
        matrix_bytes=int(matrix.nbytes),
    )


def _validate_nonnegative_finite_weights(graph: Any, topology: str) -> None:
    for src, dst, data in graph.edges(data=True):
        weight = data.get("weight", 1.0)
        try:
            value = float(weight)
        except Exception as exc:
            raise RuntimeError(f"non-numeric RTS graph weight for {topology}: {src}->{dst}") from exc
        if not math.isfinite(value) or value < 0.0:
            raise RuntimeError(f"negative or non-finite RTS graph weight for {topology}: {src}->{dst}={weight!r}")


def _graph_identity_hash(graph: Any, topology: str) -> str:
    _DIAGNOSTICS.graph_hash_count += 1
    if graph is None:
        return ""
    payload = {
        "topology": str(topology),
        "directed": bool(getattr(graph, "is_directed", lambda: False)()),
        "nodes": sorted(str(node) for node in graph.nodes()),
        "edges": sorted(
            (
                str(src),
                str(dst),
                f"{float(data.get('weight', 1.0)):.9f}",
            )
            for src, dst, data in graph.edges(data=True)
        ),
    }
    return _digest(payload)


def _layout_identity_hash(warehouse: Any) -> str:
    _DIAGNOSTICS.layout_hash_count += 1
    layout = getattr(warehouse, "layout", None)
    storage_manager = getattr(warehouse, "storage_manager", None)
    storages = tuple(getattr(storage_manager, "storages", []) or ())
    payload = {
        "layout_class": type(layout).__name__ if layout is not None else None,
        "storage_coordinates": [
            [stable_storage_id(storage), *(coordinate_pair(storage) or (None, None))]
            for storage in sorted(storages, key=stable_storage_id)
        ],
    }
    return _digest(payload)


def _storage_coordinate_hash(storages: tuple[Any, ...]) -> str:
    _DIAGNOSTICS.storage_hash_count += 1
    payload = [
        [stable_storage_id(storage), *(coordinate_pair(storage) or (None, None))]
        for storage in sorted(storages, key=stable_storage_id)
    ]
    return _digest(payload)


def _layout_normalization_metadata(warehouse: Any) -> dict[str, float]:
    coords = [coordinate_pair(obj) for obj in getattr(warehouse, "_objects", []) or []]
    coords = [coord for coord in coords if coord is not None]
    xs = [coord[0] for coord in coords]
    ys = [coord[1] for coord in coords]
    return {
        "x_min": float(min(xs)) if xs else 0.0,
        "x_max": float(max(xs)) if xs else 1.0,
        "y_min": float(min(ys)) if ys else 0.0,
        "y_max": float(max(ys)) if ys else 1.0,
    }


def _macro_region_for_storage(storage: Any, normalization: Mapping[str, float]) -> str:
    pair = coordinate_pair(storage)
    if pair is None:
        return ""
    x_min = float(normalization.get("x_min", 0.0))
    x_max = float(normalization.get("x_max", 1.0))
    y_min = float(normalization.get("y_min", 0.0))
    y_max = float(normalization.get("y_max", 1.0))
    x_norm = 0.0 if x_max <= x_min else (float(pair[0]) - x_min) / (x_max - x_min)
    y_norm = 0.0 if y_max <= y_min else (float(pair[1]) - y_min) / (y_max - y_min)
    col = min(2, int(max(0.0, min(1.0, x_norm)) * 3.0))
    row = min(2, int(max(0.0, min(1.0, y_norm)) * 3.0))
    return f"mr3x3_r{row}_c{col}"


def _digest(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()[:16]


def _clear_pair_distance_cache(warehouse: Any) -> None:
    cache = getattr(warehouse, "_rts_rl_distance_cache", None)
    if cache is None:
        return
    try:
        cache._cache.clear()
    except Exception:
        pass


def static_runtime_index_diagnostics() -> dict[str, int]:
    return _DIAGNOSTICS.to_json_dict()


def reset_static_runtime_index_diagnostics() -> None:
    global _DIAGNOSTICS
    _DIAGNOSTICS = RTSStaticRuntimeIndexDiagnostics()
