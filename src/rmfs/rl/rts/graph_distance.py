"""Directed graph-distance helpers for RTS-RL features and resolvers."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any


DISTANCE_SEMANTICS_VERSION = "rts_directed_graph_distance.v1"
DISTANCE_SOURCE_GRAPH = "directed_graph_shortest_path"
DISTANCE_SOURCE_CACHE_HIT = "directed_graph_shortest_path_cache_hit"
DISTANCE_SOURCE_EXPLICIT_FALLBACK = "explicit_metric_fallback_unavailable_graph"
DISTANCE_STATUS_AVAILABLE = "available"
DISTANCE_STATUS_FALLBACK = "fallback_explicit"
DISTANCE_STATUS_UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class RTSDistanceResult:
    distance: float | None
    source: str
    status: str
    fallback_used: bool = False

    @property
    def value_or_zero(self) -> float:
        return float(self.distance) if self.distance is not None else 0.0


class RTSDistanceCache:
    """Small in-memory directed shortest-path cache bound to a warehouse graph."""

    def __init__(self, warehouse: Any):
        self.warehouse = warehouse
        self._cache: dict[tuple[int, int, int, int], RTSDistanceResult] = {}
        self.hit_count = 0
        self.miss_count = 0
        self.graph_compute_count = 0
        self.fallback_count = 0

    def path_distance(
        self,
        src: Any,
        dst: Any,
        *,
        allow_metric_fallback: bool = True,
    ) -> RTSDistanceResult:
        key = _directed_key(src, dst)
        if key is None:
            return RTSDistanceResult(
                distance=None,
                source="invalid_coordinate",
                status=DISTANCE_STATUS_UNAVAILABLE,
            )
        if key in self._cache:
            self.hit_count += 1
            cached = self._cache[key]
            return RTSDistanceResult(
                distance=cached.distance,
                source=DISTANCE_SOURCE_CACHE_HIT if cached.source == DISTANCE_SOURCE_GRAPH else cached.source,
                status=cached.status,
                fallback_used=cached.fallback_used,
            )
        self.miss_count += 1
        self.graph_compute_count += 1
        graph_distance = _graph_shortest_path_length(self.warehouse, key)
        if graph_distance is not None:
            result = RTSDistanceResult(
                distance=graph_distance,
                source=DISTANCE_SOURCE_GRAPH,
                status=DISTANCE_STATUS_AVAILABLE,
            )
            self._cache[key] = result
            return result
        if allow_metric_fallback:
            self.fallback_count += 1
            result = RTSDistanceResult(
                distance=_metric_distance(key),
                source=DISTANCE_SOURCE_EXPLICIT_FALLBACK,
                status=DISTANCE_STATUS_FALLBACK,
                fallback_used=True,
            )
            self._cache[key] = result
            return result
        return RTSDistanceResult(
            distance=None,
            source="directed_graph_unavailable",
            status=DISTANCE_STATUS_UNAVAILABLE,
        )

    def cycle_distance(
        self,
        src: Any,
        dst: Any,
        *,
        allow_metric_fallback: bool = True,
    ) -> RTSDistanceResult:
        forward = self.path_distance(src, dst, allow_metric_fallback=allow_metric_fallback)
        backward = self.path_distance(dst, src, allow_metric_fallback=allow_metric_fallback)
        if forward.distance is None or backward.distance is None:
            return RTSDistanceResult(
                distance=None,
                source=f"{forward.source}+{backward.source}",
                status=DISTANCE_STATUS_UNAVAILABLE,
                fallback_used=forward.fallback_used or backward.fallback_used,
            )
        fallback_used = forward.fallback_used or backward.fallback_used
        status = DISTANCE_STATUS_FALLBACK if fallback_used else DISTANCE_STATUS_AVAILABLE
        source = f"{forward.source}+{backward.source}" if forward.source != backward.source else forward.source
        return RTSDistanceResult(
            distance=float(forward.distance + backward.distance),
            source=source,
            status=status,
            fallback_used=fallback_used,
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "distance_semantics_version": DISTANCE_SEMANTICS_VERSION,
            "distance_topology": "warehouse.graph_pod",
            "distance_source": "directed graph shortest path with explicit fallback status",
            "distance_cache_hit_count": int(self.hit_count),
            "distance_cache_miss_count": int(self.miss_count),
            "distance_graph_compute_count": int(self.graph_compute_count),
            "distance_fallback_count": int(self.fallback_count),
        }


def get_distance_cache(warehouse: Any) -> RTSDistanceCache | None:
    if warehouse is None:
        return None
    cache = getattr(warehouse, "_rts_rl_distance_cache", None)
    if cache is None:
        cache = RTSDistanceCache(warehouse)
        setattr(warehouse, "_rts_rl_distance_cache", cache)
    return cache


def distance_cache_metadata(warehouse: Any) -> dict[str, Any]:
    cache = get_distance_cache(warehouse)
    if cache is None:
        return {
            "distance_semantics_version": DISTANCE_SEMANTICS_VERSION,
            "distance_topology": "warehouse.graph_pod",
            "distance_source": "unavailable",
            "distance_fallback_count": 0,
        }
    return cache.metadata()


def graph_distance_or_fallback(
    warehouse: Any,
    src: Any,
    dst: Any,
    *,
    allow_metric_fallback: bool = True,
) -> RTSDistanceResult:
    cache = get_distance_cache(warehouse)
    if cache is None:
        return RTSDistanceResult(
            distance=None,
            source="warehouse_unavailable",
            status=DISTANCE_STATUS_UNAVAILABLE,
        )
    return cache.path_distance(src, dst, allow_metric_fallback=allow_metric_fallback)


def graph_cycle_distance_or_fallback(
    warehouse: Any,
    src: Any,
    dst: Any,
    *,
    allow_metric_fallback: bool = True,
) -> RTSDistanceResult:
    cache = get_distance_cache(warehouse)
    if cache is None:
        return RTSDistanceResult(
            distance=None,
            source="warehouse_unavailable",
            status=DISTANCE_STATUS_UNAVAILABLE,
        )
    return cache.cycle_distance(src, dst, allow_metric_fallback=allow_metric_fallback)


def _graph_shortest_path_length(warehouse: Any, key: tuple[int, int, int, int]) -> float | None:
    graph_wrapper = getattr(warehouse, "graph_pod", None)
    graph = getattr(graph_wrapper, "graph", None)
    if graph is None:
        return None
    src = f"{key[0]},{key[1]}"
    dst = f"{key[2]},{key[3]}"
    try:
        if src not in graph or dst not in graph:
            return None
        distance = _networkx_shortest_path_length(graph, src, dst)
    except Exception:
        return None
    if distance is None or not math.isfinite(distance) or distance < 0.0:
        return None
    return float(distance)


def _networkx_shortest_path_length(graph: Any, src: str, dst: str) -> float | None:
    try:
        import networkx as nx

        return float(nx.shortest_path_length(graph, source=src, target=dst, weight="weight"))
    except Exception:
        return None


def _directed_key(src: Any, dst: Any) -> tuple[int, int, int, int] | None:
    src_pair = _coordinate_pair(src)
    dst_pair = _coordinate_pair(dst)
    if src_pair is None or dst_pair is None:
        return None
    return src_pair[0], src_pair[1], dst_pair[0], dst_pair[1]


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


def _metric_distance(key: tuple[int, int, int, int]) -> float:
    return float(abs(key[0] - key[2]) + abs(key[1] - key[3]))

