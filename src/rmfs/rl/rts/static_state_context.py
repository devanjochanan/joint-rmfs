"""Run-stable RTS-RL state context snapshots."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd


HISTORICAL_POD_RANK_VERSION = "rts_historical_pod_request_rank.v1"
LAYOUT_NORMALIZATION_VERSION = "rts_layout_normalization.v1"
DISTANCE_NORMALIZATION_VERSION = "rts_distance_normalization.v1"


@dataclass(frozen=True)
class RTSStaticStateContext:
    pod_request_rank: dict[str, float]
    pod_request_count: dict[str, int]
    historical_metadata: dict[str, Any]
    layout_metadata: dict[str, Any]
    distance_metadata: dict[str, Any]

    def pod_rank(self, pod: Any) -> float:
        return float(self.pod_request_rank.get(str(getattr(pod, "pod_id", "")), 0.0))

    def pod_count(self, pod: Any) -> int:
        return int(self.pod_request_count.get(str(getattr(pod, "pod_id", "")), 0))

    def norm_x(self, value: Any) -> float:
        return _norm(value, self.layout_metadata["x_min"], self.layout_metadata["x_max"])

    def norm_y(self, value: Any) -> float:
        return _norm(value, self.layout_metadata["y_min"], self.layout_metadata["y_max"])

    def norm_distance(self, value: Any) -> float:
        denom = max(1.0, float(self.distance_metadata.get("distance_normalization_denominator", 1.0)))
        return max(0.0, min(1.0, _float(value) / denom))


def get_or_build_static_state_context(warehouse: Any) -> RTSStaticStateContext:
    if warehouse is None:
        raise ValueError("RTS-RL state requires a warehouse to build static context")
    context = getattr(warehouse, "_rts_rl_static_state_context", None)
    if isinstance(context, RTSStaticStateContext):
        return context
    context = build_static_state_context(warehouse)
    setattr(warehouse, "_rts_rl_static_state_context", context)
    return context


def build_static_state_context(warehouse: Any) -> RTSStaticStateContext:
    pods = list(getattr(getattr(warehouse, "pod_manager", None), "pods", []) or [])
    order_source = _historical_order_source(warehouse)
    pod_request_count = _pod_request_counts(pods, order_source["orders"])
    pod_request_rank = _pod_request_ranks(pods, pod_request_count)
    layout_metadata = _layout_metadata(warehouse)
    distance_metadata = _distance_metadata(layout_metadata)
    historical_metadata = {
        "historical_pod_rank_version": HISTORICAL_POD_RANK_VERSION,
        "historical_source_identity": order_source["identity"],
        "historical_source_kind": order_source["kind"],
        "historical_source_hash": order_source["hash"],
        "valid_unique_source_order_count": order_source["valid_unique_source_order_count"],
        "rank_algorithm": "count distinct source orders intersecting pod positive or assigned SKU set",
        "tie_breaking_rule": "descending request_count then ascending stable pod_id",
        "pod_sku_allocation_identity": _pod_sku_allocation_hash(pods),
    }
    return RTSStaticStateContext(
        pod_request_rank=pod_request_rank,
        pod_request_count=pod_request_count,
        historical_metadata=historical_metadata,
        layout_metadata=layout_metadata,
        distance_metadata=distance_metadata,
    )


def _historical_order_source(warehouse: Any) -> dict[str, Any]:
    path_value = None
    try:
        path_value = warehouse.generated_order_csv
    except Exception:
        path_value = None
    if path_value:
        path = Path(path_value)
        if path.exists():
            df = pd.read_csv(path)
            orders = _orders_from_dataframe(df)
            return {
                "kind": "generated_order_csv",
                "identity": str(path),
                "hash": _stable_hash(orders),
                "valid_unique_source_order_count": len(orders),
                "orders": orders,
            }
    orders = _orders_from_order_manager(getattr(warehouse, "order_manager", None))
    return {
        "kind": "live_order_manager",
        "identity": "warehouse.order_manager.orders",
        "hash": _stable_hash(orders),
        "valid_unique_source_order_count": len(orders),
        "orders": orders,
    }


def _orders_from_dataframe(df: pd.DataFrame) -> dict[str, set[str]]:
    if df.empty:
        return {}
    order_col = "source_order_id" if "source_order_id" in df.columns else "order_id"
    sku_col = "item_id" if "item_id" in df.columns else "item"
    if order_col not in df.columns or sku_col not in df.columns:
        raise ValueError(f"historical order source lacks required columns: {list(df.columns)}")
    orders: dict[str, set[str]] = {}
    for row in df[[order_col, sku_col]].itertuples(index=False):
        order_id = str(getattr(row, order_col)).strip()
        sku_id = str(getattr(row, sku_col)).strip()
        if order_id and sku_id and order_id.lower() != "nan" and sku_id.lower() != "nan":
            orders.setdefault(order_id, set()).add(sku_id)
    return orders


def _orders_from_order_manager(order_manager: Any) -> dict[str, set[str]]:
    orders: dict[str, set[str]] = {}
    for order in getattr(order_manager, "orders", []) or []:
        order_id = str(getattr(order, "source_order_id", getattr(order, "order_id", ""))).strip()
        skus = {str(sku) for sku in (getattr(order, "skus", {}) or {}).keys()}
        if order_id and skus:
            orders[order_id] = skus
    return orders


def _pod_request_counts(pods: Iterable[Any], orders: Mapping[str, set[str]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for pod in pods:
        pod_id = str(getattr(pod, "pod_id", ""))
        pod_skus = {str(sku) for sku in (getattr(pod, "skus", {}) or {}).keys()}
        counts[pod_id] = sum(1 for skus in orders.values() if pod_skus.intersection(skus))
    return counts


def _pod_request_ranks(pods: Iterable[Any], counts: Mapping[str, int]) -> dict[str, float]:
    ordered = sorted(
        (str(getattr(pod, "pod_id", "")) for pod in pods),
        key=lambda pid: (-int(counts.get(pid, 0)), _stable_pod_id(pid)),
    )
    if not ordered:
        return {}
    if len(ordered) == 1:
        return {ordered[0]: 1.0}
    denom = float(len(ordered) - 1)
    return {pod_id: 1.0 - (index / denom) for index, pod_id in enumerate(ordered)}


def _layout_metadata(warehouse: Any) -> dict[str, Any]:
    coords: list[tuple[float, float, str]] = []
    for storage in getattr(getattr(warehouse, "storage_manager", None), "storages", []) or []:
        coords.append((_float(getattr(storage, "pos_x", 0.0)), _float(getattr(storage, "pos_y", 0.0)), "storage"))
    for station in getattr(getattr(warehouse, "station_manager", None), "stations", []) or []:
        stype = str(getattr(station, "station_type", "station"))
        if stype in {"picker", "replenishment"}:
            coords.append((_float(getattr(station, "pos_x", 0.0)), _float(getattr(station, "pos_y", 0.0)), f"{stype}_station"))
    if not coords:
        coords = [(0.0, 0.0, "fallback_origin")]
    xs = [item[0] for item in coords]
    ys = [item[1] for item in coords]
    included = tuple(sorted({item[2] for item in coords}))
    payload = {
        "layout_normalization_version": LAYOUT_NORMALIZATION_VERSION,
        "x_min": float(min(xs)),
        "x_max": float(max(xs)),
        "y_min": float(min(ys)),
        "y_max": float(max(ys)),
        "included_object_categories": list(included),
    }
    payload["layout_identity_hash"] = _stable_hash(
        {
            "bounds": {k: payload[k] for k in ("x_min", "x_max", "y_min", "y_max")},
            "coordinates": sorted((x, y, kind) for x, y, kind in coords),
        }
    )
    return payload


def _distance_metadata(layout_metadata: Mapping[str, Any]) -> dict[str, Any]:
    denominator = max(
        1.0,
        float(layout_metadata["x_max"]) - float(layout_metadata["x_min"])
        + float(layout_metadata["y_max"]) - float(layout_metadata["y_min"]),
    )
    return {
        "distance_normalization_version": DISTANCE_NORMALIZATION_VERSION,
        "distance_normalization_denominator": float(denominator),
        "distance_normalization_source": "layout manhattan span over RTS-relevant storages and stations",
    }


def _pod_sku_allocation_hash(pods: Iterable[Any]) -> str:
    payload = {
        str(getattr(pod, "pod_id", "")): sorted(str(sku) for sku in (getattr(pod, "skus", {}) or {}).keys())
        for pod in pods
    }
    return _stable_hash(payload)


def _stable_hash(payload: Any) -> str:
    normalized = {
        str(key): sorted(str(item) for item in value) if isinstance(value, set) else value
        for key, value in (payload.items() if isinstance(payload, Mapping) else {"payload": payload}.items())
    }
    return hashlib.sha256(
        json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]


def _stable_pod_id(value: str) -> tuple[int, str]:
    try:
        return int(value), value
    except Exception:
        return 10**12, value


def _norm(value: Any, low: Any, high: Any) -> float:
    lo = _float(low)
    hi = _float(high)
    span = max(1e-9, hi - lo)
    return max(0.0, min(1.0, (_float(value) - lo) / span))


def _float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0
