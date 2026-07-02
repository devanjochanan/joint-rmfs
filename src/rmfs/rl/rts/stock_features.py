"""Stock-risk feature extraction for RTS-RL."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np


STOCK_SOURCE_VERSION = "rts_stock_source.v2"
STOCK_FEATURE_SCHEMA_VERSION = "rts_stock_features.v3"

STOCK_FEATURE_NAMES: tuple[str, ...] = (
    "local_fill_ratio",
    "local_below_threshold",
    "global_fill_ratio",
    "global_below_threshold",
)


def stock_rows_from_pod(
    pod: Any,
    warehouse: Any | None = None,
    *,
    strict_global: bool = True,
) -> list[dict[str, float | str]]:
    rows = []
    for sku_id, details in sorted((getattr(pod, "skus", {}) or {}).items(), key=lambda item: str(item[0])):
        details = dict(details or {})
        current_qty = _finite_float(details.get("current_qty", 0.0))
        limit_qty = _finite_float(details.get("limit_qty", 0.0))
        threshold = _finite_float(details.get("threshold", 0.0))
        local_fill = _ratio(current_qty, limit_qty)
        global_data = _global_sku_data(warehouse, sku_id, strict=strict_global)
        global_current = _finite_float(global_data.get("current_global_qty", 0.0))
        global_limit = _finite_float(global_data.get("max_global_qty", 0.0))
        global_threshold = _finite_float(global_data.get("global_threshold_inv_level", 0.0))
        global_fill = _ratio(global_current, global_limit)
        local_below = 1.0 if local_fill <= threshold else 0.0
        global_below = 1.0 if global_fill <= global_threshold else 0.0
        local_zero = 1.0 if current_qty <= 0.0 else 0.0
        rows.append(
            {
                "sku_id": str(sku_id),
                "local_current_qty": current_qty,
                "local_limit_qty": limit_qty,
                "local_threshold": threshold,
                "global_current_qty": global_current,
                "global_limit_qty": global_limit,
                "global_threshold": global_threshold,
                "local_fill_ratio": _bounded(local_fill),
                "local_shortage_depth": _bounded(max(0.0, threshold - local_fill)),
                "local_below_threshold": local_below,
                "local_zero_qty": local_zero,
                "global_fill_ratio": _bounded(global_fill),
                "global_shortage_depth": _bounded(max(0.0, global_threshold - global_fill)),
                "global_below_threshold": global_below,
                "local_zero_and_global_low": 1.0 if local_zero and global_below else 0.0,
            }
        )
    return rows


def derive_stock_feature_row(stock_row: Mapping[str, Any]) -> dict[str, float]:
    row = dict(stock_row or {})
    return {name: _bounded(_finite_float(row.get(name, 0.0))) for name in STOCK_FEATURE_NAMES}


def stock_summary(stock_rows: list[Mapping[str, Any]]) -> dict[str, float]:
    if not stock_rows:
        return {
            "pod_fill_ratio": 0.0,
            "capacity_weighted_pod_fill_ratio": 0.0,
            "pod_below_threshold_ratio": 0.0,
            "pod_global_low_ratio": 0.0,
            "replenishment_signal_active": 0.0,
            "pod_has_zero_and_global_low_sku": 0.0,
            "zero_global_low_sku_count": 0.0,
            "zero_global_low_sku_ratio": 0.0,
            "below_threshold_sku_count": 0.0,
            "below_threshold_sku_ratio": 0.0,
            "global_low_sku_count": 0.0,
            "global_low_sku_ratio": 0.0,
            "min_local_fill_ratio": 0.0,
            "min_sku_fill_ratio": 0.0,
            "mean_sku_fill_ratio": 0.0,
        }
    rows = [dict(row) for row in stock_rows]
    count = float(len(rows))
    local_qty_sum = sum(_finite_float(row.get("local_current_qty", 0.0)) for row in rows)
    local_limit_sum = sum(_finite_float(row.get("local_limit_qty", 0.0)) for row in rows)
    fill = [_bounded(_finite_float(row.get("local_fill_ratio", 0.0))) for row in rows]
    below_count = float(sum(1 for row in rows if _finite_float(row.get("local_below_threshold", 0.0)) > 0.0))
    global_low_count = float(sum(1 for row in rows if _finite_float(row.get("global_below_threshold", 0.0)) > 0.0))
    zero_global_count = float(sum(1 for row in rows if _finite_float(row.get("local_zero_and_global_low", 0.0)) > 0.0))
    capacity_weighted = _ratio(local_qty_sum, local_limit_sum)
    return {
        "pod_fill_ratio": _bounded(capacity_weighted),
        "capacity_weighted_pod_fill_ratio": _bounded(capacity_weighted),
        "pod_below_threshold_ratio": _bounded(below_count / count),
        "pod_global_low_ratio": _bounded(global_low_count / count),
        "replenishment_signal_active": 1.0 if below_count > 0.0 or zero_global_count > 0.0 else 0.0,
        "pod_has_zero_and_global_low_sku": 1.0 if zero_global_count > 0.0 else 0.0,
        "zero_global_low_sku_count": zero_global_count,
        "zero_global_low_sku_ratio": _bounded(zero_global_count / count),
        "below_threshold_sku_count": below_count,
        "below_threshold_sku_ratio": _bounded(below_count / count),
        "global_low_sku_count": global_low_count,
        "global_low_sku_ratio": _bounded(global_low_count / count),
        "min_local_fill_ratio": float(min(fill)),
        "min_sku_fill_ratio": float(min(fill)),
        "mean_sku_fill_ratio": _bounded(sum(fill) / count),
    }


def build_stock_feature_matrix(stock_rows: list[Mapping[str, Any]]) -> np.ndarray:
    sorted_rows = sorted((dict(row) for row in stock_rows), key=lambda row: str(row.get("sku_id", "")))
    if not sorted_rows:
        return np.zeros((0, len(STOCK_FEATURE_NAMES)), dtype=np.float32)
    return np.asarray(
        [[derive_stock_feature_row(row)[name] for name in STOCK_FEATURE_NAMES] for row in sorted_rows],
        dtype=np.float32,
    )


def stock_source_metadata() -> dict[str, Any]:
    return {
        "stock_source_version": STOCK_SOURCE_VERSION,
        "stock_feature_schema_version": STOCK_FEATURE_SCHEMA_VERSION,
        "local_stock_source": "pod.skus[sku]",
        "global_stock_source": "warehouse.pod_manager.skus_data[sku]",
    }


def _global_sku_data(warehouse: Any | None, sku_id: Any, *, strict: bool) -> Mapping[str, Any]:
    pod_manager = getattr(warehouse, "pod_manager", None)
    all_global = getattr(pod_manager, "skus_data", None)
    if isinstance(all_global, Mapping) and sku_id in all_global:
        return dict(all_global[sku_id] or {})
    try:
        int_sku = int(sku_id)
    except Exception:
        int_sku = None
    if isinstance(all_global, Mapping) and int_sku is not None and int_sku in all_global:
        return dict(all_global[int_sku] or {})
    if strict:
        raise ValueError(f"missing RTS-RL global SKU data for pod SKU {sku_id!r}")
    return {}


def _ratio(numerator: Any, denominator: Any) -> float:
    denom = _finite_float(denominator)
    if denom <= 0.0:
        return 0.0
    return _finite_float(numerator) / denom


def _bounded(value: Any) -> float:
    return max(0.0, min(1.0, _finite_float(value)))


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        normalized = float(value)
    except Exception:
        return default
    if not np.isfinite(normalized):
        return default
    return float(normalized)
