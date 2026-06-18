"""Stock-risk feature extraction for RTS-RL."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np


STOCK_FEATURE_NAMES: tuple[str, ...] = (
    "current_qty",
    "limit_qty",
    "fill_ratio",
    "pod_below_threshold",
    "below_threshold",
    "is_zero_qty",
    "is_zero_and_global_low",
    "shortage_depth",
    "global_low_depth",
)


def stock_rows_from_pod(pod: Any) -> list[dict[str, float]]:
    rows = []
    for sku_id, details in (getattr(pod, "skus", {}) or {}).items():
        details = dict(details or {})
        current_qty = _finite_float(details.get("current_qty", 0.0))
        limit_qty = _finite_float(details.get("limit_qty", 0.0))
        fill_ratio = current_qty / limit_qty if limit_qty > 0.0 else 0.0
        threshold = _finite_float(details.get("threshold", 0.0))
        global_inv_level = _finite_float(details.get("global_inv_level", 0.0))
        global_low = _finite_float(details.get("global_low_depth", 0.0))
        below = 1.0 if fill_ratio <= threshold else 0.0
        rows.append(
            {
                "sku_id": str(sku_id),
                "current_qty": current_qty,
                "limit_qty": limit_qty,
                "fill_ratio": fill_ratio,
                "pod_below_threshold": below,
                "below_threshold": below,
                "is_zero_qty": 1.0 if current_qty <= 0.0 else 0.0,
                "is_zero_and_global_low": 1.0 if current_qty <= 0.0 and (global_inv_level <= 0.0 or global_low > 0.0) else 0.0,
                "shortage_depth": max(0.0, threshold - fill_ratio),
                "global_low_depth": max(0.0, global_low),
            }
        )
    return rows


def derive_stock_feature_row(stock_row: Mapping[str, Any]) -> dict[str, float]:
    row = dict(stock_row or {})
    values = {name: _finite_float(row.get(name, 0.0)) for name in STOCK_FEATURE_NAMES}
    if values["limit_qty"] > 0.0 and "fill_ratio" not in row:
        values["fill_ratio"] = values["current_qty"] / values["limit_qty"]
    return values


def stock_summary(stock_rows: list[Mapping[str, Any]]) -> dict[str, float]:
    if not stock_rows:
        return {
            "pod_fill_ratio": 0.0,
            "pod_below_threshold_ratio": 0.0,
            "replenishment_signal_active": 0.0,
            "pod_has_zero_and_global_low_sku": 0.0,
            "zero_global_low_sku_count": 0.0,
            "zero_global_low_sku_ratio": 0.0,
            "below_threshold_sku_count": 0.0,
            "below_threshold_sku_ratio": 0.0,
            "min_sku_fill_ratio": 0.0,
            "mean_sku_fill_ratio": 0.0,
        }
    rows = [derive_stock_feature_row(row) for row in stock_rows]
    count = float(len(rows))
    fill = [row["fill_ratio"] for row in rows]
    below_count = float(sum(1 for row in rows if row["below_threshold"] > 0.0))
    zero_global_count = float(sum(1 for row in rows if row["is_zero_and_global_low"] > 0.0))
    return {
        "pod_fill_ratio": float(sum(fill) / count),
        "pod_below_threshold_ratio": float(below_count / count),
        "replenishment_signal_active": 1.0 if below_count > 0.0 or zero_global_count > 0.0 else 0.0,
        "pod_has_zero_and_global_low_sku": 1.0 if zero_global_count > 0.0 else 0.0,
        "zero_global_low_sku_count": zero_global_count,
        "zero_global_low_sku_ratio": float(zero_global_count / count),
        "below_threshold_sku_count": below_count,
        "below_threshold_sku_ratio": float(below_count / count),
        "min_sku_fill_ratio": float(min(fill)),
        "mean_sku_fill_ratio": float(sum(fill) / count),
    }


def build_stock_feature_matrix(stock_rows: list[Mapping[str, Any]]) -> np.ndarray:
    sorted_rows = sorted((dict(row) for row in stock_rows), key=lambda row: str(row.get("sku_id", "")))
    if not sorted_rows:
        return np.zeros((0, len(STOCK_FEATURE_NAMES)), dtype=np.float32)
    return np.asarray(
        [[derive_stock_feature_row(row)[name] for name in STOCK_FEATURE_NAMES] for row in sorted_rows],
        dtype=np.float32,
    )


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        normalized = float(value)
    except Exception:
        return default
    if not np.isfinite(normalized):
        return default
    return float(normalized)
