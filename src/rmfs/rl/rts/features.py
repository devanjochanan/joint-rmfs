"""Model-ready RTS-RL action and stock feature matrices."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .action_space import REPLENISH_STORE, STORE, decode_action, validate_action_mask
from .stock_features import STOCK_FEATURE_NAMES, build_stock_feature_matrix, stock_summary
from .validation import validate_no_raw_threshold_features


ACTION_FEATURE_BASE_NAMES: tuple[str, ...] = (
    "is_store_action",
    "is_replenish_store_action",
    "turnover_rank",
    "turnover_value",
    "estimated_queue_time",
    "next_retrieval_zone_known",
    "source_station_is_picking",
    "source_station_is_replenishment",
    "source_station_x_norm",
    "source_station_y_norm",
    "picking_station_count",
    "replenishment_station_count",
    "pod_fill_ratio",
    "pod_below_threshold_ratio",
    "replenishment_signal_active",
    "pod_has_zero_and_global_low_sku",
    "zero_global_low_sku_count",
    "zero_global_low_sku_ratio",
    "below_threshold_sku_count",
    "below_threshold_sku_ratio",
    "shortage_depth",
    "global_low_depth",
    "min_sku_fill_ratio",
    "mean_sku_fill_ratio",
    "zone_row_norm",
    "zone_col_norm",
    "occupation_level",
    "free_slot_count",
    "zone_destination_robot_count",
    "neighbor_zone_destination_robot_count",
    "superzone_destination_robot_count",
    "zone_present_robot_count",
    "neighbor_zone_present_robot_count",
    "superzone_present_robot_count",
    "storage_cycle_time_estimate",
    "replenish_cycle_time_estimate",
    "sku_similarity",
    "selected_replenishment_station_x_norm",
    "selected_replenishment_station_y_norm",
    "selected_replenishment_station_logical_load",
    "candidate_zone_to_selected_replenishment_station_distance",
    "candidate_zone_to_nearest_replenishment_station_distance",
    "total_robot_count",
    "active_pod_total",
    "arrival_rate_order_cycle_time",
    "store_action_valid",
    "replenish_store_action_valid",
)


@dataclass(frozen=True)
class RTSFeatureBundle:
    X_actions: np.ndarray
    M_actions: np.ndarray
    X_stock: np.ndarray
    M_stock: np.ndarray
    action_feature_names: tuple[str, ...]
    stock_feature_names: tuple[str, ...]
    zone_ids: tuple[str, ...]


def build_action_feature_names(zone_ids: Sequence[str]) -> tuple[str, ...]:
    names = list(ACTION_FEATURE_BASE_NAMES[:6])
    names.extend(f"next_retrieval_zone_one_hot__{zone_id}" for zone_id in zone_ids)
    names.extend(ACTION_FEATURE_BASE_NAMES[6:])
    validate_no_raw_threshold_features(names)
    return tuple(names)


def build_stock_feature_names() -> tuple[str, ...]:
    validate_no_raw_threshold_features(STOCK_FEATURE_NAMES)
    return STOCK_FEATURE_NAMES


def build_action_feature_matrix(
    zone_ids: Sequence[str],
    action_mask: Sequence[int],
    state_json: Mapping[str, Any],
) -> np.ndarray:
    zones = tuple(str(zone_id) for zone_id in zone_ids)
    mask = validate_action_mask(zones, action_mask, require_valid=False)
    zone_rows = list(state_json.get("zone_rows", []) or [])
    if len(zone_rows) != len(zones):
        raise ValueError("RTS-RL zone_rows must align with zone_ids")
    stock_rows = list(state_json.get("stock_rows", []) or [])
    stock = stock_summary(stock_rows)
    spatial = dict(state_json.get("spatial_context", {}) or {})
    next_hot = dict(state_json.get("next_retrieval_zone_one_hot", {}) or {})
    names = build_action_feature_names(zones)
    rows = []
    for action_index in range(len(mask)):
        action = decode_action(action_index, zones)
        zone_row = dict(zone_rows[zones.index(action.zone_id)])
        branch_values = [1.0 if action.branch == STORE else 0.0, 1.0 if action.branch == REPLENISH_STORE else 0.0]
        values = [
            *branch_values,
            _float(state_json.get("turnover_rank", 0.0)),
            _float(state_json.get("turnover_value", 0.0)),
            _float(state_json.get("estimated_queue_time", 0.0)),
            _float(state_json.get("next_retrieval_zone_known", 0.0)),
        ]
        values.extend(_float(next_hot.get(zone_id, 0.0)) for zone_id in zones)
        values.extend(
            [
                _float(spatial.get("source_station_is_picking", 0.0)),
                _float(spatial.get("source_station_is_replenishment", 0.0)),
                _float(spatial.get("source_station_x_norm", 0.0)),
                _float(spatial.get("source_station_y_norm", 0.0)),
                _float(spatial.get("picking_station_count", 0.0)),
                _float(spatial.get("replenishment_station_count", 0.0)),
                stock["pod_fill_ratio"],
                stock["pod_below_threshold_ratio"],
                stock["replenishment_signal_active"],
                stock["pod_has_zero_and_global_low_sku"],
                stock["zero_global_low_sku_count"],
                stock["zero_global_low_sku_ratio"],
                stock["below_threshold_sku_count"],
                stock["below_threshold_sku_ratio"],
                _stock_sum(stock_rows, "shortage_depth"),
                _stock_sum(stock_rows, "global_low_depth"),
                stock["min_sku_fill_ratio"],
                stock["mean_sku_fill_ratio"],
                _zone_norm(zone_row.get("zone_row_index", 0.0), spatial.get("zone_row_min", 0.0), spatial.get("zone_row_max", 1.0)),
                _zone_norm(zone_row.get("zone_col_index", 0.0), spatial.get("zone_col_min", 0.0), spatial.get("zone_col_max", 1.0)),
                _float(zone_row.get("occupation_level", 0.0)),
                _float(zone_row.get("free_slot_count", 0.0)),
                _float(zone_row.get("zone_destination_robot_count", 0.0)),
                _float(zone_row.get("neighbor_zone_destination_robot_count", 0.0)),
                _float(zone_row.get("superzone_destination_robot_count", 0.0)),
                _float(zone_row.get("zone_present_robot_count", 0.0)),
                _float(zone_row.get("neighbor_zone_present_robot_count", 0.0)),
                _float(zone_row.get("superzone_present_robot_count", 0.0)),
                _float(zone_row.get("storage_cycle_time_estimate", 0.0)),
                _float(zone_row.get("replenish_cycle_time_estimate", 0.0)),
                _float(zone_row.get("sku_similarity", 0.0)),
                _float(spatial.get("selected_replenishment_station_x_norm", 0.0)),
                _float(spatial.get("selected_replenishment_station_y_norm", 0.0)),
                _float(spatial.get("selected_replenishment_station_logical_load", 0.0)),
                _float(zone_row.get("candidate_zone_to_selected_replenishment_station_distance", 0.0)),
                _float(zone_row.get("candidate_zone_to_nearest_replenishment_station_distance", 0.0)),
                _float(spatial.get("total_robot_count", 0.0)),
                _float(spatial.get("active_pod_total", 0.0)),
                _float(spatial.get("arrival_rate_order_cycle_time", 0.0)),
                _float(zone_row.get("store_action_valid", 0.0)),
                _float(zone_row.get("replenish_store_action_valid", 0.0)),
            ]
        )
        if len(values) != len(names):
            raise ValueError(f"RTS-RL action feature width mismatch: {len(values)} != {len(names)}")
        rows.append(values)
    return np.asarray(rows, dtype=np.float32)


def build_feature_bundle(zone_ids: Sequence[str], action_mask: Sequence[int], state_json: Mapping[str, Any]) -> RTSFeatureBundle:
    zones = tuple(str(zone_id) for zone_id in zone_ids)
    X_actions = build_action_feature_matrix(zones, action_mask, state_json)
    M_actions = np.asarray(validate_action_mask(zones, action_mask, require_valid=False), dtype=np.int64)
    X_stock = build_stock_feature_matrix(list(state_json.get("stock_rows", []) or []))
    M_stock = np.ones((X_stock.shape[0],), dtype=np.int64)
    return RTSFeatureBundle(
        X_actions=X_actions,
        M_actions=M_actions,
        X_stock=X_stock,
        M_stock=M_stock,
        action_feature_names=build_action_feature_names(zones),
        stock_feature_names=build_stock_feature_names(),
        zone_ids=zones,
    )


def compute_feature_standardization(feature_matrix: np.ndarray) -> tuple[tuple[float, ...], tuple[float, ...]]:
    if feature_matrix.ndim != 2 or feature_matrix.shape[0] == 0:
        raise ValueError("RTS-RL standardization requires a non-empty 2D matrix")
    means = feature_matrix.mean(axis=0)
    stds = np.where(feature_matrix.std(axis=0) < 1e-6, 1.0, feature_matrix.std(axis=0))
    return tuple(float(x) for x in means.tolist()), tuple(float(x) for x in stds.tolist())


def standardize_feature_matrix(feature_matrix: np.ndarray, means: Sequence[float], stds: Sequence[float]) -> np.ndarray:
    validate_feature_standardization(feature_matrix.shape[1], means, stds)
    return ((feature_matrix.astype(np.float32) - np.asarray(means, dtype=np.float32)) / np.asarray(stds, dtype=np.float32)).astype(np.float32)


def validate_feature_standardization(width: int, means: Sequence[float], stds: Sequence[float]) -> None:
    if len(means) != width or len(stds) != width:
        raise ValueError("RTS-RL feature standardization length mismatch")
    if any(float(std) <= 0.0 for std in stds):
        raise ValueError("RTS-RL feature standard deviations must be positive")


def _float(value: Any) -> float:
    try:
        result = float(value)
    except Exception:
        return 0.0
    return float(result) if np.isfinite(result) else 0.0


def _zone_norm(value: Any, low: Any, high: Any) -> float:
    lo = _float(low)
    hi = _float(high)
    span = max(1.0, hi - lo)
    return (_float(value) - lo) / span


def _stock_sum(stock_rows: list[Mapping[str, Any]], name: str) -> float:
    return float(sum(_float(row.get(name, 0.0)) for row in stock_rows))
