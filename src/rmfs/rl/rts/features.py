"""Model-ready RTS-RL action and stock feature matrices."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .action_space import REPLENISH_STORE, STORE, decode_action, validate_action_mask
from .stock_features import STOCK_FEATURE_NAMES, build_stock_feature_matrix, stock_summary
from .validation import validate_no_raw_threshold_features


ACTION_FEATURE_SCHEMA_VERSION = "rts_action_features.v3"

ACTION_FEATURE_BASE_NAMES: tuple[str, ...] = (
    "is_store_action",
    "is_replenish_store_action",
    "historical_pod_request_rank",
    "pod_fill_ratio",
    "pod_below_threshold_ratio",
    "pod_global_low_ratio",
    "pod_has_zero_and_global_low_sku",
    "min_local_fill_ratio",
    "source_station_is_picking",
    "source_station_is_replenishment",
    "source_station_x_norm",
    "source_station_y_norm",
    "zone_row_norm",
    "zone_col_norm",
    "occupation_level",
    "free_slot_ratio",
    "zone_destination_robot_pressure",
    "neighbor_zone_destination_robot_pressure",
    "superzone_destination_robot_pressure",
    "zone_present_robot_pressure",
    "neighbor_zone_present_robot_pressure",
    "superzone_present_robot_pressure",
    "sku_similarity_fraction",
    "store_action_valid",
    "replenish_store_action_valid",
    "next_job_known",
    "cycle_estimate_known",
    "estimated_cycle_time",
    "estimated_travel_time",
    "estimated_queue_time",
    "estimated_replenishment_service_time",
    "estimated_handling_time",
    "candidate_storage_to_next_pod_distance_norm",
    "next_pod_to_picker_distance_norm",
    "allocator_cost_norm",
    "regret_score_norm",
    "one_robot_degenerate",
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
    split_at = ACTION_FEATURE_BASE_NAMES.index("candidate_storage_to_next_pod_distance_norm")
    names = list(ACTION_FEATURE_BASE_NAMES[:split_at])
    names.extend(f"next_pod_zone_one_hot__{zone_id}" for zone_id in zone_ids)
    names.extend(ACTION_FEATURE_BASE_NAMES[split_at:])
    validate_no_raw_threshold_features(names)
    _validate_removed_placeholders_absent(names)
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
    action_proposals = dict(state_json.get("committed_next_action_proposals", {}) or {})
    action_context_rows = {
        int(row.get("action_index")): dict(row)
        for row in list(state_json.get("rts_action_contexts", []) or [])
        if row.get("action_index") is not None
    }
    names = build_action_feature_names(zones)
    distance_denominator = max(1.0, _float(spatial.get("distance_normalization_denominator", 1.0)))
    rows = []
    for action_index in range(len(mask)):
        action = decode_action(action_index, zones)
        action_context = dict(action_context_rows.get(action_index, {}) or {})
        zone_row = dict(action_context.get("state_feature_values") or zone_rows[zones.index(action.zone_id)])
        proposal = dict(
            action_context.get("next_job_proposal")
            or action_proposals.get(f"{action.branch}:{action.zone_id}", {})
            or {}
        )
        cycle_estimate = dict(action_context.get("cycle_estimate") or {})
        next_pod_zone = str(proposal.get("committed_next_zone_id", "") or "")
        branch_values = [1.0 if action.branch == STORE else 0.0, 1.0 if action.branch == REPLENISH_STORE else 0.0]
        values = [
            *branch_values,
            _bounded(state_json.get("historical_pod_request_rank", 0.0)),
            _bounded(stock["pod_fill_ratio"]),
            _bounded(stock["pod_below_threshold_ratio"]),
            _bounded(stock["pod_global_low_ratio"]),
            _bounded(stock["pod_has_zero_and_global_low_sku"]),
            _bounded(stock["min_local_fill_ratio"]),
            _bounded(spatial.get("source_station_is_picking", 0.0)),
            _bounded(spatial.get("source_station_is_replenishment", 0.0)),
            _bounded(spatial.get("source_station_x_norm", 0.0)),
            _bounded(spatial.get("source_station_y_norm", 0.0)),
            _zone_norm(zone_row.get("zone_row_index", 0.0), spatial.get("zone_row_min", 0.0), spatial.get("zone_row_max", 1.0)),
            _zone_norm(zone_row.get("zone_col_index", 0.0), spatial.get("zone_col_min", 0.0), spatial.get("zone_col_max", 1.0)),
            _bounded(zone_row.get("occupation_level", 0.0)),
            _bounded(zone_row.get("free_slot_ratio", 0.0)),
            _bounded(zone_row.get("zone_destination_robot_pressure", 0.0)),
            _bounded(zone_row.get("neighbor_zone_destination_robot_pressure", 0.0)),
            _bounded(zone_row.get("superzone_destination_robot_pressure", 0.0)),
            _bounded(zone_row.get("zone_present_robot_pressure", 0.0)),
            _bounded(zone_row.get("neighbor_zone_present_robot_pressure", 0.0)),
            _bounded(zone_row.get("superzone_present_robot_pressure", 0.0)),
            _bounded(zone_row.get("sku_similarity_fraction", zone_row.get("sku_similarity", 0.0))),
            _bounded(action_context.get("store_valid", zone_row.get("store_action_valid", 0.0))),
            _bounded(action_context.get("replenish_store_valid", zone_row.get("replenish_store_action_valid", 0.0))),
            _bounded(proposal.get("next_job_known", 0.0)),
            1.0 if bool(cycle_estimate.get("known", False)) else 0.0,
            _finite_if_known(cycle_estimate, "estimated_cycle_seconds"),
            _finite_if_known(cycle_estimate, "estimated_travel_seconds"),
            _finite_if_known(cycle_estimate, "estimated_queue_seconds"),
            _finite_if_known(cycle_estimate, "estimated_replenishment_service_seconds"),
            _finite_if_known(cycle_estimate, "estimated_handling_seconds"),
        ]
        values.extend(1.0 if next_pod_zone == zone_id else 0.0 for zone_id in zones)
        values.extend(
            [
                _distance_norm(proposal.get("candidate_storage_to_next_pod_distance", 0.0), distance_denominator),
                _distance_norm(proposal.get("next_pod_to_picker_distance", 0.0), distance_denominator),
                _distance_norm(proposal.get("allocator_cost", 0.0), distance_denominator),
                _distance_norm(proposal.get("regret_score", 0.0), distance_denominator),
                _bounded(proposal.get("one_robot_degenerate", 0.0)),
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


def feature_schema_metadata() -> dict[str, Any]:
    from .cycle_estimator import CYCLE_ESTIMATE_VERSION, SEMANTICS_HOST_STRUCTURAL
    from .stock_features import STOCK_FEATURE_SCHEMA_VERSION, STOCK_SOURCE_VERSION
    from .travel_time import TIME_CONVERSION_VERSION, TRAVEL_TIME_VERSION

    return {
        "action_feature_schema_version": ACTION_FEATURE_SCHEMA_VERSION,
        "stock_feature_schema_version": STOCK_FEATURE_SCHEMA_VERSION,
        "stock_source_version": STOCK_SOURCE_VERSION,
        "cycle_estimator_version": CYCLE_ESTIMATE_VERSION,
        "cycle_estimate_semantics": SEMANTICS_HOST_STRUCTURAL,
        "travel_time_version": TRAVEL_TIME_VERSION,
        "time_conversion_version": TIME_CONVERSION_VERSION,
    }


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


def _validate_removed_placeholders_absent(names: Sequence[str]) -> None:
    forbidden = {
        "next_retrieval_zone_known",
        "turnover_value",
        "arrival_rate_order_cycle_time",
        "total_robot_count",
        "active_pod_total",
        "free_slot_count",
    }
    present = forbidden.intersection(str(name) for name in names)
    if present:
        raise ValueError(f"RTS-RL action features include removed placeholders/raw counts: {sorted(present)}")
    if any(str(name).startswith("next_retrieval_zone_one_hot__") for name in names):
        raise ValueError("RTS-RL action features must use action-conditioned next_pod_zone_one_hot fields")


def _float(value: Any) -> float:
    try:
        result = float(value)
    except Exception:
        return 0.0
    return float(result) if np.isfinite(result) else 0.0


def _bounded(value: Any) -> float:
    return max(0.0, min(1.0, _float(value)))


def _distance_norm(value: Any, denominator: float) -> float:
    return max(0.0, min(1.0, _float(value) / max(1.0, float(denominator))))


def _finite_if_known(payload: Mapping[str, Any], key: str) -> float:
    if not bool(payload.get("known", False)):
        return 0.0
    return max(0.0, _float(payload.get(key, 0.0)))


def _zone_norm(value: Any, low: Any, high: Any) -> float:
    lo = _float(low)
    hi = _float(high)
    span = max(1.0, hi - lo)
    return max(0.0, min(1.0, (_float(value) - lo) / span))
