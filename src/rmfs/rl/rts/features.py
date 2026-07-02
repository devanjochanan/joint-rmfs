"""Model-ready RTS-RL action and stock feature matrices."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .action_space import REPLENISH_STORE, STORE, decode_action, validate_action_mask
from .stock_features import STOCK_FEATURE_NAMES, build_stock_feature_matrix, stock_summary
from .validation import validate_no_raw_threshold_features


ACTION_FEATURE_SCHEMA_VERSION = "rts_action_features.v4"

ACTION_FEATURE_BASE_NAMES: tuple[str, ...] = (
    "is_replenish_store",
    "historical_pod_request_rank",
    "replenishment_eligible_sku_ratio",
    "source_picker_x_norm",
    "source_picker_y_norm",
    "candidate_storage_x_norm",
    "candidate_storage_y_norm",
    "free_slot_ratio",
    "sku_similarity_fraction",
    "zone_present_robot_pressure",
    "zone_destination_robot_pressure",
    "macro_region_present_robot_pressure",
    "macro_region_destination_robot_pressure",
    "selected_replenishment_station_destination_pressure",
    "proposed_next_job_known",
    "candidate_to_proposed_next_pod_distance_norm",
    "cycle_estimate_known",
    "estimated_cycle_time",
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
    names = list(ACTION_FEATURE_BASE_NAMES)
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
    spatial = dict(state_json.get("spatial_context", {}) or {})
    action_proposals = dict(state_json.get("committed_next_action_proposals", {}) or {})
    action_context_rows = {
        int(row.get("action_index")): dict(row)
        for row in list(state_json.get("rts_action_contexts", []) or [])
        if row.get("action_index") is not None
    }
    names = build_action_feature_names(zones)
    distance_denominator = max(1.0, _float(spatial.get("distance_normalization_denominator", 1.0)))
    layout = dict(state_json.get("layout_normalization", {}) or {})
    eligible_skus = list(dict(state_json.get("replenishment_snapshot", {}) or {}).get("eligible_skus", []) or [])
    replenishment_eligible_sku_ratio = _bounded(len(eligible_skus) / float(max(1, len(stock_rows)))) if stock_rows else 0.0
    rows = []
    for action_index in range(len(mask)):
        action = decode_action(action_index, zones)
        action_context = dict(action_context_rows.get(action_index, {}) or {})
        zone_row = dict(action_context.get("state_feature_values") or zone_rows[zones.index(action.zone_id)])
        proposal = dict(
            action_context.get("next_job_proposal")
            or action_proposals.get(action.zone_id, {})
            or {}
        )
        cycle_estimate = dict(action_context.get("cycle_estimate") or {})
        coord = action_context.get("candidate_storage_coordinate")
        if coord is None:
            coord = (zone_row.get("candidate_storage_x", 0.0), zone_row.get("candidate_storage_y", 0.0))
        candidate_x, candidate_y = _coord_pair(coord)
        values = [
            1.0 if action.branch == REPLENISH_STORE else 0.0,
            _bounded(state_json.get("historical_pod_request_rank", 0.0)),
            replenishment_eligible_sku_ratio,
            _bounded(spatial.get("source_picker_x_norm", spatial.get("source_station_x_norm", 0.0))),
            _bounded(spatial.get("source_picker_y_norm", spatial.get("source_station_y_norm", 0.0))),
            _norm(candidate_x, layout.get("x_min", 0.0), layout.get("x_max", 1.0)),
            _norm(candidate_y, layout.get("y_min", 0.0), layout.get("y_max", 1.0)),
            _bounded(zone_row.get("free_slot_ratio", 0.0)),
            _bounded(zone_row.get("sku_similarity_fraction", zone_row.get("sku_similarity", 0.0))),
            _bounded(zone_row.get("zone_present_robot_pressure", 0.0)),
            _bounded(zone_row.get("zone_destination_robot_pressure", 0.0)),
            _bounded(zone_row.get("macro_region_present_robot_pressure", 0.0)),
            _bounded(zone_row.get("macro_region_destination_robot_pressure", 0.0)),
            _bounded(action_context.get("selected_replenishment_station_destination_pressure", zone_row.get("selected_replenishment_station_destination_pressure", 0.0))),
            _bounded(proposal.get("proposed_next_job_known", proposal.get("next_job_known", 0.0))),
            _distance_norm(proposal.get("candidate_to_proposed_next_pod_distance", proposal.get("candidate_storage_to_next_pod_distance", 0.0)), distance_denominator),
            1.0 if bool(cycle_estimate.get("known", False)) else 0.0,
            _finite_if_known(cycle_estimate, "estimated_cycle_seconds"),
        ]
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
    from .action_space import ACTION_BRANCHES, ACTION_BRANCH_ORDER_VERSION
    from .graph_distance import DISTANCE_SEMANTICS_VERSION
    from .macro_region import macro_region_metadata
    from .stock_features import STOCK_FEATURE_SCHEMA_VERSION, STOCK_SOURCE_VERSION
    from .travel_time import TIME_CONVERSION_VERSION, TRAVEL_TIME_VERSION
    from .zone_registry import ZONE_GEOMETRY_VERSION

    return {
        "action_feature_schema_version": ACTION_FEATURE_SCHEMA_VERSION,
        "stock_feature_schema_version": STOCK_FEATURE_SCHEMA_VERSION,
        "stock_source_version": STOCK_SOURCE_VERSION,
        "cycle_estimator_version": CYCLE_ESTIMATE_VERSION,
        "cycle_estimate_semantics": SEMANTICS_HOST_STRUCTURAL,
        "distance_semantics_version": DISTANCE_SEMANTICS_VERSION,
        "travel_time_version": TRAVEL_TIME_VERSION,
        "time_conversion_version": TIME_CONVERSION_VERSION,
        "action_branch_order_version": ACTION_BRANCH_ORDER_VERSION,
        "action_branch_order": list(ACTION_BRANCHES),
        "zone_geometry_version": ZONE_GEOMETRY_VERSION,
        **macro_region_metadata(),
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
        raise ValueError("RTS-RL action features must not include zone one-hot fields")
    if any(str(name).startswith("next_pod_zone_one_hot__") for name in names):
        raise ValueError("RTS-RL action features must not include zone one-hot fields")


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


def _norm(value: Any, low: Any, high: Any) -> float:
    lo = _float(low)
    hi = _float(high)
    return _bounded((_float(value) - lo) / max(1e-9, hi - lo))


def _coord_pair(value: Any) -> tuple[float, float]:
    if isinstance(value, Sequence) and len(value) >= 2:
        return _float(value[0]), _float(value[1])
    return 0.0, 0.0
