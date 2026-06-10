"""RTS-RL state JSON builders for the current Rika-host object model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from .stock_features import stock_rows_from_pod
from .zone_features import build_zone_rows

STATE_CONTRACT_VERSION = "rts_rl_state.v1"
FIDELITY_EXACT = "exact"
FIDELITY_APPROX = "approx_repo_grounded"
FIDELITY_DEFAULT = "default_unavailable"


@dataclass(frozen=True)
class RTSStateBundle:
    state_json: dict[str, Any]
    zone_ids: tuple[str, ...]
    warnings: tuple[str, ...]


def build_default_feature_fidelity() -> dict[str, str]:
    return {
        "robot_context": FIDELITY_EXACT,
        "pod_context": FIDELITY_EXACT,
        "station_context": FIDELITY_EXACT,
        "zone_occupancy": FIDELITY_APPROX,
        "destination_robot_pressure": FIDELITY_DEFAULT,
        "present_robot_pressure": FIDELITY_APPROX,
        "cycle_time_estimates": FIDELITY_DEFAULT,
        "next_retrieval_context": FIDELITY_DEFAULT,
        "replenishment_station_context": FIDELITY_APPROX,
        "stock_risk": FIDELITY_APPROX,
        "action_validity": FIDELITY_APPROX,
    }


def build_state(context: Any, zone_ids: Sequence[str]) -> RTSStateBundle:
    zones = tuple(str(zone_id) for zone_id in zone_ids)
    if not zones:
        raise ValueError("RTS-RL state requires at least one zone")
    warehouse = getattr(context, "warehouse", None)
    robot = getattr(context, "robot", None)
    pod = getattr(context, "pod", None)
    station = getattr(context, "station", None)

    stock_rows = stock_rows_from_pod(pod)
    repl_signal_active = any(row.get("below_threshold", 0.0) > 0 for row in stock_rows)
    repl_station_available = _station_count(warehouse, "replenishment") > 0

    zone_rows, warnings = build_zone_rows(
        context,
        zones,
        replenishment_signal_active=repl_signal_active,
        replenishment_station_available=repl_station_available,
    )
    station_type = str(getattr(station, "station_type", ""))
    station_id = str(getattr(station, "station_id", ""))
    next_retrieval_zone_one_hot = {zone_id: 0 for zone_id in zones}
    spatial_context = {
        "source_station_is_picking": 1.0 if station_type == "picker" else 0.0,
        "source_station_is_replenishment": 1.0 if station_type == "replenishment" else 0.0,
        "source_station_x_norm": _norm(getattr(station, "pos_x", 0.0)),
        "source_station_y_norm": _norm(getattr(station, "pos_y", 0.0)),
        "picking_station_count": float(_station_count(warehouse, "picker")),
        "replenishment_station_count": float(_station_count(warehouse, "replenishment")),
        "selected_replenishment_station_x_norm": 0.0,
        "selected_replenishment_station_y_norm": 0.0,
        "selected_replenishment_station_logical_load": 0.0,
        "total_robot_count": float(sum(1 for obj in getattr(warehouse, "_objects", []) if getattr(obj, "object_type", "") == "robot")),
        "active_pod_total": float(len(getattr(getattr(warehouse, "pod_manager", None), "pods", []) or [])),
        "arrival_rate_order_cycle_time": 0.0,
        "zone_row_min": 0.0,
        "zone_row_max": float(max(1, len(zones) - 1)),
        "zone_col_min": 0.0,
        "zone_col_max": float(max(1, len(zones) - 1)),
    }
    state_json = {
        "state_contract_version": STATE_CONTRACT_VERSION,
        "robot_id": str(getattr(robot, "_id", getattr(robot, "id", ""))),
        "pod_id": str(getattr(pod, "pod_id", "")),
        "source_station_id": station_id,
        "source_station_type": station_type,
        "turnover_rank": 0.0,
        "turnover_value": 0.0,
        "turnover_mode": "current_repo_default",
        "next_retrieval_zone_known": 0,
        "next_retrieval_zone": "",
        "next_retrieval_zone_one_hot": next_retrieval_zone_one_hot,
        "estimated_queue_time": 0.0,
        "replenishment_signal_active": 1 if repl_signal_active else 0,
        "zone_rows": zone_rows,
        "stock_rows": stock_rows,
        "spatial_context": spatial_context,
        "feature_fidelity": build_default_feature_fidelity(),
        "warnings": warnings,
    }
    return RTSStateBundle(state_json=state_json, zone_ids=zones, warnings=tuple(warnings))


def _station_count(warehouse: Any, station_type: str) -> int:
    station_manager = getattr(warehouse, "station_manager", None)
    return sum(1 for station in getattr(station_manager, "stations", []) if getattr(station, "station_type", "") == station_type)


def _norm(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value) / 100.0))
    except Exception:
        return 0.0
