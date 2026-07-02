"""RTS-RL state JSON builders for the current Rika-host object model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from .action_context import ACTION_CONTEXT_VERSION, build_action_contexts, build_physical_zone_contexts
from .cycle_estimator import CYCLE_ESTIMATE_VERSION, SEMANTICS_HOST_STRUCTURAL
from .macro_region import macro_region_metadata
from .replenishment_snapshot import build_replenishment_snapshot
from .static_state_context import get_or_build_static_state_context
from .static_runtime_index import get_or_build_static_runtime_index, resolve_runtime_zone_ids
from .stock_features import stock_rows_from_pod
from .zone_features import build_zone_registry_metadata, build_zone_rows

STATE_CONTRACT_VERSION = "rts_rl_state.v4"
FIDELITY_EXACT = "exact"
FIDELITY_APPROX = "approx_repo_grounded"
FIDELITY_DEFAULT = "default_unavailable"


@dataclass(frozen=True)
class RTSStateBundle:
    state_json: dict[str, Any]
    zone_ids: tuple[str, ...]
    warnings: tuple[str, ...]
    action_contexts: tuple[Any, ...] = ()


def build_default_feature_fidelity() -> dict[str, str]:
    return {
        "robot_context": FIDELITY_EXACT,
        "pod_context": FIDELITY_EXACT,
        "station_context": FIDELITY_EXACT,
        "zone_registry": FIDELITY_EXACT,
        "zone_geometry": FIDELITY_EXACT,
        "zone_occupancy": FIDELITY_EXACT,
        "destination_robot_pressure": FIDELITY_APPROX,
        "present_robot_pressure": FIDELITY_APPROX,
        "graph_distance": FIDELITY_APPROX,
        "cycle_time_estimates": FIDELITY_APPROX,
        "next_retrieval_context": FIDELITY_EXACT,
        "replenishment_station_context": FIDELITY_APPROX,
        "stock_risk": FIDELITY_APPROX,
        "action_validity": FIDELITY_EXACT,
        "historical_pod_request_rank": FIDELITY_EXACT,
        "layout_normalization": FIDELITY_EXACT,
    }


def build_state(context: Any, zone_ids: Sequence[str]) -> RTSStateBundle:
    warehouse = getattr(context, "warehouse", None)
    try:
        static_index = get_or_build_static_runtime_index(warehouse)
    except Exception:
        static_index = None
    zones = resolve_runtime_zone_ids(warehouse, zone_ids) if static_index is not None else tuple(str(zone_id) for zone_id in zone_ids)
    if not zones:
        raise ValueError("RTS-RL state requires at least one zone")
    robot = getattr(context, "robot", None)
    pod = getattr(context, "pod", None)
    station = getattr(context, "station", None)
    station_type = str(getattr(station, "station_type", ""))
    if station_type not in {"picker", "picking"} and not bool(getattr(getattr(robot, "job", None), "rts_continuation_active", False)):
        raise ValueError(f"RTS-RL policy requires a picker source station, got {station_type!r}")
    static_context = get_or_build_static_state_context(warehouse)

    stock_rows = stock_rows_from_pod(pod, warehouse, strict_global=True)
    replenishment_snapshot = build_replenishment_snapshot(warehouse, pod)
    repl_signal_active = bool(replenishment_snapshot.eligible)
    repl_station_available = _station_count(warehouse, "replenishment") > 0

    zone_rows, warnings = build_zone_rows(
        context,
        zones,
        replenishment_signal_active=repl_signal_active,
        replenishment_station_available=repl_station_available,
        static_index=static_index,
    )
    zone_metadata = build_zone_registry_metadata(context, zones, static_index=static_index)
    rows_by_zone = {str(row.get("zone_id")): dict(row) for row in zone_rows}
    physical_contexts = build_physical_zone_contexts(
        context,
        zones,
        rows_by_zone=rows_by_zone,
        static_index=static_index,
    )
    base_action_contexts = build_action_contexts(
        context,
        zones,
        zone_rows=zone_rows,
        replenishment_snapshot=replenishment_snapshot,
        static_index=static_index,
        physical_contexts=physical_contexts,
        include_cycle_estimates=False,
    )
    action_proposals = {}
    if warehouse is not None and robot is not None and getattr(warehouse, "committed_next_reservations_enabled", False):
        try:
            action_proposals = {
                key: proposal.to_state_json()
                for key, proposal in warehouse.ensure_committed_next_action_proposals(
                    robot,
                    context,
                    zones,
                    action_contexts=base_action_contexts,
                ).items()
            }
        except Exception as exc:
            warnings.append(f"committed_next_action_proposals_unavailable:{exc}")
            action_proposals = {}
    proposal_objects = {}
    if warehouse is not None and robot is not None and getattr(warehouse, "committed_next_registry", None) is not None:
        try:
            proposal_objects = warehouse.committed_next_registry.get_action_proposals_for_robot(robot)
        except Exception:
            proposal_objects = {}
    action_contexts = build_action_contexts(
        context,
        zones,
        zone_rows=zone_rows,
        replenishment_snapshot=replenishment_snapshot,
        next_job_proposals=proposal_objects,
        static_index=static_index,
        physical_contexts=physical_contexts,
    )
    rows = [float(row.get("zone_row_index", 0.0)) for row in zone_rows]
    cols = [float(row.get("zone_col_index", 0.0)) for row in zone_rows]
    station_id = str(getattr(station, "station_id", ""))
    spatial_context = {
        "source_picker_x_norm": static_context.norm_x(getattr(station, "pos_x", 0.0)),
        "source_picker_y_norm": static_context.norm_y(getattr(station, "pos_y", 0.0)),
        "source_station_is_picking": 1.0 if station_type in {"picker", "picking"} else 0.0,
        "source_station_x_norm": static_context.norm_x(getattr(station, "pos_x", 0.0)),
        "source_station_y_norm": static_context.norm_y(getattr(station, "pos_y", 0.0)),
        "picking_station_count": float(_station_count(warehouse, "picker")),
        "replenishment_station_count": float(_station_count(warehouse, "replenishment")),
        "total_robot_count": float(sum(1 for obj in getattr(warehouse, "_objects", []) if getattr(obj, "object_type", "") == "robot")),
        "active_pod_total": float(len(getattr(getattr(warehouse, "pod_manager", None), "pods", []) or [])),
        "zone_row_min": min(rows) if rows else 0.0,
        "zone_row_max": max(rows) if rows else 1.0,
        "zone_col_min": min(cols) if cols else 0.0,
        "zone_col_max": max(cols) if cols else 1.0,
        "distance_normalization_denominator": float(static_context.distance_metadata.get("distance_normalization_denominator", 1.0)),
    }
    feature_status = {
        "next_retrieval_context": {
            "available": bool(action_proposals),
            "reason": (
                "phase2_action_conditioned_committed_next_proposals"
                if action_proposals
                else "no committed-next action proposal available at RTS decision time"
            ),
        },
        "distance_features": {
            "available": True,
            "semantics": zone_metadata.get("distance_semantics_version"),
            "fallback_count": zone_metadata.get("distance_fallback_count", 0),
        },
        "zone_geometry": {
            "available": True,
            "zone_geometry_hash": zone_metadata.get("zone_geometry_hash"),
        },
        "cycle_estimates": {
            "available": True,
            "estimator_version": CYCLE_ESTIMATE_VERSION,
            "semantics": SEMANTICS_HOST_STRUCTURAL,
            "known_count": sum(
                1
                for action_context in action_contexts
                if bool(getattr(getattr(action_context, "cycle_estimate", None), "known", False))
            ),
            "total_count": len(action_contexts),
        },
        "macro_regions": {
            "available": True,
            **macro_region_metadata(),
        },
    }
    state_json = {
        "state_contract_version": STATE_CONTRACT_VERSION,
        "zone_registry": zone_metadata,
        "robot_id": str(getattr(robot, "_id", getattr(robot, "id", ""))),
        "pod_id": str(getattr(pod, "pod_id", "")),
        "source_station_id": station_id,
        "source_station_type": station_type,
        "historical_pod_request_rank": static_context.pod_rank(pod),
        "historical_pod_request_count": static_context.pod_count(pod),
        "committed_next_action_proposals": action_proposals,
        "rts_action_context_version": ACTION_CONTEXT_VERSION,
        "rts_action_contexts": [action_context.to_json_dict() for action_context in action_contexts],
        "replenishment_signal_active": 1 if repl_signal_active else 0,
        "replenishment_snapshot": replenishment_snapshot.to_json_dict(),
        "zone_rows": zone_rows,
        "stock_rows": stock_rows,
        "spatial_context": spatial_context,
        "feature_fidelity": build_default_feature_fidelity(),
        "feature_status": feature_status,
        "historical_pod_request_rank_metadata": static_context.historical_metadata,
        "layout_normalization": static_context.layout_metadata,
        "distance_normalization": static_context.distance_metadata,
        "macro_region_contract": macro_region_metadata(),
        "warnings": warnings,
    }
    return RTSStateBundle(
        state_json=state_json,
        zone_ids=zones,
        warnings=tuple(warnings),
        action_contexts=action_contexts,
    )


def _station_count(warehouse: Any, station_type: str) -> int:
    station_manager = getattr(warehouse, "station_manager", None)
    return sum(1 for station in getattr(station_manager, "stations", []) if getattr(station, "station_type", "") == station_type)

def _is_robot_object(obj: object) -> bool:
    return str(getattr(obj, "object_type", "")).lower() == "robot"
