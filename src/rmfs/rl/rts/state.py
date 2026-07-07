"""RTS-RL state JSON builders for the current Rika-host object model."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Mapping, Sequence

from .action_context import ACTION_CONTEXT_VERSION, build_action_contexts, build_physical_zone_contexts
from .cycle_estimator import CYCLE_ESTIMATE_VERSION, SEMANTICS_HOST_STRUCTURAL
from .macro_region import macro_region_metadata
from .replenishment_snapshot import build_replenishment_snapshot
from .static_state_context import get_or_build_static_state_context
from .static_runtime_index import get_or_build_static_runtime_index, get_static_runtime_index, resolve_runtime_zone_ids
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
    timing: Mapping[str, float] | None = None


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
        "vrsla_slot_hotness_rank": FIDELITY_EXACT,
        "layout_normalization": FIDELITY_EXACT,
    }


def build_state(context: Any, zone_ids: Sequence[str]) -> RTSStateBundle:
    t_total_start = time.perf_counter()
    timing: dict[str, float] = {}
    warehouse = getattr(context, "warehouse", None)
    t_static_start = time.perf_counter()
    static_index = get_static_runtime_index(warehouse)
    if static_index is None:
        if _static_index_required(warehouse):
            raise RuntimeError("RTS static runtime index is required but is not installed for this headless RTS run")
        try:
            static_index = get_or_build_static_runtime_index(warehouse)
        except Exception:
            static_index = None
    timing["static_index_lookup_ms"] = _elapsed_ms(t_static_start)
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
    t_replenishment_start = time.perf_counter()
    replenishment_snapshot = build_replenishment_snapshot(warehouse, pod)
    timing["replenishment_snapshot_ms"] = _elapsed_ms(t_replenishment_start)
    repl_signal_active = bool(replenishment_snapshot.eligible)
    repl_station_available = _station_count(warehouse, "replenishment") > 0

    t_zone_start = time.perf_counter()
    zone_rows, warnings = build_zone_rows(
        context,
        zones,
        replenishment_signal_active=repl_signal_active,
        replenishment_station_available=repl_station_available,
        static_index=static_index,
    )
    zone_metadata = build_zone_registry_metadata(context, zones, static_index=static_index)
    rows_by_zone = {str(row.get("zone_id")): dict(row) for row in zone_rows}
    timing["zone_dynamic_features_ms"] = _elapsed_ms(t_zone_start)
    t_physical_start = time.perf_counter()
    physical_contexts = build_physical_zone_contexts(
        context,
        zones,
        rows_by_zone=rows_by_zone,
        static_index=static_index,
    )
    timing["physical_zone_contexts_ms"] = _elapsed_ms(t_physical_start)
    action_proposals = {}
    if warehouse is not None and robot is not None and getattr(warehouse, "committed_next_reservations_enabled", False):
        try:
            t_proposal_start = time.perf_counter()
            action_proposals = {
                key: proposal.to_state_json()
                for key, proposal in warehouse.ensure_committed_next_action_proposals(
                    robot,
                    context,
                    zones,
                    physical_contexts=physical_contexts,
                ).items()
            }
            proposal_diag = _proposal_diagnostics(warehouse, robot)
            timing["eligible_job_pool_ms"] = 1000.0 * float(proposal_diag.get("eligible_pool_build_seconds") or 0.0)
            timing["all_zone_proposals_ms"] = 1000.0 * float(proposal_diag.get("all_zone_proposals_seconds") or 0.0)
            timing["proposal_generation_ms"] = _elapsed_ms(t_proposal_start)
        except Exception as exc:
            if _proposal_generation_failure_is_fatal(warehouse):
                raise RuntimeError(
                    "RTS committed-next proposal generation failed for "
                    f"{_runtime_policy_mode(warehouse)!r}: {type(exc).__name__}: {exc}"
                ) from exc
            warnings.append(f"committed_next_action_proposals_unavailable:{exc}")
            action_proposals = {}
            timing.setdefault("eligible_job_pool_ms", 0.0)
            timing.setdefault("all_zone_proposals_ms", 0.0)
            timing.setdefault("proposal_generation_ms", 0.0)
    else:
        timing["eligible_job_pool_ms"] = 0.0
        timing["all_zone_proposals_ms"] = 0.0
        timing["proposal_generation_ms"] = 0.0
    proposal_objects = {}
    if warehouse is not None and robot is not None and getattr(warehouse, "committed_next_registry", None) is not None:
        try:
            proposal_objects = warehouse.committed_next_registry.get_action_proposals_for_robot(robot)
        except Exception:
            proposal_objects = {}
    store_physical_overrides = {}
    if proposal_objects and warehouse is not None and robot is not None:
        t_refine_start = time.perf_counter()
        store_physical_overrides, proposal_objects = _refine_store_storage_for_shortest_leg(
            context, zones, physical_contexts, proposal_objects, static_index, warehouse, robot,
        )
        if proposal_objects:
            action_proposals = {key: p.to_state_json() for key, p in proposal_objects.items()}
        timing["shortest_leg_refinement_ms"] = _elapsed_ms(t_refine_start)
    else:
        timing["shortest_leg_refinement_ms"] = 0.0
    t_final_action_start = time.perf_counter()
    action_contexts = build_action_contexts(
        context,
        zones,
        zone_rows=zone_rows,
        replenishment_snapshot=replenishment_snapshot,
        next_job_proposals=proposal_objects,
        static_index=static_index,
        physical_contexts=physical_contexts,
        store_physical_overrides=store_physical_overrides,
    )
    timing["final_action_contexts_ms"] = _elapsed_ms(t_final_action_start)
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
        "vrsla_slot_scores": {
            "available": static_index is not None,
            "slot_score_semantics": "directed_loaded_cycle_distance",
            "slot_score_identity": getattr(static_index, "vrsla_slot_score_identity", None),
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
        timing={**timing, "build_state_ms": _elapsed_ms(t_total_start)},
    )


def _station_count(warehouse: Any, station_type: str) -> int:
    station_manager = getattr(warehouse, "station_manager", None)
    return sum(1 for station in getattr(station_manager, "stations", []) if getattr(station, "station_type", "") == station_type)

def _is_robot_object(obj: object) -> bool:
    return str(getattr(obj, "object_type", "")).lower() == "robot"


def _static_index_required(warehouse: Any) -> bool:
    policy_mode = _runtime_policy_mode(warehouse)
    if policy_mode in {"random_valid", "rts_rl_explicit", "vrsla_teacher"}:
        return True
    return False


def _proposal_generation_failure_is_fatal(warehouse: Any) -> bool:
    return _runtime_policy_mode(warehouse) in {"rts_rl_explicit", "current_probe", "vrsla_teacher"}


def _runtime_policy_mode(warehouse: Any) -> str:
    runtime = getattr(warehouse, "rts_rollout_runtime", None)
    config = getattr(runtime, "config", None)
    return str(getattr(config, "policy_mode", "") or "")


def _proposal_diagnostics(warehouse: Any, robot: Any) -> dict[str, Any]:
    registry = getattr(warehouse, "committed_next_registry", None)
    if registry is None:
        return {}
    try:
        return registry.last_action_proposal_diagnostics(robot)
    except Exception:
        return {}


def _elapsed_ms(start: float) -> float:
    return max(0.0, (time.perf_counter() - start) * 1000.0)


def _refine_store_storage_for_shortest_leg(
    context: Any,
    zones: tuple[str, ...],
    physical_contexts: Mapping[str, Any],
    proposal_objects: dict[str, Any],
    static_index: Any,
    warehouse: Any,
    robot: Any,
) -> tuple[Mapping[str, Any], dict[str, Any]]:
    """Build store-only overrides minimizing loaded(picker->storage) + empty(storage->next_pod)."""
    from .action_context import (
        RTSPhysicalZoneContext,
        action_key,
        select_shortest_leg_storage,
        select_replenishment_station,
        storage_id as _sid,
        state_values_for_storage,
    )

    registry = getattr(warehouse, "committed_next_registry", None)
    if registry is None:
        return {}, proposal_objects

    store_overrides = {}
    refined_proposals = dict(proposal_objects)
    changed = False

    for zone_id in zones:
        zone_key = str(zone_id)
        proposal = proposal_objects.get(zone_key)
        if proposal is None or not proposal.has_next_job:
            continue
        next_pod_coord = getattr(proposal.job, "pod_coordinate", None)
        if next_pod_coord is None:
            continue

        storage, feas = select_shortest_leg_storage(
            context, zone_key, next_pod_coord, static_index=static_index,
        )
        if storage is None or not feas.available or not feas.reachable:
            continue

        physical = physical_contexts.get(zone_key)
        if physical is not None and _sid(storage) == _sid(getattr(physical, "storage", None)):
            continue

        station, station_feas = select_replenishment_station(context, storage, static_index=static_index)
        state_values = state_values_for_storage(
            context,
            storage,
            base_state_values=dict(physical.state_values) if physical is not None else {},
            static_index=static_index,
        )
        store_overrides[zone_key] = RTSPhysicalZoneContext(
            zone_id=zone_key,
            storage=storage,
            storage_feasibility=feas,
            replenishment_station=station,
            replenishment_station_feasibility=station_feas,
            state_values=state_values,
        )

        new_proposal = registry.rebuild_proposal_for_storage(
            warehouse, robot, zone_key, storage, proposal,
        )
        refined_proposals[action_key("store", zone_key)] = new_proposal
        changed = True

    if not changed:
        return {}, proposal_objects
    return store_overrides, refined_proposals
