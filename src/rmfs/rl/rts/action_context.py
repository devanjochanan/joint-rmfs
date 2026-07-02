"""Authoritative RTS action-context construction.

The context built here is the single read-only source for RTS-RL action
validity, concrete storage, replenishment station, and action-local metadata.
Unselected contexts do not reserve or mutate simulator state.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

from engine.netlogo_coordinate import NetLogoCoordinate

from .action_space import REPLENISH_STORE, STORE, decode_action, normalize_zone_ids
from .graph_distance import graph_distance_or_fallback
from .macro_region import macro_region_pressures
from .replenishment_snapshot import RTSReplenishmentSnapshot, build_replenishment_snapshot
from .zone_registry import build_zone_registry


ACTION_CONTEXT_VERSION = "rts_action_context.v3"


@dataclass(frozen=True)
class RTSStorageFeasibility:
    available: bool
    belongs_to_zone: bool
    not_assigned: bool
    not_reserved: bool
    reachable: bool
    reason_codes: tuple[str, ...]

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "available": bool(self.available),
            "belongs_to_zone": bool(self.belongs_to_zone),
            "not_assigned": bool(self.not_assigned),
            "not_reserved": bool(self.not_reserved),
            "reachable": bool(self.reachable),
            "reason_codes": list(self.reason_codes),
        }


@dataclass(frozen=True)
class RTSStationFeasibility:
    station_exists: bool
    correct_station_type: bool
    active_capacity_available: bool
    queue_capacity_available: bool
    admission_possible: bool
    structurally_reachable: bool
    reason_codes: tuple[str, ...]

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "station_exists": bool(self.station_exists),
            "correct_station_type": bool(self.correct_station_type),
            "active_capacity_available": bool(self.active_capacity_available),
            "queue_capacity_available": bool(self.queue_capacity_available),
            "admission_possible": bool(self.admission_possible),
            "structurally_reachable": bool(self.structurally_reachable),
            "reason_codes": list(self.reason_codes),
        }


@dataclass(frozen=True)
class RTSActionContext:
    action_index: int
    branch: str
    zone_id: str
    candidate_storage: Any | None
    candidate_storage_id: str | None
    candidate_storage_coordinate: tuple[float, float] | None
    replenishment_station: Any | None
    replenishment_station_id: str | None
    replenishment_station_coordinate: tuple[float, float] | None
    replenishment_eligibility: bool
    replenishment_eligible_skus: tuple[Any, ...]
    storage_feasibility: RTSStorageFeasibility
    station_feasibility: RTSStationFeasibility | None
    store_valid: bool
    replenish_store_valid: bool
    action_valid: bool
    invalid_reason_codes: tuple[str, ...]
    next_job_proposal: Any | None = None
    cycle_estimate: Any | None = None
    state_feature_values: Mapping[str, Any] | None = None
    context_version: str = ACTION_CONTEXT_VERSION
    context_id: str = ""

    def with_next_job_proposal(self, proposal: Any | None) -> "RTSActionContext":
        return replace(self, next_job_proposal=proposal)

    def with_cycle_estimate(self, cycle_estimate: Any | None) -> "RTSActionContext":
        return replace(self, cycle_estimate=cycle_estimate)

    def to_json_dict(self) -> dict[str, Any]:
        proposal = self.next_job_proposal
        return {
            "context_version": self.context_version,
            "context_id": self.context_id,
            "action_index": int(self.action_index),
            "branch": self.branch,
            "zone_id": self.zone_id,
            "candidate_storage_id": self.candidate_storage_id,
            "candidate_storage_coordinate": list(self.candidate_storage_coordinate)
            if self.candidate_storage_coordinate is not None
            else None,
            "replenishment_station_id": self.replenishment_station_id,
            "replenishment_station_coordinate": list(self.replenishment_station_coordinate)
            if self.replenishment_station_coordinate is not None
            else None,
            "replenishment_eligibility": bool(self.replenishment_eligibility),
            "replenishment_eligible_skus": list(self.replenishment_eligible_skus),
            "storage_feasibility": self.storage_feasibility.to_json_dict(),
            "station_feasibility": self.station_feasibility.to_json_dict()
            if self.station_feasibility is not None
            else None,
            "store_valid": bool(self.store_valid),
            "replenish_store_valid": bool(self.replenish_store_valid),
            "action_valid": bool(self.action_valid),
            "invalid_reason_codes": list(self.invalid_reason_codes),
            "next_job_proposal": proposal.to_state_json() if proposal is not None else None,
            "cycle_estimate": self.cycle_estimate.to_json_dict() if self.cycle_estimate is not None else None,
            "state_feature_values": dict(self.state_feature_values or {}),
            "mask_value": 1 if self.action_valid else 0,
        }


def build_action_contexts(
    context: Any,
    zone_ids: Sequence[str],
    *,
    zone_rows: Sequence[Mapping[str, Any]] | None = None,
    replenishment_snapshot: RTSReplenishmentSnapshot | None = None,
    next_job_proposals: Mapping[str, Any] | None = None,
) -> tuple[RTSActionContext, ...]:
    zones = normalize_zone_ids(zone_ids)
    snapshot = replenishment_snapshot or build_replenishment_snapshot(
        getattr(context, "warehouse", None),
        getattr(context, "pod", None),
    )
    rows_by_zone = {str(row.get("zone_id")): dict(row) for row in (zone_rows or [])}
    proposal_by_key = dict(next_job_proposals or {})
    per_zone: dict[str, dict[str, Any]] = {}
    for zone_id in zones:
        storage, storage_feasibility = select_candidate_storage(context, zone_id)
        station, station_feasibility = select_replenishment_station(context, storage)
        state_values = dict(rows_by_zone.get(zone_id, {}))
        if storage is not None:
            state_values["candidate_storage_x"] = float(getattr(storage, "pos_x", 0.0))
            state_values["candidate_storage_y"] = float(getattr(storage, "pos_y", 0.0))
            state_values.update(macro_region_pressures(context, storage))
        per_zone[zone_id] = {
            "storage": storage,
            "storage_feasibility": storage_feasibility,
            "replenishment_station": station,
            "replenishment_station_feasibility": station_feasibility,
            "state_values": state_values,
        }
    contexts: list[RTSActionContext] = []
    for action_index in range(len(zones) * 2):
        action = decode_action(action_index, zones)
        physical = per_zone[action.zone_id]
        storage = physical["storage"]
        storage_feasibility = physical["storage_feasibility"]
        station = physical["replenishment_station"] if action.branch == REPLENISH_STORE else None
        station_feasibility = physical["replenishment_station_feasibility"] if action.branch == REPLENISH_STORE else None
        store_valid = storage_feasibility.available and storage_feasibility.reachable
        replenish_valid = (
            store_valid
            and bool(snapshot.eligible)
            and bool(snapshot.eligible_skus)
            and station_feasibility is not None
            and station_feasibility.admission_possible
            and station_feasibility.structurally_reachable
        )
        action_valid = store_valid if action.branch == STORE else replenish_valid
        reasons = list(storage_feasibility.reason_codes)
        if action.branch == REPLENISH_STORE:
            if not snapshot.eligible:
                reasons.append(f"replenishment_not_eligible:{snapshot.reason}")
            if not snapshot.eligible_skus:
                reasons.append("no_eligible_replenishment_skus")
            if station_feasibility is None:
                reasons.append("no_replenishment_station")
            else:
                reasons.extend(station_feasibility.reason_codes)
        reasons = tuple(sorted(set(reasons)))
        proposal = proposal_by_key.get(action.zone_id) or proposal_by_key.get(action_key(action.branch, action.zone_id))
        storage_coord = _coord(storage) if storage is not None else None
        station_coord = _coord(station) if station is not None else None
        state_values = dict(physical["state_values"])
        state_values["selected_replenishment_station_destination_pressure"] = (
            _station_destination_pressure(context, station) if action.branch == REPLENISH_STORE else 0.0
        )
        context_id = (
            f"{ACTION_CONTEXT_VERSION}:{getattr(getattr(context, 'robot', None), '_id', '')}:"
            f"{getattr(getattr(context, 'pod', None), 'pod_id', '')}:{action.action_index}:"
            f"{storage_id(storage)}:{station_id(station)}"
        )
        action_context = RTSActionContext(
            action_index=action.action_index,
            branch=action.branch,
            zone_id=action.zone_id,
            candidate_storage=storage,
            candidate_storage_id=storage_id(storage) or None,
            candidate_storage_coordinate=storage_coord,
            replenishment_station=station,
            replenishment_station_id=station_id(station) or None,
            replenishment_station_coordinate=station_coord,
            replenishment_eligibility=bool(snapshot.eligible),
            replenishment_eligible_skus=tuple(snapshot.eligible_skus),
            storage_feasibility=storage_feasibility,
            station_feasibility=station_feasibility,
            store_valid=bool(store_valid),
            replenish_store_valid=bool(replenish_valid),
            action_valid=bool(action_valid),
            invalid_reason_codes=() if action_valid else reasons,
            next_job_proposal=proposal,
            state_feature_values=state_values,
            context_id=context_id,
        )
        contexts.append(_with_cycle_estimate(context, action_context))
    return tuple(contexts)


def build_action_mask_from_contexts(action_contexts: Sequence[RTSActionContext]) -> list[int]:
    return [1 if ctx.action_valid else 0 for ctx in sorted(action_contexts, key=lambda item: item.action_index)]


def selected_context_by_index(action_contexts: Sequence[RTSActionContext], action_index: int) -> RTSActionContext:
    for ctx in action_contexts:
        if int(ctx.action_index) == int(action_index):
            return ctx
    raise ValueError(f"RTS action context not found for action index {action_index}")


def revalidate_selected_context(context: Any, selected: RTSActionContext) -> RTSActionContext:
    storage, storage_feasibility = select_candidate_storage(context, selected.zone_id)
    if storage is None or not storage_feasibility.available or not storage_feasibility.reachable:
        raise RuntimeError(
            f"selected RTS storage became unavailable for {selected.branch}:{selected.zone_id}: "
            f"{','.join(storage_feasibility.reason_codes)}"
        )
    station = selected.replenishment_station
    station_feasibility = selected.station_feasibility
    if selected.branch == REPLENISH_STORE:
        station, station_feasibility = select_replenishment_station(context, storage)
        if station is None or station_feasibility is None or not station_feasibility.admission_possible:
            reasons = station_feasibility.reason_codes if station_feasibility is not None else ("no_replenishment_station",)
            raise RuntimeError(
                f"selected RTS replenishment station became unavailable for {selected.branch}:{selected.zone_id}: "
                f"{','.join(reasons)}"
            )
    refreshed = replace(
        selected,
        candidate_storage=storage,
        candidate_storage_id=storage_id(storage) or None,
        candidate_storage_coordinate=_coord(storage),
        replenishment_station=station,
        replenishment_station_id=station_id(station) or None,
        replenishment_station_coordinate=_coord(station) if station is not None else None,
        storage_feasibility=storage_feasibility,
        station_feasibility=station_feasibility,
        invalid_reason_codes=(),
        action_valid=True,
    )
    proposal = getattr(refreshed, "next_job_proposal", None)
    registry = getattr(getattr(context, "warehouse", None), "committed_next_registry", None)
    robot = getattr(context, "robot", None)
    if registry is not None and robot is not None:
        current = registry.get_action_proposal(robot, selected.branch, selected.zone_id)
        if current is None or storage_id(getattr(current, "candidate_storage", None)) != storage_id(storage):
            current = registry.refresh_action_proposal_for_zone(
                getattr(context, "warehouse", None),
                robot,
                context,
                selected.zone_id,
                storage,
            )
        proposal = current
    refreshed = refreshed.with_next_job_proposal(proposal)
    return _with_cycle_estimate(context, refreshed)


def select_candidate_storage(context: Any, zone_id: str) -> tuple[Any | None, RTSStorageFeasibility]:
    warehouse = getattr(context, "warehouse", None)
    storage_manager = getattr(warehouse, "storage_manager", None)
    storages = list(getattr(storage_manager, "storages", []) or [])
    registry = build_zone_registry(context, (str(zone_id),))
    candidates: list[tuple[float, tuple[float, float], str, Any, RTSStorageFeasibility]] = []
    checked_zone = False
    for storage in storages:
        belongs = registry.zone_id_for_storage(storage) == str(zone_id)
        if belongs:
            checked_zone = True
        feasibility = storage_feasibility(context, storage, zone_id)
        if feasibility.available and feasibility.reachable:
            candidates.append((_distance_from_source(context, storage), _coord(storage), storage_id(storage), storage, feasibility))
    if candidates:
        _, _, _, storage, feasibility = sorted(candidates, key=lambda item: (item[0], item[1], item[2]))[0]
        return storage, feasibility
    reasons = [] if checked_zone else ["no_storage_in_zone"]
    return None, RTSStorageFeasibility(False, checked_zone, False, False, False, tuple(reasons or ["no_available_storage"]))


def storage_feasibility(context: Any, storage: Any | None, zone_id: str) -> RTSStorageFeasibility:
    if storage is None:
        return RTSStorageFeasibility(False, False, False, False, False, ("missing_storage",))
    registry = build_zone_registry(context, (str(zone_id),))
    belongs = registry.zone_id_for_storage(storage) == str(zone_id)
    not_assigned = getattr(storage, "assigned_pod", None) is None
    not_reserved = bool(getattr(storage, "is_empty", False))
    reachable = _route_reachable(context, _source_coordinate(context), storage)
    reasons = []
    if not belongs:
        reasons.append("storage_not_in_zone")
    if not not_assigned:
        reasons.append("storage_assigned")
    if not not_reserved:
        reasons.append("storage_reserved_or_not_empty")
    if not reachable:
        reasons.append("storage_unreachable")
    return RTSStorageFeasibility(
        available=belongs and not_assigned and not_reserved,
        belongs_to_zone=belongs,
        not_assigned=not_assigned,
        not_reserved=not_reserved,
        reachable=reachable,
        reason_codes=tuple(reasons),
    )


def select_replenishment_station(context: Any, final_storage: Any | None) -> tuple[Any | None, RTSStationFeasibility | None]:
    station_manager = getattr(getattr(context, "warehouse", None), "station_manager", None)
    stations = list(getattr(station_manager, "replenishment_stations", []) or [])
    feasible: list[tuple[int, str, Any, RTSStationFeasibility]] = []
    for station in stations:
        feasibility = station_feasibility(context, station, final_storage)
        if feasibility.admission_possible and feasibility.structurally_reachable:
            feasible.append((len(getattr(station, "robot_ids", {}) or {}), station_id(station), station, feasibility))
    if not feasible:
        if not stations:
            return None, None
        station = sorted(stations, key=station_id)[0]
        return station, station_feasibility(context, station, final_storage)
    _, _, station, feasibility = sorted(feasible, key=lambda item: (item[0], item[1]))[0]
    return station, feasibility


def station_feasibility(context: Any, station: Any | None, final_storage: Any | None) -> RTSStationFeasibility:
    if station is None:
        return RTSStationFeasibility(False, False, False, False, False, False, ("missing_station",))
    correct = bool(getattr(station, "is_replenishment_station", lambda: False)())
    robot_ids = getattr(station, "robot_ids", {}) or {}
    robot_queue = getattr(station, "robot_queue", []) or []
    active_capacity = len(robot_ids) < int(getattr(station, "max_robots", 0))
    queue_capacity = len(robot_queue) < int(getattr(station, "max_robot_queue", 0))
    admission = correct and active_capacity
    station_gate = _station_gate(station)
    source_to_station = _route_reachable(context, _source_coordinate(context), station_gate)
    station_to_storage = _route_reachable(context, station_gate, final_storage)
    reachable = source_to_station and station_to_storage
    reasons = []
    if not correct:
        reasons.append("wrong_station_type")
    if not active_capacity:
        reasons.append("station_active_capacity_unavailable")
    if not queue_capacity:
        reasons.append("station_queue_capacity_unavailable")
    if not source_to_station:
        reasons.append("picker_to_replenishment_unreachable")
    if not station_to_storage:
        reasons.append("replenishment_to_storage_unreachable")
    return RTSStationFeasibility(
        station_exists=True,
        correct_station_type=correct,
        active_capacity_available=active_capacity,
        queue_capacity_available=queue_capacity,
        admission_possible=admission,
        structurally_reachable=reachable,
        reason_codes=tuple(reasons),
    )


def action_key(branch: str, zone_id: str) -> str:
    return f"{str(branch)}:{str(zone_id)}"


def storage_id(storage: Any | None) -> str:
    if storage is None:
        return ""
    if getattr(storage, "storage_id", None) is not None:
        return str(getattr(storage, "storage_id"))
    return f"{getattr(storage, 'pos_x', '')}:{getattr(storage, 'pos_y', '')}"


def station_id(station: Any | None) -> str:
    return "" if station is None else str(getattr(station, "station_id", ""))


def _coord(obj: Any | None) -> tuple[float, float] | None:
    if obj is None:
        return None
    if getattr(obj, "coordinate", None) is not None:
        coord = getattr(obj, "coordinate")
        return (float(getattr(coord, "x", 0.0)), float(getattr(coord, "y", 0.0)))
    return (float(getattr(obj, "pos_x", getattr(obj, "x", 0.0))), float(getattr(obj, "pos_y", getattr(obj, "y", 0.0))))


def _source_coordinate(context: Any) -> Any:
    station = getattr(context, "station", None)
    if station is not None:
        return station
    return getattr(context, "robot", None)


def _station_gate(station: Any | None) -> Any | None:
    if station is None:
        return None
    path = list(getattr(station, "get_path", lambda: [])() or [])
    return path[0] if path else station


def _route_reachable(context: Any, src: Any, dst: Any) -> bool:
    if src is None or dst is None:
        return False
    warehouse = getattr(context, "warehouse", None)
    graph = getattr(warehouse, "graph_pod", None) or getattr(warehouse, "graph", None)
    src_coord = _coord(src)
    dst_coord = _coord(dst)
    if src_coord is None or dst_coord is None:
        return False
    if hasattr(graph, "dijkstra"):
        try:
            start = f"{int(round(src_coord[0]))},{int(round(src_coord[1]))}"
            end = f"{int(round(dst_coord[0]))},{int(round(dst_coord[1]))}"
            return graph.dijkstra(start, end, []) is not None
        except Exception:
            return False
    result = graph_distance_or_fallback(warehouse, NetLogoCoordinate(*src_coord), NetLogoCoordinate(*dst_coord), allow_metric_fallback=True)
    return result.distance is not None


def _distance_from_source(context: Any, storage: Any) -> float:
    result = graph_distance_or_fallback(getattr(context, "warehouse", None), _source_coordinate(context), storage)
    if result.distance is not None:
        return float(result.distance)
    src = _coord(_source_coordinate(context)) or (0.0, 0.0)
    dst = _coord(storage) or (0.0, 0.0)
    return abs(src[0] - dst[0]) + abs(src[1] - dst[1])


def _station_destination_pressure(context: Any, station: Any | None) -> float:
    if station is None:
        return 0.0
    station_key = station_id(station)
    robots = [obj for obj in getattr(getattr(context, "warehouse", None), "_objects", []) or [] if _is_robot_object(obj)]
    denominator = max(1, len(robots))
    count = 0
    station_coord = _coord(station)
    for robot in robots:
        destination = getattr(robot, "destination", None)
        if destination is None:
            continue
        if getattr(destination, "station_id", None) is not None and str(getattr(destination, "station_id")) == station_key:
            count += 1
            continue
        if station_coord is not None and _coord(destination) == station_coord:
            count += 1
    return max(0.0, min(1.0, float(count) / float(denominator)))


def _is_robot_object(obj: object) -> bool:
    return str(getattr(obj, "object_type", "")).lower() == "robot"


def _with_cycle_estimate(context: Any, action_context: RTSActionContext) -> RTSActionContext:
    try:
        from .cycle_estimator import estimate_cycle_for_action_context

        estimate = estimate_cycle_for_action_context(context, action_context)
    except Exception as exc:
        estimate = _UnavailableCycleEstimate(action_context, exc)
    return action_context.with_cycle_estimate(estimate)


class _UnavailableCycleEstimate:
    def __init__(self, action_context: RTSActionContext, exc: Exception):
        self.action_context = action_context
        self.exc = exc

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "estimator_version": "rts_structural_cycle_estimator.v1",
            "known": False,
            "status": f"unavailable_estimator_error:{type(self.exc).__name__}",
            "branch": self.action_context.branch,
            "zone_id": self.action_context.zone_id,
            "estimated_cycle_seconds": None,
            "estimated_travel_seconds": 0.0,
            "estimated_queue_seconds": 0.0,
            "estimated_replenishment_service_seconds": 0.0,
            "estimated_handling_seconds": 0.0,
            "component_seconds": {},
            "route_components": [],
            "queue_estimate": None,
            "fallback_used": False,
            "next_job_known": bool(getattr(self.action_context.next_job_proposal, "has_next_job", False)),
            "next_job_proposal_id": getattr(self.action_context.next_job_proposal, "proposal_id", None),
        }
