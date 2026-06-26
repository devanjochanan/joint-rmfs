"""Structural, action-conditioned RTS paper-cycle estimates."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from engine.netlogo_coordinate import NetLogoCoordinate

from model.robot_job import replenishment_service_steps_for_skus

from .action_space import REPLENISH_STORE, STORE
from .graph_distance import RTSDistanceResult, graph_distance_or_fallback
from .queue_estimator import RTSQueueEstimate, estimate_replenishment_queue
from .travel_time import (
    EMPTY_ROBOT,
    LOADED_ROBOT,
    TIME_CONVERSION_VERSION,
    TRAVEL_TIME_VERSION,
    backend_steps_to_seconds,
    distance_to_seconds,
    nominal_robot_speed,
)


CYCLE_ESTIMATE_VERSION = "rts_structural_cycle_estimator.v1"
SEMANTICS_HOST_STRUCTURAL = "host_structural"
SEMANTICS_PAPER_STRUCTURAL = "paper_structural"
SUPPORTED_CYCLE_ESTIMATE_SEMANTICS = (SEMANTICS_HOST_STRUCTURAL, SEMANTICS_PAPER_STRUCTURAL)


@dataclass(frozen=True)
class RTSCycleEstimate:
    known: bool
    status: str
    branch: str
    zone_id: str
    estimated_cycle_seconds: float | None
    estimated_travel_seconds: float
    estimated_queue_seconds: float
    estimated_replenishment_service_seconds: float
    estimated_handling_seconds: float
    component_seconds: dict[str, float | None]
    route_components: tuple[dict[str, Any], ...]
    queue_estimate: RTSQueueEstimate | None
    semantics: str = SEMANTICS_HOST_STRUCTURAL
    estimator_version: str = CYCLE_ESTIMATE_VERSION
    travel_time_version: str = TRAVEL_TIME_VERSION
    time_conversion_version: str = TIME_CONVERSION_VERSION
    fallback_used: bool = False
    next_job_known: bool = False
    next_job_proposal_id: str | None = None

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "estimator_version": self.estimator_version,
            "semantics": self.semantics,
            "known": bool(self.known),
            "status": self.status,
            "branch": self.branch,
            "zone_id": self.zone_id,
            "estimated_cycle_seconds": self.estimated_cycle_seconds,
            "estimated_travel_seconds": float(self.estimated_travel_seconds),
            "estimated_queue_seconds": float(self.estimated_queue_seconds),
            "estimated_replenishment_service_seconds": float(self.estimated_replenishment_service_seconds),
            "estimated_handling_seconds": float(self.estimated_handling_seconds),
            "component_seconds": dict(self.component_seconds),
            "route_components": [dict(component) for component in self.route_components],
            "queue_estimate": self.queue_estimate.to_json_dict() if self.queue_estimate is not None else None,
            "travel_time_version": self.travel_time_version,
            "time_conversion_version": self.time_conversion_version,
            "fallback_used": bool(self.fallback_used),
            "next_job_known": bool(self.next_job_known),
            "next_job_proposal_id": self.next_job_proposal_id,
        }


def estimate_cycle_for_action_context(
    context: Any,
    action_context: Any,
    *,
    semantics: str = SEMANTICS_HOST_STRUCTURAL,
    allow_metric_fallback: bool = True,
) -> RTSCycleEstimate:
    if semantics not in SUPPORTED_CYCLE_ESTIMATE_SEMANTICS:
        raise ValueError(f"unsupported RTS cycle-estimate semantics: {semantics!r}")
    warehouse = getattr(context, "warehouse", None)
    robot = getattr(context, "robot", None)
    speed = nominal_robot_speed(robot)
    if speed is None:
        return _unknown(context, action_context, "unavailable_robot_speed", semantics)
    branch = str(getattr(action_context, "branch", ""))
    proposal = getattr(action_context, "next_job_proposal", None)
    next_job_known = bool(getattr(proposal, "has_next_job", False))
    proposal_id = getattr(proposal, "proposal_id", None)
    route_components: list[dict[str, Any]] = []
    component_seconds: dict[str, float | None] = {}
    fallback_used = False

    source = _source_coordinate(context)
    final_storage = getattr(action_context, "candidate_storage", None)
    if final_storage is None:
        return _unknown(context, action_context, "unavailable_missing_final_storage", semantics)

    current_drop_seconds = _handling_seconds(context)
    next_pick_seconds = _handling_seconds(context)
    component_seconds["current_pod_drop_seconds"] = current_drop_seconds
    component_seconds["next_pod_pickup_seconds"] = next_pick_seconds

    queue_estimate = None
    replenishment_service_seconds = 0.0
    queue_seconds = 0.0

    if branch == STORE:
        first = _route_seconds(
            warehouse,
            source,
            final_storage,
            speed,
            topology=LOADED_ROBOT,
            name="loaded_picker_to_final_storage",
            allow_metric_fallback=allow_metric_fallback,
        )
        route_components.append(first[0])
        if first[1] is None:
            return _unknown(context, action_context, "unavailable_route_loaded_picker_to_final_storage", semantics)
        fallback_used = fallback_used or first[2]
    elif branch == REPLENISH_STORE:
        station = getattr(action_context, "replenishment_station", None)
        if station is None:
            return _unknown(context, action_context, "unavailable_missing_replenishment_station", semantics)
        station_gate = _station_gate(station)
        first = _route_seconds(
            warehouse,
            source,
            station_gate,
            speed,
            topology=LOADED_ROBOT,
            name="loaded_picker_to_replenishment_station",
            allow_metric_fallback=allow_metric_fallback,
        )
        second = _route_seconds(
            warehouse,
            station_gate,
            final_storage,
            speed,
            topology=LOADED_ROBOT,
            name="loaded_replenishment_station_to_final_storage",
            allow_metric_fallback=allow_metric_fallback,
        )
        route_components.extend([first[0], second[0]])
        if first[1] is None:
            return _unknown(context, action_context, "unavailable_route_loaded_picker_to_replenishment_station", semantics)
        if second[1] is None:
            return _unknown(context, action_context, "unavailable_route_loaded_replenishment_station_to_final_storage", semantics)
        fallback_used = fallback_used or first[2] or second[2]
        queue_estimate = estimate_replenishment_queue(context, station)
        if not queue_estimate.known:
            return _unknown(context, action_context, queue_estimate.status, semantics, queue_estimate=queue_estimate)
        queue_seconds = float(queue_estimate.estimated_wait_seconds or 0.0)
        replenishment_service_seconds = _replenishment_service_seconds(context, action_context)
    else:
        return _unknown(context, action_context, "unavailable_unsupported_branch", semantics)

    if not next_job_known:
        return _partial_no_next(
            context,
            action_context,
            semantics,
            route_components,
            component_seconds,
            queue_estimate,
            replenishment_service_seconds,
            queue_seconds,
            fallback_used,
            proposal_id,
        )

    next_pod_coordinate = _next_pod_coordinate(proposal)
    next_station = _next_station(context, proposal)
    if next_pod_coordinate is None:
        return _unknown(context, action_context, "unavailable_missing_next_pod_coordinate", semantics)
    if next_station is None:
        return _unknown(context, action_context, "unavailable_missing_next_picking_station", semantics)
    empty_leg = _route_seconds(
        warehouse,
        final_storage,
        next_pod_coordinate,
        speed,
        topology=EMPTY_ROBOT,
        name="empty_final_storage_to_committed_next_pod",
        allow_metric_fallback=allow_metric_fallback,
    )
    loaded_leg = _route_seconds(
        warehouse,
        next_pod_coordinate,
        next_station,
        speed,
        topology=LOADED_ROBOT,
        name="loaded_committed_next_pod_to_picker",
        allow_metric_fallback=allow_metric_fallback,
    )
    route_components.extend([empty_leg[0], loaded_leg[0]])
    if empty_leg[1] is None:
        return _unknown(context, action_context, "unavailable_route_empty_final_storage_to_next_pod", semantics)
    if loaded_leg[1] is None:
        return _unknown(context, action_context, "unavailable_route_loaded_next_pod_to_picker", semantics)
    fallback_used = fallback_used or empty_leg[2] or loaded_leg[2]

    travel_seconds = _sum_known(component.get("seconds") for component in route_components)
    handling_seconds = _sum_known((current_drop_seconds, next_pick_seconds))
    total = travel_seconds + queue_seconds + replenishment_service_seconds + handling_seconds
    return RTSCycleEstimate(
        known=True,
        status="available",
        branch=branch,
        zone_id=str(getattr(action_context, "zone_id", "")),
        estimated_cycle_seconds=float(total),
        estimated_travel_seconds=float(travel_seconds),
        estimated_queue_seconds=float(queue_seconds),
        estimated_replenishment_service_seconds=float(replenishment_service_seconds),
        estimated_handling_seconds=float(handling_seconds),
        component_seconds=component_seconds,
        route_components=tuple(route_components),
        queue_estimate=queue_estimate,
        semantics=semantics,
        fallback_used=fallback_used,
        next_job_known=True,
        next_job_proposal_id=str(proposal_id) if proposal_id is not None else None,
    )


def _partial_no_next(
    context: Any,
    action_context: Any,
    semantics: str,
    route_components: list[dict[str, Any]],
    component_seconds: dict[str, float | None],
    queue_estimate: RTSQueueEstimate | None,
    replenishment_service_seconds: float,
    queue_seconds: float,
    fallback_used: bool,
    proposal_id: Any,
) -> RTSCycleEstimate:
    travel_seconds = _sum_known(component.get("seconds") for component in route_components)
    handling_seconds = _sum_known(component_seconds.values())
    return RTSCycleEstimate(
        known=False,
        status="unavailable_no_next_job",
        branch=str(getattr(action_context, "branch", "")),
        zone_id=str(getattr(action_context, "zone_id", "")),
        estimated_cycle_seconds=None,
        estimated_travel_seconds=float(travel_seconds),
        estimated_queue_seconds=float(queue_seconds),
        estimated_replenishment_service_seconds=float(replenishment_service_seconds),
        estimated_handling_seconds=float(handling_seconds),
        component_seconds=component_seconds,
        route_components=tuple(route_components),
        queue_estimate=queue_estimate,
        semantics=semantics,
        fallback_used=fallback_used,
        next_job_known=False,
        next_job_proposal_id=str(proposal_id) if proposal_id is not None else None,
    )


def _unknown(
    context: Any,
    action_context: Any,
    status: str,
    semantics: str,
    *,
    queue_estimate: RTSQueueEstimate | None = None,
) -> RTSCycleEstimate:
    return RTSCycleEstimate(
        known=False,
        status=status,
        branch=str(getattr(action_context, "branch", "")),
        zone_id=str(getattr(action_context, "zone_id", "")),
        estimated_cycle_seconds=None,
        estimated_travel_seconds=0.0,
        estimated_queue_seconds=0.0,
        estimated_replenishment_service_seconds=0.0,
        estimated_handling_seconds=0.0,
        component_seconds={},
        route_components=(),
        queue_estimate=queue_estimate,
        semantics=semantics,
        next_job_known=bool(getattr(getattr(action_context, "next_job_proposal", None), "has_next_job", False)),
        next_job_proposal_id=getattr(getattr(action_context, "next_job_proposal", None), "proposal_id", None),
    )


def _route_seconds(
    warehouse: Any,
    src: Any,
    dst: Any,
    speed: float,
    *,
    topology: str,
    name: str,
    allow_metric_fallback: bool,
) -> tuple[dict[str, Any], float | None, bool]:
    result = graph_distance_or_fallback(
        warehouse,
        src,
        dst,
        allow_metric_fallback=allow_metric_fallback,
        topology=topology,
    )
    seconds = distance_to_seconds(result.distance, speed)
    payload = _route_payload(name, topology, result, seconds)
    return payload, seconds, bool(result.fallback_used)


def _route_payload(name: str, topology: str, result: RTSDistanceResult, seconds: float | None) -> dict[str, Any]:
    return {
        "name": name,
        "topology": topology,
        "distance": result.distance,
        "seconds": seconds,
        "source": result.source,
        "status": result.status,
        "fallback_used": bool(result.fallback_used),
    }


def _handling_seconds(context: Any) -> float | None:
    robot = getattr(context, "robot", None)
    delay_steps = getattr(robot, "delay_per_task", 0)
    return backend_steps_to_seconds(getattr(context, "warehouse", None), delay_steps)


def _replenishment_service_seconds(context: Any, action_context: Any) -> float:
    robot_job = getattr(getattr(context, "robot", None), "job", None)
    delay_per_sku = getattr(robot_job, "replenishment_delay_per_sku", 20)
    steps = replenishment_service_steps_for_skus(
        getattr(context, "pod", None),
        getattr(action_context, "replenishment_eligible_skus", ()) or None,
        delay_per_sku=delay_per_sku,
    )
    seconds = backend_steps_to_seconds(getattr(context, "warehouse", None), steps)
    return float(seconds or 0.0)


def _next_pod_coordinate(proposal: Any) -> Any | None:
    job = getattr(proposal, "job", None)
    coord = getattr(job, "pod_coordinate", None)
    if coord is not None:
        return coord
    pod = getattr(job, "pod", None)
    if pod is not None:
        return NetLogoCoordinate(getattr(pod, "pos_x", 0.0), getattr(pod, "pos_y", 0.0))
    return None


def _next_station(context: Any, proposal: Any) -> Any | None:
    station_id = str(getattr(proposal, "picking_station_id", "") or "")
    station_manager = getattr(getattr(context, "warehouse", None), "station_manager", None)
    for station in getattr(station_manager, "stations", []) or []:
        if str(getattr(station, "station_id", "")) == station_id:
            return station
    return None


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


def _sum_known(values: Any) -> float:
    total = 0.0
    for value in values:
        if value is None:
            continue
        try:
            number = float(value)
        except Exception:
            continue
        if math.isfinite(number):
            total += number
    return float(total)
