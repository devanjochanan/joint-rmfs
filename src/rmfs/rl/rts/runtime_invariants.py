"""Side-effect-free RTS runtime ownership diagnostics."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any


HARD_VIOLATION_CODES = {
    "duplicate_job_queue_object",
    "job_assigned_to_multiple_robots",
    "activated_committed_next_job_still_queued",
    "robot_multiple_active_jobs",
    "robot_customer_job_with_charger_claim",
    "pod_carried_and_available",
    "pod_assigned_to_multiple_jobs",
    "pod_unavailable_without_owner",
    "charger_claimed_by_multiple_robots",
    "robot_claim_absent_from_occupied_chargers",
    "occupied_charger_without_matching_robot",
}


def check_runtime_invariants(warehouse: Any) -> dict[str, Any]:
    violations: list[dict[str, Any]] = []
    job_queue = list(getattr(warehouse, "job_queue", []) or [])
    robots = [
        obj
        for obj in getattr(warehouse, "_objects", []) or []
        if getattr(obj, "object_type", None) == "robot"
    ]

    _check_job_ownership(violations, warehouse, job_queue, robots)
    _check_robot_ownership(violations, robots)
    _check_pod_ownership(violations, warehouse, job_queue, robots)
    _check_charger_ownership(violations, warehouse, robots)

    hard = [item for item in violations if item["code"] in HARD_VIOLATION_CODES]
    return {
        "checked": True,
        "violation_count": len(violations),
        "hard_violation_count": len(hard),
        "violations": violations,
    }


def _add(violations: list[dict[str, Any]], code: str, **details: Any) -> None:
    violations.append({"code": code, **{key: _safe(value) for key, value in details.items()}})


def _check_job_ownership(violations, warehouse, job_queue, robots) -> None:
    queue_object_counts = Counter(id(job) for job in job_queue)
    for identity, count in queue_object_counts.items():
        if count > 1:
            _add(violations, "duplicate_job_queue_object", object_id=identity, count=count)

    active_jobs = defaultdict(list)
    for robot in robots:
        job = getattr(robot, "job", None)
        if job is not None and getattr(robot, "current_state", None) != "idle":
            active_jobs[id(job)].append(robot)
    for _identity, owners in active_jobs.items():
        if len(owners) > 1:
            _add(
                violations,
                "job_assigned_to_multiple_robots",
                job_id=_job_id(getattr(owners[0], "job", None)),
                robot_ids=[_robot_id(robot) for robot in owners],
            )

    queued_ids = {id(job) for job in job_queue}
    for robot in robots:
        job = getattr(robot, "job", None)
        if job is None:
            continue
        if (
            getattr(job, "committed_next_activated_by_robot_id", None)
            and id(job) in queued_ids
        ):
            _add(
                violations,
                "activated_committed_next_job_still_queued",
                robot_id=_robot_id(robot),
                job_id=_job_id(job),
            )

    registry = getattr(warehouse, "committed_next_registry", None)
    reservations = getattr(registry, "reservations_by_id", {}) if registry is not None else {}
    for reservation in reservations.values():
        if getattr(reservation, "status", None) != "committed":
            continue
        if getattr(reservation, "job", None) not in job_queue:
            _add(
                violations,
                "reserved_job_missing_from_queue_before_activation",
                reservation_id=getattr(reservation, "reservation_id", None),
                job_id=getattr(reservation, "job_id", None),
            )


def _check_robot_ownership(violations, robots) -> None:
    for robot in robots:
        job = getattr(robot, "job", None)
        active_jobs = 1 if job is not None and getattr(robot, "current_state", None) != "idle" else 0
        if active_jobs > 1:
            _add(violations, "robot_multiple_active_jobs", robot_id=_robot_id(robot))
        if job is not None and getattr(robot, "_claimed_charger", None) is not None:
            _add(
                violations,
                "robot_customer_job_with_charger_claim",
                robot_id=_robot_id(robot),
                job_id=_job_id(job),
                charger=getattr(robot, "_claimed_charger", None),
            )
        if getattr(robot, "current_state", None) in {"going_to_charge", "charging", "dead"}:
            if getattr(robot, "job", None) is None and getattr(robot, "is_charging_pending", False):
                continue


def _check_pod_ownership(violations, warehouse, job_queue, robots) -> None:
    pod_jobs = defaultdict(list)
    for job in job_queue:
        pod = getattr(job, "pod", None)
        if pod is not None:
            pod_jobs[id(pod)].append(job)
    for robot in robots:
        job = getattr(robot, "job", None)
        pod = getattr(job, "pod", None)
        if pod is not None:
            pod_jobs[id(pod)].append(job)
            if getattr(pod, "is_idle", False):
                _add(
                    violations,
                    "pod_carried_and_available",
                    robot_id=_robot_id(robot),
                    pod_id=getattr(pod, "pod_id", None),
                )
    for _identity, jobs in pod_jobs.items():
        unique_jobs = {id(job) for job in jobs}
        if len(unique_jobs) > 1:
            _add(
                violations,
                "pod_assigned_to_multiple_jobs",
                pod_id=getattr(getattr(jobs[0], "pod", None), "pod_id", None),
                job_ids=[_job_id(job) for job in jobs],
            )

    pod_manager = getattr(warehouse, "pod_manager", None)
    if pod_manager is None:
        return
    all_pods = getattr(pod_manager, "get_all_pods", None)
    if all_pods is None:
        return
    pods = list(all_pods() or [])
    pod_idle = getattr(pod_manager, "pod_idle", {}) or {}

    # Legitimate ownership sources for an unavailable pod.
    queued_pod_ids = {
        getattr(getattr(job, "pod", None), "pod_id", None)
        for job in job_queue
        if getattr(job, "pod", None) is not None
    }
    active_pod_ids = set()
    for robot in robots:
        job = getattr(robot, "job", None)
        pod = getattr(job, "pod", None)
        if pod is not None and getattr(robot, "current_state", None) != "idle":
            active_pod_ids.add(getattr(pod, "pod_id", None))
    pending_pod_ids = {
        int(request["pod_id"])
        for request in (getattr(warehouse, "pending_replenishment_dispatches", []) or [])
    }
    registry = getattr(warehouse, "committed_next_registry", None)
    reserved_pod_ids = set(getattr(registry, "pod_id_to_reservation", {}) or {}) if registry is not None else set()

    for pod in pods:
        pod_id = getattr(pod, "pod_id", None)
        raw_idle = bool(pod_idle.get(int(str(pod_id)), True)) if pod_id is not None else True
        if raw_idle:
            # An available pod must not still be owned by committed-next / RTS
            # return continuations.
            if getattr(pod, "committed_next_owner_robot_id", None):
                _add(
                    violations,
                    "pod_committed_next_and_available",
                    pod_id=pod_id,
                    owner=getattr(pod, "committed_next_owner_robot_id", None),
                )
            if getattr(pod, "rts_return_in_progress", False):
                _add(
                    violations,
                    "pod_return_in_progress_and_available",
                    pod_id=pod_id,
                )
            continue
        # Unavailable pod: must be owned by at least one legitimate state.
        owner_states = []
        if pod_id in queued_pod_ids:
            owner_states.append("queued_job")
        if pod_id in active_pod_ids:
            owner_states.append("active_robot_job")
        if (
            getattr(pod, "is_awaiting_replenishment", False)
            or getattr(pod, "has_pending_replenishment_dispatch", False)
            or (pod_id in pending_pod_ids)
        ):
            owner_states.append("replenishment_commitment")
        if getattr(pod, "rts_return_in_progress", False):
            owner_states.append("rts_return_continuation")
        if getattr(pod, "committed_next_owner_robot_id", None) or (pod_id in reserved_pod_ids):
            owner_states.append("committed_next_reservation")
        if not owner_states:
            _add(
                violations,
                "pod_unavailable_without_owner",
                pod_id=pod_id,
                observed={
                    "is_awaiting_replenishment": bool(getattr(pod, "is_awaiting_replenishment", False)),
                    "has_pending_replenishment_dispatch": bool(getattr(pod, "has_pending_replenishment_dispatch", False)),
                    "must_replenish_before_pick": bool(getattr(pod, "must_replenish_before_pick", False)),
                    "rts_return_in_progress": bool(getattr(pod, "rts_return_in_progress", False)),
                    "committed_next_owner_robot_id": getattr(pod, "committed_next_owner_robot_id", None),
                    "station": str(getattr(pod, "station", None)),
                },
            )


def _check_charger_ownership(violations, warehouse, robots) -> None:
    occupied = dict(getattr(warehouse, "occupied_chargers", {}) or {})
    claimed_by_robot = {}
    for robot in robots:
        claim = getattr(robot, "_claimed_charger", None)
        if claim is None:
            continue
        if claim in claimed_by_robot:
            _add(
                violations,
                "charger_claimed_by_multiple_robots",
                charger=claim,
                robot_ids=[claimed_by_robot[claim], _robot_id(robot)],
            )
        claimed_by_robot[claim] = _robot_id(robot)
        if occupied.get(claim) != getattr(robot, "id", None):
            _add(
                violations,
                "robot_claim_absent_from_occupied_chargers",
                robot_id=_robot_id(robot),
                charger=claim,
            )
    for charger, robot_id in occupied.items():
        matched = any(getattr(robot, "id", None) == robot_id and getattr(robot, "_claimed_charger", None) == charger for robot in robots)
        if not matched:
            _add(
                violations,
                "occupied_charger_without_matching_robot",
                charger=charger,
                robot_id=robot_id,
            )


def _job_id(job: Any) -> str:
    if job is None:
        return ""
    if getattr(job, "my_id", None) is not None:
        return str(getattr(job, "my_id"))
    return str(getattr(job, "job_id", ""))


def _robot_id(robot: Any) -> str:
    if getattr(robot, "_id", None) is not None:
        return str(getattr(robot, "_id"))
    return str(getattr(robot, "id", ""))


def _safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    return str(value)
