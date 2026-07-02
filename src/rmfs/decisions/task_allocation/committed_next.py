"""Committed-next retrieval reservations for RTS paper cycles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.rmfs.rl.rts.graph_distance import graph_distance_or_fallback
from src.rmfs.rl.rts.travel_time import EMPTY_ROBOT, LOADED_ROBOT
from src.rmfs.rl.rts.zone_registry import build_zone_registry

PROPOSAL_SEMANTICS_VERSION = "rts_nearest_next_job_proposal.v1"
STATUS_COMMITTED = "committed"
STATUS_ACTIVATED = "activated"
STATUS_CANCELLED = "cancelled"
STATUS_COMPLETED = "completed"
TERMINAL_STATUSES = {STATUS_CANCELLED, STATUS_COMPLETED}


@dataclass
class CommittedNextReservation:
    reservation_id: str
    owner_robot_id: str
    job: Any
    job_id: str
    pod_id: str
    picking_station_id: str
    original_queue_index: int
    created_time_seconds: float
    status: str = STATUS_COMMITTED
    linked_rts_decision_event_id: str | None = None
    cancellation_reason: str | None = None
    candidate_count: int = 0
    retrieval_cost: float | None = None
    committed_next_zone_id: str | None = None
    activation_time_seconds: float | None = None
    selected_branch: str | None = None
    selected_zone_id: str | None = None
    selected_action_index: int | None = None
    selected_storage_id: str | None = None

    def is_active(self) -> bool:
        return self.status in {STATUS_COMMITTED, STATUS_ACTIVATED}


@dataclass(frozen=True)
class CommittedNextProposal:
    proposal_id: str
    owner_robot_id: str
    zone_id: str
    candidate_storage: Any
    candidate_storage_id: str
    destination_x: float
    destination_y: float
    job: Any | None
    job_id: str | None
    pod_id: str | None
    picking_station_id: str | None
    original_queue_index: int | None
    created_time_seconds: float
    candidate_count: int = 0
    candidate_to_proposed_next_pod_distance: float | None = None
    next_pod_to_picker_distance: float | None = None
    proposal_cost: float | None = None
    committed_next_zone_id: str | None = None
    candidate_to_proposed_next_pod_distance_source: str | None = None
    candidate_to_proposed_next_pod_distance_status: str | None = None
    next_pod_to_picker_distance_source: str | None = None
    next_pod_to_picker_distance_status: str | None = None
    fallback_used: bool = False

    @property
    def has_next_job(self) -> bool:
        return self.job is not None and self.job_id is not None and self.pod_id is not None

    def to_state_json(self) -> dict[str, Any]:
        next_zone = str(self.committed_next_zone_id or "")
        return {
            "proposal_semantics_version": PROPOSAL_SEMANTICS_VERSION,
            "proposal_id": self.proposal_id,
            "zone_id": self.zone_id,
            "candidate_storage_id": self.candidate_storage_id,
            "destination_x": float(self.destination_x),
            "destination_y": float(self.destination_y),
            "proposed_next_job_known": 1 if self.has_next_job else 0,
            "next_job_known": 1 if self.has_next_job else 0,
            "job_id": str(self.job_id or ""),
            "pod_id": str(self.pod_id or ""),
            "picking_station_id": str(self.picking_station_id or ""),
            "committed_next_zone_id": next_zone,
            "candidate_count": int(self.candidate_count),
            "candidate_to_proposed_next_pod_distance": _finite(self.candidate_to_proposed_next_pod_distance),
            "candidate_storage_to_next_pod_distance": _finite(self.candidate_to_proposed_next_pod_distance),
            "next_pod_to_picker_distance": _finite(self.next_pod_to_picker_distance),
            "proposal_cost": _finite(self.proposal_cost),
            "candidate_to_proposed_next_pod_distance_source": str(self.candidate_to_proposed_next_pod_distance_source or ""),
            "candidate_to_proposed_next_pod_distance_status": str(self.candidate_to_proposed_next_pod_distance_status or ""),
            "next_pod_to_picker_distance_source": str(self.next_pod_to_picker_distance_source or ""),
            "next_pod_to_picker_distance_status": str(self.next_pod_to_picker_distance_status or ""),
            "fallback_used": bool(self.fallback_used),
        }


class CommittedNextRegistry:
    def __init__(self):
        self.reservations_by_id: dict[str, CommittedNextReservation] = {}
        self.robot_id_to_reservation: dict[str, str] = {}
        self.pod_id_to_reservation: dict[str, str] = {}
        self.job_id_to_reservation: dict[str, str] = {}
        self.robot_id_to_action_proposals: dict[str, dict[str, CommittedNextProposal]] = {}
        self._counter = 0

    def rebuild_indexes(self) -> None:
        self.robot_id_to_reservation = {}
        self.pod_id_to_reservation = {}
        self.job_id_to_reservation = {}
        for reservation_id, reservation in self.reservations_by_id.items():
            if not reservation.is_active():
                continue
            self.robot_id_to_reservation[reservation.owner_robot_id] = reservation_id
            self.pod_id_to_reservation[reservation.pod_id] = reservation_id
            self.job_id_to_reservation[reservation.job_id] = reservation_id

    def get_for_robot(self, robot: Any) -> CommittedNextReservation | None:
        return self.get_by_robot_id(robot_id(robot))

    def get_by_robot_id(self, owner_robot_id: str) -> CommittedNextReservation | None:
        reservation_id = self.robot_id_to_reservation.get(str(owner_robot_id))
        if reservation_id is None:
            return None
        return self.reservations_by_id.get(reservation_id)

    def has_for_robot(self, robot: Any) -> bool:
        reservation = self.get_for_robot(robot)
        return reservation is not None and reservation.status == STATUS_COMMITTED

    def build_action_proposals(
        self,
        inventory: Any,
        robot: Any,
        context: Any,
        zone_ids: tuple[str, ...],
        action_contexts: tuple[Any, ...] | None = None,
    ) -> dict[str, CommittedNextProposal]:
        owner_id = robot_id(robot)
        proposals: dict[str, CommittedNextProposal] = {}
        if self.get_for_robot(robot) is not None:
            self.robot_id_to_action_proposals[owner_id] = proposals
            return proposals
        registry = build_zone_registry(inventory, zone_ids=zone_ids)
        contexts_by_zone = {}
        for action_context in action_contexts or ():
            zone_id = str(getattr(action_context, "zone_id", ""))
            contexts_by_zone.setdefault(zone_id, action_context)
        for zone_id in zone_ids:
            action_context = contexts_by_zone.get(str(zone_id))
            storage = getattr(action_context, "candidate_storage", None)
            if storage is None:
                storage = self._candidate_storage_for_action(context, registry, str(zone_id))
            if storage is None:
                continue
            proposal = self._build_proposal_for_storage(
                inventory=inventory,
                robot=robot,
                zone_id=str(zone_id),
                storage=storage,
            )
            proposals[proposal_key("", zone_id)] = proposal
        self.robot_id_to_action_proposals[owner_id] = proposals
        return proposals

    def get_action_proposals_for_robot(self, robot: Any) -> dict[str, CommittedNextProposal]:
        return dict(self.robot_id_to_action_proposals.get(robot_id(robot), {}))

    def get_action_proposal(self, robot: Any, branch: str, zone_id: str) -> CommittedNextProposal | None:
        return self.robot_id_to_action_proposals.get(robot_id(robot), {}).get(proposal_key(branch, zone_id))

    def refresh_action_proposal_for_zone(
        self,
        inventory: Any,
        robot: Any,
        context: Any,
        zone_id: str,
        storage: Any | None = None,
    ) -> CommittedNextProposal | None:
        if storage is None:
            try:
                registry = build_zone_registry(inventory, zone_ids=(str(zone_id),))
            except Exception:
                registry = None
            storage = self._candidate_storage_for_action(context, registry, str(zone_id)) if registry is not None else None
        if storage is None:
            return None
        proposal = self._build_proposal_for_storage(
            inventory=inventory,
            robot=robot,
            zone_id=str(zone_id),
            storage=storage,
        )
        self.robot_id_to_action_proposals.setdefault(robot_id(robot), {})[proposal_key("", zone_id)] = proposal
        return proposal

    def clear_action_proposals_for_robot(self, robot: Any) -> None:
        self.robot_id_to_action_proposals.pop(robot_id(robot), None)

    def commit_for_decision(self, inventory: Any, robot: Any, decision: Any) -> CommittedNextReservation | None:
        existing = self.get_for_robot(robot)
        if existing is not None and existing.status == STATUS_COMMITTED:
            return existing
        branch = str(getattr(decision, "branch", "") or "")
        zone_id = str(getattr(decision, "zone_id", "") or "")
        proposal = self.get_action_proposal(robot, branch, zone_id)
        if proposal is None:
            proposal = self._proposal_from_decision(inventory, robot, decision)
        if proposal is None or not proposal.has_next_job:
            self.clear_action_proposals_for_robot(robot)
            return None
        decision_storage = getattr(decision, "storage", None)
        if decision_storage is not None and proposal.candidate_storage is not decision_storage:
            if storage_id(proposal.candidate_storage) != storage_id(decision_storage):
                proposal = self._proposal_from_decision(inventory, robot, decision)
                if proposal is None or not proposal.has_next_job:
                    self.clear_action_proposals_for_robot(robot)
                    return None
        if self.job_id_to_reservation.get(str(proposal.job_id)) is not None:
            return None
        if self.pod_id_to_reservation.get(str(proposal.pod_id)) is not None:
            return None
        valid, reason = self._validate_proposal_for_commit(inventory, proposal)
        if not valid:
            proposal = self._proposal_from_decision(inventory, robot, decision)
            valid, reason = self._validate_proposal_for_commit(inventory, proposal) if proposal is not None else (False, "missing_proposal")
            if not valid:
                raise ValueError(f"Committed-next proposal is no longer valid: {reason}")
        self._counter += 1
        reservation = CommittedNextReservation(
            reservation_id=f"cnr-{int(getattr(inventory, '_tick', 0))}-{robot_id(robot)}-{self._counter}",
            owner_robot_id=robot_id(robot),
            job=proposal.job,
            job_id=str(proposal.job_id),
            pod_id=str(proposal.pod_id),
            picking_station_id=str(proposal.picking_station_id),
            original_queue_index=int(proposal.original_queue_index or 0),
            created_time_seconds=float(getattr(inventory, "_tick", 0.0)),
            candidate_count=int(proposal.candidate_count),
            retrieval_cost=float(proposal.proposal_cost) if proposal.proposal_cost is not None else None,
            committed_next_zone_id=proposal.committed_next_zone_id,
            selected_branch=branch,
            selected_zone_id=zone_id,
            selected_action_index=int(getattr(decision, "action_index", -1) if getattr(decision, "action_index", None) is not None else -1),
            selected_storage_id=proposal.candidate_storage_id,
        )
        self._set_markers(reservation)
        self.reservations_by_id[reservation.reservation_id] = reservation
        self.robot_id_to_reservation[reservation.owner_robot_id] = reservation.reservation_id
        self.pod_id_to_reservation[reservation.pod_id] = reservation.reservation_id
        self.job_id_to_reservation[reservation.job_id] = reservation.reservation_id
        self.robot_id_to_action_proposals.pop(reservation.owner_robot_id, None)
        return reservation

    def reserve_for_robot(self, inventory: Any, robot: Any) -> CommittedNextReservation | None:
        existing = self.get_for_robot(robot)
        if existing is not None and existing.status == STATUS_COMMITTED:
            return existing
        candidates = self._eligible_candidates(inventory, robot)
        if not candidates:
            return None
        candidates.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
        cost, queue_index, job_id, pod_id, job, station_id, zone_id = candidates[0]
        self._counter += 1
        reservation = CommittedNextReservation(
            reservation_id=f"cnr-{int(getattr(inventory, '_tick', 0))}-{robot_id(robot)}-{self._counter}",
            owner_robot_id=robot_id(robot),
            job=job,
            job_id=job_id,
            pod_id=pod_id,
            picking_station_id=station_id,
            original_queue_index=queue_index,
            created_time_seconds=float(getattr(inventory, "_tick", 0.0)),
            candidate_count=len(candidates),
            retrieval_cost=float(cost),
            committed_next_zone_id=zone_id,
        )
        self._set_markers(reservation)
        self.reservations_by_id[reservation.reservation_id] = reservation
        self.robot_id_to_reservation[reservation.owner_robot_id] = reservation.reservation_id
        self.pod_id_to_reservation[reservation.pod_id] = reservation.reservation_id
        self.job_id_to_reservation[reservation.job_id] = reservation.reservation_id
        return reservation

    def _candidate_storage_for_action(self, context: Any, registry: Any, zone_id: str) -> Any | None:
        warehouse = getattr(context, "warehouse", None)
        storage_manager = getattr(warehouse, "storage_manager", None)
        storages = list(getattr(storage_manager, "storages", []) or [])
        candidates = [
            storage
            for storage in storages
            if registry.zone_id_for_storage(storage) == str(zone_id)
            and _storage_available(storage)
        ]
        if not candidates:
            return None
        station = getattr(context, "station", None)
        robot = getattr(context, "robot", None)
        origin_x = float(getattr(station, "pos_x", getattr(robot, "pos_x", 0.0)))
        origin_y = float(getattr(station, "pos_y", getattr(robot, "pos_y", 0.0)))
        return sorted(
            candidates,
            key=lambda storage: (
                _distance_between(inventory=warehouse, src=_Coord(origin_x, origin_y), dst=storage, topology=LOADED_ROBOT),
                _coord(storage),
            ),
        )[0]

    def _build_proposal_for_storage(
        self,
        inventory: Any,
        robot: Any,
        zone_id: str,
        storage: Any,
    ) -> CommittedNextProposal:
        candidates = self._eligible_candidates_from_storage(inventory, storage)
        candidates.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
        best = candidates[0] if candidates else None
        if best is None:
            return CommittedNextProposal(
                proposal_id=f"cnp-{int(getattr(inventory, '_tick', 0))}-{robot_id(robot)}-{zone_id}",
                owner_robot_id=robot_id(robot),
                zone_id=str(zone_id),
                candidate_storage=storage,
                candidate_storage_id=storage_id(storage),
                destination_x=float(getattr(storage, "pos_x", 0.0)),
                destination_y=float(getattr(storage, "pos_y", 0.0)),
                job=None,
                job_id=None,
                pod_id=None,
                picking_station_id=None,
                original_queue_index=None,
                created_time_seconds=float(getattr(inventory, "_tick", 0.0)),
                candidate_count=0,
            )
        cost, queue_index, job_key, pod_key, job, station_id, next_zone_id, storage_to_pod, pod_to_picker, empty_source, empty_status, loaded_source, loaded_status, fallback_used = best
        return CommittedNextProposal(
            proposal_id=f"cnp-{int(getattr(inventory, '_tick', 0))}-{robot_id(robot)}-{zone_id}",
            owner_robot_id=robot_id(robot),
            zone_id=str(zone_id),
            candidate_storage=storage,
            candidate_storage_id=storage_id(storage),
            destination_x=float(getattr(storage, "pos_x", 0.0)),
            destination_y=float(getattr(storage, "pos_y", 0.0)),
            job=job,
            job_id=job_key,
            pod_id=pod_key,
            picking_station_id=station_id,
            original_queue_index=queue_index,
            created_time_seconds=float(getattr(inventory, "_tick", 0.0)),
            candidate_count=len(candidates),
            candidate_to_proposed_next_pod_distance=float(storage_to_pod),
            next_pod_to_picker_distance=float(pod_to_picker),
            proposal_cost=float(cost),
            committed_next_zone_id=next_zone_id,
            candidate_to_proposed_next_pod_distance_source=empty_source,
            candidate_to_proposed_next_pod_distance_status=empty_status,
            next_pod_to_picker_distance_source=loaded_source,
            next_pod_to_picker_distance_status=loaded_status,
            fallback_used=bool(fallback_used),
        )

    def _eligible_candidates_from_storage(
        self,
        inventory: Any,
        storage: Any,
    ) -> list[tuple[float, int, str, str, Any, str, str, float, float, str, str, str, str, bool]]:
        candidates = []
        runtime = getattr(inventory, "rts_rollout_runtime", None)
        config = getattr(runtime, "config", None)
        try:
            registry = build_zone_registry(inventory, zone_ids=getattr(config, "zone_ids", ()) or ())
        except Exception:
            registry = None
        for queue_index, job in enumerate(getattr(inventory, "job_queue", []) or []):
            eligible, station, pod, zone_id = self._candidate_context(inventory, job, registry)
            if not eligible:
                continue
            if self.job_id_to_reservation.get(job_id(job)) is not None:
                continue
            if self.pod_id_to_reservation.get(pod_id(pod)) is not None:
                continue
            if _active_robot_has_job_or_pod(inventory, job, pod):
                continue
            try:
                empty_leg = graph_distance_or_fallback(inventory, storage, job.pod_coordinate, topology=EMPTY_ROBOT)
                loaded_leg = graph_distance_or_fallback(inventory, job.pod_coordinate, station, topology=LOADED_ROBOT)
                if empty_leg.distance is None or loaded_leg.distance is None:
                    continue
                storage_to_pod = float(empty_leg.distance)
                pod_to_picker = float(loaded_leg.distance)
            except Exception:
                continue
            cost = storage_to_pod + pod_to_picker
            candidates.append((
                cost,
                queue_index,
                job_id(job),
                pod_id(pod),
                job,
                station.station_id,
                zone_id,
                storage_to_pod,
                pod_to_picker,
                empty_leg.source,
                empty_leg.status,
                loaded_leg.source,
                loaded_leg.status,
                bool(empty_leg.fallback_used or loaded_leg.fallback_used),
            ))
        return candidates

    def _proposal_from_decision(self, inventory: Any, robot: Any, decision: Any) -> CommittedNextProposal | None:
        storage = getattr(decision, "storage", None)
        zone_id = str(getattr(decision, "zone_id", "") or "")
        if storage is None or not zone_id:
            return None
        return self._build_proposal_for_storage(inventory, robot, zone_id, storage)

    def _validate_proposal_for_commit(self, inventory: Any, proposal: CommittedNextProposal) -> tuple[bool, str]:
        if proposal.job is None:
            return False, "missing_job"
        if _identity_index(getattr(inventory, "job_queue", []), proposal.job) is None:
            return False, "job_missing_from_queue"
        eligible, station, pod, zone_id = self._candidate_context(
            inventory,
            proposal.job,
            build_zone_registry(inventory, zone_ids=getattr(getattr(getattr(inventory, "rts_rollout_runtime", None), "config", None), "zone_ids", ()) or ()),
        )
        if not eligible:
            return False, "job_no_longer_eligible"
        if str(getattr(station, "station_id", "")) != str(proposal.picking_station_id):
            return False, "station_changed"
        if pod_id(pod) != str(proposal.pod_id):
            return False, "pod_changed"
        if zone_id != str(proposal.committed_next_zone_id):
            return False, "zone_changed"
        return True, ""

    def link_decision_event(self, robot: Any, decision_event_id: str | None) -> None:
        if not decision_event_id:
            return
        reservation = self.get_for_robot(robot)
        if reservation is not None and reservation.status == STATUS_COMMITTED:
            reservation.linked_rts_decision_event_id = str(decision_event_id)

    def cancel_for_robot(self, robot: Any, reason: str) -> CommittedNextReservation | None:
        return self.cancel_reservation(self.get_for_robot(robot), reason)

    def cancel_reservation(
        self,
        reservation: CommittedNextReservation | None,
        reason: str,
    ) -> CommittedNextReservation | None:
        if reservation is None:
            return None
        if reservation.status in TERMINAL_STATUSES:
            return reservation
        self._clear_markers(reservation)
        reservation.status = STATUS_CANCELLED
        reservation.cancellation_reason = str(reason)
        self._drop_indexes(reservation)
        return reservation

    def activate_for_robot(self, inventory: Any, robot: Any) -> CommittedNextReservation | None:
        reservation = self.get_for_robot(robot)
        if reservation is None:
            return None
        if reservation.status != STATUS_COMMITTED:
            self.cancel_reservation(reservation, "reservation_not_committed")
            return None
        valid, reason = self._validate_activation(inventory, reservation)
        if not valid:
            self.cancel_reservation(reservation, reason)
            return None

        job = reservation.job
        queue_index = _identity_index(inventory.job_queue, job)
        del inventory.job_queue[queue_index]
        previous_job = getattr(robot, "job", None)
        previous_state = getattr(robot, "current_state", None)
        self._clear_markers(reservation)
        try:
            reservation.status = STATUS_ACTIVATED
            reservation.activation_time_seconds = float(getattr(inventory, "_tick", 0.0))
            setattr(job, "committed_next_activated_by_robot_id", reservation.owner_robot_id)
            setattr(job, "committed_next_reservation_id", None)
            setattr(job, "committed_next_owner_robot_id", None)
            robot.assign_job_and_set_move_to_take_pod(job)
        except Exception:
            insert_at = max(0, min(int(reservation.original_queue_index), len(inventory.job_queue)))
            if job not in inventory.job_queue:
                inventory.job_queue.insert(insert_at, job)
            robot.job = previous_job
            robot.current_state = previous_state
            reservation.status = STATUS_CANCELLED
            reservation.cancellation_reason = "activation_failed_after_queue_removal"
            self._drop_indexes(reservation)
            return None
        self._drop_indexes(reservation)
        return reservation

    def complete_for_robot(self, robot: Any) -> CommittedNextReservation | None:
        reservation = self.get_by_robot_id(robot_id(robot))
        if reservation is None:
            return None
        reservation.status = STATUS_COMPLETED
        self._drop_indexes(reservation)
        return reservation

    def cancel_all(self, reason: str) -> list[CommittedNextReservation]:
        cancelled = []
        for reservation in list(self.reservations_by_id.values()):
            if reservation.is_active():
                cancelled.append(self.cancel_reservation(reservation, reason))
        return [item for item in cancelled if item is not None]

    def _eligible_candidates(self, inventory: Any, robot: Any) -> list[tuple[float, int, str, str, Any, str, str]]:
        candidates = []
        registry = None
        try:
            runtime = getattr(inventory, "rts_rollout_runtime", None)
            config = getattr(runtime, "config", None)
            registry = build_zone_registry(inventory, zone_ids=getattr(config, "zone_ids", ()) or ())
        except Exception:
            registry = None
        for queue_index, job in enumerate(getattr(inventory, "job_queue", []) or []):
            eligible, station, pod, zone_id = self._candidate_context(inventory, job, registry)
            if not eligible:
                continue
            if self.job_id_to_reservation.get(job_id(job)) is not None:
                continue
            if self.pod_id_to_reservation.get(pod_id(pod)) is not None:
                continue
            if _active_robot_has_job_or_pod(inventory, job, pod):
                continue
            cost = _distance_between(inventory=inventory, src=robot, dst=job.pod_coordinate, topology=EMPTY_ROBOT)
            candidates.append((cost, queue_index, job_id(job), pod_id(pod), job, station.station_id, zone_id))
        return candidates

    def _candidate_context(self, inventory: Any, job: Any, registry: Any):
        if job is None or getattr(job, "is_finished", False):
            return False, None, None, ""
        if getattr(job, "rts_continuation_active", False):
            return False, None, None, ""
        if getattr(job, "committed_next_reservation_id", None):
            return False, None, None, ""
        if not list(getattr(job, "orders", []) or []):
            return False, None, None, ""
        pod = getattr(job, "pod", None)
        if pod is None:
            return False, None, None, ""
        if getattr(pod, "rts_return_in_progress", False):
            return False, None, None, ""
        if getattr(pod, "committed_next_owner_robot_id", None):
            return False, None, None, ""
        if getattr(pod, "is_awaiting_replenishment", False):
            return False, None, None, ""
        if getattr(pod, "must_replenish_before_pick", False):
            return False, None, None, ""
        try:
            station = inventory.station_manager.get_station_by_id(job.station_id)
        except Exception:
            return False, None, None, ""
        if station is None or not station.is_picker_station():
            return False, None, None, ""
        storage = inventory.storage_manager.getStorageByPod(pod)
        if storage is None:
            return False, None, None, ""
        if registry is None:
            return False, None, None, ""
        try:
            zone_id = registry.zone_id_for_storage(storage)
        except Exception:
            zone_id = ""
        if not zone_id:
            return False, None, None, ""
        return True, station, pod, str(zone_id)

    def _validate_activation(self, inventory: Any, reservation: CommittedNextReservation) -> tuple[bool, str]:
        job = reservation.job
        if _identity_index(getattr(inventory, "job_queue", []), job) is None:
            return False, "reserved_job_missing_from_queue"
        if getattr(job, "committed_next_reservation_id", None) != reservation.reservation_id:
            return False, "job_marker_mismatch"
        if getattr(job, "committed_next_owner_robot_id", None) != reservation.owner_robot_id:
            return False, "job_owner_mismatch"
        pod = getattr(job, "pod", None)
        if pod is None or pod_id(pod) != reservation.pod_id:
            return False, "pod_mismatch"
        if getattr(pod, "committed_next_owner_robot_id", None) != reservation.owner_robot_id:
            return False, "pod_owner_mismatch"
        try:
            station = inventory.station_manager.get_station_by_id(job.station_id)
        except Exception:
            return False, "picking_station_missing"
        if station is None or station.station_id != reservation.picking_station_id or not station.is_picker_station():
            return False, "picking_station_mismatch"
        return True, ""

    def _set_markers(self, reservation: CommittedNextReservation) -> None:
        setattr(reservation.job, "committed_next_reservation_id", reservation.reservation_id)
        setattr(reservation.job, "committed_next_owner_robot_id", reservation.owner_robot_id)
        setattr(reservation.job.pod, "committed_next_owner_robot_id", reservation.owner_robot_id)
        setattr(reservation.job.pod, "committed_next_reservation_id", reservation.reservation_id)

    def _clear_markers(self, reservation: CommittedNextReservation) -> None:
        if getattr(reservation.job, "committed_next_reservation_id", None) == reservation.reservation_id:
            setattr(reservation.job, "committed_next_reservation_id", None)
            setattr(reservation.job, "committed_next_owner_robot_id", None)
        pod = getattr(reservation.job, "pod", None)
        if pod is not None and getattr(pod, "committed_next_reservation_id", None) == reservation.reservation_id:
            setattr(pod, "committed_next_owner_robot_id", None)
            setattr(pod, "committed_next_reservation_id", None)

    def _drop_indexes(self, reservation: CommittedNextReservation) -> None:
        self.robot_id_to_reservation.pop(reservation.owner_robot_id, None)
        self.pod_id_to_reservation.pop(reservation.pod_id, None)
        self.job_id_to_reservation.pop(reservation.job_id, None)


def has_committed_next_retrieval(inventory: Any, robot: Any) -> bool:
    registry = getattr(inventory, "committed_next_registry", None)
    return bool(registry and registry.has_for_robot(robot))


def get_committed_next_reservation(inventory: Any, robot: Any) -> CommittedNextReservation | None:
    registry = getattr(inventory, "committed_next_registry", None)
    return registry.get_for_robot(robot) if registry is not None else None


def get_committed_next_job(inventory: Any, robot: Any) -> Any | None:
    reservation = get_committed_next_reservation(inventory, robot)
    return reservation.job if reservation is not None else None


def get_committed_next_pod(inventory: Any, robot: Any) -> Any | None:
    job = get_committed_next_job(inventory, robot)
    return getattr(job, "pod", None)


def get_committed_next_station(inventory: Any, robot: Any) -> Any | None:
    reservation = get_committed_next_reservation(inventory, robot)
    if reservation is None:
        return None
    try:
        return inventory.station_manager.get_station_by_id(reservation.picking_station_id)
    except Exception:
        return None


def get_committed_next_zone(inventory: Any, robot: Any) -> str | None:
    reservation = get_committed_next_reservation(inventory, robot)
    return reservation.committed_next_zone_id if reservation is not None else None


def get_committed_next_action_proposals(inventory: Any, robot: Any) -> dict[str, CommittedNextProposal]:
    registry = getattr(inventory, "committed_next_registry", None)
    return registry.get_action_proposals_for_robot(robot) if registry is not None else {}


def get_committed_next_action_proposal(
    inventory: Any,
    robot: Any,
    branch: str,
    zone_id: str,
) -> CommittedNextProposal | None:
    registry = getattr(inventory, "committed_next_registry", None)
    return registry.get_action_proposal(robot, branch, zone_id) if registry is not None else None


def proposal_key(branch: str, zone_id: str) -> str:
    return str(zone_id)


def robot_id(robot: Any) -> str:
    if getattr(robot, "_id", None) is not None:
        return str(getattr(robot, "_id"))
    return str(getattr(robot, "id", ""))


def job_id(job: Any) -> str:
    if getattr(job, "my_id", None) is not None:
        return str(getattr(job, "my_id"))
    return str(getattr(job, "job_id", id(job)))


def pod_id(pod: Any) -> str:
    return str(getattr(pod, "pod_id", ""))


def storage_id(storage: Any) -> str:
    if storage is None:
        return ""
    if getattr(storage, "storage_id", None) is not None:
        return str(getattr(storage, "storage_id"))
    return f"{getattr(storage, 'pos_x', '')}:{getattr(storage, 'pos_y', '')}"


def _storage_available(storage: Any) -> bool:
    return bool(getattr(storage, "is_empty", False)) and getattr(storage, "assigned_pod", None) is None


def _coord(storage: Any) -> tuple[float, float]:
    return (float(getattr(storage, "pos_x", 0.0)), float(getattr(storage, "pos_y", 0.0)))


def _distance_between(*, inventory: Any, src: Any, dst: Any, topology: str) -> float:
    result = graph_distance_or_fallback(inventory, src, dst, topology=topology)
    if result.distance is not None:
        return float(result.distance)
    src_coord = _coord(src)
    dst_coord = _coord(dst)
    return abs(src_coord[0] - dst_coord[0]) + abs(src_coord[1] - dst_coord[1])


class _Coord:
    def __init__(self, x: Any, y: Any):
        self.x = x
        self.y = y


def _finite(value: Any) -> float:
    try:
        result = float(value)
    except Exception:
        return 0.0
    if result != result or result in (float("inf"), float("-inf")):
        return 0.0
    return result


def _identity_index(items: list[Any], target: Any) -> int | None:
    for index, item in enumerate(items):
        if item is target:
            return index
    return None


def _active_robot_has_job_or_pod(inventory: Any, job: Any, pod: Any) -> bool:
    for obj in getattr(inventory, "_objects", []) or []:
        if getattr(obj, "object_type", "") != "robot":
            continue
        active_job = getattr(obj, "job", None)
        if active_job is None or getattr(obj, "current_state", "") == "idle":
            continue
        if active_job is job or getattr(active_job, "pod", None) is pod:
            return True
    return False
