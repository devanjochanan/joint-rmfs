"""Event-driven VRSLA teacher policy for RTS behavior cloning."""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any, Mapping, Sequence

from engine.netlogo_coordinate import NetLogoCoordinate
from src.rmfs.decisions.rts.types import RTSDestinationContext, RTSDecision

from .action_context import (
    action_context_with_storage,
    selected_context_by_index,
    storage_feasibility,
)
from .action_space import REPLENISH_STORE, STORE, build_action_mask_from_contexts, encode_action, validate_action_mask
from .features import build_feature_bundle, feature_schema_identity, feature_schema_metadata
from .state import build_state
from .static_runtime_index import get_or_build_static_runtime_index, stable_storage_id
from .static_state_context import get_or_build_static_state_context


@dataclass(frozen=True)
class VRSLAMatchingAssignment:
    pod_id: str
    storage: Any
    storage_id: str
    zone_id: str
    picker_id: str
    pod_velocity_count: int
    pod_velocity_rank: float
    slot_cycle_distance: float | None
    slot_hotness_rank: float

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "pod_id": self.pod_id,
            "storage_id": self.storage_id,
            "zone_id": self.zone_id,
            "picker_id": self.picker_id,
            "pod_velocity_count": int(self.pod_velocity_count),
            "pod_velocity_rank": float(self.pod_velocity_rank),
            "slot_cycle_distance": self.slot_cycle_distance,
            "slot_hotness_rank": float(self.slot_hotness_rank),
        }


@dataclass(frozen=True)
class _VisibleReturnPod:
    context: Any
    robot: Any
    pod: Any
    station: Any
    pod_id: str


class RTSVRSLATeacherPolicy:
    """Selects RTS actions from static VRSLA scores for supervised collection."""

    def __init__(self, *, zone_ids: Sequence[str] = (), random_seed: int | None = None):
        self.zone_ids = tuple(str(zone_id) for zone_id in zone_ids)
        self.rng = random.Random(random_seed)

    def select_destination(self, context: Any) -> RTSDecision:
        zones = self.zone_ids
        if not zones:
            zones = tuple(getattr(get_or_build_static_runtime_index(context.warehouse), "zone_ids", ()))
        if not zones:
            raise RuntimeError("vrsla_teacher requires zone_ids or inferable runtime zones")

        state = build_state(context, zones)
        action_mask = list(build_action_mask_from_contexts(zones, state.action_contexts))
        validate_action_mask(zones, action_mask, require_valid=True)

        static_index = get_or_build_static_runtime_index(context.warehouse)
        static_context = get_or_build_static_state_context(context.warehouse)
        assignment = select_vrsla_assignment_for_current_pod(
            context=context,
            zone_ids=zones,
            static_index=static_index,
            static_context=static_context,
        )
        selected_zone = assignment.zone_id
        store_index = encode_action(STORE, selected_zone, zones)
        replenish_index = encode_action(REPLENISH_STORE, selected_zone, zones)
        store_mask = int(action_mask[store_index])
        replenish_mask = int(action_mask[replenish_index])

        proposal = _proposal_for_exact_storage(context, selected_zone, assignment.storage, state.action_contexts)
        exact_contexts = {}
        if store_mask:
            try:
                exact_contexts[STORE] = action_context_with_storage(
                    context,
                    selected_context_by_index(state.action_contexts, store_index),
                    assignment.storage,
                    static_index=static_index,
                    proposal=proposal,
                )
            except Exception:
                pass
        if replenish_mask:
            try:
                exact_contexts[REPLENISH_STORE] = action_context_with_storage(
                    context,
                    selected_context_by_index(state.action_contexts, replenish_index),
                    assignment.storage,
                    static_index=static_index,
                    proposal=proposal,
                )
            except Exception:
                pass

        store_valid = STORE in exact_contexts and bool(exact_contexts[STORE].action_valid)
        replenish_valid = REPLENISH_STORE in exact_contexts and bool(exact_contexts[REPLENISH_STORE].action_valid)
        branch_choice = choose_vrsla_teacher_branch(
            rng=self.rng,
            store_valid=store_valid,
            replenish_valid=replenish_valid,
            store_index=store_index,
            replenish_index=replenish_index,
            store_mask=store_mask,
            replenish_mask=replenish_mask,
        )
        if branch_choice is None:
            raise RuntimeError(
                f"vrsla_teacher selected zone {selected_zone!r}, but neither branch is valid "
                f"for exact storage {assignment.storage_id!r}"
            )
        branch = branch_choice["branch"]
        draw = branch_choice["rng_draw"]
        branch_probability = branch_choice["branch_probability"]

        selected_index = int(branch_choice["action_index"])
        action_context = exact_contexts[branch]
        storage = action_context.candidate_storage
        if storage is None:
            raise RuntimeError("vrsla_teacher selected an exact action without storage")

        exact_state_json = _state_json_with_exact_context(state.state_json, action_context)
        exact_mask = list(action_mask)
        features = build_feature_bundle(zones, exact_mask, exact_state_json)
        schema_id = _feature_schema_id(features.action_feature_names, features.stock_feature_names)
        cycle_estimate = (
            action_context.cycle_estimate.to_json_dict()
            if getattr(action_context, "cycle_estimate", None) is not None
            else None
        )
        row = _teacher_row(
            config=getattr(getattr(context.warehouse, "rts_rollout_runtime", None), "config", None),
            features=features,
            teacher_action_index=selected_index,
            teacher_branch=branch,
            teacher_zone_id=selected_zone,
            teacher_storage_id=action_context.candidate_storage_id,
            assignment=assignment,
            feature_schema_id=schema_id,
            branch_probability=branch_probability,
            rng_draw=draw,
            store_mask=store_mask,
            replenish_mask=replenish_mask,
            score_identities={
                "pod_velocity_score_cache_identity": static_context.historical_metadata.get("vrsla_score_cache_identity"),
                "historical_source_hash": static_context.historical_metadata.get("historical_source_hash"),
                "pod_sku_allocation_identity": static_context.historical_metadata.get("pod_sku_allocation_identity"),
                "slot_score_cache_identity": static_index.vrsla_slot_score_identity,
            },
        )
        proposal_obj = action_context.next_job_proposal
        proposal_has_next_job = bool(getattr(proposal_obj, "has_next_job", False))
        return RTSDecision(
            storage=storage,
            destination=NetLogoCoordinate(storage.pos_x, storage.pos_y),
            policy_name="vrsla_teacher",
            mode="rl",
            action_index=selected_index,
            branch=branch,
            zone_id=selected_zone,
            storage_id=action_context.candidate_storage_id,
            replenishment_station=action_context.replenishment_station,
            replenishment_station_id=action_context.replenishment_station_id,
            action_context_id=action_context.context_id,
            action_context_version=action_context.context_version,
            next_job_proposal_id=getattr(proposal_obj, "proposal_id", None),
            reason="event-driven VRSLA teacher",
            metadata={
                "actor_kind": "vrsla_teacher",
                "policy_mode": "vrsla_teacher",
                "selected_action_index": selected_index,
                "selected_action_branch": branch,
                "selected_zone_id": selected_zone,
                "action_mask": [int(value) for value in exact_mask],
                "raw_action_mask": [int(value) for value in action_mask],
                "action_context_id": action_context.context_id,
                "action_context_version": action_context.context_version,
                "candidate_storage_id": action_context.candidate_storage_id,
                "selected_replenishment_station_id": action_context.replenishment_station_id,
                "validity_reason_codes": list(action_context.invalid_reason_codes),
                "next_job_proposal_id": getattr(proposal_obj, "proposal_id", None),
                "selected_proposal_id": getattr(proposal_obj, "proposal_id", None),
                "selected_proposal_has_next_job": proposal_has_next_job,
                "selected_proposal_job_id": getattr(proposal_obj, "job_id", None) if proposal_has_next_job else None,
                "selected_proposal_pod_id": getattr(proposal_obj, "pod_id", None) if proposal_has_next_job else None,
                "selected_proposal_picker_id": getattr(proposal_obj, "picking_station_id", None) if proposal_has_next_job else None,
                "selected_proposal_candidate_count": int(getattr(proposal_obj, "candidate_count", 0) or 0),
                "selected_cycle_estimate": cycle_estimate,
                "state_json": exact_state_json,
                "state_capture_timing": "vrsla_teacher_decision_state",
                "feature_schema_id": schema_id,
                "teacher_supervised_row": row,
                "teacher_policy": "vrsla_event_driven",
                "teacher_branch_probability": branch_probability,
                "teacher_rng_draw": draw,
                "teacher_selected_branch": branch,
                "teacher_store_mask_value": store_mask,
                "teacher_replenish_store_mask_value": replenish_mask,
                "teacher_selected_storage_id": action_context.candidate_storage_id,
                "teacher_selected_zone_id": selected_zone,
                "vrsla_assignment": assignment.to_json_dict(),
            },
        )


def select_vrsla_assignment_for_current_pod(
    *,
    context: Any,
    zone_ids: Sequence[str],
    static_index: Any | None = None,
    static_context: Any | None = None,
) -> VRSLAMatchingAssignment:
    warehouse = getattr(context, "warehouse", None)
    static_index = static_index or get_or_build_static_runtime_index(warehouse)
    static_context = static_context or get_or_build_static_state_context(warehouse)
    zones = tuple(str(zone_id) for zone_id in zone_ids)
    zone_set = set(zones)
    visible = _visible_return_ready_pods(context)
    current_pod_id = str(getattr(getattr(context, "pod", None), "pod_id", ""))
    assignments = greedy_vrsla_assignments(
        visible,
        zone_ids=zones,
        static_index=static_index,
        static_context=static_context,
    )
    assignment = assignments.get(current_pod_id)
    if assignment is not None and assignment.zone_id in zone_set:
        return assignment
    raise RuntimeError(f"vrsla_teacher found no feasible VRSLA slot for current pod {current_pod_id!r}")


def choose_vrsla_teacher_branch(
    *,
    rng: random.Random,
    store_valid: bool,
    replenish_valid: bool,
    store_index: int,
    replenish_index: int,
    store_mask: int,
    replenish_mask: int,
) -> dict[str, Any] | None:
    """Choose the post-pick branch for a VRSLA-selected zone without rerolling zones."""
    if store_valid and replenish_valid:
        draw = float(rng.random())
        branch = STORE if draw < 0.5 else REPLENISH_STORE
        return {
            "branch": branch,
            "action_index": int(store_index if branch == STORE else replenish_index),
            "branch_probability": 0.5,
            "rng_draw": draw,
            "store_mask": int(store_mask),
            "replenish_mask": int(replenish_mask),
        }
    if store_valid:
        return {
            "branch": STORE,
            "action_index": int(store_index),
            "branch_probability": 1.0,
            "rng_draw": None,
            "store_mask": int(store_mask),
            "replenish_mask": int(replenish_mask),
        }
    if replenish_valid:
        return {
            "branch": REPLENISH_STORE,
            "action_index": int(replenish_index),
            "branch_probability": 1.0,
            "rng_draw": None,
            "store_mask": int(store_mask),
            "replenish_mask": int(replenish_mask),
        }
    return None


def greedy_vrsla_assignments(
    visible_pods: Sequence[Any],
    *,
    zone_ids: Sequence[str],
    static_index: Any,
    static_context: Any,
) -> dict[str, VRSLAMatchingAssignment]:
    zone_set = set(str(zone_id) for zone_id in zone_ids)
    allowed_storage_ids = {
        stable_storage_id(storage)
        for zone_id in zone_set
        for storage in getattr(static_index, "storages_by_zone", {}).get(str(zone_id), ())
    }
    remaining = set(allowed_storage_ids)
    ordered = sorted(
        visible_pods,
        key=lambda item: (
            -int(static_context.vrsla_pod_velocity_count(getattr(item, "pod", None))),
            _stable_id_key(str(getattr(getattr(item, "pod", None), "pod_id", ""))),
        ),
    )
    assignments: dict[str, VRSLAMatchingAssignment] = {}
    for item in ordered:
        pod = getattr(item, "pod", None)
        station = getattr(item, "station", None)
        pod_id = str(getattr(pod, "pod_id", ""))
        for storage_id in static_index.vrsla_sorted_storage_ids(station):
            if storage_id not in remaining:
                continue
            storage = getattr(static_index, "storage_by_id", {}).get(storage_id)
            zone_id = str(getattr(static_index, "storage_id_to_zone_id", {}).get(storage_id, ""))
            if storage is None or zone_id not in zone_set:
                continue
            feasibility = storage_feasibility(item.context, storage, zone_id, static_index=static_index)
            if not feasibility.available or not feasibility.reachable:
                continue
            remaining.remove(storage_id)
            assignments[pod_id] = VRSLAMatchingAssignment(
                pod_id=pod_id,
                storage=storage,
                storage_id=storage_id,
                zone_id=zone_id,
                picker_id=str(getattr(station, "station_id", "")),
                pod_velocity_count=static_context.vrsla_pod_velocity_count(pod),
                pod_velocity_rank=static_context.vrsla_pod_velocity_rank(pod),
                slot_cycle_distance=static_index.vrsla_cycle_distance(station, storage),
                slot_hotness_rank=static_index.vrsla_hotness_rank(station, storage),
            )
            break
    return assignments


def _visible_return_ready_pods(context: Any) -> tuple[_VisibleReturnPod, ...]:
    warehouse = getattr(context, "warehouse", None)
    current = _VisibleReturnPod(
        context=context,
        robot=getattr(context, "robot", None),
        pod=getattr(context, "pod", None),
        station=getattr(context, "station", None),
        pod_id=str(getattr(getattr(context, "pod", None), "pod_id", "")),
    )
    items: list[_VisibleReturnPod] = [current]
    for robot in list(getattr(warehouse, "get_movable_objects", lambda: [])() or []):
        if robot is current.robot or str(getattr(robot, "object_type", "")).lower() != "robot":
            continue
        job = getattr(robot, "job", None)
        pod = getattr(job, "pod", None)
        if job is None or pod is None or not bool(getattr(job, "is_finished", False)):
            continue
        if str(getattr(robot, "current_state", "")) != "station_processing":
            continue
        station = _station_for_job(warehouse, job)
        if station is None or str(getattr(station, "station_type", "")).lower() not in {"picker", "picking"}:
            continue
        pod_id = str(getattr(pod, "pod_id", ""))
        if pod_id == current.pod_id:
            continue
        items.append(
            _VisibleReturnPod(
                context=RTSDestinationContext(warehouse=warehouse, robot=robot, pod=pod, station=station),
                robot=robot,
                pod=pod,
                station=station,
                pod_id=pod_id,
            )
        )
    return tuple(items)


def _station_for_job(warehouse: Any, job: Any) -> Any | None:
    station_manager = getattr(warehouse, "station_manager", None)
    getter = getattr(station_manager, "get_station_by_id", None)
    if getter is None:
        return None
    try:
        return getter(getattr(job, "station_id", None))
    except Exception:
        return None


def _proposal_for_exact_storage(context: Any, zone_id: str, storage: Any, action_contexts: Sequence[Any]) -> Any | None:
    warehouse = getattr(context, "warehouse", None)
    robot = getattr(context, "robot", None)
    registry = getattr(warehouse, "committed_next_registry", None)
    if registry is None or robot is None:
        return None
    original = None
    for branch in (STORE, REPLENISH_STORE):
        try:
            ctx = selected_context_by_index(action_contexts, encode_action(branch, zone_id, getattr(getattr(warehouse, "rts_rollout_runtime", None).config, "zone_ids", ())))
        except Exception:
            continue
        proposal = getattr(ctx, "next_job_proposal", None)
        if proposal is not None:
            original = proposal
            break
    if original is not None:
        try:
            return registry.rebuild_proposal_for_storage(warehouse, robot, zone_id, storage, original)
        except Exception:
            return original
    try:
        return registry.refresh_action_proposal_for_zone(warehouse, robot, context, zone_id, storage)
    except Exception:
        return None


def _state_json_with_exact_context(state_json: Mapping[str, Any], action_context: Any) -> dict[str, Any]:
    payload = dict(state_json)
    rows = []
    for row in list(state_json.get("rts_action_contexts", []) or []):
        if int(row.get("action_index", -1)) == int(getattr(action_context, "action_index", -2)):
            rows.append(action_context.to_json_dict())
        else:
            rows.append(dict(row))
    payload["rts_action_contexts"] = rows
    return payload


def _teacher_row(
    *,
    config: Any,
    features: Any,
    teacher_action_index: int,
    teacher_branch: str,
    teacher_zone_id: str,
    teacher_storage_id: str | None,
    assignment: VRSLAMatchingAssignment,
    feature_schema_id: str,
    branch_probability: float,
    rng_draw: float | None,
    store_mask: int,
    replenish_mask: int,
    score_identities: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    identities = dict(score_identities or {})
    return {
        "teacher_policy": "vrsla_event_driven",
        "X_actions": features.X_actions.tolist(),
        "M_actions": features.M_actions.tolist(),
        "X_stock": features.X_stock.tolist(),
        "M_stock": features.M_stock.tolist(),
        "action_feature_names": list(features.action_feature_names),
        "stock_feature_names": list(features.stock_feature_names),
        "zone_ids": list(features.zone_ids),
        "teacher_action_index": int(teacher_action_index),
        "teacher_branch": str(teacher_branch),
        "teacher_zone_id": str(teacher_zone_id),
        "teacher_storage_id": teacher_storage_id,
        "pod_velocity_count": int(assignment.pod_velocity_count),
        "pod_velocity_rank": float(assignment.pod_velocity_rank),
        "slot_cycle_distance": assignment.slot_cycle_distance,
        "slot_hotness_rank": float(assignment.slot_hotness_rank),
        "run_id": getattr(config, "run_id", None),
        "seed": getattr(config, "random_seed", None),
        "worker": getattr(config, "worker_id", None),
        "batch_id": getattr(config, "batch_id", None),
        "artifact_label": getattr(config, "artifact_label", None),
        "feature_schema_id": feature_schema_id,
        "feature_schema_version": feature_schema_metadata()["action_feature_schema_version"],
        "teacher_branch_probability": float(branch_probability),
        "teacher_rng_draw": rng_draw,
        "teacher_store_mask_value": int(store_mask),
        "teacher_replenish_store_mask_value": int(replenish_mask),
        "selected_slot": assignment.to_json_dict(),
        **identities,
    }


def _feature_schema_id(action_names: Sequence[str], stock_names: Sequence[str]) -> str:
    schema = {
        **feature_schema_metadata(),
        "action_feature_names": list(action_names),
        "stock_feature_names": list(stock_names),
        "action_feature_dim": len(tuple(action_names)),
        "stock_feature_dim": len(tuple(stock_names)),
    }
    return feature_schema_identity(schema)


def _stable_id_key(value: str) -> tuple[int, str]:
    try:
        return int(value), value
    except Exception:
        return 10**12, value
