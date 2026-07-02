"""RTS rollout decision/outcome linking runtime."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .action_space import STORE, build_action_mask_from_contexts, encode_action
from .cycle_reference import read_cycle_reference
from .evaluation_policy import infer_zone_ids_from_context
from .evaluation_summary import RolloutSummaryAccumulator, write_rollout_summary
from .features import build_feature_bundle
from .reward import RTSRewardReference, build_reward_components_from_paper_cycle, compute_reward
from .rollout_schema import build_decision_event, build_outcome_event, make_decision_event_id
from .rollout_writer import RTSRolloutWriter
from .runtime_config import RTSRuntimeConfig
from .state import build_state
from .zone_features import infer_zone_id
from .training.timebase import warehouse_time_to_netlogo_steps
from .training.reward_normalizer import pending_cold_start_reward_json

PAPER_CYCLE_STATUS_PENDING = "pending"
PAPER_CYCLE_STATUS_COMPLETE = "complete"
PAPER_CYCLE_STATUS_CENSORED_NEXT_TASK_REPLENISHMENT = "censored_next_task_replenishment"
PAPER_CYCLE_STATUS_CENSORED_NEXT_TASK_CHARGING = "censored_next_task_charging"
PAPER_CYCLE_STATUS_CENSORED_NO_NEXT_TASK = "censored_no_next_task"
PAPER_CYCLE_STATUS_CENSORED_COMMITTED_NEXT_CANCELLED = "censored_committed_next_cancelled"
PAPER_CYCLE_STATUS_CENSORED_RUN_END = "censored_run_end"
PAPER_CYCLE_STATUS_CENSORED_MAXIMUM_HORIZON = "censored_maximum_horizon"
PAPER_CYCLE_STATUS_CENSORED_CONGESTION = "censored_congestion"
PAPER_CYCLE_STATUS_CENSORED_NO_ACTIVE_WORK = "censored_no_active_work"
PAPER_CYCLE_STATUS_CENSORED_MANUAL_CANCELLATION = "censored_manual_cancellation"
PAPER_CYCLE_STATUS_CENSORED_WORKER_EXCEPTION = "censored_worker_exception"
PAPER_CYCLE_COMPLETION_RULE_NEXT_ORDER_RETRIEVAL_ARRIVAL = "next_order_retrieval_arrival"


@dataclass
class PendingRTSDecision:
    decision_event_id: str
    robot_id: str
    job_id: str
    pod_id: str
    return_start_tick: float
    selected_action_branch: str
    metadata: dict[str, Any]
    tick_to_second: float | None = None
    return_finish_tick: float | None = None
    return_duration: float | None = None
    paper_cycle_status: str = PAPER_CYCLE_STATUS_PENDING
    committed_next_reservation_id: str | None = None
    committed_next_job_id: str | None = None
    committed_next_pod_id: str | None = None
    committed_next_station_id: str | None = None
    committed_next_zone_id: str | None = None
    committed_next_created_time_seconds: float | None = None
    committed_next_activation_time_seconds: float | None = None
    cycle_estimate_known: bool = False
    estimated_cycle_time_at_decision: float | None = None
    estimated_queue_time_at_decision: float | None = None
    estimated_replenishment_service_time_at_decision: float | None = None
    cycle_estimate_status: str | None = None


class RTSOutcomeTracker:
    def __init__(self):
        self.pending: dict[tuple[str, str, str], PendingRTSDecision] = {}
        self.pending_by_robot_id: dict[str, PendingRTSDecision] = {}
        self.completed_count = 0
        self.censored_count = 0

    def record_decision(self, pending: PendingRTSDecision) -> None:
        self.pending[(pending.robot_id, pending.job_id, pending.pod_id)] = pending
        self.pending_by_robot_id[pending.robot_id] = pending

    def complete_return(self, *, robot_id: str, job_id: str, pod_id: str) -> PendingRTSDecision | None:
        pending = self.pending.pop((robot_id, job_id, pod_id), None)
        if pending is not None and self.pending_by_robot_id.get(robot_id) is pending:
            self.pending_by_robot_id.pop(robot_id, None)
        return pending

    def mark_return_completed(
        self,
        *,
        robot_id: str,
        job_id: str,
        pod_id: str,
        return_finish_tick: float,
    ) -> PendingRTSDecision | None:
        pending = self.pending.pop((robot_id, job_id, pod_id), None)
        if pending is None:
            return None
        if float(return_finish_tick) < pending.return_start_tick:
            raise ValueError(
                "RTS return chronology is invalid: "
                f"return_finish_tick={return_finish_tick} < decision_tick={pending.return_start_tick}"
            )
        pending.return_finish_tick = float(return_finish_tick)
        pending.return_duration = float(return_finish_tick) - pending.return_start_tick
        self.pending_by_robot_id[robot_id] = pending
        return pending

    def complete_paper_cycle_for_robot(self, *, robot_id: str) -> PendingRTSDecision | None:
        pending = self.pending_by_robot_id.pop(robot_id, None)
        if pending is not None:
            pending.paper_cycle_status = PAPER_CYCLE_STATUS_COMPLETE
            self.completed_count += 1
        return pending

    def censor_paper_cycle_for_robot(self, *, robot_id: str, status: str) -> PendingRTSDecision | None:
        pending = self.pending_by_robot_id.pop(robot_id, None)
        if pending is not None:
            pending.paper_cycle_status = status
            self.censored_count += 1
        return pending

    def orphan_pending(self) -> list[PendingRTSDecision]:
        seen: set[int] = set()
        pending: list[PendingRTSDecision] = []
        for item in list(self.pending.values()) + list(self.pending_by_robot_id.values()):
            identity = id(item)
            if identity not in seen:
                pending.append(item)
                seen.add(identity)
        return pending


class NoopRTSRolloutRuntime:
    def on_decision(self, *args, **kwargs) -> None:
        return None

    def on_return_completed(self, *args, **kwargs) -> None:
        return None

    def on_station_arrival(self, *args, **kwargs) -> None:
        return None

    def on_committed_next_activated(self, *args, **kwargs) -> None:
        return None

    def censor_pending_for_robot(self, *args, **kwargs) -> None:
        return None

    def censor_all_pending(self, *args, **kwargs) -> None:
        return None

    def close(self, *args, **kwargs) -> None:
        return None


class RTSRolloutRuntime:
    def __init__(self, *, config: RTSRuntimeConfig, runtime_root: Path):
        self.config = config
        self.runtime_root = Path(runtime_root)
        self.writer = RTSRolloutWriter(
            self.runtime_root / config.rollout_filename,
            enabled=config.rollout_enabled,
            max_events=config.max_events,
        )
        self.summary_path = self.runtime_root / config.summary_filename
        self.tracker = RTSOutcomeTracker()
        self.summary = RolloutSummaryAccumulator(policy_mode=self.config.policy_mode)
        self._summary_dirty_events = 0
        self._summary_write_cadence = 100
        self.reward_reference = _load_reward_reference(config.reward_reference_path)
        if self.config.rollout_enabled:
            self._write_summary(force=True)

    def on_decision(self, *, robot: Any, context: Any, decision: Any) -> str | None:
        if not self.config.rollout_enabled:
            return None
        zones = self.config.zone_ids or infer_zone_ids_from_context(context)
        if not zones:
            raise RuntimeError("RTS rollout requires configured or inferable zone_ids when a decision occurs")
        metadata = dict(getattr(decision, "metadata", {}) or {})
        captured_state = metadata.get("state_json")
        if isinstance(captured_state, dict):
            state_json = captured_state
            captured_mask = metadata.get("action_mask")
            mask = list(captured_mask) if captured_mask is not None else [
                int(row.get("mask_value", 0))
                for row in sorted(
                    state_json.get("rts_action_contexts", []) or [],
                    key=lambda row: int(row.get("action_index", 0)),
                )
            ]
        else:
            state = build_state(context, zones)
            state_json = state.state_json
            mask = build_action_mask_from_contexts(zones, state.action_contexts)
        selected = _selected_action(decision, zones)
        selected_context_payload = _selected_action_context_payload(state_json, selected["index"])
        selected_cycle_estimate = dict(selected_context_payload.get("cycle_estimate") or {})
        features = build_feature_bundle(zones, mask, state_json)
        warehouse = getattr(context, "warehouse", None)
        tick = getattr(warehouse, "_tick", None)
        tick_to_second = getattr(warehouse, "tick_to_second", None)
        warehouse_time = _float_or_none(tick)
        netlogo_step = _netlogo_step_or_none(warehouse_time, tick_to_second)
        robot_id = _robot_id(robot)
        job = getattr(robot, "job", None)
        job_id = _text(getattr(job, "my_id", ""))
        pod_id = _text(getattr(getattr(job, "pod", None), "pod_id", getattr(context.pod, "pod_id", "")))
        decision_event_id = make_decision_event_id(robot_id=robot_id, job_id=job_id, pod_id=pod_id, tick=tick)
        committed_next = _committed_next_payload(context, robot)
        row = build_decision_event(
            decision_event_id=decision_event_id,
            tick=tick,
            robot_id=robot_id,
            job_id=job_id,
            pod_id=pod_id,
            source_station_id=getattr(context.station, "station_id", ""),
            source_station_type=getattr(context.station, "station_type", ""),
            policy_name=getattr(decision, "policy_name", self.config.policy_mode),
            zone_ids=zones,
            action_mask=mask,
            selected_action_index=selected["index"],
            selected_action_branch=selected["branch"],
            selected_zone_id=selected["zone_id"],
            selected_storage=getattr(decision, "storage", None),
            state_json=state_json,
            feature_shapes={
                "X_actions": list(features.X_actions.shape),
                "M_actions": list(features.M_actions.shape),
                "X_stock": list(features.X_stock.shape),
                "M_stock": list(features.M_stock.shape),
            },
            actor_kind=metadata.get("actor_kind"),
            policy_checkpoint_id=metadata.get("policy_checkpoint_id"),
            policy_mode=metadata.get("policy_mode") or self.config.policy_mode,
            old_log_prob=metadata.get("old_log_prob"),
            old_value=metadata.get("old_value"),
            policy_entropy=metadata.get("policy_entropy"),
            feature_schema_id=metadata.get("feature_schema_id"),
            netlogo_step=netlogo_step,
            warehouse_time=warehouse_time,
            tick_to_second=tick_to_second,
        )
        row.update(committed_next)
        row.update(
            {
                "action_context_id": getattr(decision, "action_context_id", None) or metadata.get("action_context_id"),
                "action_context_version": getattr(decision, "action_context_version", None)
                or metadata.get("action_context_version"),
                "candidate_storage_id": metadata.get("candidate_storage_id"),
                "selected_replenishment_station_id": getattr(decision, "replenishment_station_id", None)
                or metadata.get("selected_replenishment_station_id"),
                "validity_reason_codes": list(metadata.get("validity_reason_codes", []) or []),
                "next_job_proposal_id": getattr(decision, "next_job_proposal_id", None)
                or metadata.get("next_job_proposal_id"),
            }
        )
        row.update(_cycle_estimate_payload(selected_cycle_estimate))
        self.writer.write_decision(row)
        self.summary.add_event(row)
        self.tracker.record_decision(
            PendingRTSDecision(
                decision_event_id=decision_event_id,
                robot_id=robot_id,
                job_id=job_id,
                pod_id=pod_id,
                return_start_tick=float(tick or 0.0),
                selected_action_branch=selected["branch"] or STORE,
                metadata=metadata,
                tick_to_second=float(tick_to_second) if tick_to_second is not None else None,
                committed_next_reservation_id=committed_next["committed_next_reservation_id"],
                committed_next_job_id=committed_next["committed_next_job_id"],
                committed_next_pod_id=committed_next["committed_next_pod_id"],
                committed_next_station_id=committed_next["committed_next_station_id"],
                committed_next_zone_id=committed_next["committed_next_zone_id"],
                committed_next_created_time_seconds=committed_next["committed_next_created_time_seconds"],
                cycle_estimate_known=bool(selected_cycle_estimate.get("known", False)),
                estimated_cycle_time_at_decision=_float_or_none(
                    selected_cycle_estimate.get("estimated_cycle_seconds")
                ),
                estimated_queue_time_at_decision=_float_or_none(
                    selected_cycle_estimate.get("estimated_queue_seconds")
                ),
                estimated_replenishment_service_time_at_decision=_float_or_none(
                    selected_cycle_estimate.get("estimated_replenishment_service_seconds")
                ),
                cycle_estimate_status=_text(selected_cycle_estimate.get("status")) or None,
            )
        )
        self._write_summary()
        return decision_event_id

    def on_return_completed(self, *, robot: Any) -> None:
        if not self.config.rollout_enabled:
            return
        job = getattr(robot, "job", None)
        if job is None:
            return
        robot_id = _robot_id(robot)
        job_id = _text(getattr(job, "my_id", ""))
        pod_id = _text(getattr(getattr(job, "pod", None), "pod_id", ""))
        tick = float(getattr(getattr(robot, "universe", None), "_tick", getattr(getattr(robot, "warehouse", None), "_tick", 0.0)))
        pending = self.tracker.mark_return_completed(
            robot_id=robot_id,
            job_id=job_id,
            pod_id=pod_id,
            return_finish_tick=tick,
        )
        if pending is None:
            return
        return_finish_warehouse_time = tick
        return_start_warehouse_time = pending.return_start_tick
        realized_warehouse_time = return_finish_warehouse_time - return_start_warehouse_time
        
        tick_to_second = pending.tick_to_second
        if tick_to_second is not None:
            netlogo_steps_elapsed_since_decision = warehouse_time_to_netlogo_steps(realized_warehouse_time, tick_to_second)
        else:
            netlogo_steps_elapsed_since_decision = 0
            
        warehouse_time_elapsed_since_decision = realized_warehouse_time
        reward_json = pending_cold_start_reward_json(
            realized_cycle_time=realized_warehouse_time,
            paper_cycle_status=PAPER_CYCLE_STATUS_PENDING,
        )
        destination = getattr(robot, "destination", None)
        netlogo_step = _netlogo_step_or_none(tick, tick_to_second)
        row = build_outcome_event(
            decision_event_id=pending.decision_event_id,
            tick=tick,
            robot_id=robot_id,
            job_id=job_id,
            pod_id=pod_id,
            outcome_status="return_completed",
            return_start_tick=pending.return_start_tick,
            return_finish_tick=tick,
            realized_cycle_time=None,
            destination_x=getattr(destination, "x", None),
            destination_y=getattr(destination, "y", None),
            reward_json=reward_json,
            netlogo_step=netlogo_step,
            warehouse_time=tick,
            tick_to_second=tick_to_second,
            netlogo_steps_elapsed_since_decision=netlogo_steps_elapsed_since_decision,
            warehouse_time_elapsed_since_decision=warehouse_time_elapsed_since_decision,
            return_duration=realized_warehouse_time,
            paper_cycle_status=PAPER_CYCLE_STATUS_PENDING,
            paper_cycle_complete=0,
            paper_cycle_start_tick=pending.return_start_tick,
            paper_cycle_storage_arrival_tick=tick,
            paper_cycle_duration=None,
            paper_cycle_censor_reason="",
            paper_cycle_completion_rule="",
        )
        row.update(_pending_committed_next_payload(pending))
        row.update(_pending_cycle_estimate_payload(pending, realized_cycle_time=None))
        self.writer.write_outcome(row)
        self.summary.add_event(row)
        self._write_summary()
        if pending.committed_next_reservation_id is None:
            self._censor_paper_cycle(
                robot=robot,
                station=None,
                pending=pending,
                status=PAPER_CYCLE_STATUS_CENSORED_NO_NEXT_TASK,
                reason="no_committed_next_reservation",
            )

    def on_station_arrival(self, *, robot: Any, station: Any) -> None:
        if not self.config.rollout_enabled:
            return
        robot_id = _robot_id(robot)
        pending = self.tracker.pending_by_robot_id.get(robot_id)
        if pending is None:
            return
        if pending.return_finish_tick is None:
            return
        station_type = str(getattr(station, "station_type", "")).strip().lower()
        if station_type in {"picker", "picking"}:
            if self._matches_committed_next_arrival(robot=robot, station=station, pending=pending):
                self._complete_paper_cycle(robot=robot, station=station, pending=pending)
        elif station_type == "replenishment":
            self._censor_paper_cycle(
                robot=robot,
                station=station,
                pending=pending,
                status=PAPER_CYCLE_STATUS_CENSORED_NEXT_TASK_REPLENISHMENT,
                reason="next_task_replenishment",
            )

    def on_committed_next_activated(self, *, robot: Any, reservation: Any) -> None:
        if not self.config.rollout_enabled:
            return
        pending = self.tracker.pending_by_robot_id.get(_robot_id(robot))
        if pending is None:
            return
        if pending.committed_next_reservation_id != _text(getattr(reservation, "reservation_id", "")):
            return
        pending.committed_next_activation_time_seconds = _float_or_none(
            getattr(reservation, "activation_time_seconds", None)
        )

    def censor_pending_for_robot(self, *, robot: Any, status: str, reason: str) -> None:
        if not self.config.rollout_enabled:
            return
        pending = self.tracker.pending_by_robot_id.get(_robot_id(robot))
        if pending is None:
            return
        self._censor_paper_cycle(robot=robot, station=None, pending=pending, status=status, reason=reason)

    def censor_all_pending(self, *, status: str, reason: str, warehouse: Any | None = None) -> None:
        if not self.config.rollout_enabled:
            return
        for pending in self.tracker.orphan_pending():
            robot = type("_PendingRobot", (), {"warehouse": warehouse, "universe": warehouse, "job": None, "destination": None})()
            setattr(robot, "_id", pending.robot_id)
            self._censor_paper_cycle(robot=robot, station=None, pending=pending, status=status, reason=reason)

    def close(
        self,
        *,
        censor_status: str = PAPER_CYCLE_STATUS_CENSORED_RUN_END,
        reason: str = "runtime_close",
    ) -> None:
        self.censor_all_pending(status=censor_status, reason=reason)
        self._write_summary(force=True)
        self.writer.close()

    def _matches_committed_next_arrival(self, *, robot: Any, station: Any, pending: PendingRTSDecision) -> bool:
        job = getattr(robot, "job", None)
        if pending.committed_next_reservation_id is None:
            return False
        if _text(getattr(job, "committed_next_activated_by_robot_id", "")) != pending.robot_id:
            return False
        if _text(getattr(job, "committed_next_reservation_id", "")) not in {"", "None"}:
            return False
        if _text(getattr(job, "my_id", "")) != pending.committed_next_job_id:
            return False
        if _text(getattr(getattr(job, "pod", None), "pod_id", "")) != pending.committed_next_pod_id:
            return False
        return _text(getattr(station, "station_id", "")) == pending.committed_next_station_id

    def _complete_paper_cycle(self, *, robot: Any, station: Any, pending: PendingRTSDecision) -> None:
        tick = float(getattr(getattr(robot, "universe", None), "_tick", getattr(getattr(robot, "warehouse", None), "_tick", 0.0)))
        completed = self.tracker.complete_paper_cycle_for_robot(robot_id=pending.robot_id)
        if completed is None:
            return
        duration = tick - completed.return_start_tick
        if duration < 0:
            raise ValueError(
                "RTS paper-cycle chronology is invalid: "
                f"arrival_tick={tick} < decision_tick={completed.return_start_tick}"
            )
        if completed.return_finish_tick is not None and tick < completed.return_finish_tick:
            raise ValueError(
                "RTS paper-cycle chronology is invalid: "
                f"arrival_tick={tick} < storage_arrival_tick={completed.return_finish_tick}"
            )
        tick_to_second = completed.tick_to_second
        netlogo_step = _netlogo_step_or_none(tick, tick_to_second)
        reward_json = self._reward_json(completed.selected_action_branch, duration)
        job = getattr(robot, "job", None)
        row = build_outcome_event(
            decision_event_id=completed.decision_event_id,
            tick=tick,
            robot_id=pending.robot_id,
            job_id=_text(getattr(job, "my_id", "")),
            pod_id=_text(getattr(getattr(job, "pod", None), "pod_id", completed.pod_id)),
            outcome_status="paper_cycle_completed",
            return_start_tick=completed.return_start_tick,
            return_finish_tick=completed.return_finish_tick,
            realized_cycle_time=duration,
            destination_x=getattr(getattr(robot, "destination", None), "x", None),
            destination_y=getattr(getattr(robot, "destination", None), "y", None),
            reward_json=reward_json,
            netlogo_step=netlogo_step,
            warehouse_time=tick,
            tick_to_second=tick_to_second,
            netlogo_steps_elapsed_since_decision=_netlogo_step_or_none(duration, tick_to_second),
            warehouse_time_elapsed_since_decision=duration,
            return_duration=completed.return_duration,
            paper_cycle_status=PAPER_CYCLE_STATUS_COMPLETE,
            paper_cycle_complete=1,
            paper_cycle_start_tick=completed.return_start_tick,
            paper_cycle_storage_arrival_tick=completed.return_finish_tick,
            paper_cycle_next_station_arrival_tick=tick,
            paper_cycle_duration=duration,
            paper_cycle_censor_reason="",
            paper_cycle_completion_rule=PAPER_CYCLE_COMPLETION_RULE_NEXT_ORDER_RETRIEVAL_ARRIVAL,
        )
        row.update(_pending_committed_next_payload(completed))
        row.update(_pending_cycle_estimate_payload(completed, realized_cycle_time=duration))
        self.writer.write_outcome(row)
        self.summary.add_event(row)
        self._write_summary()

    def _censor_paper_cycle(
        self,
        *,
        robot: Any,
        station: Any | None,
        pending: PendingRTSDecision,
        status: str,
        reason: str,
    ) -> None:
        tick = float(getattr(getattr(robot, "universe", None), "_tick", getattr(getattr(robot, "warehouse", None), "_tick", 0.0)))
        censored = self.tracker.censor_paper_cycle_for_robot(robot_id=pending.robot_id, status=status)
        if censored is None:
            return
        row = build_outcome_event(
            decision_event_id=censored.decision_event_id,
            tick=tick,
            robot_id=pending.robot_id,
            job_id=_text(getattr(getattr(robot, "job", None), "my_id", "")),
            pod_id=censored.pod_id,
            outcome_status="paper_cycle_censored",
            return_start_tick=censored.return_start_tick,
            return_finish_tick=censored.return_finish_tick,
            realized_cycle_time=None,
            destination_x=getattr(getattr(robot, "destination", None), "x", None),
            destination_y=getattr(getattr(robot, "destination", None), "y", None),
            reward_json=pending_cold_start_reward_json(paper_cycle_status=status),
            netlogo_step=_netlogo_step_or_none(tick, censored.tick_to_second),
            warehouse_time=tick,
            tick_to_second=censored.tick_to_second,
            return_duration=censored.return_duration,
            paper_cycle_status=status,
            paper_cycle_complete=0,
            paper_cycle_start_tick=censored.return_start_tick,
            paper_cycle_storage_arrival_tick=censored.return_finish_tick,
            paper_cycle_next_station_arrival_tick=tick if station is not None else None,
            paper_cycle_duration=None,
            paper_cycle_censor_reason=reason,
            paper_cycle_completion_rule=status,
        )
        row.update(_pending_committed_next_payload(censored))
        row.update(_pending_cycle_estimate_payload(censored, realized_cycle_time=None))
        self.writer.write_outcome(row)
        self.summary.add_event(row)
        self._write_summary()

    def _reward_json(self, branch: str, paper_cycle_duration: float) -> dict[str, Any]:
        components = build_reward_components_from_paper_cycle(
            selected_action_branch=branch,
            paper_cycle_duration=max(1e-9, paper_cycle_duration),
        )
        if self.reward_reference is None:
            return pending_cold_start_reward_json(paper_cycle_status=PAPER_CYCLE_STATUS_COMPLETE)
        reward = compute_reward(components, self.reward_reference)
        return reward.to_json_dict()

    def _write_summary(self, *, force: bool = False) -> None:
        if not self.config.rollout_enabled:
            return
        self._summary_dirty_events += 1
        if not force and self._summary_dirty_events < self._summary_write_cadence:
            return
        write_rollout_summary(self.summary_path, self.summary.to_summary())
        self._summary_dirty_events = 0


def _selected_action(decision: Any, zones: tuple[str, ...]) -> dict[str, Any]:
    metadata: Mapping[str, Any] = getattr(decision, "metadata", {}) or {}
    decision_branch = getattr(decision, "branch", None)
    decision_zone_id = getattr(decision, "zone_id", None)
    if getattr(decision, "action_index", None) is not None:
        return {
            "index": int(getattr(decision, "action_index")),
            "branch": decision_branch or metadata.get("selected_action_branch"),
            "zone_id": decision_zone_id or metadata.get("selected_zone_id"),
        }
    if metadata.get("selected_action_index") is not None:
        return {
            "index": int(metadata["selected_action_index"]),
            "branch": decision_branch or metadata.get("selected_action_branch"),
            "zone_id": decision_zone_id or metadata.get("selected_zone_id"),
        }
    storage = getattr(decision, "storage", None)
    if storage is None:
        return {"index": None, "branch": None, "zone_id": None}
    zone_id = decision_zone_id or infer_zone_id(storage)
    branch = decision_branch or STORE
    try:
        index = encode_action(branch, zone_id, zones)
    except ValueError:
        index = None
    return {"index": index, "branch": branch, "zone_id": zone_id}


def _committed_next_payload(context: Any, robot: Any) -> dict[str, Any]:
    warehouse = getattr(context, "warehouse", None)
    registry = getattr(warehouse, "committed_next_registry", None)
    reservation = registry.get_for_robot(robot) if registry is not None else None
    if reservation is None:
        return _empty_committed_next_payload()
    return {
        "committed_next_reservation_id": _text(getattr(reservation, "reservation_id", "")) or None,
        "committed_next_job_id": _text(getattr(reservation, "job_id", "")) or None,
        "committed_next_pod_id": _text(getattr(reservation, "pod_id", "")) or None,
        "committed_next_station_id": _text(getattr(reservation, "picking_station_id", "")) or None,
        "committed_next_zone_id": _text(getattr(reservation, "committed_next_zone_id", "")) or None,
        "committed_next_created_time_seconds": _float_or_none(
            getattr(reservation, "created_time_seconds", None)
        ),
        "committed_next_activation_time_seconds": _float_or_none(
            getattr(reservation, "activation_time_seconds", None)
        ),
        "committed_next_candidate_count": getattr(reservation, "candidate_count", None),
        "committed_next_retrieval_cost": _float_or_none(getattr(reservation, "retrieval_cost", None)),
    }


def _pending_committed_next_payload(pending: PendingRTSDecision) -> dict[str, Any]:
    return {
        "committed_next_reservation_id": pending.committed_next_reservation_id,
        "committed_next_job_id": pending.committed_next_job_id,
        "committed_next_pod_id": pending.committed_next_pod_id,
        "committed_next_station_id": pending.committed_next_station_id,
        "committed_next_zone_id": pending.committed_next_zone_id,
        "committed_next_created_time_seconds": pending.committed_next_created_time_seconds,
        "committed_next_activation_time_seconds": pending.committed_next_activation_time_seconds,
    }


def _selected_action_context_payload(state_json: Mapping[str, Any], action_index: Any) -> dict[str, Any]:
    if action_index is None:
        return {}
    try:
        selected_index = int(action_index)
    except Exception:
        return {}
    for row in state_json.get("rts_action_contexts", []) or []:
        try:
            if int(row.get("action_index")) == selected_index:
                return dict(row)
        except Exception:
            continue
    return {}


def _cycle_estimate_payload(cycle_estimate: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "cycle_estimate_known": 1 if bool(cycle_estimate.get("known", False)) else 0,
        "cycle_estimate_status": _text(cycle_estimate.get("status")) or None,
        "cycle_estimate_semantics": _text(cycle_estimate.get("semantics")) or None,
        "cycle_estimator_version": _text(cycle_estimate.get("estimator_version")) or None,
        "estimated_cycle_time_at_decision": _float_or_none(cycle_estimate.get("estimated_cycle_seconds")),
        "estimated_travel_time_at_decision": _float_or_none(cycle_estimate.get("estimated_travel_seconds")),
        "estimated_queue_time_at_decision": _float_or_none(cycle_estimate.get("estimated_queue_seconds")),
        "estimated_replenishment_service_time_at_decision": _float_or_none(
            cycle_estimate.get("estimated_replenishment_service_seconds")
        ),
        "estimated_handling_time_at_decision": _float_or_none(cycle_estimate.get("estimated_handling_seconds")),
        "cycle_estimate_fallback_used": 1 if bool(cycle_estimate.get("fallback_used", False)) else 0,
    }


def _pending_cycle_estimate_payload(pending: PendingRTSDecision, *, realized_cycle_time: float | None) -> dict[str, Any]:
    estimate = pending.estimated_cycle_time_at_decision
    payload = {
        "cycle_estimate_known": 1 if pending.cycle_estimate_known else 0,
        "cycle_estimate_status": pending.cycle_estimate_status,
        "estimated_cycle_time_at_decision": estimate,
        "estimated_queue_time_at_decision": pending.estimated_queue_time_at_decision,
        "estimated_replenishment_service_time_at_decision": pending.estimated_replenishment_service_time_at_decision,
        "cycle_estimate_error": None,
        "cycle_estimate_absolute_error": None,
        "cycle_estimate_relative_error": None,
    }
    if realized_cycle_time is None or estimate is None:
        return payload
    error = float(realized_cycle_time) - float(estimate)
    payload["cycle_estimate_error"] = error
    payload["cycle_estimate_absolute_error"] = abs(error)
    payload["cycle_estimate_relative_error"] = abs(error) / max(1e-9, abs(float(realized_cycle_time)))
    return payload


def _empty_committed_next_payload() -> dict[str, Any]:
    return {
        "committed_next_reservation_id": None,
        "committed_next_job_id": None,
        "committed_next_pod_id": None,
        "committed_next_station_id": None,
        "committed_next_zone_id": None,
        "committed_next_created_time_seconds": None,
        "committed_next_activation_time_seconds": None,
        "committed_next_candidate_count": 0,
        "committed_next_retrieval_cost": None,
    }


def _load_reward_reference(path: str | None) -> RTSRewardReference | None:
    if not path:
        return None
    ref_path = Path(path)
    if not ref_path.exists():
        return None
    cycle_ref = read_cycle_reference(ref_path)
    return RTSRewardReference(
        reference_overall_cycle_time=cycle_ref.reference_overall_cycle_time,
        reference_avg_storage_cycle_time=cycle_ref.reference_avg_storage_cycle_time,
        reference_avg_replenish_cycle_time=cycle_ref.reference_avg_replenish_cycle_time,
        alpha=cycle_ref.alpha,
        source=cycle_ref.source,
        source_run_id=cycle_ref.source_run_id,
        semantics="paper_cycle_duration",
    )


def _robot_id(robot: Any) -> str:
    if getattr(robot, "_id", None) is not None:
        return _text(getattr(robot, "_id"))
    return _text(getattr(robot, "id", ""))


def _text(value: Any) -> str:
    return "" if value is None else str(value)


def _float_or_none(val: Any) -> float | None:
    if val is None:
        return None
    try:
        return float(val)
    except Exception:
        return None


def _netlogo_step_or_none(warehouse_time: Any, tick_to_second: Any) -> int | None:
    if warehouse_time is None or tick_to_second is None:
        return None
    try:
        return warehouse_time_to_netlogo_steps(float(warehouse_time), float(tick_to_second))
    except Exception:
        return None
