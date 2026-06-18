"""RTS rollout decision/outcome linking runtime."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .action_space import STORE, build_action_mask, encode_action
from .cycle_reference import read_cycle_reference
from .evaluation_policy import infer_zone_ids_from_context
from .evaluation_summary import summarize_rollout_events, write_rollout_summary
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
        pending.return_finish_tick = float(return_finish_tick)
        pending.return_duration = max(0.0, float(return_finish_tick) - pending.return_start_tick)
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

    def close(self) -> None:
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
        self.reward_reference = _load_reward_reference(config.reward_reference_path)
        if self.config.rollout_enabled:
            self._write_summary()

    def on_decision(self, *, robot: Any, context: Any, decision: Any) -> None:
        if not self.config.rollout_enabled:
            return
        zones = self.config.zone_ids or infer_zone_ids_from_context(context)
        if not zones:
            raise RuntimeError("RTS rollout requires configured or inferable zone_ids when a decision occurs")
        state = build_state(context, zones)
        store_valid = {row["zone_id"]: bool(row["store_action_valid"]) for row in state.state_json["zone_rows"]}
        repl_valid = {row["zone_id"]: bool(row["replenish_store_action_valid"]) for row in state.state_json["zone_rows"]}
        mask = build_action_mask(zones, store_valid_by_zone=store_valid, replenish_valid_by_zone=repl_valid)
        selected = _selected_action(decision, zones)
        features = build_feature_bundle(zones, mask, state.state_json)
        warehouse = getattr(context, "warehouse", None)
        tick = getattr(warehouse, "_tick", None)
        tick_to_second = getattr(warehouse, "tick_to_second", None)
        warehouse_time = _float_or_none(tick)
        netlogo_step = _netlogo_step_or_none(warehouse_time, tick_to_second)
        metadata = dict(getattr(decision, "metadata", {}) or {})
        robot_id = _robot_id(robot)
        job = getattr(robot, "job", None)
        job_id = _text(getattr(job, "my_id", ""))
        pod_id = _text(getattr(getattr(job, "pod", None), "pod_id", getattr(context.pod, "pod_id", "")))
        decision_event_id = make_decision_event_id(robot_id=robot_id, job_id=job_id, pod_id=pod_id, tick=tick)
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
            state_json=state.state_json,
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
        self.writer.write_decision(row)
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
            )
        )
        self._write_summary()

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
        self.writer.write_outcome(row)
        self._write_summary()

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
            self._complete_paper_cycle(robot=robot, station=station, pending=pending)
        elif station_type == "replenishment":
            self._censor_paper_cycle(
                robot=robot,
                station=station,
                pending=pending,
                status=PAPER_CYCLE_STATUS_CENSORED_NEXT_TASK_REPLENISHMENT,
                reason="next_task_replenishment",
            )

    def close(self) -> None:
        self._write_summary()
        self.writer.close()

    def _complete_paper_cycle(self, *, robot: Any, station: Any, pending: PendingRTSDecision) -> None:
        tick = float(getattr(getattr(robot, "universe", None), "_tick", getattr(getattr(robot, "warehouse", None), "_tick", 0.0)))
        completed = self.tracker.complete_paper_cycle_for_robot(robot_id=pending.robot_id)
        if completed is None:
            return
        duration = max(0.0, tick - completed.return_start_tick)
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
        self.writer.write_outcome(row)
        self._write_summary()

    def _censor_paper_cycle(
        self,
        *,
        robot: Any,
        station: Any,
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
            paper_cycle_next_station_arrival_tick=tick,
            paper_cycle_duration=None,
            paper_cycle_censor_reason=reason,
            paper_cycle_completion_rule=status,
        )
        self.writer.write_outcome(row)
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

    def _write_summary(self) -> None:
        if not self.config.rollout_enabled:
            return
        summary = summarize_rollout_events(self.writer.events, policy_mode=self.config.policy_mode)
        write_rollout_summary(self.summary_path, summary)


def _selected_action(decision: Any, zones: tuple[str, ...]) -> dict[str, Any]:
    metadata: Mapping[str, Any] = getattr(decision, "metadata", {}) or {}
    if metadata.get("selected_action_index") is not None:
        return {
            "index": int(metadata["selected_action_index"]),
            "branch": metadata.get("selected_action_branch"),
            "zone_id": metadata.get("selected_zone_id"),
        }
    storage = getattr(decision, "storage", None)
    if storage is None:
        return {"index": None, "branch": None, "zone_id": None}
    zone_id = infer_zone_id(storage)
    try:
        index = encode_action(STORE, zone_id, zones)
    except ValueError:
        index = None
    return {"index": index, "branch": STORE, "zone_id": zone_id}


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
