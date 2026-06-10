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
from .reward import RTSRewardReference, build_reward_components_from_realized_cycle, compute_reward
from .rollout_schema import build_decision_event, build_outcome_event, make_decision_event_id
from .rollout_writer import RTSRolloutWriter
from .runtime_config import RTSRuntimeConfig
from .state import build_state
from .zone_features import infer_zone_id


@dataclass
class PendingRTSDecision:
    decision_event_id: str
    robot_id: str
    job_id: str
    pod_id: str
    return_start_tick: float
    selected_action_branch: str


class RTSOutcomeTracker:
    def __init__(self):
        self.pending: dict[tuple[str, str, str], PendingRTSDecision] = {}

    def record_decision(self, pending: PendingRTSDecision) -> None:
        self.pending[(pending.robot_id, pending.job_id, pending.pod_id)] = pending

    def complete_return(self, *, robot_id: str, job_id: str, pod_id: str) -> PendingRTSDecision | None:
        return self.pending.pop((robot_id, job_id, pod_id), None)

    def orphan_pending(self) -> list[PendingRTSDecision]:
        return list(self.pending.values())


class NoopRTSRolloutRuntime:
    def on_decision(self, *args, **kwargs) -> None:
        return None

    def on_return_completed(self, *args, **kwargs) -> None:
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
        tick = getattr(getattr(context, "warehouse", None), "_tick", None)
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
        pending = self.tracker.complete_return(robot_id=robot_id, job_id=job_id, pod_id=pod_id)
        if pending is None:
            return
        tick = float(getattr(getattr(robot, "universe", None), "_tick", getattr(getattr(robot, "warehouse", None), "_tick", 0.0)))
        realized = max(0.0, tick - pending.return_start_tick)
        reward_json = self._reward_json(pending.selected_action_branch, realized)
        destination = getattr(robot, "destination", None)
        row = build_outcome_event(
            decision_event_id=pending.decision_event_id,
            tick=tick,
            robot_id=robot_id,
            job_id=job_id,
            pod_id=pod_id,
            outcome_status="completed",
            return_start_tick=pending.return_start_tick,
            return_finish_tick=tick,
            realized_cycle_time=realized,
            destination_x=getattr(destination, "x", None),
            destination_y=getattr(destination, "y", None),
            reward_json=reward_json,
        )
        self.writer.write_outcome(row)
        self._write_summary()

    def close(self) -> None:
        self._write_summary()
        self.writer.close()

    def _reward_json(self, branch: str, realized_cycle_time: float) -> dict[str, Any]:
        reward = compute_reward(
            build_reward_components_from_realized_cycle(
                selected_action_branch=branch,
                realized_cycle_time=max(1e-9, realized_cycle_time),
            ),
            self.reward_reference,
        )
        return reward.to_json_dict()

    def _write_summary(self) -> None:
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
        semantics=cycle_ref.semantics,
    )


def _robot_id(robot: Any) -> str:
    if getattr(robot, "_id", None) is not None:
        return _text(getattr(robot, "_id"))
    return _text(getattr(robot, "id", ""))


def _text(value: Any) -> str:
    return "" if value is None else str(value)

