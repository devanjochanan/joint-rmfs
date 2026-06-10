"""JSON-safe RTS rollout event builders."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


ROLLOUT_SCHEMA_VERSION = "rts_rollout.v1"
DECISION_EVENT = "decision"
OUTCOME_EVENT = "outcome"


def make_decision_event_id(*, robot_id: Any, job_id: Any, pod_id: Any, tick: Any) -> str:
    return f"rts-{_text(robot_id)}-{_text(job_id)}-{_text(pod_id)}-{_text(tick)}"


def build_decision_event(
    *,
    decision_event_id: str,
    tick: Any,
    robot_id: Any,
    job_id: Any,
    pod_id: Any,
    source_station_id: Any,
    source_station_type: Any,
    policy_name: str,
    zone_ids: Sequence[str],
    action_mask: Sequence[int],
    selected_action_index: int | None,
    selected_action_branch: str | None,
    selected_zone_id: str | None,
    selected_storage: Any,
    state_json: Mapping[str, Any],
    feature_shapes: Mapping[str, Any],
) -> dict[str, Any]:
    return _json_safe(
        {
            "schema_version": ROLLOUT_SCHEMA_VERSION,
            "event_type": DECISION_EVENT,
            "decision_event_id": decision_event_id,
            "tick": _number_or_text(tick),
            "robot_id": _text(robot_id),
            "job_id": _text(job_id),
            "pod_id": _text(pod_id),
            "source_station_id": _text(source_station_id),
            "source_station_type": _text(source_station_type),
            "policy_name": str(policy_name),
            "zone_ids": list(zone_ids),
            "action_mask": [int(value) for value in action_mask],
            "selected_action_index": selected_action_index,
            "selected_action_branch": selected_action_branch,
            "selected_zone_id": selected_zone_id,
            "selected_storage": _storage_json(selected_storage),
            "state_json": dict(state_json),
            "feature_shapes": dict(feature_shapes),
            "reward_json": None,
            "outcome": None,
        }
    )


def build_outcome_event(
    *,
    decision_event_id: str,
    tick: Any,
    robot_id: Any,
    job_id: Any,
    pod_id: Any,
    outcome_status: str,
    return_start_tick: Any,
    return_finish_tick: Any,
    realized_cycle_time: Any,
    destination_x: Any,
    destination_y: Any,
    reward_json: Mapping[str, Any] | None,
) -> dict[str, Any]:
    return _json_safe(
        {
            "schema_version": ROLLOUT_SCHEMA_VERSION,
            "event_type": OUTCOME_EVENT,
            "decision_event_id": decision_event_id,
            "tick": _number_or_text(tick),
            "robot_id": _text(robot_id),
            "job_id": _text(job_id),
            "pod_id": _text(pod_id),
            "outcome_status": str(outcome_status),
            "return_start_tick": _number_or_text(return_start_tick),
            "return_finish_tick": _number_or_text(return_finish_tick),
            "realized_cycle_time": _number_or_text(realized_cycle_time),
            "destination_x": _number_or_text(destination_x),
            "destination_y": _number_or_text(destination_y),
            "reward_json": reward_json,
        }
    )


def _storage_json(storage: Any) -> dict[str, Any] | None:
    if storage is None:
        return None
    return {
        "x": _number_or_text(getattr(storage, "pos_x", None)),
        "y": _number_or_text(getattr(storage, "pos_y", None)),
    }


def _text(value: Any) -> str:
    return "" if value is None else str(value)


def _number_or_text(value: Any) -> Any:
    if value is None:
        return None
    try:
        number = float(value)
    except Exception:
        return str(value)
    if number.is_integer():
        return int(number)
    return number


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "tolist"):
        return _json_safe(value.tolist())
    return str(value)

