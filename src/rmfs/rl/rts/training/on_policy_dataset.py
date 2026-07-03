"""On-policy RTS rollout dataset builder using logged old-policy values."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

import numpy as np

from src.rmfs.rl.rts.action_space import action_mask_entry, validate_action_mask
from src.rmfs.rl.rts.features import build_action_feature_names, build_feature_bundle
from src.rmfs.rl.rts.ablation import resolve_ablation
from src.rmfs.rl.rts.rollout_schema import DECISION_EVENT, OUTCOME_EVENT
from src.rmfs.rl.rts.training.metrics import finite_float, mean_or_none
from src.rmfs.rl.rts.training.ppo import RTSPPORolloutBatch, compute_gae
from src.rmfs.rl.rts.training.reward_normalizer import (
    apply_cold_start_rewards,
    choose_reward_metadata_for_batch,
    derive_reward_normalizer_from_events,
)
from src.rmfs.rl.rts.training.rollout_dataset import RTSPaddedTrainingBatch, build_feature_tensors_from_steps


@dataclass(frozen=True)
class RTSOnPolicyTrainingStep:
    decision_event_id: str
    worker_run_id: str | None
    netlogo_step: int | None
    warehouse_time: float | None
    tick_to_second: float | None
    policy_checkpoint_id: str
    old_log_prob: float
    old_value: float
    reward: float
    selected_action_index: int
    action_mask: np.ndarray
    state_json: dict[str, Any]
    zone_ids: tuple[str, ...]
    terminated: bool
    truncated: bool
    feature_ablation: str


@dataclass(frozen=True)
class RTSOnPolicyRolloutDataset:
    steps: tuple[RTSOnPolicyTrainingStep, ...]
    summary: dict[str, Any]


def build_on_policy_training_steps(
    events: Sequence[Mapping[str, Any]],
    *,
    required_policy_checkpoint_id: str,
    required_feature_schema_id: str | None = None,
    required_feature_ablation_hash: str | None = None,
    expected_tick_to_second: float = 0.15,
    reward_normalizer_metadata: Mapping[str, Any] | None = None,
) -> RTSOnPolicyRolloutDataset:
    derived_reward_metadata = derive_reward_normalizer_from_events(events)
    reward_metadata = choose_reward_metadata_for_batch(
        previous_metadata=reward_normalizer_metadata,
        derived_metadata=derived_reward_metadata,
    )
    events = apply_cold_start_rewards(events, reward_metadata=reward_metadata)
    decisions: dict[str, list[dict]] = {}
    outcomes: dict[str, list[dict]] = {}
    for row in events:
        event_id = row.get("decision_event_id")
        if event_id is None:
            continue
        if row.get("event_type") == DECISION_EVENT:
            decisions.setdefault(str(event_id), []).append(dict(row))
        elif row.get("event_type") == OUTCOME_EVENT:
            outcomes.setdefault(str(event_id), []).append(dict(row))
    rejected = {
        "rejected_duplicate_decision_count": 0,
        "rejected_duplicate_outcome_count": 0,
        "rejected_missing_outcome_count": 0,
        "rejected_missing_completed_paper_cycle_count": 0,
        "rejected_reward_uncomputed_count": 0,
        "rejected_non_on_policy_count": 0,
        "rejected_checkpoint_mismatch_count": 0,
        "rejected_missing_old_log_prob_count": 0,
        "rejected_missing_old_value_count": 0,
        "rejected_invalid_selected_action_count": 0,
        "rejected_missing_state_count": 0,
        "rejected_minimal_capture_count": 0,
        "rejected_nontrainable_capture_count": 0,
        "rejected_feature_error_count": 0,
        "rejected_feature_schema_mismatch_count": 0,
        "rejected_feature_ablation_mismatch_count": 0,
        "rejected_action_mask_dimension_count": 0,
        "rejected_state_contract_count": 0,
        "rejected_timebase_mismatch_count": 0,
        "rejected_runtime_invariant_count": 0,
        "rejected_duplicate_trainable_outcome_count": 0,
    }
    censored_count_by_reason: dict[str, int] = {}
    rejected_count_by_reason: dict[str, int] = {}
    raw_steps: list[RTSOnPolicyTrainingStep] = []
    for event_id, decs in decisions.items():
        if len(decs) != 1:
            rejected["rejected_duplicate_decision_count"] += len(decs)
            continue
        outs = outcomes.get(event_id, [])
        if not outs:
            rejected["rejected_missing_outcome_count"] += 1
            continue
        _count_censored(outs, censored_count_by_reason)
        trainable_outcome, outcome_reason = _select_trainable_paper_cycle_outcome(outs)
        if trainable_outcome is None:
            rejected[outcome_reason] += 1
            continue
        step, reason = _build_step(
            decs[0],
            trainable_outcome,
            required_policy_checkpoint_id,
            required_feature_schema_id=required_feature_schema_id,
            required_feature_ablation_hash=required_feature_ablation_hash,
            expected_tick_to_second=expected_tick_to_second,
        )
        if step is None:
            rejected[reason] += 1
            rejected_count_by_reason[reason] = rejected_count_by_reason.get(reason, 0) + 1
        else:
            raw_steps.append(step)

    processed_steps: list[RTSOnPolicyTrainingStep] = [
        replace(step, terminated=True, truncated=False)
        for step in sorted(
            raw_steps,
            key=lambda s: (
                s.worker_run_id or "",
            s.netlogo_step if s.netlogo_step is not None else 0,
            s.warehouse_time if s.warehouse_time is not None else 0.0,
            s.decision_event_id
            ),
        )
    ]

    rewards = [step.reward for step in processed_steps]
    summary = {
        "decision_count": sum(len(v) for v in decisions.values()),
        "outcome_count": sum(len(v) for v in outcomes.values()),
        "completed_paper_cycle_count": sum(
            1
            for group in outcomes.values()
            for outcome in group
            if _is_completed_paper_cycle_outcome(outcome)
        ),
        "pending_paper_cycle_count": sum(
            1
            for group in outcomes.values()
            for outcome in group
            if str(outcome.get("paper_cycle_status") or "").strip() == "pending"
        ),
        "pending_count": sum(
            1
            for group in outcomes.values()
            for outcome in group
            if str(outcome.get("paper_cycle_status") or "").strip() == "pending"
        ),
        "censored_paper_cycle_count": sum(
            1
            for group in outcomes.values()
            for outcome in group
            if str(outcome.get("paper_cycle_status") or "").strip().startswith("censored")
        ),
        "censored_count": sum(censored_count_by_reason.values()),
        "censored_count_by_reason": censored_count_by_reason,
        "rejected_count_by_reason": rejected_count_by_reason,
        "trainable_step_count": len(processed_steps),
        "completion_rate": _rate(
            sum(1 for group in outcomes.values() for outcome in group if _is_completed_paper_cycle_outcome(outcome)),
            sum(len(v) for v in decisions.values()),
        ),
        "censor_rate": _rate(sum(censored_count_by_reason.values()), sum(len(v) for v in decisions.values())),
        "trainable_rate": _rate(len(processed_steps), sum(len(v) for v in decisions.values())),
        "feature_ablation": processed_steps[0].feature_ablation if processed_steps else None,
        "avg_reward": mean_or_none(rewards),
        "reward_mean": mean_or_none(rewards),
        "reward_std": float(np.std(rewards)) if rewards else 0.0,
        "reward_min": float(np.min(rewards)) if rewards else 0.0,
        "reward_max": float(np.max(rewards)) if rewards else 0.0,
        **reward_metadata,
        **rejected,
    }
    return RTSOnPolicyRolloutDataset(steps=tuple(processed_steps), summary=summary)


def build_on_policy_ppo_batch(
    dataset: RTSOnPolicyRolloutDataset,
    *,
    gamma: float,
    gae_lambda: float,
) -> RTSPPORolloutBatch:
    if not dataset.steps:
        raise ValueError("on-policy PPO batch requires at least one step")
    padded = _padded_from_on_policy_steps(dataset.steps)
    old_values = np.asarray([step.old_value for step in dataset.steps], dtype=np.float32)
    advantages, returns = compute_gae(
        padded.rewards,
        old_values,
        padded.terminated,
        padded.truncated,
        gamma,
        gae_lambda,
    )
    return RTSPPORolloutBatch(
        X_actions=padded.X_actions,
        M_actions=padded.M_actions,
        X_stock=padded.X_stock,
        M_stock=padded.M_stock,
        selected_action_indices=padded.selected_action_indices,
        old_log_probs=np.asarray([step.old_log_prob for step in dataset.steps], dtype=np.float32),
        old_values=old_values,
        rewards=padded.rewards,
        terminated=padded.terminated,
        truncated=padded.truncated,
        action_feature_names=padded.action_feature_names,
        stock_feature_names=padded.stock_feature_names,
        returns=returns.astype(np.float32),
        advantages=advantages.astype(np.float32),
    )


def _build_step(
    decision: Mapping[str, Any],
    outcome: Mapping[str, Any],
    required_policy_checkpoint_id: str,
    *,
    required_feature_schema_id: str | None,
    required_feature_ablation_hash: str | None,
    expected_tick_to_second: float,
):
    if str(decision.get("state_capture_mode") or "full") == "minimal":
        return None, "rejected_minimal_capture_count"
    if decision.get("trainable") is False:
        return None, "rejected_nontrainable_capture_count"
    if decision.get("actor_kind") != "rts_rl_explicit":
        return None, "rejected_non_on_policy_count"
    if decision.get("policy_checkpoint_id") != required_policy_checkpoint_id:
        return None, "rejected_checkpoint_mismatch_count"
    if required_feature_schema_id is not None and decision.get("feature_schema_id") != required_feature_schema_id:
        return None, "rejected_feature_schema_mismatch_count"
    if required_feature_ablation_hash is not None and decision.get("feature_ablation_hash") != required_feature_ablation_hash:
        return None, "rejected_feature_ablation_mismatch_count"
    if bool(decision.get("hard_runtime_invariant_violation") or decision.get("runtime_invariant_violation")):
        return None, "rejected_runtime_invariant_count"
    tick_to_second = finite_float(decision.get("tick_to_second"))
    if tick_to_second is None or abs(tick_to_second - float(expected_tick_to_second)) > 1e-9:
        return None, "rejected_timebase_mismatch_count"
    old_log_prob = finite_float(decision.get("old_log_prob"))
    if old_log_prob is None:
        return None, "rejected_missing_old_log_prob_count"
    old_value = finite_float(decision.get("old_value"))
    if old_value is None:
        return None, "rejected_missing_old_value_count"
    zone_ids = tuple(str(zone) for zone in decision.get("zone_ids") or ())
    state_json = decision.get("state_json")
    if not zone_ids or not isinstance(state_json, Mapping) or not decision.get("action_mask"):
        return None, "rejected_missing_state_count"
    if str(state_json.get("state_contract_version") or "").strip() != "rts_rl_state.v4":
        return None, "rejected_state_contract_count"
    selected = decision.get("selected_action_index")
    try:
        selected_index = int(selected)
        mask = np.asarray(validate_action_mask(zone_ids, decision.get("action_mask"), require_valid=True), dtype=np.int64)
        if mask.shape[0] != 2 * len(zone_ids):
            return None, "rejected_action_mask_dimension_count"
        if action_mask_entry(selected_index, zone_ids, mask) != 1:
            return None, "rejected_invalid_selected_action_count"
    except Exception:
        return None, "rejected_invalid_selected_action_count"
    reward_json = outcome.get("reward_json") or {}
    reward = finite_float(reward_json.get("reward_value"))
    if not reward_json.get("reward_computed") or reward is None:
        return None, "rejected_reward_uncomputed_count"
    try:
        bundle = build_feature_bundle(zone_ids, mask, state_json)
        if len(bundle.action_feature_names) != len(build_action_feature_names(zone_ids)):
            return None, "rejected_feature_error_count"
        if len(bundle.stock_feature_names) != 4:
            return None, "rejected_feature_error_count"
    except Exception:
        return None, "rejected_feature_error_count"
    return RTSOnPolicyTrainingStep(
        decision_event_id=str(decision.get("decision_event_id", "")),
        worker_run_id=decision.get("worker_run_id") or outcome.get("worker_run_id"),
        netlogo_step=_int_or_none(decision.get("netlogo_step")),
        warehouse_time=finite_float(decision.get("warehouse_time")),
        tick_to_second=tick_to_second,
        policy_checkpoint_id=str(decision.get("policy_checkpoint_id")),
        old_log_prob=float(old_log_prob),
        old_value=float(old_value),
        reward=float(reward),
        selected_action_index=selected_index,
        action_mask=mask,
        state_json=dict(state_json),
        zone_ids=zone_ids,
        terminated=True,
        truncated=False,
        feature_ablation=str(decision.get("feature_ablation") or "full"),
    ), ""


def _select_trainable_paper_cycle_outcome(outcomes: Sequence[Mapping[str, Any]]) -> tuple[Mapping[str, Any] | None, str]:
    completed = [outcome for outcome in outcomes if _is_completed_paper_cycle_outcome(outcome)]
    if len(completed) > 1:
        return None, "rejected_duplicate_trainable_outcome_count"
    if not completed:
        return None, "rejected_missing_completed_paper_cycle_count"
    completed.sort(
        key=lambda outcome: (
            finite_float(outcome.get("paper_cycle_next_station_arrival_tick")) or 0.0,
            finite_float(outcome.get("warehouse_time")) or 0.0,
        )
    )
    return completed[0], ""


def _is_completed_paper_cycle_outcome(outcome: Mapping[str, Any]) -> bool:
    if str(outcome.get("paper_cycle_status") or "").strip() != "complete":
        return False
    try:
        if int(outcome.get("paper_cycle_complete") or 0) != 1:
            return False
    except Exception:
        return False
    rule = str(outcome.get("paper_cycle_completion_rule") or outcome.get("completion_rule") or "").strip()
    if rule != "next_order_retrieval_arrival":
        return False
    duration = finite_float(outcome.get("paper_cycle_duration"))
    return duration is not None and duration > 0.0


def _padded_from_on_policy_steps(steps: Sequence[RTSOnPolicyTrainingStep]) -> RTSPaddedTrainingBatch:
    class _Step:
        def __init__(self, step: RTSOnPolicyTrainingStep):
            self.decision_event_id = step.decision_event_id
            self.zone_ids = step.zone_ids
            self.action_mask = step.action_mask
            self.selected_action_index = step.selected_action_index
            self.reward = step.reward
            self.terminated = step.terminated
            self.truncated = step.truncated
            self.state_json = step.state_json

    ablation_name = "full"
    for step in steps:
        if step.feature_ablation:
            ablation_name = str(step.feature_ablation)
            break
    resolve_ablation(ablation_name)
    return build_feature_tensors_from_steps([_Step(step) for step in steps], feature_ablation=ablation_name)


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except Exception:
        return None


def _count_censored(outcomes: Sequence[Mapping[str, Any]], counts: dict[str, int]) -> None:
    for outcome in outcomes:
        status = str(outcome.get("paper_cycle_status") or "").strip()
        if not status.startswith("censored"):
            continue
        reason = status
        counts[reason] = counts.get(reason, 0) + 1


def _rate(numerator: int, denominator: int) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0
