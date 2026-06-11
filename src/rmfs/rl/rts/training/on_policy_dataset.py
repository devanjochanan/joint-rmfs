"""On-policy RTS rollout dataset builder using logged old-policy values."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from src.rmfs.rl.rts.action_space import action_mask_entry, validate_action_mask
from src.rmfs.rl.rts.features import build_feature_bundle
from src.rmfs.rl.rts.rollout_schema import DECISION_EVENT, OUTCOME_EVENT
from src.rmfs.rl.rts.training.metrics import finite_float, mean_or_none
from src.rmfs.rl.rts.training.ppo import RTSPPORolloutBatch, compute_gae
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


@dataclass(frozen=True)
class RTSOnPolicyRolloutDataset:
    steps: tuple[RTSOnPolicyTrainingStep, ...]
    summary: dict[str, Any]


def build_on_policy_training_steps(
    events: Sequence[Mapping[str, Any]],
    *,
    required_policy_checkpoint_id: str,
) -> RTSOnPolicyRolloutDataset:
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
        "rejected_reward_uncomputed_count": 0,
        "rejected_non_on_policy_count": 0,
        "rejected_checkpoint_mismatch_count": 0,
        "rejected_missing_old_log_prob_count": 0,
        "rejected_missing_old_value_count": 0,
        "rejected_invalid_selected_action_count": 0,
        "rejected_missing_state_count": 0,
        "rejected_feature_error_count": 0,
    }
    steps: list[RTSOnPolicyTrainingStep] = []
    for event_id, decs in decisions.items():
        if len(decs) != 1:
            rejected["rejected_duplicate_decision_count"] += len(decs)
            continue
        outs = outcomes.get(event_id, [])
        if len(outs) > 1:
            rejected["rejected_duplicate_outcome_count"] += len(outs)
            continue
        if not outs:
            rejected["rejected_missing_outcome_count"] += 1
            continue
        step, reason = _build_step(decs[0], outs[0], required_policy_checkpoint_id)
        if step is None:
            rejected[reason] += 1
        else:
            steps.append(step)
    rewards = [step.reward for step in steps]
    summary = {
        "decision_count": sum(len(v) for v in decisions.values()),
        "outcome_count": sum(len(v) for v in outcomes.values()),
        "trainable_step_count": len(steps),
        "avg_reward": mean_or_none(rewards),
        **rejected,
    }
    return RTSOnPolicyRolloutDataset(steps=tuple(steps), summary=summary)


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
    if advantages.size > 1 and float(advantages.std()) > 1e-8:
        advantages = ((advantages - advantages.mean()) / advantages.std()).astype(np.float32)
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


def _build_step(decision: Mapping[str, Any], outcome: Mapping[str, Any], required_policy_checkpoint_id: str):
    if decision.get("actor_kind") != "rts_rl_explicit":
        return None, "rejected_non_on_policy_count"
    if decision.get("policy_checkpoint_id") != required_policy_checkpoint_id:
        return None, "rejected_checkpoint_mismatch_count"
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
    selected = decision.get("selected_action_index")
    try:
        selected_index = int(selected)
        mask = np.asarray(validate_action_mask(zone_ids, decision.get("action_mask"), require_valid=True), dtype=np.int64)
        if action_mask_entry(selected_index, zone_ids, mask) != 1:
            return None, "rejected_invalid_selected_action_count"
    except Exception:
        return None, "rejected_invalid_selected_action_count"
    reward_json = outcome.get("reward_json") or {}
    reward = finite_float(reward_json.get("reward_value"))
    if not reward_json.get("reward_computed") or reward is None:
        return None, "rejected_reward_uncomputed_count"
    try:
        build_feature_bundle(zone_ids, mask, state_json)
    except Exception:
        return None, "rejected_feature_error_count"
    return RTSOnPolicyTrainingStep(
        decision_event_id=str(decision.get("decision_event_id", "")),
        worker_run_id=decision.get("worker_run_id") or outcome.get("worker_run_id"),
        netlogo_step=_int_or_none(decision.get("netlogo_step")),
        warehouse_time=finite_float(decision.get("warehouse_time")),
        tick_to_second=finite_float(decision.get("tick_to_second")),
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
    ), ""


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

    return build_feature_tensors_from_steps([_Step(step) for step in steps])


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except Exception:
        return None

