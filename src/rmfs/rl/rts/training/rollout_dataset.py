"""Rollout JSONL loading and feature reconstruction for RTS PPO smokes."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from src.rmfs.rl.rts.action_space import action_mask_entry, validate_action_mask
from src.rmfs.rl.rts.features import build_feature_bundle
from src.rmfs.rl.rts.rollout_schema import DECISION_EVENT, OUTCOME_EVENT

from .metrics import finite_float


SKIP_REASONS = (
    "skipped_missing_outcome",
    "skipped_missing_selected_action",
    "skipped_invalid_selected_action",
    "skipped_reward_uncomputed",
    "skipped_missing_state",
    "skipped_feature_error",
    "skipped_nonpositive_cycle_time",
)


@dataclass(frozen=True)
class RTSTrainingStep:
    decision_event_id: str
    zone_ids: tuple[str, ...]
    action_mask: np.ndarray
    selected_action_index: int
    reward: float
    terminated: bool
    truncated: bool
    state_json: dict[str, Any]
    selected_action_branch: str | None
    selected_zone_id: str | None
    realized_cycle_time: float
    policy_name: str


@dataclass(frozen=True)
class RTSRolloutDataset:
    steps: tuple[RTSTrainingStep, ...]
    summary: dict[str, Any]


@dataclass(frozen=True)
class RTSPaddedTrainingBatch:
    X_actions: np.ndarray
    M_actions: np.ndarray
    X_stock: np.ndarray
    M_stock: np.ndarray
    selected_action_indices: np.ndarray
    rewards: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    action_feature_names: tuple[str, ...]
    stock_feature_names: tuple[str, ...]
    decision_event_ids: tuple[str, ...]


def load_rollout_jsonl(path: Path) -> list[dict[str, Any]]:
    with Path(path).open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


def pair_decision_outcome_events(events: Sequence[Mapping[str, Any]]) -> tuple[list[tuple[dict, dict]], dict]:
    decisions = {str(row.get("decision_event_id")): dict(row) for row in events if row.get("event_type") == DECISION_EVENT}
    outcomes = {str(row.get("decision_event_id")): dict(row) for row in events if row.get("event_type") == OUTCOME_EVENT}
    pairs = [(decision, outcomes[event_id]) for event_id, decision in decisions.items() if event_id in outcomes]
    return pairs, {
        "decision_count": len(decisions),
        "outcome_count": len(outcomes),
        "paired_count": len(pairs),
        "missing_outcome_count": len(decisions) - len(pairs),
    }


def build_training_steps(events: Sequence[Mapping[str, Any]]) -> RTSRolloutDataset:
    pairs, pair_summary = pair_decision_outcome_events(events)
    decisions_with_outcome = {decision.get("decision_event_id") for decision, _outcome in pairs}
    skipped = {reason: 0 for reason in SKIP_REASONS}
    for row in events:
        if row.get("event_type") == DECISION_EVENT and row.get("decision_event_id") not in decisions_with_outcome:
            skipped["skipped_missing_outcome"] += 1
    steps: list[RTSTrainingStep] = []
    for decision, outcome in pairs:
        step, reason = _build_step(decision, outcome)
        if step is None:
            skipped[reason] += 1
        else:
            steps.append(step)
    summary = {
        **pair_summary,
        **skipped,
        "eligible_step_count": len(steps),
    }
    return RTSRolloutDataset(steps=tuple(steps), summary=summary)


def build_feature_tensors_from_steps(steps: Sequence[RTSTrainingStep]) -> RTSPaddedTrainingBatch:
    if not steps:
        raise ValueError("RTS training batch requires at least one step")
    bundles = [build_feature_bundle(step.zone_ids, step.action_mask, step.state_json) for step in steps]
    action_names = bundles[0].action_feature_names
    stock_names = bundles[0].stock_feature_names
    if any(bundle.action_feature_names != action_names for bundle in bundles):
        raise ValueError("action feature schemas must match within a training batch")
    if any(bundle.stock_feature_names != stock_names for bundle in bundles):
        raise ValueError("stock feature schemas must match within a training batch")
    a_max = max(bundle.X_actions.shape[0] for bundle in bundles)
    k_max = max(bundle.X_stock.shape[0] for bundle in bundles)
    action_dim = len(action_names)
    stock_dim = len(stock_names)
    batch = len(steps)
    X_actions = np.zeros((batch, a_max, action_dim), dtype=np.float32)
    M_actions = np.zeros((batch, a_max), dtype=np.int64)
    X_stock = np.zeros((batch, k_max, stock_dim), dtype=np.float32)
    M_stock = np.zeros((batch, k_max), dtype=np.int64)
    for index, bundle in enumerate(bundles):
        actions = bundle.X_actions.shape[0]
        stocks = bundle.X_stock.shape[0]
        X_actions[index, :actions, :] = bundle.X_actions
        M_actions[index, :actions] = bundle.M_actions
        if stocks:
            X_stock[index, :stocks, :] = bundle.X_stock
            M_stock[index, :stocks] = bundle.M_stock
    return RTSPaddedTrainingBatch(
        X_actions=X_actions,
        M_actions=M_actions,
        X_stock=X_stock,
        M_stock=M_stock,
        selected_action_indices=np.asarray([step.selected_action_index for step in steps], dtype=np.int64),
        rewards=np.asarray([step.reward for step in steps], dtype=np.float32),
        terminated=np.asarray([step.terminated for step in steps], dtype=np.bool_),
        truncated=np.asarray([step.truncated for step in steps], dtype=np.bool_),
        action_feature_names=action_names,
        stock_feature_names=stock_names,
        decision_event_ids=tuple(step.decision_event_id for step in steps),
    )


def _build_step(decision: Mapping[str, Any], outcome: Mapping[str, Any]) -> tuple[RTSTrainingStep | None, str]:
    zone_ids = tuple(str(zone) for zone in decision.get("zone_ids") or ())
    state_json = decision.get("state_json")
    if not zone_ids or not isinstance(state_json, Mapping):
        return None, "skipped_missing_state"
    selected = decision.get("selected_action_index")
    if selected is None:
        return None, "skipped_missing_selected_action"
    try:
        selected_index = int(selected)
        mask = np.asarray(validate_action_mask(zone_ids, decision.get("action_mask") or (), require_valid=True), dtype=np.int64)
        if action_mask_entry(selected_index, zone_ids, mask) != 1:
            return None, "skipped_invalid_selected_action"
    except Exception:
        return None, "skipped_invalid_selected_action"
    reward_json = outcome.get("reward_json") or {}
    reward_value = finite_float(reward_json.get("reward_value"))
    if not reward_json.get("reward_computed") or reward_value is None:
        return None, "skipped_reward_uncomputed"
    realized = finite_float(outcome.get("realized_cycle_time"))
    if realized is None or realized <= 0.0:
        return None, "skipped_nonpositive_cycle_time"
    try:
        build_feature_bundle(zone_ids, mask, state_json)
    except Exception:
        return None, "skipped_feature_error"
    return RTSTrainingStep(
        decision_event_id=str(decision.get("decision_event_id", "")),
        zone_ids=zone_ids,
        action_mask=mask,
        selected_action_index=selected_index,
        reward=float(reward_value),
        terminated=True,
        truncated=False,
        state_json=dict(state_json),
        selected_action_branch=decision.get("selected_action_branch"),
        selected_zone_id=decision.get("selected_zone_id"),
        realized_cycle_time=float(realized),
        policy_name=str(decision.get("policy_name", "")),
    ), ""

