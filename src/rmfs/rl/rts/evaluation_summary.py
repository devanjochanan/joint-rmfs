"""Aggregation helpers for RTS rollout events."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from .action_space import action_mask_entry
from .rollout_schema import DECISION_EVENT, OUTCOME_EVENT


SUMMARY_SCHEMA_VERSION = "rts_rollout_summary.v1"


class RolloutSummaryAccumulator:
    def __init__(self, policy_mode: str | None = None):
        self.policy_mode = policy_mode or "unknown"
        self.decision_count = 0
        self.outcome_count = 0
        self._decision_ids: set[Any] = set()
        self._outcome_ids: set[Any] = set()
        self.reward_computed_count = 0
        self._realized_total = 0.0
        self._realized_count = 0
        self._paper_cycle_total = 0.0
        self._paper_cycle_count = 0
        self.completed_paper_cycle_count = 0
        self.pending_paper_cycle_count = 0
        self.censored_paper_cycle_count = 0
        self.paper_cycle_status_counts: dict[str, int] = {}
        self.selected_action_counts: dict[str, int] = {}
        self.invalid_action_selected_count = 0
        self._timing_rows: list[Mapping[str, Any]] = []

    def add_event(self, event: Mapping[str, Any]) -> None:
        row = dict(event)
        event_type = row.get("event_type")
        if event_type == DECISION_EVENT:
            self._add_decision(row)
        elif event_type == OUTCOME_EVENT:
            self._add_outcome(row)

    def _add_decision(self, row: Mapping[str, Any]) -> None:
        self.decision_count += 1
        self._decision_ids.add(row.get("decision_event_id"))
        if self.policy_mode == "unknown":
            self.policy_mode = str(row.get("policy_name", "unknown"))
        branch = row.get("selected_action_branch")
        zone = row.get("selected_zone_id")
        key = f"{branch}:{zone}" if branch and zone else "unselected"
        self.selected_action_counts[key] = self.selected_action_counts.get(key, 0) + 1
        try:
            index = row.get("selected_action_index")
            zone_ids = row.get("zone_ids") or []
            mask = row.get("action_mask") or []
            if index is not None and action_mask_entry(int(index), zone_ids, mask) != 1:
                self.invalid_action_selected_count += 1
        except Exception:
            self.invalid_action_selected_count += 1
        timing = row.get("rts_decision_timing")
        if timing is not None:
            self._timing_rows.append(timing)

    def _add_outcome(self, row: Mapping[str, Any]) -> None:
        self.outcome_count += 1
        self._outcome_ids.add(row.get("decision_event_id"))
        realized = _float(row.get("realized_cycle_time"))
        if realized is not None:
            self._realized_total += realized
            self._realized_count += 1
        paper_cycle = _float(row.get("paper_cycle_duration"))
        if paper_cycle is not None:
            self._paper_cycle_total += paper_cycle
            self._paper_cycle_count += 1
        status = str(row.get("paper_cycle_status") or "").strip() or "unknown"
        self.paper_cycle_status_counts[status] = self.paper_cycle_status_counts.get(status, 0) + 1
        if status == "complete":
            self.completed_paper_cycle_count += 1
        elif status == "pending":
            self.pending_paper_cycle_count += 1
        elif status.startswith("censored"):
            self.censored_paper_cycle_count += 1
        reward = row.get("reward_json")
        if isinstance(reward, Mapping) and reward.get("reward_computed"):
            self.reward_computed_count += 1

    def to_summary(self) -> dict[str, Any]:
        timing_count = len(self._timing_rows)
        if timing_count > 0:
            mean_build_state = sum(float(r.get("build_state_ms") or 0.0) for r in self._timing_rows) / timing_count
            mean_feature_bundle = sum(float(r.get("build_feature_bundle_ms") or 0.0) for r in self._timing_rows) / timing_count
            mean_forward = sum(float(r.get("tensor_and_forward_ms") or 0.0) for r in self._timing_rows) / timing_count
            mean_revalidation = sum(float(r.get("selected_context_revalidation_ms") or 0.0) for r in self._timing_rows) / timing_count
            mean_total = sum(float(r.get("total_select_destination_ms") or 0.0) for r in self._timing_rows) / timing_count
            max_total = max(float(r.get("total_select_destination_ms") or 0.0) for r in self._timing_rows)
        else:
            mean_build_state = None
            mean_feature_bundle = None
            mean_forward = None
            mean_revalidation = None
            mean_total = None
            max_total = None

        return {
            "schema_version": SUMMARY_SCHEMA_VERSION,
            "policy_mode": self.policy_mode,
            "decision_count": self.decision_count,
            "outcome_count": self.outcome_count,
            "orphan_count": sum(1 for event_id in self._decision_ids if event_id not in self._outcome_ids),
            "reward_computed_count": self.reward_computed_count,
            "avg_realized_cycle_time": (
                self._realized_total / self._realized_count
                if self._realized_count
                else None
            ),
            "avg_paper_cycle_duration": (
                self._paper_cycle_total / self._paper_cycle_count
                if self._paper_cycle_count
                else None
            ),
            "completed_paper_cycle_count": self.completed_paper_cycle_count,
            "pending_paper_cycle_count": self.pending_paper_cycle_count,
            "censored_paper_cycle_count": self.censored_paper_cycle_count,
            "paper_cycle_status_counts": dict(self.paper_cycle_status_counts),
            "selected_action_counts": dict(self.selected_action_counts),
            "invalid_action_selected_count": self.invalid_action_selected_count,
            "decision_timing_count": timing_count,
            "mean_build_state_ms": mean_build_state,
            "mean_feature_bundle_ms": mean_feature_bundle,
            "mean_forward_ms": mean_forward,
            "mean_revalidation_ms": mean_revalidation,
            "mean_total_decision_ms": mean_total,
            "max_total_decision_ms": max_total,
        }


def summarize_rollout_events(events: Iterable[Mapping[str, Any]], policy_mode: str | None = None) -> dict[str, Any]:
    rows = [dict(event) for event in events]
    decisions = [row for row in rows if row.get("event_type") == DECISION_EVENT]
    outcomes = [row for row in rows if row.get("event_type") == OUTCOME_EVENT]
    outcome_ids = {row.get("decision_event_id") for row in outcomes}
    realized = [_float(row.get("realized_cycle_time")) for row in outcomes]
    realized = [value for value in realized if value is not None]
    paper_cycles = [_float(row.get("paper_cycle_duration")) for row in outcomes]
    paper_cycles = [value for value in paper_cycles if value is not None]
    paper_status_counts: dict[str, int] = {}
    completed_paper_cycles = 0
    pending_paper_cycles = 0
    censored_paper_cycles = 0
    for row in outcomes:
        status = str(row.get("paper_cycle_status") or "").strip() or "unknown"
        paper_status_counts[status] = paper_status_counts.get(status, 0) + 1
        if status == "complete":
            completed_paper_cycles += 1
        elif status == "pending":
            pending_paper_cycles += 1
        elif status.startswith("censored"):
            censored_paper_cycles += 1
    counts: dict[str, int] = {}
    invalid = 0
    for row in decisions:
        branch = row.get("selected_action_branch")
        zone = row.get("selected_zone_id")
        key = f"{branch}:{zone}" if branch and zone else "unselected"
        counts[key] = counts.get(key, 0) + 1
        try:
            index = row.get("selected_action_index")
            zone_ids = row.get("zone_ids") or []
            mask = row.get("action_mask") or []
            if index is not None and action_mask_entry(int(index), zone_ids, mask) != 1:
                invalid += 1
        except Exception:
            invalid += 1
    reward_count = sum(
        1
        for row in outcomes
        if isinstance(row.get("reward_json"), Mapping) and row["reward_json"].get("reward_computed")
    )
    detected_mode = policy_mode
    if detected_mode is None and decisions:
        detected_mode = str(decisions[0].get("policy_name", "unknown"))

    timing_rows = [
        row.get("rts_decision_timing")
        for row in decisions
        if row.get("rts_decision_timing") is not None
    ]
    timing_count = len(timing_rows)
    if timing_count > 0:
        mean_build_state = sum(float(r.get("build_state_ms") or 0.0) for r in timing_rows) / timing_count
        mean_feature_bundle = sum(float(r.get("build_feature_bundle_ms") or 0.0) for r in timing_rows) / timing_count
        mean_forward = sum(float(r.get("tensor_and_forward_ms") or 0.0) for r in timing_rows) / timing_count
        mean_revalidation = sum(float(r.get("selected_context_revalidation_ms") or 0.0) for r in timing_rows) / timing_count
        mean_total = sum(float(r.get("total_select_destination_ms") or 0.0) for r in timing_rows) / timing_count
        max_total = max(float(r.get("total_select_destination_ms") or 0.0) for r in timing_rows)
    else:
        mean_build_state = None
        mean_feature_bundle = None
        mean_forward = None
        mean_revalidation = None
        mean_total = None
        max_total = None

    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "policy_mode": detected_mode or "unknown",
        "decision_count": len(decisions),
        "outcome_count": len(outcomes),
        "orphan_count": sum(1 for row in decisions if row.get("decision_event_id") not in outcome_ids),
        "reward_computed_count": reward_count,
        "avg_realized_cycle_time": (sum(realized) / len(realized)) if realized else None,
        "avg_paper_cycle_duration": (sum(paper_cycles) / len(paper_cycles)) if paper_cycles else None,
        "completed_paper_cycle_count": completed_paper_cycles,
        "pending_paper_cycle_count": pending_paper_cycles,
        "censored_paper_cycle_count": censored_paper_cycles,
        "paper_cycle_status_counts": paper_status_counts,
        "selected_action_counts": counts,
        "invalid_action_selected_count": invalid,
        "decision_timing_count": timing_count,
        "mean_build_state_ms": mean_build_state,
        "mean_feature_bundle_ms": mean_feature_bundle,
        "mean_forward_ms": mean_forward,
        "mean_revalidation_ms": mean_revalidation,
        "mean_total_decision_ms": mean_total,
        "max_total_decision_ms": max_total,
    }


def write_rollout_summary(path: Path, events_or_summary: Iterable[Mapping[str, Any]] | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(events_or_summary, Mapping):
        summary = dict(events_or_summary)
    else:
        summary = summarize_rollout_events(events_or_summary)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    with tmp_path.open("w") as fh:
        json.dump(summary, fh, indent=2)
    tmp_path.replace(path)
    return summary


def _float(value: Any) -> float | None:
    try:
        return float(value)
    except Exception:
        return None
