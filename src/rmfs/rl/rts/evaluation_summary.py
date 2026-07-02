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
        self.decisions_with_valid_candidate_storage = 0
        self.decisions_with_nonempty_eligible_job_pool = 0
        self.decisions_with_known_proposed_next_job = 0
        self.proposals_selected_for_commitment = 0
        self.reservations_committed = 0
        self.reservations_activated = 0
        self.trainable_transition_count = 0
        self._reservation_ids: set[str] = set()
        self._activated_reservation_ids: set[str] = set()
        self._eligible_pool_size_total = 0.0
        self._eligible_pool_size_max = 0
        self._proposal_candidate_count_total = 0.0
        self._proposal_candidate_count_seen = 0
        self._proposal_build_ms_total = 0.0
        self._eligible_pool_build_ms_total = 0.0
        self._proposal_build_seen = 0
        self.cancellation_counts_by_reason: dict[str, int] = {}

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
        if row.get("selected_storage") is not None or row.get("candidate_storage_id"):
            self.decisions_with_valid_candidate_storage += 1
        pool_size = _int(row.get("eligible_job_pool_size"))
        if pool_size is not None:
            self._eligible_pool_size_total += float(pool_size)
            self._eligible_pool_size_max = max(self._eligible_pool_size_max, int(pool_size))
            if pool_size > 0:
                self.decisions_with_nonempty_eligible_job_pool += 1
        candidate_count = _int(row.get("committed_next_candidate_count"))
        if candidate_count is not None:
            self._proposal_candidate_count_total += float(candidate_count)
            self._proposal_candidate_count_seen += 1
        proposal_ms = _float(row.get("proposal_build_ms"))
        eligible_ms = _float(row.get("eligible_pool_build_ms"))
        if proposal_ms is not None or eligible_ms is not None:
            self._proposal_build_seen += 1
            self._proposal_build_ms_total += float(proposal_ms or 0.0)
            self._eligible_pool_build_ms_total += float(eligible_ms or 0.0)
        if row.get("next_job_proposal_id") or row.get("committed_next_job_id"):
            self.decisions_with_known_proposed_next_job += 1
        if row.get("next_job_proposal_id"):
            self.proposals_selected_for_commitment += 1
        reservation_id = row.get("committed_next_reservation_id")
        if reservation_id:
            self._reservation_ids.add(str(reservation_id))
            self.reservations_committed = len(self._reservation_ids)
        if row.get("trainable") is True:
            self.trainable_transition_count += 1
        try:
            index = row.get("selected_action_index")
            zone_ids = row.get("zone_ids") or []
            mask = row.get("action_mask") or []
            if row.get("state_capture_mode") == "minimal" or not mask:
                pass
            elif index is not None and action_mask_entry(int(index), zone_ids, mask) != 1:
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
            reason = str(row.get("paper_cycle_censor_reason") or status).strip() or status
            self.cancellation_counts_by_reason[reason] = self.cancellation_counts_by_reason.get(reason, 0) + 1
        reservation_id = row.get("committed_next_reservation_id")
        if reservation_id and row.get("committed_next_activation_time_seconds") is not None:
            self._activated_reservation_ids.add(str(reservation_id))
            self.reservations_activated = len(self._activated_reservation_ids)
        reward = row.get("reward_json")
        if isinstance(reward, Mapping) and reward.get("reward_computed"):
            self.reward_computed_count += 1

    def to_summary(self) -> dict[str, Any]:
        timing_count = len(self._timing_rows)
        timing_summary = _timing_summary(self._timing_rows)
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

        mean_pool_size = self._eligible_pool_size_total / self.decision_count if self.decision_count else None
        mean_candidate_count = (
            self._proposal_candidate_count_total / self._proposal_candidate_count_seen
            if self._proposal_candidate_count_seen
            else None
        )
        mean_proposal_ms = (
            self._proposal_build_ms_total / self._proposal_build_seen
            if self._proposal_build_seen
            else None
        )
        mean_eligible_pool_ms = (
            self._eligible_pool_build_ms_total / self._proposal_build_seen
            if self._proposal_build_seen
            else None
        )
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
            "decisions_with_valid_candidate_storage": self.decisions_with_valid_candidate_storage,
            "decisions_with_nonempty_eligible_job_pool": self.decisions_with_nonempty_eligible_job_pool,
            "decisions_with_known_proposed_next_job": self.decisions_with_known_proposed_next_job,
            "proposals_selected_for_commitment": self.proposals_selected_for_commitment,
            "reservations_committed": self.reservations_committed,
            "reservations_activated": self.reservations_activated,
            "reservations_cancelled": sum(self.cancellation_counts_by_reason.values()),
            "cancellation_counts_by_reason": dict(self.cancellation_counts_by_reason),
            "trainable_transition_count": self.trainable_transition_count,
            "proposal_availability_rate": _rate(self.decisions_with_known_proposed_next_job, self.decision_count),
            "reservation_success_rate": _rate(self.reservations_committed, self.decision_count),
            "activation_rate": _rate(self.reservations_activated, self.reservations_committed),
            "completed_cycle_rate": _rate(self.completed_paper_cycle_count, self.decision_count),
            "trainable_transition_rate": _rate(self.trainable_transition_count, self.decision_count),
            "mean_eligible_job_pool_size": mean_pool_size,
            "maximum_eligible_job_pool_size": self._eligible_pool_size_max,
            "mean_proposal_candidate_count": mean_candidate_count,
            "mean_proposal_build_ms": mean_proposal_ms,
            "mean_eligible_pool_build_ms": mean_eligible_pool_ms,
            "decision_timing_count": timing_count,
            "decision_timing_ms_summary": timing_summary,
            "mean_build_state_ms": mean_build_state,
            "mean_feature_bundle_ms": mean_feature_bundle,
            "mean_forward_ms": mean_forward,
            "mean_revalidation_ms": mean_revalidation,
            "mean_total_decision_ms": mean_total,
            "max_total_decision_ms": max_total,
        }


def summarize_rollout_events(events: Iterable[Mapping[str, Any]], policy_mode: str | None = None) -> dict[str, Any]:
    accumulator = RolloutSummaryAccumulator(policy_mode=policy_mode)
    for event in events:
        accumulator.add_event(event)
    return accumulator.to_summary()
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


def _int(value: Any) -> int | None:
    try:
        return int(value)
    except Exception:
        return None


def _rate(numerator: int, denominator: int) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _timing_summary(rows: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, float]]:
    accum: dict[str, dict[str, float]] = {}
    for row in rows:
        for key, value in dict(row).items():
            number = _float(value)
            if number is None:
                continue
            slot = accum.setdefault(str(key), {"count": 0.0, "total": 0.0, "max": float("-inf")})
            slot["count"] += 1.0
            slot["total"] += float(number)
            slot["max"] = max(slot["max"], float(number))
    return {
        key: {
            "mean": values["total"] / values["count"] if values["count"] else 0.0,
            "max": 0.0 if values["max"] == float("-inf") else values["max"],
        }
        for key, values in sorted(accum.items())
    }
