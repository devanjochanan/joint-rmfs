"""Aggregation helpers for RTS rollout events."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from .action_space import action_mask_entry
from .rollout_schema import DECISION_EVENT, OUTCOME_EVENT


SUMMARY_SCHEMA_VERSION = "rts_rollout_summary.v1"


def summarize_rollout_events(events: Iterable[Mapping[str, Any]], policy_mode: str | None = None) -> dict[str, Any]:
    rows = [dict(event) for event in events]
    decisions = [row for row in rows if row.get("event_type") == DECISION_EVENT]
    outcomes = [row for row in rows if row.get("event_type") == OUTCOME_EVENT]
    outcome_ids = {row.get("decision_event_id") for row in outcomes}
    realized = [_float(row.get("realized_cycle_time")) for row in outcomes]
    realized = [value for value in realized if value is not None]
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
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "policy_mode": detected_mode or "unknown",
        "decision_count": len(decisions),
        "outcome_count": len(outcomes),
        "orphan_count": sum(1 for row in decisions if row.get("decision_event_id") not in outcome_ids),
        "reward_computed_count": reward_count,
        "avg_realized_cycle_time": (sum(realized) / len(realized)) if realized else None,
        "selected_action_counts": counts,
        "invalid_action_selected_count": invalid,
    }


def write_rollout_summary(path: Path, events_or_summary: Iterable[Mapping[str, Any]] | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(events_or_summary, Mapping):
        summary = dict(events_or_summary)
    else:
        summary = summarize_rollout_events(events_or_summary)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w") as fh:
        json.dump(summary, fh, indent=2)
    return summary


def _float(value: Any) -> float | None:
    try:
        return float(value)
    except Exception:
        return None

