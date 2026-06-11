"""Cycle-reference update proposal gate."""

from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Any, Mapping

from src.rmfs.experiments.identity import short_hash


def validate_eval_summary_for_cycle_proposal(summary: Mapping[str, Any], *, max_failed_replications: int = 0) -> None:
    """Validate that the evaluation summary is complete and successful."""
    valid = summary.get("valid")
    status = summary.get("status")

    if valid is not True and not status:
        raise ValueError("evaluation summary is missing both 'valid' flag and 'status'")

    if valid is True:
        # Accept if explicitly valid=True, check failed replications if present
        failed_reps = summary.get("failed_replications")
        if failed_reps is not None and failed_reps > max_failed_replications:
            raise ValueError(f"failed replications ({failed_reps}) exceeds maximum allowed ({max_failed_replications})")
        return

    # Check status
    if status in {"dry_run", "failed", "partial", "skipped"}:
        raise ValueError(f"evaluation status is '{status}' which is rejected for cycle proposals")

    if status not in {"success", "completed"}:
        raise ValueError(f"invalid evaluation status '{status}', must be 'success' or 'completed'")

    failed_reps = summary.get("failed_replications")
    if failed_reps is not None and failed_reps > max_failed_replications:
        raise ValueError(f"failed replications ({failed_reps}) exceeds maximum allowed ({max_failed_replications})")


def build_cycle_reference_update_proposal(
    *,
    source_eval_run_id: str,
    source_checkpoint_id: str,
    current_reference: Mapping[str, Any],
    candidate_reference: Mapping[str, Any],
) -> dict[str, Any]:
    payload = {
        "source_eval_run_id": source_eval_run_id,
        "source_checkpoint_id": source_checkpoint_id,
        "candidate_reference": dict(candidate_reference),
    }
    return {
        "schema_version": "cycle_reference_update_proposal.v1",
        "proposal_id": f"cycle_prop_{short_hash(payload)}",
        "source_eval_run_id": source_eval_run_id,
        "source_checkpoint_id": source_checkpoint_id,
        "current_reference": dict(current_reference),
        "candidate_reference": dict(candidate_reference),
        "decision": "proposed",
        "requires_manual_approval": True,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }


def write_cycle_reference_update_proposal(path: Path, proposal: Mapping[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w") as fh:
        json.dump(dict(proposal), fh, indent=2)
