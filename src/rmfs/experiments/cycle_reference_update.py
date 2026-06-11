"""Cycle-reference update proposal gate."""

from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Any, Mapping

from src.rmfs.experiments.identity import short_hash


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

