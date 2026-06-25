#!/usr/bin/env python3
"""Smoke test cycle-reference proposal gate."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.rmfs.experiments.cycle_reference_update import (
    build_cycle_reference_update_proposal,
    write_cycle_reference_update_proposal,
    validate_eval_summary_for_cycle_proposal,
)


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        json.dump(payload, fh)


def main():
    root = REPO_ROOT / "data" / "runtime" / "phase10_cycle_prop_smoke"
    shutil.rmtree(root, ignore_errors=True)

    current = {"reference_overall_cycle_time": 10.0}
    candidate = {"reference_overall_cycle_time": 9.0}

    current_path = root / "cycle_reference.json"
    write_json(current_path, current)

    # 1. Test valid completed evaluation summary -> proposal written
    valid_summary = {"status": "completed", "failed_replications": 0}
    validate_eval_summary_for_cycle_proposal(valid_summary)  # Should not raise

    valid_explicit = {"valid": True}
    validate_eval_summary_for_cycle_proposal(valid_explicit)  # Should not raise

    proposal = build_cycle_reference_update_proposal(
        source_eval_run_id="eval_1",
        source_checkpoint_id="batch_1",
        current_reference=current,
        candidate_reference=candidate,
    )
    path = root / "cycle_reference_update_proposal.json"
    write_cycle_reference_update_proposal(path, proposal)

    assert json.load(path.open())["requires_manual_approval"] is True
    # Ensure original reference remains unchanged
    assert json.load(current_path.open()) == current

    # 2. Test failed evaluation summary -> raises
    failed_summary = {"status": "failed", "failed_replications": 1}
    try:
        validate_eval_summary_for_cycle_proposal(failed_summary)
        assert False, "Expected ValueError for failed status"
    except ValueError:
        pass

    # 3. Test dry_run evaluation summary -> raises
    dry_run_summary = {"status": "dry_run", "failed_replications": 0}
    try:
        validate_eval_summary_for_cycle_proposal(dry_run_summary)
        assert False, "Expected ValueError for dry_run status"
    except ValueError:
        pass

    # 4. Test failed replications exceeded limit -> raises
    failed_rep_summary = {"status": "completed", "failed_replications": 2}
    try:
        validate_eval_summary_for_cycle_proposal(failed_rep_summary, max_failed_replications=1)
        assert False, "Expected ValueError for failed replications exceeding limit"
    except ValueError:
        pass

    shutil.rmtree(root)
    print("cycle reference update proposal smoke ok")


if __name__ == "__main__":
    main()
