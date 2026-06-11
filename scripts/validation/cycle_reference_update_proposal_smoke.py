#!/usr/bin/env python3
"""Smoke test cycle-reference proposal gate."""

from __future__ import annotations

from pathlib import Path
import json
import shutil
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.rmfs.experiments.cycle_reference_update import build_cycle_reference_update_proposal, write_cycle_reference_update_proposal


def main():
    root = REPO_ROOT / "data" / "runtime" / "phase10_cycle_prop_smoke"
    shutil.rmtree(root, ignore_errors=True)
    current = {"reference_overall_cycle_time": 10.0}
    candidate = {"reference_overall_cycle_time": 9.0}
    current_path = root / "cycle_reference.json"
    current_path.parent.mkdir(parents=True, exist_ok=True)
    current_path.write_text(json.dumps(current))
    proposal = build_cycle_reference_update_proposal(source_eval_run_id="eval_1", source_checkpoint_id="batch_1", current_reference=current, candidate_reference=candidate)
    path = root / "cycle_reference_update_proposal.json"
    write_cycle_reference_update_proposal(path, proposal)
    assert json.load(path.open())["requires_manual_approval"] is True
    assert json.load(current_path.open()) == current
    shutil.rmtree(root)
    print("cycle reference update proposal smoke ok")


if __name__ == "__main__":
    main()

