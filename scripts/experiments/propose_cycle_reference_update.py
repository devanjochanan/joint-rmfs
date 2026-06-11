#!/usr/bin/env python3
"""Write a cycle-reference update proposal."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.rmfs.experiments.cycle_reference_update import build_cycle_reference_update_proposal, write_cycle_reference_update_proposal


def main(argv=None):
    parser = argparse.ArgumentParser(description="Propose cycle-reference update.")
    parser.add_argument("--current-reference", required=True)
    parser.add_argument("--candidate-reference", required=True)
    parser.add_argument("--source-eval-run-id", required=True)
    parser.add_argument("--source-checkpoint-id", required=True)
    parser.add_argument("--output", default="cycle_reference_update_proposal.json")
    args = parser.parse_args(argv)
    current = json.load(Path(args.current_reference).open())
    candidate = json.load(Path(args.candidate_reference).open())
    proposal = build_cycle_reference_update_proposal(
        source_eval_run_id=args.source_eval_run_id,
        source_checkpoint_id=args.source_checkpoint_id,
        current_reference=current,
        candidate_reference=candidate,
    )
    write_cycle_reference_update_proposal(Path(args.output), proposal)
    print(proposal["proposal_id"])


if __name__ == "__main__":
    raise SystemExit(main())

