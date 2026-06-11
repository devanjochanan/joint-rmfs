#!/usr/bin/env python3
"""Create dry-run evaluation specs for an RTS checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.rmfs.experiments.evaluation.controller import write_eval_dry_run


def main(argv=None):
    parser = argparse.ArgumentParser(description="Dry-run RTS checkpoint evaluation.")
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--policy-checkpoint-id", required=True)
    parser.add_argument("--zone-ids", required=True)
    parser.add_argument("--seed-pack", required=True)
    parser.add_argument("--output-root", default="data/runtime/rts_evaluation")
    parser.add_argument("--policy-action-mode", choices=("greedy", "sample"), default="greedy")
    parser.add_argument("--dry-run", action="store_true", default=True)
    args = parser.parse_args(argv)
    summary = write_eval_dry_run(
        checkpoint_dir=Path(args.checkpoint_dir),
        policy_checkpoint_id=args.policy_checkpoint_id,
        zone_ids=tuple(zone.strip() for zone in args.zone_ids.split(",") if zone.strip()),
        seed_pack_path=Path(args.seed_pack),
        output_root=Path(args.output_root),
        policy_action_mode=args.policy_action_mode,
    )
    print(summary["eval_run_id"])


if __name__ == "__main__":
    raise SystemExit(main())

