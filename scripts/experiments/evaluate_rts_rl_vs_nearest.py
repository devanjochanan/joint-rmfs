#!/usr/bin/env python3
"""Run a balanced local RTS--RL versus Nearest evaluation campaign."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.rmfs.experiments.evaluation.paired_campaign import run_paired_rts_rl_vs_nearest_evaluation


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Run seed-matched RTS--RL versus Nearest in balanced local waves."
    )
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--zone-ids", required=True)
    parser.add_argument("--seed-pack", required=True)
    parser.add_argument("--output-root", default="data/runtime/rts_rl_vs_nearest_evaluation")
    parser.add_argument("--policy-action-mode", choices=("greedy", "sample"), default="greedy")
    parser.add_argument("--feature-ablation", default="full")
    parser.add_argument("--state-capture-mode", choices=("auto", "full", "minimal"), default="auto")
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--machine-id", default="local")
    parser.add_argument(
        "--resume-campaign-id",
        default=None,
        help="Resume this exact paired campaign after checking its scientific identity; worker/thread limits may change.",
    )
    parser.add_argument(
        "--pairs-per-wave",
        type=int,
        default=None,
        help="Matched pairs admitted together before the next wave; default is max-workers / 2.",
    )
    parser.add_argument("--rts-torch-threads", type=int, default=None)
    parser.add_argument("--rts-torch-interop-threads", type=int, default=None)
    parser.add_argument("--execute", action="store_true", default=False)
    parser.add_argument("--dry-run", action="store_true", default=False)
    charging = parser.add_mutually_exclusive_group()
    charging.add_argument("--charging-enabled", action="store_const", const="enabled", dest="charging_mode")
    charging.add_argument("--charging-disabled", action="store_const", const="disabled", dest="charging_mode")
    charging.add_argument("--charging-inherit-default", action="store_const", const="inherit", dest="charging_mode")
    parser.set_defaults(charging_mode="inherit")
    args = parser.parse_args(argv)

    summary = run_paired_rts_rl_vs_nearest_evaluation(
        repo_root=REPO_ROOT,
        checkpoint_dir=Path(args.checkpoint_dir).resolve(),
        zone_ids=tuple(zone.strip() for zone in args.zone_ids.split(",") if zone.strip()),
        seed_pack_path=Path(args.seed_pack),
        output_root=Path(args.output_root),
        policy_action_mode=args.policy_action_mode,
        feature_ablation=args.feature_ablation,
        charging_mode=args.charging_mode,
        dry_run=not args.execute or args.dry_run,
        rts_torch_threads=args.rts_torch_threads,
        rts_torch_interop_threads=args.rts_torch_interop_threads,
        state_capture_mode=args.state_capture_mode,
        max_workers=args.max_workers,
        pairs_per_wave=args.pairs_per_wave,
        machine_id=args.machine_id,
        resume_campaign_id=args.resume_campaign_id,
    )
    print(summary["campaign_id"])
    return 0 if summary.get("valid") or summary.get("status") == "dry_run" else 1


if __name__ == "__main__":
    raise SystemExit(main())
