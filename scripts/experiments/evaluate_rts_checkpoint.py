#!/usr/bin/env python3
"""Evaluate or dry-run an RTS checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.rmfs.experiments.evaluation.controller import run_rts_evaluation


def main(argv=None):
    parser = argparse.ArgumentParser(description="RTS checkpoint evaluation.")
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--policy-mode", choices=("current", "random_valid", "rts_rl_explicit"), default="rts_rl_explicit")
    parser.add_argument("--zone-ids", required=True)
    parser.add_argument("--seed-pack", default=None)
    parser.add_argument("--output-root", default="data/runtime/rts_evaluation")
    parser.add_argument("--policy-action-mode", choices=("greedy", "sample"), default="greedy")
    parser.add_argument("--feature-ablation", default="full")
    parser.add_argument("--state-capture-mode", choices=("auto", "full", "minimal"), default="auto")
    parser.add_argument("--min-completed-cycles", type=int, default=1)
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--ledger-path", default=None)
    parser.add_argument("--execute", action="store_true", default=False)
    parser.add_argument("--dry-run", action="store_true", default=False)
    charging = parser.add_mutually_exclusive_group()
    charging.add_argument("--charging-enabled", action="store_const", const="enabled", dest="charging_mode")
    charging.add_argument("--charging-disabled", action="store_const", const="disabled", dest="charging_mode")
    charging.add_argument("--charging-inherit-default", action="store_const", const="inherit", dest="charging_mode")
    parser.add_argument("--rts-torch-threads", type=int, default=None)
    parser.add_argument("--rts-torch-interop-threads", type=int, default=None)
    parser.add_argument("--debug-rollouts", action="store_true", default=False,
                        help="Persist per-decision rollout JSONL and worker stdout (adds ~300 MB/worker). Default: off.")
    
    # Paired comparison arguments
    parser.add_argument("--paired-current-vs-rl", action="store_true", default=False)
    parser.add_argument("--replications", type=int, default=60)
    parser.add_argument("--seed-base", type=int, default=42)
    parser.add_argument("--simulated-seconds", type=float, default=10000.0)
    parser.add_argument("--robot-count", type=int, default=20)
    parser.add_argument("--order-rate", type=int, default=500)
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--progress", action="store_true", default=False)

    parser.set_defaults(charging_mode="inherit")
    args = parser.parse_args(argv)

    if args.paired_current_vs_rl:
        from src.rmfs.experiments.evaluation.controller import run_rts_paired_evaluation
        summary = run_rts_paired_evaluation(
            repo_root=REPO_ROOT,
            checkpoint_dir=Path(args.checkpoint_dir).resolve() if args.checkpoint_dir else None,
            zone_ids=tuple(zone.strip() for zone in args.zone_ids.split(",") if zone.strip()),
            output_root=Path(args.output_root),
            replications=args.replications,
            seed_base=args.seed_base,
            simulated_seconds=args.simulated_seconds,
            robot_count=args.robot_count,
            order_rate=args.order_rate,
            charging_enabled=(args.charging_mode == "enabled"),
            max_workers=args.max_workers,
            resume=args.resume,
            progress=args.progress,
            dry_run=not args.execute or args.dry_run,
            rts_torch_threads=args.rts_torch_threads,
            rts_torch_interop_threads=args.rts_torch_interop_threads,
        )
    else:
        if not args.seed_pack:
            parser.error("--seed-pack is required when not in paired mode")
        summary = run_rts_evaluation(
            repo_root=REPO_ROOT,
            checkpoint_dir=Path(args.checkpoint_dir).resolve() if args.checkpoint_dir else None,
            policy_mode=args.policy_mode,
            zone_ids=tuple(zone.strip() for zone in args.zone_ids.split(",") if zone.strip()),
            seed_pack_path=Path(args.seed_pack),
            output_root=Path(args.output_root),
            policy_action_mode=args.policy_action_mode,
            feature_ablation=args.feature_ablation,
            state_capture_mode=args.state_capture_mode,
            charging_mode=args.charging_mode,
            dry_run=not args.execute or args.dry_run,
            min_completed_cycles=args.min_completed_cycles,
            max_workers=args.max_workers,
            ledger_path=Path(args.ledger_path).resolve() if args.ledger_path else None,
            rts_torch_threads=args.rts_torch_threads,
            rts_torch_interop_threads=args.rts_torch_interop_threads,
            debug_rollouts=args.debug_rollouts,
        )
    print(summary["eval_run_id"])


if __name__ == "__main__":
    raise SystemExit(main())

