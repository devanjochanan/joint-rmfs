#!/usr/bin/env python3
"""CLI wrapper for the local isolated executor smoke."""

import argparse
from pathlib import Path
import sys


def main():
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))

    parser = argparse.ArgumentParser(description="Run local isolated RMFS executor smoke.")
    parser.add_argument("--runs", type=int, default=4)
    parser.add_argument("--ticks", type=int, default=3)
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--debug-trace", action="store_true", default=False)
    parser.add_argument("--trace-cadence", type=int, default=1000)
    parser.add_argument("--trace-first-n", type=int, default=0)
    parser.add_argument("--snapshot-inputs", action="store_true", default=False)
    parser.add_argument("--rts-policy-mode", choices=("current", "current_probe", "random_valid"), default="current")
    parser.add_argument("--rts-rollout", action="store_true", default=False)
    parser.add_argument("--rts-zone-ids", default=None)
    parser.add_argument("--rts-reward-reference", default=None)
    parser.add_argument("--rts-random-seed", type=int, default=None)
    parser.add_argument("--rts-max-events", type=int, default=None)
    args = parser.parse_args()

    if args.runs < 1:
        parser.error("--runs must be >= 1")
    if args.ticks < 0:
        parser.error("--ticks must be >= 0")
    if args.max_workers < 1:
        parser.error("--max-workers must be >= 1")
    if args.trace_cadence < 0:
        parser.error("--trace-cadence must be >= 0")
    if args.trace_first_n < 0:
        parser.error("--trace-first-n must be >= 0")
    if args.rts_max_events is not None and args.rts_max_events <= 0:
        parser.error("--rts-max-events must be positive")
    if args.rts_policy_mode in {"current_probe", "random_valid"} and not args.rts_rollout:
        parser.error("--rts-policy-mode current_probe/random_valid requires --rts-rollout")
    rts_zone_ids = None
    if args.rts_zone_ids:
        rts_zone_ids = [zone.strip() for zone in args.rts_zone_ids.split(",") if zone.strip()]

    from src.rmfs.orchestration.local_executor import run_controller

    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = repo_root / output_root

    run_controller(
        repo_root=repo_root,
        output_root=output_root.resolve(),
        runs=args.runs,
        ticks=args.ticks,
        max_workers=args.max_workers,
        python_executable=sys.executable,
        debug_trace=args.debug_trace,
        trace_cadence=args.trace_cadence,
        trace_first_n=args.trace_first_n,
        snapshot_inputs=args.snapshot_inputs,
        rts_policy_mode=args.rts_policy_mode,
        rts_rollout_enabled=args.rts_rollout,
        rts_zone_ids=rts_zone_ids,
        rts_reward_reference_path=args.rts_reward_reference,
        rts_random_seed=args.rts_random_seed,
        rts_max_events=args.rts_max_events,
    )


if __name__ == "__main__":
    main()
