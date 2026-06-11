#!/usr/bin/env python3
"""RTS on-policy PPO controller CLI."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.rmfs.rl.rts.training.controller import run_on_policy_training_controller
from src.rmfs.rl.rts.training.on_policy_config import RTSOnPolicyTrainingConfig


def main(argv=None):
    parser = argparse.ArgumentParser(description="RTS on-policy PPO controller.")
    parser.add_argument("--artifact-label", required=True)
    parser.add_argument("--output-root", default="data/runtime/rts_training")
    parser.add_argument("--batches", type=int, required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--netlogo-steps-per-run", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cycle-reference", required=True)
    parser.add_argument("--initial-checkpoint-dir", default=None)
    parser.add_argument("--resume-latest", action="store_true", default=False)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--worker-device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--policy-action-mode", choices=("sample", "greedy"), default="sample")
    progress = parser.add_mutually_exclusive_group()
    progress.add_argument("--progress", action="store_true", dest="progress")
    progress.add_argument("--no-progress", action="store_false", dest="progress")
    parser.set_defaults(progress=None)
    tb = parser.add_mutually_exclusive_group()
    tb.add_argument("--tensorboard", action="store_true", dest="tensorboard_enabled")
    tb.add_argument("--no-tensorboard", action="store_false", dest="tensorboard_enabled")
    parser.set_defaults(tensorboard_enabled=False)
    parser.add_argument("--dry-run", action="store_true", default=False)
    args = parser.parse_args(argv)

    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = REPO_ROOT / output_root
    config = RTSOnPolicyTrainingConfig(
        artifact_label=args.artifact_label,
        output_root=output_root,
        batches=args.batches,
        workers=args.workers,
        netlogo_steps_per_run=args.netlogo_steps_per_run,
        seed=args.seed,
        cycle_reference_path=Path(args.cycle_reference).resolve(),
        device=args.device,
        worker_device=args.worker_device,
        policy_action_mode=args.policy_action_mode,
        progress=args.progress,
        tensorboard_enabled=args.tensorboard_enabled,
    )
    result = run_on_policy_training_controller(
        config=config,
        repo_root=REPO_ROOT,
        initial_checkpoint_dir=Path(args.initial_checkpoint_dir).resolve() if args.initial_checkpoint_dir else None,
        resume_latest=args.resume_latest,
        dry_run=args.dry_run,
    )
    print(result["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

