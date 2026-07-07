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
from src.rmfs.rl.rts.training.timebase import warehouse_horizon_seconds_to_netlogo_steps
from src.rmfs.runtime_io.run_profiles import TICK_TO_SECOND


def main(argv=None):
    parser = argparse.ArgumentParser(description="RTS on-policy PPO training controller.")
    parser.add_argument("--artifact-label", default=None, help="Optional artifact label. If omitted, one is auto-generated.")
    parser.add_argument("--output-root", default="data/runtime/rts_training", help="Training run output root directory.")
    parser.add_argument("--batches", type=int, required=True, help="Total number of PPO batches (end batch ID). With --resume-latest, training continues from the last checkpoint up to this batch.")
    parser.add_argument("--workers", type=int, required=True, help="Number of isolated worker rollouts (workers) inside a batch.")
    horizon = parser.add_mutually_exclusive_group(required=True)
    horizon.add_argument(
        "--simulated-seconds-per-run",
        "--simulated-seconds",
        "--warehouse-seconds-per-run",
        type=float,
        default=None,
        help="Requested simulated warehouse seconds for one worker run. Converted to backend steps using tick_to_second=0.15.",
    )
    horizon.add_argument("--netlogo-steps-per-run", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, default=42, help="Seed base for training.")
    parser.add_argument("--cycle-reference", default=None, help="Optional legacy cycle_reference.json path (compatibility only).")
    parser.add_argument("--initial-checkpoint-dir", default=None, help="Initial policy checkpoint directory.")
    parser.add_argument("--resume-latest", action="store_true", default=False, help="Resume training using the latest checkpoint of the run.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--worker-device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--policy-action-mode", choices=("sample", "greedy"), default="sample")
    parser.add_argument("--zone-ids", default=None, help="Comma-separated list of explicit zone IDs.")
    parser.add_argument("--robot-task-allocator", choices=("regret_k", "legacy_nearest"), default="legacy_nearest")
    parser.add_argument("--regret-k", type=int, default=2)
    parser.add_argument("--min-trainable-steps", type=int, default=1)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--entropy-coef", type=float, default=None, help="Entropy bonus coefficient (default 0.01).")
    parser.add_argument("--order-rate-per-hour", type=int, default=500, help="Orders per simulated hour for training demand generation.")
    parser.add_argument("--feature-ablation", default="full")
    parser.add_argument("--ledger-path", default=None)
    charging = parser.add_mutually_exclusive_group()
    charging.add_argument("--charging-enabled", action="store_const", const="enabled", dest="charging_mode")
    charging.add_argument("--charging-disabled", action="store_const", const="disabled", dest="charging_mode")
    charging.add_argument("--charging-inherit-default", action="store_const", const="inherit", dest="charging_mode")
    progress = parser.add_mutually_exclusive_group()
    progress.add_argument("--progress", action="store_true", dest="progress")
    progress.add_argument("--no-progress", action="store_false", dest="progress")
    parser.set_defaults(progress=True)
    tb = parser.add_mutually_exclusive_group()
    tb.add_argument("--tensorboard", action="store_true", dest="tensorboard_enabled")
    tb.add_argument("--no-tensorboard", action="store_false", dest="tensorboard_enabled")
    parser.set_defaults(tensorboard_enabled=True)
    parser.add_argument("--dry-run", action="store_true", default=False, help="Deprecated no-op flag. Dry run is default.")
    parser.add_argument("--execute", action="store_true", default=False, help="Run real training instead of a dry run.")
    parser.add_argument("--debug-worker-logs", action="store_true", default=False, help="Persist worker stdout/stderr logs for diagnosis.")
    parser.add_argument("--worker-spawn-delay", type=float, default=5.0, help="Seconds to wait between spawning each worker subprocess (default 5.0). Prevents memory spikes from simultaneous startup.")
    parser.add_argument("--rts-torch-threads", type=int, default=None)
    parser.add_argument("--rts-torch-interop-threads", type=int, default=None)
    parser.set_defaults(charging_mode="inherit")
    args = parser.parse_args(argv)

    if args.execute:
        if not args.initial_checkpoint_dir and not args.resume_latest:
            parser.error("real execution (--execute) requires either --initial-checkpoint-dir or --resume-latest")

    simulated_seconds_per_run = args.simulated_seconds_per_run
    if simulated_seconds_per_run is not None:
        netlogo_steps_per_run = warehouse_horizon_seconds_to_netlogo_steps(simulated_seconds_per_run, TICK_TO_SECOND)
    else:
        print(
            "warning: --netlogo-steps-per-run is deprecated for RTS training; "
            "use --simulated-seconds-per-run so the requested horizon is expressed in warehouse seconds.",
            file=sys.stderr,
        )
        netlogo_steps_per_run = args.netlogo_steps_per_run

    zone_ids = ()
    if args.zone_ids:
        zone_ids = tuple(z.strip() for z in args.zone_ids.split(",") if z.strip())

    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = REPO_ROOT / output_root
    config = RTSOnPolicyTrainingConfig(
        artifact_label=args.artifact_label,
        output_root=output_root,
        batches=args.batches,
        workers=args.workers,
        netlogo_steps_per_run=netlogo_steps_per_run,
        seed=args.seed,
        simulated_seconds_per_run=simulated_seconds_per_run,
        cycle_reference_path=Path(args.cycle_reference).resolve() if args.cycle_reference else None,
        device=args.device,
        worker_device=args.worker_device,
        policy_action_mode=args.policy_action_mode,
        progress=args.progress,
        tensorboard_enabled=args.tensorboard_enabled,
        min_trainable_steps=args.min_trainable_steps,
        ppo_epochs=args.ppo_epochs,
        minibatch_size=args.minibatch_size,
        learning_rate=args.learning_rate,
        entropy_coef=args.entropy_coef,
        order_rate_per_hour=args.order_rate_per_hour,
        zone_ids=zone_ids,
        feature_ablation=args.feature_ablation,
        ledger_path=Path(args.ledger_path).resolve() if args.ledger_path else None,
        charging_mode=args.charging_mode,
        robot_task_allocator=args.robot_task_allocator,
        regret_k=args.regret_k if args.robot_task_allocator == "regret_k" else None,
        debug_worker_logs=args.debug_worker_logs,
        worker_spawn_delay_seconds=args.worker_spawn_delay,
        rts_torch_threads=args.rts_torch_threads,
        rts_torch_interop_threads=args.rts_torch_interop_threads,
    )
    result = run_on_policy_training_controller(
        config=config,
        repo_root=REPO_ROOT,
        initial_checkpoint_dir=Path(args.initial_checkpoint_dir).resolve() if args.initial_checkpoint_dir else None,
        resume_latest=args.resume_latest,
        dry_run=not args.execute,
    )
    print(result["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
