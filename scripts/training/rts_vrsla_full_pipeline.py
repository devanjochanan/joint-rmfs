#!/usr/bin/env python3
"""Combined VRSLA teacher pretraining + PPO pipeline.

Runs two phases back-to-back, each using the same tqdm progress bar
from the orchestration layer (run_specs):
  Phase 1: VRSLA teacher data collection (10 batches) + behavior cloning
  Phase 2: PPO fine-tuning from the BC checkpoint (30 batches)

Usage:
  cd ~/Project\ Ta/Fresh\ Start\ Structure\ V1/Rika\'s\ Version
  source ~/torch-gpu/bin/activate
  python scripts/training/rts_vrsla_full_pipeline.py --execute
"""

from __future__ import annotations

import argparse
import datetime
import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def fmt_duration(seconds: float) -> str:
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    return f"{m}m {s:02d}s"


def run_phase(*, label: str, cmd: list[str]) -> float:
    border = "=" * 64
    print(f"\n{border}")
    print(f"  PHASE: {label}")
    print(f"  Started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{border}\n")

    start = time.time()
    proc = subprocess.Popen(cmd, cwd=str(REPO_ROOT))
    proc.wait()
    elapsed = time.time() - start

    if proc.returncode != 0:
        print(f"\n  FAILED after {fmt_duration(elapsed)} (exit code {proc.returncode})")
        sys.exit(1)

    print(f"\n  Completed in {fmt_duration(elapsed)}")
    return elapsed


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="VRSLA pretrain + PPO full pipeline.")
    parser.add_argument("--teacher-batches", type=int, default=10)
    parser.add_argument("--teacher-simulated-seconds", type=float, default=10000)
    parser.add_argument("--ppo-batches", type=int, default=30)
    parser.add_argument("--ppo-simulated-seconds", type=float, default=10000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--entropy-coef", type=float, default=0.05)
    parser.add_argument("--teacher-label", default="dewa_rts_v7_vrsla_teacher_10000s_8w_10b_seed42")
    parser.add_argument("--ppo-label", default="dewa_rts_v7_vrsla_ppo_10000s_8w_30b_lr3e4_ent05_seed42")
    parser.add_argument("--execute", action="store_true", default=False)
    args = parser.parse_args(argv)

    train_root = REPO_ROOT / "data" / "runtime" / "rts_training"
    teacher_dir = train_root / args.teacher_label
    python = sys.executable

    min_per_10k = 27.0
    teacher_est = args.teacher_batches * min_per_10k * (args.teacher_simulated_seconds / 10000.0)
    ppo_est = args.ppo_batches * min_per_10k * (args.ppo_simulated_seconds / 10000.0)
    total_est = teacher_est + ppo_est + 1
    eta = datetime.datetime.now() + datetime.timedelta(minutes=total_est)

    print(f"\n{'#' * 64}")
    print(f"  VRSLA PRETRAIN + PPO PIPELINE")
    print(f"{'#' * 64}")
    print(f"  Teacher:  {args.teacher_batches} batches x {args.teacher_simulated_seconds:.0f}s x {args.workers}w")
    print(f"  PPO:      {args.ppo_batches} batches x {args.ppo_simulated_seconds:.0f}s x {args.workers}w")
    print(f"  PPO LR:   {args.learning_rate}  |  entropy: {args.entropy_coef}")
    print(f"  Seed:     {args.seed}")
    print(f"")
    print(f"  ETA:  Phase 1 ~{teacher_est:.0f} min | Phase 2 ~{ppo_est:.0f} min | Total ~{total_est:.0f} min ({total_est/60:.1f} hr)")
    print(f"  Finish at: ~{eta.strftime('%Y-%m-%d %H:%M')}")
    print()

    if not args.execute:
        print("  DRY RUN — pass --execute to start.\n")
        return 0

    pipeline_start = time.time()

    # --- Phase 1: VRSLA teacher data collection + behavior cloning ---
    phase1_time = run_phase(
        label=f"VRSLA Teacher ({args.teacher_batches} batches) + Behavior Cloning",
        cmd=[
            python, "scripts/training/rts_vrsla_pretrain.py",
            "--artifact-label", args.teacher_label,
            "--teacher-batches", str(args.teacher_batches),
            "--workers", str(args.workers),
            "--simulated-seconds", str(args.teacher_simulated_seconds),
            "--seed", str(args.seed),
            "--execute",
        ],
    )

    # Find BC checkpoint
    checkpoint_dir = None
    summary_path = teacher_dir / "controller_summary.json"
    if summary_path.exists():
        with summary_path.open() as fh:
            summary = json.load(fh)
        checkpoint_dir = summary.get("checkpoint_dir")
        print(f"\n  BC checkpoint: {checkpoint_dir}")
        bc = summary.get("behavior_cloning", {})
        if bc:
            print(f"  BC result: {bc.get('epochs_completed')} epochs, "
                  f"action agreement {bc.get('overall_action_agreement', 0):.1%}, "
                  f"zone agreement {bc.get('zone_agreement', 0):.1%}")

    if checkpoint_dir is None or not Path(checkpoint_dir).exists():
        fallback = teacher_dir / f"batch_{args.teacher_batches:06d}" / "checkpoint"
        if fallback.exists():
            checkpoint_dir = str(fallback)
        else:
            print(f"\n  ERROR: Cannot find BC checkpoint at {fallback}")
            return 1

    # --- Phase 2: PPO from BC checkpoint ---
    phase2_time = run_phase(
        label=f"PPO Training ({args.ppo_batches} batches, LR={args.learning_rate}, entropy={args.entropy_coef})",
        cmd=[
            python, "scripts/training/rts_train_controller.py",
            "--artifact-label", args.ppo_label,
            "--batches", str(args.ppo_batches),
            "--workers", str(args.workers),
            "--simulated-seconds", str(args.ppo_simulated_seconds),
            "--seed", str(args.seed),
            "--learning-rate", str(args.learning_rate),
            "--entropy-coef", str(args.entropy_coef),
            "--initial-checkpoint-dir", str(checkpoint_dir),
            "--execute",
            "--progress",
        ],
    )

    total_time = time.time() - pipeline_start
    ppo_dir = train_root / args.ppo_label
    print(f"\n{'#' * 64}")
    print(f"  PIPELINE COMPLETE")
    print(f"  Phase 1 (teacher + BC): {fmt_duration(phase1_time)}")
    print(f"  Phase 2 (PPO):          {fmt_duration(phase2_time)}")
    print(f"  Total wall time:        {fmt_duration(total_time)}")
    print(f"  Finished:               {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Teacher output:  {teacher_dir}")
    print(f"  PPO output:      {ppo_dir}")
    print(f"  Final checkpoint: {ppo_dir / f'batch_{args.ppo_batches:06d}' / 'checkpoint'}")
    print(f"{'#' * 64}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
