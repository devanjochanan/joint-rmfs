#!/usr/bin/env python3
"""Collect VRSLA teacher rows and behavior-clone an RTS actor checkpoint."""

from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path
import subprocess
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.rmfs.decisions.task_allocation import scheduler_metadata
from src.rmfs.experiments.feature_flags import default_rika_rts_rl_feature_flags
from src.rmfs.experiments.identity import make_experiment_run_id, make_scenario_id
from src.rmfs.orchestration.local_executor import git_value, run_specs
from src.rmfs.orchestration.run_spec import RunSpec
from src.rmfs.rl.rts.training.config import RTSTrainingConfig
from src.rmfs.rl.rts.training.device import resolve_rts_torch_device
from src.rmfs.rl.rts.training.metrics import atomic_write_json
from src.rmfs.rl.rts.training.seeding import derive_worker_seed
from src.rmfs.rl.rts.training.timebase import warehouse_horizon_seconds_to_netlogo_steps
from src.rmfs.rl.rts.training.vrsla_pretrain import (
    load_teacher_rows,
    run_behavior_cloning,
    save_behavior_cloning_checkpoint,
    teacher_batch_from_rows,
)
from src.rmfs.runtime_io.run_profiles import TICK_TO_SECOND


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="RTS VRSLA teacher behavior-cloning pretraining.")
    parser.add_argument("--artifact-label", default=None)
    parser.add_argument("--output-root", default="data/runtime/rts_training")
    parser.add_argument("--teacher-batches", type=int, default=10, help="Teacher batch-equivalents to collect.")
    parser.add_argument("--workers", type=int, required=True)
    horizon = parser.add_mutually_exclusive_group(required=True)
    horizon.add_argument("--simulated-seconds-per-run", "--simulated-seconds", type=float, default=None)
    horizon.add_argument("--netlogo-steps-per-run", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--worker-device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--zone-ids", default=None)
    parser.add_argument("--robot-task-allocator", choices=("regret_k", "legacy_nearest"), default="legacy_nearest")
    parser.add_argument("--regret-k", type=int, default=2)
    parser.add_argument("--order-rate-per-hour", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--bc-epochs", type=int, default=50)
    parser.add_argument("--bc-patience", type=int, default=5)
    parser.add_argument("--worker-spawn-delay", type=float, default=5.0)
    parser.add_argument("--execute", action="store_true", default=False)
    args = parser.parse_args(argv)

    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = REPO_ROOT / output_root
    if args.simulated_seconds_per_run is not None:
        netlogo_steps = warehouse_horizon_seconds_to_netlogo_steps(args.simulated_seconds_per_run, TICK_TO_SECOND)
        simulated_seconds = float(args.simulated_seconds_per_run)
    else:
        netlogo_steps = int(args.netlogo_steps_per_run)
        simulated_seconds = float(netlogo_steps) * TICK_TO_SECOND

    branch = git_value(REPO_ROOT, "rev-parse", "--abbrev-ref", "HEAD")
    commit = git_value(REPO_ROOT, "rev-parse", "HEAD")
    timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    artifact_label = args.artifact_label or f"dewa_rts_vrsla_bc_{timestamp}_{(commit or 'nogit')[:7]}"
    run_root = output_root / artifact_label
    zone_ids = tuple(z.strip() for z in (args.zone_ids or "").split(",") if z.strip()) or ("auto",)
    scheduler = scheduler_metadata(
        robot_task_allocator=args.robot_task_allocator,
        regret_k=args.regret_k if args.robot_task_allocator == "regret_k" else None,
        committed_next_reservations_enabled=True,
    )
    feature_flags = default_rika_rts_rl_feature_flags()
    scenario_id = make_scenario_id(
        {
            "teacher_policy": "vrsla_event_driven",
            "teacher_batches": int(args.teacher_batches),
            "workers": int(args.workers),
            "netlogo_steps_per_run": int(netlogo_steps),
            "order_rate_per_hour": int(args.order_rate_per_hour),
            "zone_ids": list(zone_ids),
            "feature_flags": feature_flags,
            **scheduler,
        }
    )
    experiment_id = make_experiment_run_id(
        {
            "artifact_label": artifact_label,
            "seed": int(args.seed),
            "scenario_id": scenario_id,
            "repo_commit": commit,
            "netlogo_steps_per_run": int(netlogo_steps),
            "workers": int(args.workers),
            "teacher_batches": int(args.teacher_batches),
        }
    )
    config_summary = {
        "artifact_label": artifact_label,
        "output_root": str(output_root),
        "run_root": str(run_root),
        "teacher_policy": "vrsla_event_driven",
        "teacher_batch_count": int(args.teacher_batches),
        "workers": int(args.workers),
        "seed": int(args.seed),
        "netlogo_steps_per_run": int(netlogo_steps),
        "simulated_seconds_per_run": simulated_seconds,
        "order_rate_per_hour": int(args.order_rate_per_hour),
        "zone_ids": list(zone_ids),
        "branch": branch,
        "commit": commit,
        "experiment_id": experiment_id,
        "scenario_id": scenario_id,
        "execute": bool(args.execute),
        "initialization_method": "vrsla_behavior_cloning",
        "pod_velocity_semantics": "distinct_order_union",
        "slot_score_semantics": "directed_loaded_cycle_distance",
        "replenishment_teacher_rule": "seeded_50_50_when_both_masked_valid",
        "critic_pretrained": False,
        **scheduler,
    }
    run_root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(run_root / "vrsla_pretrain_config.json", config_summary)

    if not args.execute:
        atomic_write_json(run_root / "controller_summary.json", {"status": "dry_run", **config_summary})
        print("dry_run")
        print(run_root)
        return 0

    for batch_id in range(1, int(args.teacher_batches) + 1):
        batch_dir = run_root / f"batch_{batch_id:06d}"
        batch_summary_path = batch_dir / "batch_summary.json"
        if batch_summary_path.exists():
            with batch_summary_path.open() as fh:
                bs = json.load(fh)
            if bs.get("status") == "teacher_collected":
                print(f"\n[teacher batch {batch_id}/{args.teacher_batches}] already completed — skipping")
                continue
        workers_dir = batch_dir / "workers"
        workers_dir.mkdir(parents=True, exist_ok=True)
        specs = []
        for worker_index in range(int(args.workers)):
            run_id = f"run_{worker_index + 1:03d}"
            worker_root = workers_dir / run_id
            worker_root.mkdir(parents=True, exist_ok=True)
            spec = RunSpec(
                run_id=run_id,
                ticks=int(netlogo_steps),
                runtime_root=worker_root,
                repo_root=REPO_ROOT,
                branch=branch,
                commit=commit,
                python_executable=sys.executable,
                timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                rts_policy_mode="vrsla_teacher",
                rts_rollout_enabled=True,
                rts_seed_base=int(args.seed),
                rts_random_seed=derive_worker_seed(int(args.seed), batch_id, worker_index),
                rts_zone_ids=list(zone_ids),
                rts_state_capture_mode="full",
                experiment_id=experiment_id,
                scenario_id=scenario_id,
                artifact_label=artifact_label,
                batch_id=batch_id,
                worker_id=worker_index + 1,
                run_profile="training",
                run_horizon_ticks=int(netlogo_steps),
                demand_horizon_ticks=int(netlogo_steps) + 1000,
                demand_buffer_ticks=1000,
                order_generation_mode="shuffled_historical_cycle",
                full_raw_order_replay=False,
                order_rate_per_hour=int(args.order_rate_per_hour),
                pod_location_mode="randomize_slots",
                pod_location_seed=derive_worker_seed(int(args.seed), batch_id, worker_index),
                **scheduler,
            )
            atomic_write_json(worker_root / "run_spec.json", spec.to_json_dict())
            specs.append(spec)
        atomic_write_json(batch_dir / "worker_specs.json", [spec.to_json_dict() for spec in specs])
        print(f"\n[teacher batch {batch_id}/{args.teacher_batches}]")
        results = run_specs(specs, max_workers=int(args.workers), progress=True)
        failures = [r for r in results if r.get("return_code", 0) != 0]
        if failures:
            raise RuntimeError(f"VRSLA teacher workers failed: {[(r['spec'].run_id, r['return_code']) for r in failures]}")
        atomic_write_json(batch_dir / "batch_summary.json", {"status": "teacher_collected", "batch_id": batch_id})

    rows = load_teacher_rows(run_root.glob("batch_*/workers/run_*/rts_vrsla_teacher.jsonl"))
    teacher_batch = teacher_batch_from_rows(rows)
    train_config = RTSTrainingConfig(
        artifact_label=artifact_label,
        output_root=output_root,
        seed=int(args.seed),
        learning_rate=float(args.learning_rate),
        zone_ids=tuple(z for z in zone_ids if z != "auto"),
        tensorboard_enabled=False,
    )
    model, result = run_behavior_cloning(
        teacher_batch,
        config=train_config,
        device=resolve_rts_torch_device(args.device),
        max_epochs=int(args.bc_epochs),
        patience=int(args.bc_patience),
    )
    checkpoint_dir = save_behavior_cloning_checkpoint(
        model=model,
        config=train_config,
        batch_id=int(args.teacher_batches),
        teacher_batch=teacher_batch,
        result=result,
        teacher_batch_count=int(args.teacher_batches),
    )
    summary = {
        "status": "completed",
        **config_summary,
        "teacher_row_count": len(rows),
        "checkpoint_dir": str(checkpoint_dir),
        "behavior_cloning": result.to_json_dict(),
    }
    atomic_write_json(run_root / "controller_summary.json", summary)
    print("completed")
    print(checkpoint_dir)
    return 0



if __name__ == "__main__":
    raise SystemExit(main())
