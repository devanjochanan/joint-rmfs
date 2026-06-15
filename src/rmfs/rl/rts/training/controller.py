"""RTS on-policy PPO training controller spine."""

from __future__ import annotations

import datetime
import json
import subprocess
from pathlib import Path
import shutil
import sys
import time
from typing import Any

import torch

from src.rmfs.orchestration.run_spec import RunSpec
from src.rmfs.orchestration.local_executor import git_value
from src.rmfs.rl.rts.training.checkpoint import load_training_checkpoint, save_training_checkpoint
from src.rmfs.rl.rts.training.config import RTSTrainingConfig
from src.rmfs.rl.rts.training.on_policy_dataset import build_on_policy_ppo_batch, build_on_policy_training_steps
from src.rmfs.rl.rts.training.policy_loader import load_policy_from_checkpoint

from .device import resolve_rts_torch_device
from .metrics import append_jsonl, atomic_write_json
from .on_policy_config import RTSOnPolicyTrainingConfig, validate_on_policy_training_config
from .progress import progress_bar, resolve_progress_enabled, RTSTrainingProgressBar
from .seeding import derive_worker_seed
from .tensorboard import RTSTensorBoardLogger
from .reward_normalizer import (
    apply_cold_start_rewards,
    choose_reward_metadata_for_batch,
    default_reward_normalizer_metadata,
    derive_reward_normalizer_from_events,
    load_reward_normalizer_metadata,
)
from src.rmfs.experiments.identity import make_experiment_run_id, make_scenario_id
from src.rmfs.experiments.feature_flags import default_rika_rts_rl_feature_flags
from src.rmfs.decisions.task_allocation import scheduler_metadata
from src.rmfs.rl.rts.zone_registry import validate_no_col_zone_ids


def run_on_policy_training_controller(
    *,
    config: RTSOnPolicyTrainingConfig,
    repo_root: Path,
    initial_checkpoint_dir: Path | None = None,
    resume_latest: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    # Root-sensitive artifact path check
    try:
        Path(config.output_root).resolve().relative_to(repo_root.resolve() / "data" / "runtime")
    except ValueError as exc:
        raise RuntimeError("training outputs must stay under data/runtime") from exc

    # Resolve or generate artifact label if omitted
    resolved_label = config.artifact_label
    pointer_path = Path(config.output_root) / "latest_local_artifact.json"
    if resolved_label is None or not str(resolved_label).strip():
        if resume_latest:
            if pointer_path.exists():
                try:
                    with pointer_path.open() as fh:
                        pointer_data = json.load(fh)
                    resolved_label = pointer_data.get("latest_artifact_label")
                except Exception as exc:
                    raise RuntimeError(f"failed to read latest local artifact pointer: {exc}")
            if not resolved_label:
                raise RuntimeError("resume_latest requested but no latest local artifact label could be resolved")
        else:
            operation = "dry_run" if dry_run else "train"
            short_commit = "nogit"
            try:
                full_commit = git_value(repo_root, "rev-parse", "HEAD")
                if full_commit:
                    short_commit = full_commit[:7]
            except Exception:
                pass
            timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
            resolved_label = f"dewa_rts_{operation}_{timestamp}_{short_commit}"

    from dataclasses import replace
    config = replace(config, artifact_label=resolved_label)

    validate_on_policy_training_config(config, require_cycle_reference_exists=True)
    run_root = Path(config.output_root) / config.artifact_label
    run_root.mkdir(parents=True, exist_ok=True)

    # Write/update pointer file
    atomic_write_json(pointer_path, {"latest_artifact_label": config.artifact_label})

    branch = git_value(repo_root, "rev-parse", "--abbrev-ref", "HEAD")
    commit = git_value(repo_root, "rev-parse", "HEAD")

    config_dict = config.to_json_dict()
    scheduler = scheduler_metadata(
        robot_task_allocator=config.robot_task_allocator,
        regret_k=config.regret_k,
    )
    config_dict.update(scheduler)
    reward_metadata = default_reward_normalizer_metadata()
    config_dict.update(reward_metadata)
    if config.cycle_reference_path is not None:
        config_dict["cycle_reference_enabled"] = True
    feature_flags = config_dict.get("feature_flags") or default_rika_rts_rl_feature_flags()
    base_config_for_hash = {k: v for k, v in config_dict.items() if k not in ("scenario_id", "experiment_id")}
    scenario_id = make_scenario_id({"config": base_config_for_hash, "feature_flags": feature_flags})
    experiment_id = make_experiment_run_id({
        "artifact_label": config.artifact_label,
        "seed": config.seed,
        "scenario_id": scenario_id,
        "repo_commit": commit,
        "netlogo_steps_per_run": config.netlogo_steps_per_run,
        "workers": config.workers,
    })

    config_dict["scenario_id"] = scenario_id
    config_dict["experiment_id"] = experiment_id
    config_dict["branch"] = branch
    config_dict["commit"] = commit
    config_dict["python_executable"] = sys.executable

    atomic_write_json(run_root / "training_config.json", config_dict)

    # Write initial starting status
    initial_summary = {
        "status": "started",
        "experiment_id": experiment_id,
        "scenario_id": scenario_id,
        "branch": branch,
        "commit": commit,
        "run_root": str(run_root),
        **scheduler,
        **reward_metadata,
        "batches": [],
    }
    atomic_write_json(run_root / "controller_summary.json", initial_summary)

    tb = RTSTensorBoardLogger(run_root / "tb", enabled=config.tensorboard_enabled)
    progress_enabled = resolve_progress_enabled(config.progress)
    latest = _read_latest(run_root) if resume_latest else None
    active_checkpoint_dir = Path(initial_checkpoint_dir) if initial_checkpoint_dir else None
    active_checkpoint_id = _checkpoint_id(active_checkpoint_dir) if active_checkpoint_dir else "dry_run_uninitialized"
    if latest:
        active_checkpoint_dir = Path(latest["checkpoint_dir"])
        active_checkpoint_id = _checkpoint_id(active_checkpoint_dir)
    batch_summaries = []
    last_update_time = 0.0

    try:
        for batch_id in range(1, config.batches + 1):
            batch_dir = run_root / f"batch_{batch_id:06d}"
            rollout_input = batch_dir / "rollout_input"
            workers_dir = batch_dir / "workers"
            rollout_input.mkdir(parents=True, exist_ok=True)
            workers_dir.mkdir(parents=True, exist_ok=True)
            reward_reference_path = None
            if config.cycle_reference_path is not None:
                reward_reference_path = rollout_input / "cycle_reference.json"
                shutil.copyfile(config.cycle_reference_path, reward_reference_path)
            previous_reward_metadata = load_reward_normalizer_metadata(active_checkpoint_dir)
            active_ref = {
                "policy_checkpoint_dir": str(active_checkpoint_dir) if active_checkpoint_dir else None,
                "policy_checkpoint_id": active_checkpoint_id,
                "reward_normalizer": previous_reward_metadata,
            }
            atomic_write_json(rollout_input / "active_checkpoint_ref.json", active_ref)
            worker_specs = []
            # Prefer explicit config.zone_ids
            if config.zone_ids:
                active_zone_ids = config.zone_ids
            else:
                # Try to load from checkpoint metadata
                active_zone_ids = _zone_ids_from_checkpoint_metadata(active_checkpoint_dir)
                # As last resort for dry-run/smoke, infer from checkpoint feature names
                if not active_zone_ids and active_checkpoint_dir:
                    try:
                        active_zone_ids = _zone_ids_from_checkpoint(active_checkpoint_dir)
                    except Exception:
                        active_zone_ids = None

            if not dry_run:
                if not active_zone_ids:
                    raise RuntimeError("non-dry training execution requires zone_ids to be explicitly determined")
                if any(not str(z).strip() for z in active_zone_ids) or len(set(active_zone_ids)) != len(active_zone_ids):
                    raise ValueError("zone_ids must be nonblank and unique for non-dry training")
                validate_no_col_zone_ids(active_zone_ids, context="RTS on-policy controller")

            # Initialize metrics for TB
            worker_successes = 0
            worker_failures = 0
            worker_elapsed_avg = 0.0
            wait_time = 0.0
            update_time = 0.0
            energy_avg = 0.0
            orders_completed_total = 0
            avg_order_cycle_time = 0.0
            cuda_memory_allocated = 0.0
            if torch.cuda.is_available():
                cuda_memory_allocated = float(torch.cuda.memory_allocated())

            for worker_index in range(config.workers):
                run_id = f"run_{worker_index + 1:03d}"
                worker_root = workers_dir / run_id
                spec = RunSpec(
                    run_id=run_id,
                    ticks=config.netlogo_steps_per_run,
                    runtime_root=worker_root,
                    repo_root=repo_root,
                    branch=branch,
                    commit=commit,
                    python_executable=sys.executable,
                    timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    rts_policy_mode="rts_rl_explicit",
                    rts_rollout_enabled=True,
                    rts_reward_reference_path=str(reward_reference_path) if reward_reference_path is not None else None,
                    rts_seed_base=config.seed,
                    rts_random_seed=derive_worker_seed(config.seed, batch_id, worker_index),
                    rts_zone_ids=list(active_zone_ids or ()),
                    rts_policy_checkpoint_dir=str(active_checkpoint_dir) if active_checkpoint_dir else None,
                    rts_policy_checkpoint_id=active_checkpoint_id,
                    rts_policy_action_mode=config.policy_action_mode,
                    rts_policy_device=config.worker_device,
                    experiment_id=experiment_id,
                    scenario_id=scenario_id,
                    artifact_label=config.artifact_label,
                    batch_id=batch_id,
                    worker_id=worker_index + 1,
                    **scheduler,
                )
                worker_root.mkdir(parents=True, exist_ok=True)
                atomic_write_json(worker_root / "run_spec.json", spec.to_json_dict())
                worker_specs.append(spec.to_json_dict())
            
            atomic_write_json(batch_dir / "worker_specs.json", worker_specs)

            if dry_run:
                summary = {
                    "experiment_id": experiment_id,
                    "scenario_id": scenario_id,
                    "artifact_label": config.artifact_label,
                    "seed_base": config.seed,
                    "batch_id": batch_id,
                    "status": "dry_run",
                    "netlogo_steps_per_run": config.netlogo_steps_per_run,
                    "workers": config.workers,
                    "active_checkpoint_id": active_checkpoint_id,
                    "worker_specs_written": len(worker_specs),
                    "trainable_step_count": 0,
                    "ppo_update": None,
                    "latest_updated": False,
                    "zone_ids": list(active_zone_ids or ()),
                    **scheduler,
                    **previous_reward_metadata,
                }
                atomic_write_json(batch_dir / "batch_summary.json", summary)
                append_jsonl(run_root / "training_events.jsonl", {"event_type": "batch_dry_run", **summary})
                append_jsonl(run_root / "batch_metrics.jsonl", summary)
                tb.log_scalars(
                    {
                        "time/netlogo_steps_per_run": config.netlogo_steps_per_run,
                        "rollout/trainable_step_count": 0,
                        "checkpoint/batch_id": batch_id,
                        "checkpoint/latest_updated": 0,
                        "train/ppo_total_loss": 0.0,
                        "train/policy_loss": 0.0,
                        "train/value_loss": 0.0,
                        "train/entropy": 0.0,
                        "train/approx_kl": 0.0,
                        "train/clip_fraction": 0.0,
                        "train/optimizer_steps": 0,
                        "train/update_time_sec": 0.0,
                        "ppo/total_loss": 0.0,
                        "ppo/policy_loss": 0.0,
                        "ppo/value_loss": 0.0,
                        "ppo/entropy": 0.0,
                        "ppo/approx_kl": 0.0,
                        "ppo/clip_fraction": 0.0,
                        "rollout/decision_count": 0,
                        "rollout/outcome_count": 0,
                        "rollout/rejected_non_on_policy_count": 0,
                        "rollout/rejected_checkpoint_mismatch_count": 0,
                        "rollout/avg_reward": 0.0,
                        "rollout/reward_mean": 0.0,
                        "rollout/reward_std": 0.0,
                        "worker/success_count": 0,
                        "worker/failure_count": 0,
                        "worker/elapsed_wall_time_avg_sec": 0.0,
                        "worker/controller_wait_time_sec": 0.0,
                        "warehouse/orders_completed_total": 0,
                        "warehouse/avg_order_cycle_time": 0.0,
                        "warehouse/avg_energy": 0.0,
                        "warehouse/congestion_rate": 0.0,
                        "warehouse/robot_idle_time": 0.0,
                        "device/cuda_memory_allocated_bytes": cuda_memory_allocated,
                    },
                    batch_id,
                )
                batch_summaries.append(summary)
                continue

            if active_checkpoint_dir is None:
                raise RuntimeError("non-dry on-policy training requires an active checkpoint")
            processes = []
            log_files = []
            try:
                for spec in worker_specs:
                    runtime_root = Path(spec["runtime_root"])
                    if config.debug_worker_logs:
                        stdout_file = open(runtime_root / "worker_stdout.log", "w", encoding="utf-8")
                        stderr_file = open(runtime_root / "worker_stderr.log", "w", encoding="utf-8")
                        log_files.append((stdout_file, stderr_file))
                        stdout_arg = stdout_file
                        stderr_arg = stderr_file
                    else:
                        stdout_arg = subprocess.DEVNULL
                        stderr_arg = subprocess.DEVNULL

                    p = subprocess.Popen(
                        [
                            sys.executable,
                            "-m",
                            "src.rmfs.orchestration.local_executor",
                            "worker",
                            "--spec",
                            str(runtime_root / "run_spec.json"),
                        ],
                        cwd=repo_root,
                        stdout=stdout_arg,
                        stderr=stderr_arg,
                    )
                    processes.append(p)
                
                # Wait for all worker subprocesses to complete, updating progress bar
                wait_start = time.perf_counter()
                
                progress_target = config.workers * config.netlogo_steps_per_run
                pbar = RTSTrainingProgressBar(
                    enabled=progress_enabled,
                    batch_id=batch_id,
                    workers=config.workers,
                    progress_target=progress_target,
                    latest_checkpoint_id=active_checkpoint_id,
                    update_time=last_update_time,
                )
                
                completed_workers = {}
                worker_wall_times = [0.0] * config.workers
                
                while len(completed_workers) < len(processes):
                    for idx, p in enumerate(processes):
                        if idx in completed_workers:
                            continue
                        exit_code = p.poll()
                        if exit_code is not None:
                            completed_workers[idx] = exit_code
                            worker_wall_times[idx] = time.perf_counter() - wait_start
                    
                    # Check status files of all workers
                    progress_current = 0
                    workers_done = len(completed_workers)
                    worker_failed_count = sum(1 for ec in completed_workers.values() if ec != 0)
                    
                    for idx, spec in enumerate(worker_specs):
                        status_file = Path(spec["runtime_root"]) / "worker_status.json"
                        if idx in completed_workers:
                            progress_current += spec["ticks"]
                        elif status_file.exists():
                            try:
                                with status_file.open() as fh:
                                    payload = json.load(fh)
                                progress_current += payload.get("current_progress_steps", 0)
                            except Exception:
                                pass
                    
                    current_wait_time = time.perf_counter() - wait_start
                    
                    # Straggler ratio calculation
                    current_durations = []
                    for idx in range(config.workers):
                        if idx in completed_workers:
                            current_durations.append(worker_wall_times[idx])
                        else:
                            current_durations.append(current_wait_time)
                    
                    import numpy as np
                    median_dur = float(np.median(current_durations)) if current_durations else 0.0
                    max_dur = float(np.max(current_durations)) if current_durations else 0.0
                    straggler_ratio = (max_dur / median_dur) if median_dur > 0.0 else 0.0
                    
                    pbar.update_live(
                        progress_current=progress_current,
                        workers_done=workers_done,
                        worker_failed_count=worker_failed_count,
                        wait_time=current_wait_time,
                        straggler_ratio=straggler_ratio,
                    )
                    time.sleep(0.1)
                    
                pbar.close()
                wait_time = time.perf_counter() - wait_start
            finally:
                for stdout_file, stderr_file in log_files:
                    try:
                        stdout_file.close()
                    except Exception:
                        pass
            
            # Check for failures
            failed_exit_code = None
            failed_worker_idx = None
            for idx, exit_code in completed_workers.items():
                if exit_code != 0:
                    failed_exit_code = exit_code
                    failed_worker_idx = idx
            
            if failed_exit_code is not None:
                failed_worker_id = failed_worker_idx + 1
                failed_spec = worker_specs[failed_worker_idx]
                failed_runtime_root = Path(failed_spec["runtime_root"])
                
                worker_summary_path = failed_runtime_root / "worker_summary.json"
                error_type = None
                error_message = None
                if worker_summary_path.exists():
                    try:
                        with worker_summary_path.open() as fh:
                            summary_data = json.load(fh)
                        error_type = summary_data.get("error_type")
                        error_message = summary_data.get("error_message")
                    except Exception:
                        pass
                
                lines = []
                lines.append(f"Worker {failed_worker_id} failed with exit code {failed_exit_code}.")
                lines.append(f"  Worker Runtime Root: {failed_runtime_root}")
                if worker_summary_path.exists():
                    lines.append(f"  Worker Summary Path: {worker_summary_path}")
                if error_type:
                    lines.append(f"  Worker Error Type: {error_type}")
                if error_message:
                    lines.append(f"  Worker Error Message: {error_message}")
                
                if config.debug_worker_logs:
                    stdout_path = failed_runtime_root / "worker_stdout.log"
                    stderr_path = failed_runtime_root / "worker_stderr.log"
                    lines.append(f"  Worker Stdout Log: {stdout_path}")
                    lines.append(f"  Worker Stderr Log: {stderr_path}")
                    
                    if stderr_path.exists():
                        try:
                            with stderr_path.open("r", errors="replace") as fh:
                                stderr_lines = fh.readlines()
                            if stderr_lines:
                                tail_lines = stderr_lines[-15:]
                                lines.append("  Worker Stderr Tail:")
                                for l in tail_lines:
                                    lines.append(f"    {l.rstrip()}")
                        except Exception:
                            pass
                else:
                    lines.append("  Worker output logs were not persisted; rerun with --debug-worker-logs for stdout/stderr capture.")
                
                full_error_msg = "\n".join(lines)
                print(full_error_msg, file=sys.stderr)
                raise RuntimeError(f"Worker {failed_worker_id} failed with exit code {failed_exit_code}")
            
            # Load worker summaries and rollout summaries to extract metrics
            worker_summaries = []
            rollout_summaries = []
            for worker_index in range(config.workers):
                run_id = f"run_{worker_index + 1:03d}"
                worker_summary_path = workers_dir / run_id / "worker_summary.json"
                if worker_summary_path.exists():
                    with worker_summary_path.open() as fh:
                        worker_summaries.append(json.load(fh))
                rollout_summary_path = workers_dir / run_id / "rts_rollout_summary.json"
                if rollout_summary_path.exists():
                    with rollout_summary_path.open() as fh:
                        rollout_summaries.append(json.load(fh))

            worker_successes = sum(1 for s in worker_summaries if s.get("status") == "success")
            worker_failures = sum(1 for s in worker_summaries if s.get("status") == "failure")
            import numpy as np
            worker_elapsed_avg = float(np.mean([s.get("worker_wall_time_elapsed", 0.0) for s in worker_summaries])) if worker_summaries else 0.0
            energy_avg = float(np.mean([s.get("final_metrics", {}).get("total_energy", 0.0) for s in worker_summaries if s.get("final_metrics")])) if worker_summaries else 0.0
            orders_completed_total = sum(s.get("completed_paper_cycle_count", 0) for s in rollout_summaries)
            cycle_times = [s.get("avg_paper_cycle_duration") for s in rollout_summaries if s.get("avg_paper_cycle_duration") is not None]
            avg_order_cycle_time = float(np.mean(cycle_times)) if cycle_times else 0.0

            cuda_memory_allocated = 0.0
            if torch.cuda.is_available():
                cuda_memory_allocated = float(torch.cuda.memory_allocated())

            events = _load_rollout_shards(workers_dir)
            derived_reward_metadata = derive_reward_normalizer_from_events(events, batch_id=batch_id)
            batch_reward_metadata = choose_reward_metadata_for_batch(
                previous_metadata=previous_reward_metadata,
                derived_metadata=derived_reward_metadata,
            )
            normalized_events = apply_cold_start_rewards(events, reward_metadata=batch_reward_metadata)
            dataset = build_on_policy_training_steps(
                normalized_events,
                required_policy_checkpoint_id=active_checkpoint_id,
                reward_normalizer_metadata=batch_reward_metadata,
            )
            if dataset.summary["trainable_step_count"] < config.min_trainable_steps:
                summary = {
                    "experiment_id": experiment_id,
                    "scenario_id": scenario_id,
                    "artifact_label": config.artifact_label,
                    "seed_base": config.seed,
                    "batch_id": batch_id,
                    "status": "skipped_insufficient_trainable_steps",
                    "active_checkpoint_id": active_checkpoint_id,
                    "dataset_summary": dataset.summary,
                    "latest_updated": False,
                    "zone_ids": list(active_zone_ids or ()),
                    "derived_reward_normalizer": derived_reward_metadata,
                    **scheduler,
                    **batch_reward_metadata,
                }
                atomic_write_json(batch_dir / "batch_summary.json", summary)
                append_jsonl(run_root / "training_events.jsonl", {"event_type": "batch_skipped", **summary})
                append_jsonl(run_root / "batch_metrics.jsonl", summary)
                batch_summaries.append(summary)
                continue

            update_start = time.perf_counter()
            loaded = load_policy_from_checkpoint(active_checkpoint_dir, device=_controller_device(config.device))
            optimizer = torch.optim.Adam(loaded.model.parameters(), lr=_learning_rate(loaded.metadata))
            load_training_checkpoint(active_checkpoint_dir, model=loaded.model, optimizer=optimizer, device=_controller_device(config.device))
            train_config = _ppo_training_config(config, loaded.metadata)
            ppo_batch = build_on_policy_ppo_batch(dataset, gamma=train_config.gamma, gae_lambda=train_config.gae_lambda)
            from src.rmfs.rl.rts.training.ppo import run_ppo_update

            update_result = run_ppo_update(loaded.model, optimizer, ppo_batch, train_config, _controller_device(config.device))
            checkpoint_dir = save_training_checkpoint(
                model=loaded.model,
                optimizer=optimizer,
                config=train_config,
                batch_id=batch_id,
                dataset_summary=dataset.summary,
                ppo_update_result=update_result,
                action_feature_names=ppo_batch.action_feature_names,
                stock_feature_names=ppo_batch.stock_feature_names,
                cycle_reference_path=reward_reference_path,
                reward_normalizer_metadata=derived_reward_metadata,
                lineage_metadata={
                    "experiment_id": experiment_id,
                    "scenario_id": scenario_id,
                    "artifact_label": config.artifact_label,
                    "seed_base": config.seed,
                    "repo_branch": branch,
                    "repo_commit": commit,
                    "python_executable": sys.executable,
                    **scheduler,
                    **batch_reward_metadata,
                },
                checkpoint_id_before=active_checkpoint_id,
            )
            update_time = time.perf_counter() - update_start
            last_update_time = update_time
            active_checkpoint_dir = checkpoint_dir
            active_checkpoint_id = _checkpoint_id(checkpoint_dir)
            summary = {
                "experiment_id": experiment_id,
                "scenario_id": scenario_id,
                "artifact_label": config.artifact_label,
                "seed_base": config.seed,
                "batch_id": batch_id,
                "status": "updated",
                "netlogo_steps_per_run": config.netlogo_steps_per_run,
                "workers": config.workers,
                "active_checkpoint_id": active_checkpoint_id,
                "worker_specs_written": len(worker_specs),
                "trainable_step_count": dataset.summary["trainable_step_count"],
                "dataset_summary": dataset.summary,
                "ppo_update": update_result.to_json_dict(),
                "checkpoint_dir": str(checkpoint_dir),
                "latest_updated": True,
                "zone_ids": list(active_zone_ids or ()),
                "derived_reward_normalizer": derived_reward_metadata,
                **scheduler,
                **batch_reward_metadata,
            }
            atomic_write_json(batch_dir / "batch_summary.json", summary)
            append_jsonl(run_root / "training_events.jsonl", {"event_type": "batch_updated", **summary})
            append_jsonl(run_root / "batch_metrics.jsonl", summary)
            tb.log_scalars(
                {
                    "train/ppo_total_loss": update_result.total_loss_mean,
                    "train/policy_loss": update_result.policy_loss_mean,
                    "train/value_loss": update_result.value_loss_mean,
                    "train/entropy": update_result.entropy_mean,
                    "train/approx_kl": update_result.approx_kl_mean,
                    "train/clip_fraction": update_result.clip_fraction_mean,
                    "train/optimizer_steps": update_result.optimizer_steps,
                    "train/update_time_sec": update_time,
                    "ppo/total_loss": update_result.total_loss_mean,
                    "ppo/policy_loss": update_result.policy_loss_mean,
                    "ppo/value_loss": update_result.value_loss_mean,
                    "ppo/entropy": update_result.entropy_mean,
                    "ppo/approx_kl": update_result.approx_kl_mean,
                    "ppo/clip_fraction": update_result.clip_fraction_mean,
                    "rollout/trainable_step_count": dataset.summary["trainable_step_count"],
                    "rollout/decision_count": dataset.summary["decision_count"],
                    "rollout/outcome_count": dataset.summary["outcome_count"],
                    "rollout/rejected_non_on_policy_count": dataset.summary["rejected_non_on_policy_count"],
                    "rollout/rejected_checkpoint_mismatch_count": dataset.summary["rejected_checkpoint_mismatch_count"],
                    "rollout/avg_reward": dataset.summary["avg_reward"],
                    "rollout/reward_mean": dataset.summary.get("reward_mean", 0.0),
                    "rollout/reward_std": dataset.summary.get("reward_std", 0.0),
                    "worker/success_count": worker_successes,
                    "worker/failure_count": worker_failures,
                    "worker/elapsed_wall_time_avg_sec": worker_elapsed_avg,
                    "worker/controller_wait_time_sec": wait_time,
                    "warehouse/orders_completed_total": orders_completed_total,
                    "warehouse/avg_order_cycle_time": avg_order_cycle_time,
                    "warehouse/avg_energy": energy_avg,
                    "warehouse/congestion_rate": 0.0,
                    "warehouse/robot_idle_time": 0.0,
                    "checkpoint/batch_id": batch_id,
                    "checkpoint/latest_updated": 1,
                    "device/cuda_memory_allocated_bytes": cuda_memory_allocated,
                },
                batch_id,
            )
            batch_summaries.append(summary)
        tb.close()

        if dry_run:
            status = "dry_run"
        else:
            updated_any = any(b["status"] == "updated" for b in batch_summaries)
            if updated_any:
                status = "completed"
            else:
                status = "completed_with_skips"

        result = {
            "status": status,
            "experiment_id": experiment_id,
            "scenario_id": scenario_id,
            "branch": branch,
            "commit": commit,
            "run_root": str(run_root),
            **scheduler,
            **reward_metadata,
            "batches": batch_summaries,
        }
        atomic_write_json(run_root / "controller_summary.json", result)
        return result

    except Exception as exc:
        tb.close()
        failure_summary = {
            "status": "failure",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "experiment_id": experiment_id,
            "scenario_id": scenario_id,
            "branch": branch,
            "commit": commit,
            "run_root": str(run_root),
            **scheduler,
            **reward_metadata,
            "batches": batch_summaries,
        }
        atomic_write_json(run_root / "controller_summary.json", failure_summary)
        raise exc


def _read_latest(run_root: Path) -> dict[str, Any] | None:
    path = run_root / "latest.json"
    if not path.exists():
        return None
    with path.open() as fh:
        return json.load(fh)


def _checkpoint_id(checkpoint_dir: Path | None) -> str:
    if checkpoint_dir is None:
        return "dry_run_uninitialized"
    parent = checkpoint_dir.parent
    return parent.name if parent.name.startswith("batch_") else checkpoint_dir.name


def _load_rollout_shards(workers_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(Path(workers_dir).glob("run_*/rts_rollout.jsonl")):
        worker_run_id = path.parent.name
        with path.open() as fh:
            for line in fh:
                if line.strip():
                    item = json.loads(line)
                    item["worker_run_id"] = worker_run_id
                    rows.append(item)
    return rows


def _zone_ids_from_checkpoint(checkpoint_dir: Path | None) -> tuple[str, ...] | None:
    if checkpoint_dir is None:
        return None
    loaded = load_policy_from_checkpoint(checkpoint_dir, device="cpu")
    names = loaded.feature_schema.get("action_feature_names", []) or []
    zones = [
        name.removeprefix("next_retrieval_zone_one_hot__")
        for name in names
        if str(name).startswith("next_retrieval_zone_one_hot__")
    ]
    if not zones:
        raise RuntimeError("could not infer RTS zone_ids from checkpoint feature schema")
    return tuple(zones)


def _controller_device(device: str) -> str:
    return resolve_rts_torch_device(device)


def _zone_ids_from_checkpoint_metadata(checkpoint_dir: Path | None) -> tuple[str, ...] | None:
    if checkpoint_dir is None:
        return None
    try:
        with (checkpoint_dir / "metadata.json").open() as fh:
            metadata = json.load(fh)
        config_data = metadata.get("training_config", {}) or {}
        zones = config_data.get("zone_ids")
        if zones:
            return tuple(str(z) for z in zones)
    except Exception:
        pass
    return None


def _learning_rate(metadata: dict[str, Any]) -> float:
    return float((metadata.get("training_config") or {}).get("learning_rate", 1e-4))


def _ppo_training_config(config: RTSOnPolicyTrainingConfig, metadata: dict[str, Any]) -> RTSTrainingConfig:
    base = metadata.get("training_config") or {}
    return RTSTrainingConfig(
        artifact_label=config.artifact_label,
        output_root=config.output_root,
        seed=config.seed,
        learning_rate=float(base.get("learning_rate", 1e-4)),
        gamma=float(base.get("gamma", 0.99)),
        gae_lambda=float(base.get("gae_lambda", 0.95)),
        ppo_epochs=config.ppo_epochs,
        minibatch_size=config.minibatch_size,
        hidden_sizes=tuple(base.get("hidden_sizes", (64, 64))),
        stock_hidden_sizes=tuple(base.get("stock_hidden_sizes", (32, 32))),
        stock_embedding_dim=int(base.get("stock_embedding_dim", 16)),
        tensorboard_enabled=config.tensorboard_enabled,
    )
