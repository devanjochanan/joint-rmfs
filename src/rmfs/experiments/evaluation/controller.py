"""Dry-run RTS checkpoint evaluation controller."""

from __future__ import annotations

import datetime
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

from src.rmfs.experiments.identity import short_hash
from src.rmfs.experiments.ledger.ingest_evaluation import ingest_evaluation_summary
from src.rmfs.orchestration.local_executor import git_value
from src.rmfs.orchestration.run_spec import RunSpec
from src.rmfs.rl.rts.ablation import resolve_ablation
from src.rmfs.rl.rts.training.checkpoint import resolve_policy_checkpoint_id
from src.rmfs.rl.rts.training.metrics import atomic_write_json, append_jsonl, finite_float
from src.rmfs.rl.rts.training.policy_loader import load_policy_from_checkpoint
from src.rmfs.runtime_io.run_profiles import DEFAULT_RTS_ORDER_RATE_PER_HOUR


def build_eval_run_id(config: dict[str, Any]) -> str:
    return f"eval_{short_hash(config)}"


def write_eval_dry_run(
    *,
    checkpoint_dir: Path,
    policy_checkpoint_id: str,
    zone_ids: tuple[str, ...],
    seed_pack_path: Path,
    output_root: Path,
    policy_action_mode: str = "greedy",
) -> dict[str, Any]:
    with Path(seed_pack_path).open() as fh:
        seed_pack = json.load(fh)
    config = {
        "checkpoint_dir": str(checkpoint_dir),
        "policy_checkpoint_id": policy_checkpoint_id,
        "zone_ids": list(zone_ids),
        "seed_pack_id": seed_pack["seed_pack_id"],
        "netlogo_steps_per_run": seed_pack["netlogo_steps_per_run"],
        "replications": seed_pack["replications"],
        "policy_action_mode": policy_action_mode,
    }
    eval_run_id = build_eval_run_id(config)
    run_root = Path(output_root) / eval_run_id
    run_root.mkdir(parents=True, exist_ok=True)
    worker_specs = [
        {
            "worker_run_id": f"eval_{seed['replication']:03d}",
            "seed": seed["seed"],
            "netlogo_steps_per_run": seed_pack["netlogo_steps_per_run"],
            "policy_checkpoint_id": policy_checkpoint_id,
            "policy_action_mode": policy_action_mode,
            "zone_ids": list(zone_ids),
        }
        for seed in seed_pack["seeds"]
    ]
    summary = {
        "status": "dry_run",
        "eval_run_id": eval_run_id,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        **config,
        "worker_count": len(worker_specs),
    }
    for name, payload in (("eval_config.json", config), ("worker_specs.json", worker_specs), ("eval_summary.json", summary)):
        with (run_root / name).open("w") as fh:
            json.dump(payload, fh, indent=2)
    return summary


def run_rts_evaluation(
    *,
    repo_root: Path,
    checkpoint_dir: Path | None,
    zone_ids: tuple[str, ...],
    seed_pack_path: Path,
    output_root: Path,
    policy_mode: str = "rts_rl_explicit",
    policy_action_mode: str = "greedy",
    feature_ablation: str = "full",
    charging_mode: str = "inherit",
    dry_run: bool = False,
    min_completed_cycles: int = 1,
    ledger_path: Path | None = None,
    rts_torch_threads: int | None = None,
    rts_torch_interop_threads: int | None = None,
    state_capture_mode: str = "auto",
) -> dict[str, Any]:
    repo_root = Path(repo_root)
    with Path(seed_pack_path).open() as fh:
        seed_pack = json.load(fh)
    if charging_mode not in {"inherit", "enabled", "disabled"}:
        raise ValueError("charging_mode must be inherit, enabled, or disabled")
    ablation = resolve_ablation(feature_ablation)
    checkpoint = Path(checkpoint_dir) if checkpoint_dir is not None else None
    policy_checkpoint_id = None
    if policy_mode == "rts_rl_explicit":
        if checkpoint is None:
            raise ValueError("rts_rl_explicit evaluation requires checkpoint_dir")
        loaded = load_policy_from_checkpoint(checkpoint, device="cpu")
        policy_checkpoint_id = loaded.policy_checkpoint_id
    elif policy_mode not in {"current", "random_valid"}:
        raise ValueError("policy_mode must be current, random_valid, or rts_rl_explicit")

    config = {
        "checkpoint_dir": str(checkpoint) if checkpoint is not None else None,
        "policy_checkpoint_id": policy_checkpoint_id,
        "zone_ids": list(zone_ids) if zone_ids else ["auto"],
        "seed_pack_id": seed_pack["seed_pack_id"],
        "netlogo_steps_per_run": seed_pack["netlogo_steps_per_run"],
        "replications": seed_pack["replications"],
        "policy_mode": policy_mode,
        "policy_action_mode": policy_action_mode,
        "feature_ablation": ablation.name,
        "feature_ablation_hash": ablation.hash,
        "charging_mode": charging_mode,
        "state_capture_mode": state_capture_mode,
        "rts_torch_threads": rts_torch_threads if rts_torch_threads is not None else (1 if policy_mode == "rts_rl_explicit" else None),
        "rts_torch_interop_threads": rts_torch_interop_threads if rts_torch_interop_threads is not None else (1 if policy_mode == "rts_rl_explicit" else None),
        "run_profile": "training",
        "order_generation_mode": "shuffled_historical_cycle",
        "full_raw_order_replay": False,
        "order_rate_per_hour": DEFAULT_RTS_ORDER_RATE_PER_HOUR,
        "tick_to_second": 0.15,
    }
    eval_run_id = build_eval_run_id(config)
    run_root = Path(output_root) / eval_run_id
    run_root.mkdir(parents=True, exist_ok=True)
    branch = git_value(repo_root, "rev-parse", "--abbrev-ref", "HEAD")
    commit = git_value(repo_root, "rev-parse", "HEAD")
    specs = []
    for seed in seed_pack["seeds"]:
        run_id = f"eval_{int(seed['replication']):03d}"
        worker_root = run_root / "workers" / run_id
        spec = RunSpec(
            run_id=run_id,
            ticks=int(seed_pack["netlogo_steps_per_run"]),
            runtime_root=worker_root,
            repo_root=repo_root,
            branch=branch,
            commit=commit,
            python_executable=sys.executable,
            timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            rts_policy_mode=policy_mode,
            rts_rollout_enabled=True,
            rts_zone_ids=list(zone_ids) if zone_ids else ["auto"],
            rts_seed_base=int(seed_pack["seed_base"]),
            rts_random_seed=int(seed["seed"]),
            rts_policy_checkpoint_dir=str(checkpoint) if checkpoint is not None else None,
            rts_policy_checkpoint_id=policy_checkpoint_id,
            rts_policy_action_mode=policy_action_mode,
            rts_policy_device="cpu",
            rts_feature_ablation=ablation.name,
            rts_feature_ablation_hash=ablation.hash,
            rts_state_capture_mode=state_capture_mode,
            rts_charging_mode=charging_mode,
            committed_next_reservations_enabled=(policy_mode in {"rts_rl_explicit", "current", "random_valid"}),
            experiment_id=eval_run_id,
            scenario_id=f"eval_scenario_{short_hash(config)}",
            artifact_label=eval_run_id,
            worker_id=int(seed["replication"]),
            rts_torch_threads=rts_torch_threads,
            rts_torch_interop_threads=rts_torch_interop_threads,
            run_profile="training",
            run_horizon_ticks=int(seed_pack["netlogo_steps_per_run"]),
            demand_horizon_ticks=int(seed_pack["netlogo_steps_per_run"]) + 1000,
            demand_buffer_ticks=1000,
            order_generation_mode="shuffled_historical_cycle",
            full_raw_order_replay=False,
            order_rate_per_hour=DEFAULT_RTS_ORDER_RATE_PER_HOUR,
            pod_location_mode="randomize_slots",
            pod_location_seed=int(seed["seed"]),
        )
        worker_root.mkdir(parents=True, exist_ok=True)
        atomic_write_json(worker_root / "run_spec.json", spec.to_json_dict())
        specs.append(spec)
    atomic_write_json(run_root / "eval_config.json", config)
    atomic_write_json(run_root / "worker_specs.json", [spec.to_json_dict() for spec in specs])

    if dry_run:
        summary = {
            "status": "dry_run",
            "valid": False,
            "eval_run_id": eval_run_id,
            "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            **config,
            "worker_count": len(specs),
        }
        atomic_write_json(run_root / "eval_summary.json", summary)
        return summary

    worker_summaries: list[dict[str, Any]] = []
    rollout_summaries: list[dict[str, Any]] = []
    failures = 0
    for spec in specs:
        code = subprocess.call(
            [
                sys.executable,
                "-m",
                "src.rmfs.orchestration.local_executor",
                "worker",
                "--spec",
                str(spec.runtime_root / "run_spec.json"),
            ],
            cwd=repo_root,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if code != 0:
            failures += 1
        worker_summary_path = spec.runtime_root / "worker_summary.json"
        if worker_summary_path.exists():
            with worker_summary_path.open() as fh:
                worker_summaries.append(json.load(fh))
        rollout_summary_path = spec.runtime_root / "rts_rollout_summary.json"
        if rollout_summary_path.exists():
            with rollout_summary_path.open() as fh:
                rollout_summaries.append(json.load(fh))

    metrics = _aggregate_eval_metrics(worker_summaries, rollout_summaries, run_root=run_root)
    valid = (
        len(worker_summaries) == int(seed_pack["replications"])
        and failures == 0
        and metrics["worker_failures"] == 0
        and metrics["completed_cycle_count"] >= int(min_completed_cycles)
        and metrics["runtime_invariant_violations"] == 0
    )
    status = "valid" if valid else "invalid"
    summary = {
        "status": status,
        "valid": valid,
        "eval_run_id": eval_run_id,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        **config,
        "expected_replications": int(seed_pack["replications"]),
        "completed_replications": len(worker_summaries) - metrics["worker_failures"],
        "failed_replications": failures + metrics["worker_failures"],
        "min_completed_cycles": int(min_completed_cycles),
        "metrics": metrics,
    }
    atomic_write_json(run_root / "eval_summary.json", summary)
    for key, value in metrics.items():
        append_jsonl(run_root / "eval_metrics_long.jsonl", {"eval_run_id": eval_run_id, "metric": key, "value": value})
    if ledger_path is not None:
        try:
            row = ingest_evaluation_summary(run_root / "eval_summary.json", ledger_path)
            summary["ledger_ingest"] = {"status": "success", "row": row}
        except Exception as exc:
            summary["ledger_ingest"] = {"status": "failure", "error_type": type(exc).__name__, "error_message": str(exc)}
            summary["valid"] = False
            summary["status"] = "invalid_ledger_ingest_failed"
        atomic_write_json(run_root / "eval_summary.json", summary)
    return summary


def _aggregate_eval_metrics(
    worker_summaries: list[dict[str, Any]],
    rollout_summaries: list[dict[str, Any]],
    run_root: Path | None = None,
) -> dict[str, Any]:
    import statistics

    completed_durations_all = []
    has_individual_cycles = False
    if run_root is not None:
        try:
            for path in run_root.glob("workers/*/rts_rollout.jsonl"):
                if path.exists():
                    with path.open() as fh:
                        for line in fh:
                            if line.strip():
                                row = json.loads(line)
                                if row.get("paper_cycle_status") == "complete":
                                    dur = finite_float(row.get("paper_cycle_duration"))
                                    if dur is not None:
                                        completed_durations_all.append(dur)
            if completed_durations_all:
                has_individual_cycles = True
        except Exception:
            has_individual_cycles = False

    completed_counts = [int(summary.get("completed_paper_cycle_count") or 0) for summary in rollout_summaries]
    total_completed = sum(completed_counts)

    if has_individual_cycles:
        mean_duration = statistics.mean(completed_durations_all)
        median_duration = statistics.median(completed_durations_all)
        std_duration = statistics.pstdev(completed_durations_all) if len(completed_durations_all) > 1 else None
    else:
        # Fallback: calculate weighted mean
        total_duration_sum = sum(
            finite_float(summary.get("avg_paper_cycle_duration")) * int(summary.get("completed_paper_cycle_count") or 0)
            for summary in rollout_summaries
            if finite_float(summary.get("avg_paper_cycle_duration")) is not None
            and int(summary.get("completed_paper_cycle_count") or 0) > 0
        )
        mean_duration = (total_duration_sum / total_completed) if total_completed > 0 else None
        median_duration = None
        std_duration = None

    paper_status_counts: dict[str, int] = {}
    for summary in rollout_summaries:
        for key, value in dict(summary.get("paper_cycle_status_counts") or {}).items():
            paper_status_counts[key] = paper_status_counts.get(key, 0) + int(value or 0)

    # Warehouse metrics extraction helpers
    def sum_metric(name: str) -> tuple[float | None, bool]:
        vals = []
        for summary in worker_summaries:
            fm = summary.get("final_metrics") or {}
            val = fm.get(name)
            if val is not None:
                fval = finite_float(val)
                if fval is not None:
                    vals.append(fval)
        if not vals:
            return None, False
        return sum(vals), True

    def mean_metric(name: str) -> tuple[float | None, bool]:
        vals = []
        for summary in worker_summaries:
            fm = summary.get("final_metrics") or {}
            val = fm.get(name)
            if val is not None:
                fval = finite_float(val)
                if fval is not None:
                    vals.append(fval)
        if not vals:
            return None, False
        return sum(vals) / len(vals), True

    orders_completed, orders_completed_available = sum_metric("orders_completed")
    average_order_cycle_time, average_order_cycle_time_available = mean_metric("average_order_cycle_time")
    total_energy, total_energy_available = sum_metric("total_energy")

    energy_per_order, energy_per_order_available = mean_metric("energy_per_order")
    if not energy_per_order_available:
        if total_energy_available and orders_completed_available and orders_completed is not None and orders_completed > 0:
            energy_per_order = total_energy / orders_completed
            energy_per_order_available = True
        else:
            energy_per_order = None
            energy_per_order_available = False

    robot_distance_per_order, robot_distance_per_order_available = mean_metric("robot_distance_per_order")
    loaded_distance_per_order, loaded_distance_per_order_available = mean_metric("loaded_distance_per_order")
    empty_distance_per_order, empty_distance_per_order_available = mean_metric("empty_distance_per_order")
    congestion_rate, congestion_rate_available = mean_metric("congestion_rate")
    replenishment_count, replenishment_count_available = sum_metric("replenishment_count")
    charging_count, charging_count_available = sum_metric("charging_count")

    worker_failures = sum(1 for summary in worker_summaries if summary.get("status") != "success")
    wall_times = [finite_float(summary.get("worker_wall_time_elapsed")) for summary in worker_summaries]
    wall_times = [value for value in wall_times if value is not None]
    decision_total = sum(int(summary.get("decision_count") or 0) for summary in rollout_summaries)

    return {
        "mean_completed_paper_cycle_duration": mean_duration,
        "median_completed_paper_cycle_duration": median_duration,
        "std_completed_paper_cycle_duration": std_duration,
        "completed_cycle_count": total_completed,
        "censored_counts_by_reason": {k: v for k, v in paper_status_counts.items() if str(k).startswith("censored")},
        "completion_rate": (total_completed / decision_total) if decision_total else 0.0,
        "rts_cycle_completion_rate": (total_completed / decision_total) if decision_total else 0.0,
        "orders_completed": orders_completed,
        "orders_completed_available": orders_completed_available,
        "average_order_cycle_time": average_order_cycle_time,
        "average_order_cycle_time_available": average_order_cycle_time_available,
        "total_energy": total_energy,
        "total_energy_available": total_energy_available,
        "energy_per_order": energy_per_order,
        "energy_per_order_available": energy_per_order_available,
        "robot_distance_per_order": robot_distance_per_order,
        "robot_distance_per_order_available": robot_distance_per_order_available,
        "loaded_distance_per_order": loaded_distance_per_order,
        "loaded_distance_per_order_available": loaded_distance_per_order_available,
        "empty_distance_per_order": empty_distance_per_order,
        "empty_distance_per_order_available": empty_distance_per_order_available,
        "congestion_rate": congestion_rate,
        "congestion_rate_available": congestion_rate_available,
        "replenishment_count": replenishment_count,
        "replenishment_count_available": replenishment_count_available,
        "charging_count": charging_count,
        "charging_count_available": charging_count_available,
        "charging_censor_count": paper_status_counts.get("censored_next_task_charging", 0),
        "worker_failures": worker_failures,
        "runtime_invariant_violations": sum(int(summary.get("runtime_invariant_violations") or 0) for summary in worker_summaries),
        "wall_clock_duration": sum(wall_times) if wall_times else None,
    }

