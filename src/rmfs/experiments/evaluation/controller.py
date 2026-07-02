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
        "zone_ids": list(zone_ids),
        "seed_pack_id": seed_pack["seed_pack_id"],
        "netlogo_steps_per_run": seed_pack["netlogo_steps_per_run"],
        "replications": seed_pack["replications"],
        "policy_mode": policy_mode,
        "policy_action_mode": policy_action_mode,
        "feature_ablation": ablation.name,
        "feature_ablation_hash": ablation.hash,
        "charging_mode": charging_mode,
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
            rts_zone_ids=list(zone_ids),
            rts_seed_base=int(seed_pack["seed_base"]),
            rts_random_seed=int(seed["seed"]),
            rts_policy_checkpoint_dir=str(checkpoint) if checkpoint is not None else None,
            rts_policy_checkpoint_id=policy_checkpoint_id,
            rts_policy_action_mode=policy_action_mode,
            rts_policy_device="cpu",
            rts_feature_ablation=ablation.name,
            rts_feature_ablation_hash=ablation.hash,
            rts_charging_mode=charging_mode,
            committed_next_reservations_enabled=(policy_mode == "rts_rl_explicit"),
            experiment_id=eval_run_id,
            scenario_id=f"eval_scenario_{short_hash(config)}",
            artifact_label=eval_run_id,
            worker_id=int(seed["replication"]),
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

    metrics = _aggregate_eval_metrics(worker_summaries, rollout_summaries)
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


def _aggregate_eval_metrics(worker_summaries: list[dict[str, Any]], rollout_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    import statistics

    completed_durations = [
        finite_float(summary.get("avg_paper_cycle_duration"))
        for summary in rollout_summaries
        if finite_float(summary.get("avg_paper_cycle_duration")) is not None
    ]
    completed_counts = [int(summary.get("completed_paper_cycle_count") or 0) for summary in rollout_summaries]
    total_completed = sum(completed_counts)
    paper_status_counts: dict[str, int] = {}
    for summary in rollout_summaries:
        for key, value in dict(summary.get("paper_cycle_status_counts") or {}).items():
            paper_status_counts[key] = paper_status_counts.get(key, 0) + int(value or 0)
    energy_values = [
        finite_float((summary.get("final_metrics") or {}).get("total_energy"))
        for summary in worker_summaries
        if finite_float((summary.get("final_metrics") or {}).get("total_energy")) is not None
    ]
    orders_completed = total_completed
    worker_failures = sum(1 for summary in worker_summaries if summary.get("status") != "success")
    wall_times = [finite_float(summary.get("worker_wall_time_elapsed")) for summary in worker_summaries]
    wall_times = [value for value in wall_times if value is not None]
    mean_duration = statistics.mean(completed_durations) if completed_durations else None
    decision_total = sum(int(summary.get("decision_count") or 0) for summary in rollout_summaries)
    return {
        "mean_completed_paper_cycle_duration": mean_duration,
        "median_paper_cycle_duration": statistics.median(completed_durations) if completed_durations else None,
        "std_paper_cycle_duration": statistics.pstdev(completed_durations) if len(completed_durations) > 1 else None,
        "completed_cycle_count": total_completed,
        "censored_counts_by_reason": {k: v for k, v in paper_status_counts.items() if str(k).startswith("censored")},
        "completion_rate": (total_completed / decision_total) if decision_total else 0.0,
        "orders_completed": orders_completed,
        "average_order_cycle_time": mean_duration,
        "congestion_rate": None,
        "energy_per_order": (sum(energy_values) / orders_completed) if energy_values and orders_completed else None,
        "robot_distance_per_order": None,
        "loaded_distance_per_order": None,
        "empty_distance_per_order": None,
        "replenishment_count": paper_status_counts.get("censored_next_task_replenishment", 0),
        "charging_count": None,
        "charging_censor_count": paper_status_counts.get("censored_next_task_charging", 0),
        "worker_failures": worker_failures,
        "runtime_invariant_violations": sum(int(summary.get("runtime_invariant_violations") or 0) for summary in worker_summaries),
        "wall_clock_duration": sum(wall_times) if wall_times else None,
    }

