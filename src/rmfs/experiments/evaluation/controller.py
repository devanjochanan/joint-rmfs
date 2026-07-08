"""Dry-run RTS checkpoint evaluation controller."""

from __future__ import annotations

import datetime
import json
from pathlib import Path
import sys
from typing import Any

from src.rmfs.experiments.identity import short_hash
from src.rmfs.experiments.ledger.ingest_evaluation import ingest_evaluation_summary
from src.rmfs.orchestration.local_executor import git_value, run_specs
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
    max_workers: int = 1,
    debug_rollouts: bool = False,
) -> dict[str, Any]:
    repo_root = Path(repo_root)
    if max_workers < 1:
        raise ValueError("max_workers must be >= 1")
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
        "max_workers": int(max_workers),
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
            rts_rollout_write_disk=debug_rollouts,
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
    specs_to_run: list[RunSpec] = []
    skipped_completed_workers = 0
    for spec in specs:
        worker_summary_path = spec.runtime_root / "worker_summary.json"
        if worker_summary_path.exists():
            with worker_summary_path.open() as fh:
                existing_summary = json.load(fh)
            if existing_summary.get("status") == "success":
                worker_summaries.append(existing_summary)
                rollout_summary_path = spec.runtime_root / "rts_rollout_summary.json"
                if rollout_summary_path.exists():
                    with rollout_summary_path.open() as fh:
                        rollout_summaries.append(json.load(fh))
                skipped_completed_workers += 1
                continue
        specs_to_run.append(spec)

    failures = 0
    for result in run_specs(specs_to_run, max_workers=int(max_workers), progress=True):
        spec = result["spec"]
        if int(result.get("return_code") or 0) != 0:
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

    if run_root is not None and not debug_rollouts:
        cleaned_bytes = 0
        for bulky in run_root.glob("workers/*/rts_rollout.jsonl"):
            cleaned_bytes += bulky.stat().st_size
            bulky.unlink()
        for bulky in run_root.glob("workers/*/worker_stdout.log"):
            cleaned_bytes += bulky.stat().st_size
            bulky.unlink()
        if cleaned_bytes > 0:
            print(f"[eval] cleaned {cleaned_bytes / 1_000_000:.0f} MB of rollout/log artifacts from {run_root.name}")

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
        "skipped_completed_workers": skipped_completed_workers,
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

    orders_completed, orders_completed_available = sum_metric("warehouse_orders_completed")
    average_order_cycle_time, average_order_cycle_time_available = mean_metric("warehouse_average_order_cycle_time")
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
    congestion_rate, congestion_rate_available = mean_metric("stop_and_go")
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


def verify_paired_checkpoint(checkpoint_dir: Path) -> dict[str, Any]:
    checkpoint_dir = Path(checkpoint_dir)
    metadata_path = checkpoint_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Checkpoint metadata.json missing: {metadata_path}")
    with metadata_path.open("r", encoding="utf-8") as fh:
        meta = json.load(fh)

    # 1. Verify PPO optimizer steps
    ppo_res = meta.get("ppo_update_result")
    if not ppo_res:
        raise ValueError(f"Checkpoint at {checkpoint_dir} does not contain 'ppo_update_result' (not trained with PPO)")
    opt_steps = ppo_res.get("optimizer_steps", 0)
    if opt_steps <= 0:
        raise ValueError(f"Checkpoint PPO optimizer steps must be > 0, got {opt_steps}")

    # 2. Verify lineage
    lineage = meta.get("lineage", {})
    init_method = lineage.get("initialization_method")
    if init_method != "vrsla_behavior_cloning":
        raise ValueError(f"Expected lineage initialization_method = 'vrsla_behavior_cloning', got '{init_method}'")
    teacher = lineage.get("teacher_policy")
    if teacher != "vrsla_event_driven":
        raise ValueError(f"Expected lineage teacher_policy = 'vrsla_event_driven', got '{teacher}'")

    # 3. Verify feature schema exists
    schema_path = checkpoint_dir / "feature_schema.json"
    if not schema_path.exists():
        raise FileNotFoundError(f"feature_schema.json missing in checkpoint: {schema_path}")

    # 4. Verify model weights exist
    model_path = checkpoint_dir / "model.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"model.pt missing in checkpoint: {model_path}")

    return meta


def assert_paired_plan(specs: list[RunSpec], seed_pack: dict[str, Any], replications: int, steps: int) -> None:
    if len(specs) != replications * 2:
        raise AssertionError(f"Expected {replications * 2} runs, got {len(specs)}")

    current_specs = [s for s in specs if s.rts_policy_mode == "current"]
    rl_specs = [s for s in specs if s.rts_policy_mode == "rts_rl_explicit"]

    if len(current_specs) != replications:
        raise AssertionError(f"Expected {replications} current specs, got {len(current_specs)}")
    if len(rl_specs) != replications:
        raise AssertionError(f"Expected {replications} RTS RL specs, got {len(rl_specs)}")

    current_seeds = sorted([s.rts_random_seed for s in current_specs])
    rl_seeds = sorted([s.rts_random_seed for s in rl_specs])
    pack_seeds = sorted([s["seed"] for s in seed_pack["seeds"]])
    if current_seeds != pack_seeds or rl_seeds != pack_seeds:
        raise AssertionError("Seeds do not match seed pack seeds")

    for idx in range(replications):
        if specs[idx * 2].rts_policy_mode != "current":
            raise AssertionError(f"Expected specs[{idx * 2}] to be current, got {specs[idx * 2].rts_policy_mode}")
        if specs[idx * 2 + 1].rts_policy_mode != "rts_rl_explicit":
            raise AssertionError(f"Expected specs[{idx * 2 + 1}] to be rts_rl_explicit, got {specs[idx * 2 + 1].rts_policy_mode}")
        if specs[idx * 2].rts_random_seed != specs[idx * 2 + 1].rts_random_seed:
            raise AssertionError(f"Random seeds do not match at interleaved index {idx}")

    for spec in specs:
        if spec.ticks != steps:
            raise AssertionError(f"Expected {steps} steps, got {spec.ticks}")
        if spec.robot_count != 20:
            raise AssertionError(f"Expected 20 robots, got {spec.robot_count}")
        if spec.order_rate_per_hour != 500:
            raise AssertionError(f"Expected 500 orders/hour, got {spec.order_rate_per_hour}")
        if not spec.charging_enabled:
            raise AssertionError("Charging must be enabled")
        if spec.pps_mode != "heuristic":
            raise AssertionError(f"Expected heuristic PPS, got {spec.pps_mode}")
        if spec.pps_model_path is not None:
            raise AssertionError("PPS model path must be None (disabled)")
        if spec.kpi_schema_version != "sensitivity_full_kpi.v1":
            raise AssertionError(f"Expected kpi_schema_version = sensitivity_full_kpi.v1, got {spec.kpi_schema_version}")
        if spec.replication is None:
            raise AssertionError("replication must not be None")
        if spec.campaign_id is None:
            raise AssertionError("campaign_id must not be None")
        if spec.machine_id is None:
            raise AssertionError("machine_id must not be None")
        if spec.stage_first_requested is None:
            raise AssertionError("stage_first_requested must not be None")
        if spec.pps_model_sha256 != "none":
            raise AssertionError("pps_model_sha256 must be 'none'")

    for idx in range(replications):
        spec_current = specs[idx * 2]
        spec_rl = specs[idx * 2 + 1]

        current_dict = spec_current.to_json_dict()
        rl_dict = spec_rl.to_json_dict()

        expected_diff_keys = {
            "run_id", "runtime_root", "rts_policy_mode", "rts_rollout_enabled",
            "rts_policy_checkpoint_dir", "rts_policy_checkpoint_id", "rts_policy_action_mode",
            "rts_torch_threads", "rts_torch_interop_threads", "timestamp",
            "rts_checkpoint_sha256", "policy_configuration"
        }
        for key in expected_diff_keys:
            current_dict.pop(key, None)
            rl_dict.pop(key, None)

        if current_dict != rl_dict:
            mismatches = {k: (current_dict[k], rl_dict[k]) for k in current_dict if current_dict[k] != rl_dict[k]}
            raise AssertionError(f"Unrelated RunSpec differences detected: {mismatches}")


def write_paired_evaluation_outputs(
    run_root: Path,
    seed_pack: dict[str, Any],
    eval_config: dict[str, Any],
    specs: list[RunSpec],
) -> None:
    from src.rmfs.orchestration.local_executor import SENSITIVITY_KPI_FIELDS
    import csv

    # 1. Load all worker summaries that exist and are successful
    worker_summaries = {}
    for spec in specs:
        summary_path = spec.runtime_root / "worker_summary.json"
        if summary_path.exists():
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                if summary.get("status") == "success":
                    worker_summaries[spec.run_id] = summary
            except Exception:
                pass

    # 2. Write seed_pack.json
    atomic_write_json(run_root / "seed_pack.json", seed_pack)

    # 3. Write eval_config.json
    atomic_write_json(run_root / "eval_config.json", eval_config)

    # 4. Write worker_specs.json
    atomic_write_json(run_root / "worker_specs.json", [spec.to_json_dict() for spec in specs])

    # Let's define the run IDs grouped by replication:
    replications_count = seed_pack["replications"]

    # 5. Write paired_replication_status.csv
    paired_rows = []
    for rep in range(1, replications_count + 1):
        seed_info = seed_pack["seeds"][rep - 1]
        seed_val = seed_info["seed"]

        current_run_id = f"current_{rep:03d}"
        rl_run_id = f"rts_rl_{rep:03d}"

        current_sum = worker_summaries.get(current_run_id, {})
        rl_sum = worker_summaries.get(rl_run_id, {})

        row = {
            "replication": rep,
            "seed": seed_val,
            "current_status": current_sum.get("status", "pending"),
            "rts_rl_status": rl_sum.get("status", "pending"),
            "current_orders_completed": current_sum.get("orders_completed"),
            "rts_rl_orders_completed": rl_sum.get("orders_completed"),
            "current_avg_cycle_time": current_sum.get("average_order_cycle_time"),
            "rts_rl_avg_cycle_time": rl_sum.get("average_order_cycle_time"),
            "current_total_energy": current_sum.get("total_energy_kj"),
            "rts_rl_total_energy": rl_sum.get("total_energy_kj"),
        }
        paired_rows.append(row)

    paired_csv_path = run_root / "paired_replication_status.csv"
    if paired_rows:
        fields = list(paired_rows[0].keys())
        tmp_csv = paired_csv_path.with_name(f".{paired_csv_path.name}.tmp")
        with tmp_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(paired_rows)
        tmp_csv.replace(paired_csv_path)

    # 6. Write full_kpi_summary.csv and .json
    kpi_rows = []
    for spec in specs:
        summary = worker_summaries.get(spec.run_id)
        if summary:
            kpi_row = {}
            for field in SENSITIVITY_KPI_FIELDS:
                kpi_row[field] = summary.get(field)
            kpi_rows.append(kpi_row)

    atomic_write_json(run_root / "full_kpi_summary.json", kpi_rows)
    full_kpi_csv_path = run_root / "full_kpi_summary.csv"
    if kpi_rows:
        fields = list(kpi_rows[0].keys())
        tmp_csv = full_kpi_csv_path.with_name(f".{full_kpi_csv_path.name}.tmp")
        with tmp_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(kpi_rows)
        tmp_csv.replace(full_kpi_csv_path)

    # 7. Write condition_summary.csv and .json
    import statistics
    condition_stats = {}
    for treatment in ("current", "rts_rl_explicit"):
        t_runs = [f"{'rts_rl' if treatment == 'rts_rl_explicit' else 'current'}_{rep:03d}" for rep in range(1, replications_count + 1)]
        t_sums = [worker_summaries[r_id] for r_id in t_runs if r_id in worker_summaries]

        stats = {"treatment": treatment, "completed_runs": len(t_sums)}
        for metric in ("orders_completed", "average_order_cycle_time", "total_energy_kj", "stop_and_go_count", "turning_count"):
            vals = [finite_float(s.get(metric)) for s in t_sums]
            vals = [v for v in vals if v is not None]
            if vals:
                stats[f"{metric}_mean"] = statistics.mean(vals)
                stats[f"{metric}_std"] = statistics.stdev(vals) if len(vals) > 1 else 0.0
                stats[f"{metric}_min"] = min(vals)
                stats[f"{metric}_max"] = max(vals)
            else:
                stats[f"{metric}_mean"] = None
                stats[f"{metric}_std"] = None
                stats[f"{metric}_min"] = None
                stats[f"{metric}_max"] = None
        condition_stats[treatment] = stats

    atomic_write_json(run_root / "condition_summary.json", condition_stats)
    cond_csv_path = run_root / "condition_summary.csv"
    if condition_stats:
        fields = list(condition_stats["current"].keys())
        tmp_csv = cond_csv_path.with_name(f".{cond_csv_path.name}.tmp")
        with tmp_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(list(condition_stats.values()))
        tmp_csv.replace(cond_csv_path)

    # 8. Write campaign_status.json
    completed_runs_count = len(worker_summaries)
    failed_runs_count = 0
    for spec in specs:
        summary_path = spec.runtime_root / "worker_summary.json"
        if summary_path.exists():
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                if summary.get("status") == "failure":
                    failed_runs_count += 1
            except Exception:
                failed_runs_count += 1

    pending_runs_count = len(specs) - completed_runs_count - failed_runs_count
    campaign_status = {
        "campaign_id": eval_config.get("eval_run_id"),
        "total_runs": len(specs),
        "completed_runs": completed_runs_count,
        "failed_runs": failed_runs_count,
        "pending_runs": pending_runs_count,
        "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    atomic_write_json(run_root / "campaign_status.json", campaign_status)

    # 9. Write eval_summary.json
    eval_summary = {
        "status": "valid" if completed_runs_count == len(specs) and failed_runs_count == 0 else "running",
        "eval_run_id": eval_config.get("eval_run_id"),
        "created_at": eval_config.get("created_at"),
        "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        **eval_config,
        "campaign_status": campaign_status,
        "condition_summary": condition_stats,
    }
    atomic_write_json(run_root / "eval_summary.json", eval_summary)


def run_rts_paired_evaluation(
    *,
    repo_root: Path,
    checkpoint_dir: Path | None,
    zone_ids: tuple[str, ...],
    output_root: Path,
    replications: int = 60,
    seed_base: int = 42,
    simulated_seconds: float = 10000.0,
    robot_count: int = 20,
    order_rate: int = 500,
    charging_enabled: bool = True,
    max_workers: int = 1,
    resume: bool = False,
    progress: bool = False,
    dry_run: bool = False,
    rts_torch_threads: int | None = 1,
    rts_torch_interop_threads: int | None = 1,
) -> dict[str, Any]:
    from src.rmfs.experiments.evaluation.seed_pack import build_seed_pack
    from src.rmfs.decisions.charging.config import canonical_charging_config_path
    from src.rmfs.orchestration.local_executor import run_specs

    repo_root = Path(repo_root)
    if max_workers < 1:
        raise ValueError("max_workers must be >= 1")

    # Verify checkpoint and calculate SHA-256 hash
    if checkpoint_dir is None:
        raise ValueError("checkpoint_dir is required for paired evaluation")
    checkpoint_dir = Path(checkpoint_dir).resolve()
    meta = verify_paired_checkpoint(checkpoint_dir)
    policy_checkpoint_id = meta.get("policy_checkpoint_id", "batch_000005")

    model_path = checkpoint_dir / "model.pt"
    import hashlib
    digest = hashlib.sha256()
    with model_path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            digest.update(chunk)
    rts_checkpoint_sha256 = digest.hexdigest()

    # Determine steps
    netlogo_steps_per_run = int(round(simulated_seconds / 0.15))
    if simulated_seconds == 10000.0:
        netlogo_steps_per_run = 66667

    # Build seed pack
    seed_pack = build_seed_pack(
        seed_base=seed_base,
        replications=replications,
        netlogo_steps_per_run=netlogo_steps_per_run,
        purpose="paired_current_vs_rl_evaluation",
    )

    eval_run_id = f"paired_current_vs_vrsla_{replications}rep_{int(simulated_seconds)}s"
    run_root = Path(output_root) / eval_run_id
    run_root.mkdir(parents=True, exist_ok=True)

    branch = git_value(repo_root, "rev-parse", "--abbrev-ref", "HEAD")
    commit = git_value(repo_root, "rev-parse", "HEAD")

    # Config dict
    eval_config = {
        "eval_run_id": eval_run_id,
        "checkpoint_dir": str(checkpoint_dir),
        "policy_checkpoint_id": policy_checkpoint_id,
        "rts_checkpoint_sha256": rts_checkpoint_sha256,
        "zone_ids": list(zone_ids) if zone_ids else ["auto"],
        "seed_pack_id": seed_pack["seed_pack_id"],
        "replications": replications,
        "seed_base": seed_base,
        "simulated_seconds": simulated_seconds,
        "netlogo_steps_per_run": netlogo_steps_per_run,
        "robot_count": robot_count,
        "order_rate": order_rate,
        "charging_enabled": charging_enabled,
        "charging_config_path": str(canonical_charging_config_path()),
        "max_workers": max_workers,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "default_lukman_allocation_path": "src/rmfs/order_generation/pod_sku.py",
    }

    # Generate the 120 interleaved run specs
    specs = []
    for rep in range(1, replications + 1):
        seed_info = seed_pack["seeds"][rep - 1]
        seed_val = seed_info["seed"]

        # Current spec
        spec_current = RunSpec(
            run_id=f"current_{rep:03d}",
            ticks=netlogo_steps_per_run,
            runtime_root=run_root / "workers" / f"current_{rep:03d}",
            repo_root=repo_root,
            branch=branch,
            commit=commit,
            python_executable=sys.executable,
            timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            rts_policy_mode="current",
            rts_rollout_enabled=False,
            rts_rollout_write_disk=False,
            rts_zone_ids=list(zone_ids) if zone_ids else ["auto"],
            rts_seed_base=seed_base,
            rts_random_seed=seed_val,
            rts_policy_checkpoint_dir=None,
            rts_policy_checkpoint_id="none",
            rts_policy_action_mode="sample",
            rts_policy_device="cpu",
            rts_feature_ablation="full",
            rts_state_capture_mode="auto",
            rts_charging_mode="enabled" if charging_enabled else "disabled",
            committed_next_reservations_enabled=True,
            experiment_id=eval_run_id,
            scenario_id=f"paired_scenario_{rep:03d}",
            artifact_label=eval_run_id,
            worker_id=rep,
            rts_torch_threads=None,
            rts_torch_interop_threads=None,
            robot_count=robot_count,
            pps_mode="heuristic",
            pps_model_path=None,
            charging_enabled=charging_enabled,
            charging_config_path=str(canonical_charging_config_path()),
            run_profile="training",
            run_horizon_ticks=netlogo_steps_per_run,
            demand_horizon_ticks=netlogo_steps_per_run + 1000,
            demand_buffer_ticks=1000,
            order_generation_mode="shuffled_historical_cycle",
            full_raw_order_replay=False,
            order_rate_per_hour=order_rate,
            pod_location_mode="randomize_slots",
            pod_location_seed=seed_val,
            kpi_schema_version="sensitivity_full_kpi.v1",
            replication=rep,
            campaign_id=eval_run_id,
            machine_id="local",
            stage_first_requested=1,
            rts_checkpoint_sha256="none",
            pps_model_sha256="none",
            policy_configuration="current",
        )

        # RTS RL spec
        spec_rl = RunSpec(
            run_id=f"rts_rl_{rep:03d}",
            ticks=netlogo_steps_per_run,
            runtime_root=run_root / "workers" / f"rts_rl_{rep:03d}",
            repo_root=repo_root,
            branch=branch,
            commit=commit,
            python_executable=sys.executable,
            timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            rts_policy_mode="rts_rl_explicit",
            rts_rollout_enabled=True,
            rts_rollout_write_disk=False,
            rts_zone_ids=list(zone_ids) if zone_ids else ["auto"],
            rts_seed_base=seed_base,
            rts_random_seed=seed_val,
            rts_policy_checkpoint_dir=str(checkpoint_dir),
            rts_policy_checkpoint_id=policy_checkpoint_id,
            rts_policy_action_mode="greedy",
            rts_policy_device="cpu",
            rts_feature_ablation="full",
            rts_state_capture_mode="auto",
            rts_charging_mode="enabled" if charging_enabled else "disabled",
            committed_next_reservations_enabled=True,
            experiment_id=eval_run_id,
            scenario_id=f"paired_scenario_{rep:03d}",
            artifact_label=eval_run_id,
            worker_id=rep,
            rts_torch_threads=rts_torch_threads,
            rts_torch_interop_threads=rts_torch_interop_threads,
            robot_count=robot_count,
            pps_mode="heuristic",
            pps_model_path=None,
            charging_enabled=charging_enabled,
            charging_config_path=str(canonical_charging_config_path()),
            run_profile="training",
            run_horizon_ticks=netlogo_steps_per_run,
            demand_horizon_ticks=netlogo_steps_per_run + 1000,
            demand_buffer_ticks=1000,
            order_generation_mode="shuffled_historical_cycle",
            full_raw_order_replay=False,
            order_rate_per_hour=order_rate,
            pod_location_mode="randomize_slots",
            pod_location_seed=seed_val,
            kpi_schema_version="sensitivity_full_kpi.v1",
            replication=rep,
            campaign_id=eval_run_id,
            machine_id="local",
            stage_first_requested=1,
            rts_checkpoint_sha256=rts_checkpoint_sha256,
            pps_model_sha256="none",
            policy_configuration="rts_rl_explicit",
        )

        specs.append(spec_current)
        specs.append(spec_rl)

    # Enforce strict plan assertions
    assert_paired_plan(specs, seed_pack, replications, netlogo_steps_per_run)

    # Initial write of outputs
    write_paired_evaluation_outputs(run_root, seed_pack, eval_config, specs)

    if dry_run:
        summary_result = {
            "status": "dry_run",
            "valid": False,
            "eval_run_id": eval_run_id,
            "created_at": eval_config["created_at"],
            **eval_config,
            "worker_count": len(specs),
        }
        atomic_write_json(run_root / "eval_summary.json", summary_result)
        return summary_result

    # If resume is enabled, find completed specs to skip
    specs_to_run = []
    skipped_count = 0
    for spec in specs:
        summary_path = spec.runtime_root / "worker_summary.json"
        if resume and summary_path.exists():
            try:
                existing_summary = json.loads(summary_path.read_text(encoding="utf-8"))
                if existing_summary.get("status") == "success":
                    skipped_count += 1
                    continue
            except Exception:
                pass
        specs_to_run.append(spec)

    print(f"[paired_eval] Starting campaign with {len(specs_to_run)} runs ({skipped_count} skipped/resumed)")

    # Define callbacks
    def progress_postfix_callback(completed_list) -> dict[str, Any]:
        completed_run_ids = {item["spec"].run_id for item in completed_list if item.get("return_code") == 0}
        current_completed = sum(1 for item in completed_list if item["spec"].rts_policy_mode == "current" and item.get("return_code") == 0)
        rts_rl_completed = sum(1 for item in completed_list if item["spec"].rts_policy_mode == "rts_rl_explicit" and item.get("return_code") == 0)
        
        # Include skipped runs in completed counts for postfix display
        for spec in specs:
            if spec.run_id not in [item["spec"].run_id for item in completed_list]:
                summary_path = spec.runtime_root / "worker_summary.json"
                if summary_path.exists():
                    try:
                        existing = json.loads(summary_path.read_text(encoding="utf-8"))
                        if existing.get("status") == "success":
                            completed_run_ids.add(spec.run_id)
                            if spec.rts_policy_mode == "current":
                                current_completed += 1
                            elif spec.rts_policy_mode == "rts_rl_explicit":
                                rts_rl_completed += 1
                    except Exception:
                        pass

        paired_completed = 0
        for r in range(1, replications + 1):
            if f"current_{r:03d}" in completed_run_ids and f"rts_rl_{r:03d}" in completed_run_ids:
                paired_completed += 1

        failed = sum(1 for item in completed_list if item.get("return_code", 0) != 0)

        return {
            "completed": len(completed_run_ids),
            "current_completed": current_completed,
            "rts_rl_completed": rts_rl_completed,
            "paired_completed": paired_completed,
            "failed": failed,
        }

    # Write refresh on each worker completion
    def on_worker_completed_callback(result, completed_list):
        write_paired_evaluation_outputs(run_root, seed_pack, eval_config, specs)

    # Run!
    failures = 0
    results = run_specs(
        specs_to_run,
        max_workers=max_workers,
        progress=progress,
        postfix_callback=progress_postfix_callback,
        on_worker_completed=on_worker_completed_callback,
    )
    for res in results:
        if int(res.get("return_code") or 0) != 0:
            failures += 1

    # Final refresh of aggregate outputs
    write_paired_evaluation_outputs(run_root, seed_pack, eval_config, specs)

    # Reclaim bulky runtime files if debug_rollouts is False
    # (By default debug_rollouts is false)
    # We clean rts_rollout.jsonl and worker_stdout.log for successful runs
    cleaned_bytes = 0
    for spec in specs:
        summary_path = spec.runtime_root / "worker_summary.json"
        if summary_path.exists():
            try:
                sum_data = json.loads(summary_path.read_text(encoding="utf-8"))
                if sum_data.get("status") == "success":
                    for name in ("rts_rollout.jsonl", "worker_stdout.log"):
                        bulky = spec.runtime_root / name
                        if bulky.exists():
                            cleaned_bytes += bulky.stat().st_size
                            bulky.unlink()
            except Exception:
                pass
    if cleaned_bytes > 0:
        print(f"[paired_eval] Cleaned {cleaned_bytes / 1_000_000:.2f} MB of temporary rollout/log files")

    # Load final eval_summary
    with (run_root / "eval_summary.json").open("r", encoding="utf-8") as fh:
        final_summary = json.load(fh)
    return final_summary


