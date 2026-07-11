#!/usr/bin/env python3
"""Run a 1000-tick short-horizon comparison for requested RMFS policies."""

from __future__ import annotations

import datetime as _dt
import json
import sys
from pathlib import Path


def _metric(summary: dict, name: str):
    if name in summary:
        return summary.get(name)
    kpi = summary.get("kpi")
    if isinstance(kpi, dict):
        return kpi.get(name)
    final_metrics = summary.get("final_metrics")
    if isinstance(final_metrics, dict):
        aliases = {
            "orders_completed": "warehouse_orders_completed",
            "average_order_cycle_time": "warehouse_average_order_cycle_time",
            "total_energy_kj": "total_energy",
        }
        return final_metrics.get(name, final_metrics.get(aliases.get(name, "")))
    return None


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))

    from src.rmfs.orchestration.local_executor import git_value, load_worker_summary, run_specs
    from src.rmfs.orchestration.run_spec import RunSpec
    from scripts.data.build_adaptive_hybrid import build as build_salsa_adaptive_config
    from scripts.data.build_adaptive_hybrid import load_grid as load_salsa_grid
    from scripts.data.build_baseline_random import build as build_reference_config
    from scripts.data.build_baseline_random import load_grid as load_reference_grid

    timestamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_root = repo_root / "data" / "runtime" / "tmp" / f"short_horizon_policy_compare_{timestamp}"
    output_root.mkdir(parents=True, exist_ok=True)

    branch = git_value(repo_root, "rev-parse", "--abbrev-ref", "HEAD")
    commit = git_value(repo_root, "rev-parse", "HEAD")
    rts_checkpoint_dir = repo_root / "data" / "models" / "rts" / "batch_000014" / "checkpoint"
    pps_model_path = repo_root / "data" / "models" / "pps" / "pps_rl_policy_inference.zip"
    cindy_s3_root = repo_root / "data" / "input" / "scenarios" / "cindy_s3"
    scenario4_root = repo_root / "data" / "input" / "scenarios" / "scenario4_sij"

    off_charging_config_path = output_root / "all_off__cindy_s3__1000ticks" / "input_snapshot" / "reference_charging.json"
    off_charging_config_path.parent.mkdir(parents=True, exist_ok=True)
    off_config = build_reference_config(
        load_reference_grid(cindy_s3_root / "generated_pod.csv"),
        num_chargers=10,
        seed=42,
    )
    off_charging_config_path.write_text(json.dumps(off_config, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    on_charging_config_path = output_root / "all_on_rl__scenario4_sij__1000ticks" / "input_snapshot" / "salsa_adaptive_charging.json"
    on_charging_config_path.parent.mkdir(parents=True, exist_ok=True)
    on_config, _picker, _depot = build_salsa_adaptive_config(
        load_salsa_grid(scenario4_root / "generated_pod.csv"),
        n_robots=20,
        rho=0.6,
    )
    on_charging_config_path.write_text(json.dumps(on_config, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    common = {
        "ticks": 1000,
        "repo_root": repo_root,
        "branch": branch,
        "commit": commit,
        "python_executable": sys.executable,
        "timestamp": timestamp,
        "debug_trace": False,
        "trace_cadence": 1000,
        "trace_first_n": 0,
        "robot_count": 20,
        "expected_picking_station_count": 3,
        "expected_replenishment_station_count": 1,
        "persist_final_state": False,
        "keep_runtime_artifacts": True,
        "detail_db": False,
        "timing": False,
        "worker_status_cadence": 1000,
        "run_profile": "training",
        "run_horizon_ticks": 1000,
        "demand_horizon_ticks": 2000,
        "demand_buffer_ticks": 1000,
        "order_generation_mode": "shuffled_historical_cycle",
        "full_raw_order_replay": False,
        "order_rate_per_hour": 500,
        "pod_location_mode": "fixed",
        "pod_location_seed": 42,
        "experiment_id": "short_horizon_policy_compare",
        "batch_id": 1,
        "robot_task_allocator": "legacy_nearest",
        "regret_k": None,
        "task_allocator_scope": "active_job_queue",
        "rts_torch_threads": 1,
        "rts_torch_interop_threads": 1,
        "campaign_id": "short_horizon_policy_compare",
        "allocation_patch_id": "manual_1000_tick_pair",
        "simulation_semantics_id": "manual.short_horizon.v1",
        "machine_id": "local",
        "stage_first_requested": 1,
        "kpi_schema_version": "manual_short_horizon.v1",
        "replication": 1,
        "campaign_seed": 42,
    }

    specs = [
        RunSpec(
            **common,
            run_id="all_off__cindy_s3__1000ticks",
            runtime_root=output_root / "all_off__cindy_s3__1000ticks",
            input_root=cindy_s3_root,
            rts_policy_mode="current",
            rts_rollout_enabled=False,
            rts_rollout_write_disk=False,
            rts_seed_base=42,
            rts_random_seed=42,
            rts_policy_checkpoint_dir=None,
            rts_policy_checkpoint_id="not_applicable",
            rts_policy_action_mode="sample",
            pps_mode="heuristic",
            pps_model_path=None,
            charging_enabled=False,
            charging_config_path=str(off_charging_config_path),
            charging_placement_source="generated_reference",
            committed_next_reservations_enabled=False,
            policy_configuration="all_off",
            scenario_id="cindy_s3",
            artifact_label="all_off__cindy_s3__1000ticks",
            worker_id=1,
        ),
        RunSpec(
            **common,
            run_id="all_on_rl__scenario4_sij__1000ticks",
            runtime_root=output_root / "all_on_rl__scenario4_sij__1000ticks",
            input_root=scenario4_root,
            rts_policy_mode="rts_rl_explicit",
            rts_rollout_enabled=True,
            rts_rollout_write_disk=False,
            rts_zone_ids=["auto"],
            rts_seed_base=42,
            rts_random_seed=42,
            rts_policy_checkpoint_dir=str(rts_checkpoint_dir),
            rts_policy_checkpoint_id="batch_000014",
            rts_policy_action_mode="greedy",
            rts_policy_device="cpu",
            rts_feature_ablation="full",
            rts_state_capture_mode="full",
            pps_mode="ppo_constrained",
            pps_model_path=str(pps_model_path),
            charging_enabled=True,
            charging_config_path=str(on_charging_config_path),
            charging_placement_source="generated_salsa_adaptive",
            committed_next_reservations_enabled=True,
            policy_configuration="all_on_rl",
            scenario_id="scenario4_sij",
            artifact_label="all_on_rl__scenario4_sij__1000ticks",
            worker_id=2,
        ),
    ]

    completed = run_specs(specs, max_workers=1, progress=True)
    rows = []
    for spec in specs:
        summary = load_worker_summary(spec.runtime_root)
        rows.append({
            "policy_configuration": spec.policy_configuration,
            "scenario_id": spec.scenario_id,
            "status": summary.get("status"),
            "ticks_requested": spec.ticks,
            "ticks_completed": summary.get("ticks_completed") or summary.get("netlogo_steps_completed"),
            "orders_completed": _metric(summary, "orders_completed"),
            "order_throughput": _metric(summary, "order_throughput"),
            "average_order_cycle_time": _metric(summary, "average_order_cycle_time"),
            "total_robot_distance": _metric(summary, "total_robot_distance"),
            "total_energy_kj": _metric(summary, "total_energy_kj"),
            "pps_mode": summary.get("pps_mode"),
            "rts_policy_mode": summary.get("rts_policy_mode"),
            "charging_enabled": summary.get("charging_enabled"),
            "worker_summary": str(spec.runtime_root / "worker_summary.json"),
        })

    completed_processes = []
    for item in completed:
        normalized = dict(item)
        spec = normalized.pop("spec", None)
        if spec is not None:
            normalized["run_id"] = getattr(spec, "run_id", None)
            normalized["runtime_root"] = str(getattr(spec, "runtime_root", ""))
        completed_processes.append(normalized)

    result = {
        "output_root": str(output_root),
        "completed_processes": completed_processes,
        "comparison_note": (
            "The requested policies use different scenarios, so differences are not a controlled "
            "policy-only effect."
        ),
        "runs": rows,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if all(row["status"] == "success" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
