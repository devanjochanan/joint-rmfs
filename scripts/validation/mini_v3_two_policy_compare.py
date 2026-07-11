#!/usr/bin/env python3
"""Run a two-condition v3 sensitivity mini-compare.

This intentionally does not materialize or execute the full 720-run campaign.
It reuses the current distributed_sensitivity_campaign v3 treatment contracts,
asset resolution, charging identity generation, and RunSpec builder for exactly:

* all_off, cindy_s3, 20 robots, 500 orders/hour, replication 1
* all_on_rl, scenario4_sij, 20 robots, 500 orders/hour, replication 1
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any


def _final_metric(summary: dict[str, Any], key: str) -> Any:
    if key in summary:
        return summary.get(key)
    kpi = summary.get("kpi")
    if isinstance(kpi, dict) and key in kpi:
        return kpi.get(key)
    final_metrics = summary.get("final_metrics")
    aliases = {
        "orders_completed": "warehouse_orders_completed",
        "average_order_cycle_time": "warehouse_average_order_cycle_time",
        "total_energy": "total_energy",
        "stop_and_go": "stop_and_go",
        "total_turning": "total_turning",
        "job_queue_len": "job_queue_len",
    }
    if isinstance(final_metrics, dict):
        return final_metrics.get(key, final_metrics.get(aliases.get(key, "")))
    return None


def _condition(
    campaign,
    *,
    policy: str,
    scenario_name: str,
    scenario_root: Path,
    run_id: str,
    ticks: int,
) -> dict[str, Any]:
    robot_count = 20
    order_rate = 500
    replication = 1
    seed = campaign.seed_for_replication(replication)
    return {
        "condition_key": campaign.condition_key(policy, robot_count, order_rate, replication),
        "policy_configuration": policy,
        "robot_count": robot_count,
        "order_rate": order_rate,
        "picker_count": campaign.PICKER_COUNT,
        "replenishment_count": campaign.REPLENISHMENT_COUNT,
        "replication": replication,
        "seed": seed,
        "paired_group_id": campaign.paired_group_id(robot_count, order_rate, replication),
        "stage_first_requested": 1,
        "charging": campaign.condition_charging_identity(
            policy,
            robot_count,
            seed,
            scenario_root / "generated_pod.csv",
        ),
        "run_id": run_id,
        "ticks": ticks,
        "simulated_seconds": ticks * campaign.TICK_TO_SECOND,
        "tick_to_second": campaign.TICK_TO_SECOND,
        "branch": campaign.git_value(campaign.REPO_ROOT, "rev-parse", "--abbrev-ref", "HEAD"),
        "commit": campaign.git_value(campaign.REPO_ROOT, "rev-parse", "HEAD"),
        "allocation_patch_id": "manual_v3_two_policy_1000_ticks",
        "simulation_semantics_id": campaign.SIMULATION_SEMANTICS_ID,
        "scenario_id": scenario_name,
        "scenario_hash": scenario_name,
        "layout_hash": campaign.sha256_file(scenario_root / "generated_pod.csv"),
        "source_tree_hash": None,
        "machine_id": "local_mini",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a two-condition v3 sensitivity mini-compare.")
    parser.add_argument(
        "--backend-steps",
        type=int,
        default=6667,
        help="Backend / NetLogo steps to run. 6667 steps is approximately 1000 simulated seconds.",
    )
    args = parser.parse_args()
    if args.backend_steps <= 0:
        parser.error("--backend-steps must be positive")

    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))

    from scripts.experiments import distributed_sensitivity_campaign as campaign
    from src.rmfs.orchestration.local_executor import load_worker_summary, run_specs

    ticks = int(args.backend_steps)
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_root = repo_root / "data" / "runtime" / "tmp" / f"mini_v3_two_policy_compare_{timestamp}"

    checkpoint_dir = repo_root / "data" / "models" / "rts" / "batch_000014" / "checkpoint"
    pps_path = repo_root / "data" / "models" / "pps" / "pps_rl_policy_inference.zip"
    assets = campaign.AssetBundle(
        pps_model_relative_path=pps_path.relative_to(repo_root).as_posix(),
        pps_model_sha256=campaign.sha256_file(pps_path),
        pps_observation_schema={},
        rts_checkpoint_relative_dir=checkpoint_dir.relative_to(repo_root).as_posix(),
        rts_checkpoint_id=campaign.CANONICAL_RTS_CHECKPOINT_ID,
        rts_model_sha256=campaign.sha256_file(checkpoint_dir / "model.pt"),
        rts_metadata_sha256=campaign.sha256_file(checkpoint_dir / "metadata.json"),
        rts_feature_schema_sha256=campaign.sha256_file(checkpoint_dir / "feature_schema.json"),
        rts_feature_schema_id="not_validated_for_mini_run",
        rts_training_artifact="batch_000014",
        rts_training_latest_relative_path="",
        rts_lineage={},
        rts_lineage_source_relative_dir=None,
        charging_config_relative_path="",
        charging_config_sha256="",
    )
    assets_dict = asdict(assets)
    contract_off = campaign.treatment_execution_contract("all_off", assets)
    contract_on = campaign.treatment_execution_contract("all_on_rl", assets)
    if contract_off["pps_mode"] != "heuristic":
        raise RuntimeError(f"unexpected v3 all_off pps_mode: {contract_off['pps_mode']}")
    if contract_on["pps_mode"] != "ppo_constrained":
        raise RuntimeError(f"unexpected v3 all_on_rl pps_mode: {contract_on['pps_mode']}")

    machine = campaign.Machine(
        machine_id="local_mini",
        os=sys.platform,
        repository=str(repo_root),
        python=sys.executable,
        max_workers=1,
        effective_steps_per_second=1.0,
        eligible_stages=(1,),
    )

    common_manifest = {
        "schema_version": campaign.CAMPAIGN_SCHEMA_VERSION,
        "campaign_id": f"mini_v3_two_policy_compare_{timestamp}",
        "simulation_semantics_id": campaign.SIMULATION_SEMANTICS_ID,
        "allocation_patch_id": "manual_v3_two_policy_1000_ticks",
        "campaign_root_relative": output_root.relative_to(repo_root).as_posix(),
        "assets": assets_dict,
        "branch": campaign.git_value(repo_root, "rev-parse", "--abbrev-ref", "HEAD"),
        "commit": campaign.git_value(repo_root, "rev-parse", "HEAD"),
        "kpi_schema_version": campaign.FULL_KPI_V3_SCHEMA_VERSION,
        "machines": [asdict(machine)],
    }

    scenarios = {
        "all_off": ("cindy_s3", repo_root / "data" / "input" / "scenarios" / "cindy_s3"),
        "all_on_rl": ("scenario4_sij", repo_root / "data" / "input" / "scenarios" / "scenario4_sij"),
    }

    specs = []
    for policy in ("all_off", "all_on_rl"):
        scenario_name, scenario_root = scenarios[policy]
        manifest = {
            **common_manifest,
            "input_root_relative": scenario_root.relative_to(repo_root).as_posix(),
        }
        run_id = f"{policy}__{scenario_name}__r20__arr500__rep001__1000ticks"
        condition = _condition(
            campaign,
            policy=policy,
            scenario_name=scenario_name,
            scenario_root=scenario_root,
            run_id=run_id,
            ticks=ticks,
        )
        spec = campaign.build_run_spec_from_condition(
            condition,
            manifest=manifest,
            machine=machine,
            repo_root=repo_root,
        )
        specs.append(spec)

    completed = run_specs(specs, max_workers=1, progress=True)
    completed_normalized = []
    for item in completed:
        row = dict(item)
        spec = row.pop("spec", None)
        row.pop("worker_summary", None)
        row.pop("kpi_payload", None)
        if spec is not None:
            row["run_id"] = spec.run_id
            row["runtime_root"] = str(spec.runtime_root)
        completed_normalized.append(row)

    runs = []
    for spec in specs:
        summary = load_worker_summary(spec.runtime_root)
        runs.append({
            "policy_configuration": spec.policy_configuration,
            "scenario_id": spec.scenario_id,
            "status": summary.get("status"),
            "ticks_completed": summary.get("netlogo_steps_completed") or summary.get("ticks_completed"),
            "ticks_requested": summary.get("netlogo_steps_requested") or summary.get("ticks_requested"),
            "pps_mode": summary.get("pps_mode"),
            "rts_policy_mode": summary.get("rts_policy_mode"),
            "charging_enabled": summary.get("charging_enabled"),
            "effective_charger_count": summary.get("effective_charger_count"),
            "orders_completed": _final_metric(summary, "orders_completed"),
            "average_order_cycle_time": _final_metric(summary, "average_order_cycle_time"),
            "total_energy": _final_metric(summary, "total_energy"),
            "stop_and_go": _final_metric(summary, "stop_and_go"),
            "total_turning": _final_metric(summary, "total_turning"),
            "job_queue_len": _final_metric(summary, "job_queue_len"),
            "order_throughput": _final_metric(summary, "order_throughput"),
            "kpi_complete": summary.get("kpi_complete"),
            "worker_summary": str(spec.runtime_root / "worker_summary.json"),
        })

    payload = {
        "output_root": str(output_root),
        "ticks": ticks,
        "robot_count": 20,
        "order_rate": 500,
        "replication": 1,
        "simulation_semantics_id": campaign.SIMULATION_SEMANTICS_ID,
        "kpi_schema_version": campaign.FULL_KPI_V3_SCHEMA_VERSION,
        "note": "Two selected v3 campaign-style RunSpecs only; this does not run the 720-condition campaign.",
        "completed_processes": completed_normalized,
        "runs": runs,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if all(row["status"] == "success" for row in runs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
