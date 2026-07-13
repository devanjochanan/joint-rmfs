"""Balanced, resumable local campaign for RTS--RL versus Nearest evaluation."""

from __future__ import annotations

import csv
import datetime
import json
from pathlib import Path
import shutil
import sys
from typing import Any

from src.rmfs.experiments.identity import short_hash
from src.rmfs.orchestration.kpi_schema import (
    FULL_KPI_V3_FIELDS,
    FULL_KPI_V3_SCHEMA_VERSION,
    FULL_KPI_V3_SIDECAR_FIELDS,
)
from src.rmfs.orchestration.local_executor import git_value, load_worker_summary, run_specs
from src.rmfs.orchestration.run_spec import RunSpec
from src.rmfs.rl.rts.ablation import resolve_ablation
from src.rmfs.rl.rts.training.metrics import append_jsonl, atomic_write_json
from src.rmfs.rl.rts.training.policy_loader import load_policy_from_checkpoint
from src.rmfs.runtime_io.run_profiles import DEFAULT_RTS_ORDER_RATE_PER_HOUR


KPI_ID_FIELDS = (
    "run_id",
    "seed",
    "replication",
    "policy_configuration",
    "campaign_id",
    "simulated_seconds",
    "kpi_schema_version",
    "repo_commit",
)
KPI_STATUS_FIELDS = (
    "kpi_complete",
    "kpi_complete_strict",
    "kpi_completion_status",
)


def run_paired_rts_rl_vs_nearest_evaluation(
    *,
    repo_root: Path,
    checkpoint_dir: Path,
    zone_ids: tuple[str, ...],
    seed_pack_path: Path,
    output_root: Path,
    policy_action_mode: str = "greedy",
    feature_ablation: str = "full",
    charging_mode: str = "inherit",
    dry_run: bool = False,
    rts_torch_threads: int | None = None,
    rts_torch_interop_threads: int | None = None,
    state_capture_mode: str = "auto",
    max_workers: int = 8,
    pairs_per_wave: int | None = None,
    machine_id: str = "local",
    resume_campaign_id: str | None = None,
) -> dict[str, Any]:
    """Run both policy arms through one queue with terminal-wave barriers.

    The campaign owns a single local-executor queue.  With the requested
    eight-worker machine budget, every wave launches four Nearest and four
    RTS--RL workers.  It will not admit the next wave until all eight terminal
    records from the active wave have arrived.  This keeps the arms balanced,
    gives the controller one progress bar, and avoids the two independent
    queues that otherwise drift apart.

    A compact outcome JSONL is appended before every worker root is removed.
    It permits resume after interruption without retaining rollout, log, or
    worker-summary artifacts.
    """
    repo_root = Path(repo_root)
    checkpoint = Path(checkpoint_dir)
    if int(max_workers) < 2 or int(max_workers) % 2 != 0:
        raise ValueError("max_workers must be an even integer >= 2")
    if pairs_per_wave is None:
        pairs_per_wave = int(max_workers) // 2
    if int(pairs_per_wave) < 1 or 2 * int(pairs_per_wave) > int(max_workers):
        raise ValueError("pairs_per_wave must be >= 1 and use no more than max_workers slots")
    if charging_mode not in {"inherit", "enabled", "disabled"}:
        raise ValueError("charging_mode must be inherit, enabled, or disabled")
    if not str(machine_id).strip():
        raise ValueError("machine_id must be nonblank")

    with Path(seed_pack_path).open(encoding="utf-8") as fh:
        seed_pack = json.load(fh)
    seeds = sorted((dict(seed) for seed in seed_pack.get("seeds", [])), key=lambda seed: int(seed["replication"]))
    if len(seeds) != int(seed_pack.get("replications") or 0):
        raise ValueError("seed pack replications does not match its seed records")
    if len({int(seed["seed"]) for seed in seeds}) != len(seeds):
        raise ValueError("paired evaluation requires unique seeds")
    if len({int(seed["replication"]) for seed in seeds}) != len(seeds):
        raise ValueError("paired evaluation requires unique replication identifiers")

    loaded = load_policy_from_checkpoint(checkpoint, device="cpu")
    ablation = resolve_ablation(feature_ablation)
    branch = git_value(repo_root, "rev-parse", "--abbrev-ref", "HEAD")
    commit = git_value(repo_root, "rev-parse", "HEAD")
    charging_config_path = repo_root / "data" / "input" / "charging" / "salsa_charging_config.json"
    charging_enabled = {"inherit": None, "enabled": True, "disabled": False}[charging_mode]
    allocation_patch_id = "rts_eval_local_executor.v1"
    simulation_semantics_id = "rts_eval_simulation_semantics.v1"
    config = {
        "campaign_kind": "paired_rts_rl_vs_nearest.v1",
        "repo_commit": commit,
        "checkpoint_dir": str(checkpoint),
        "policy_checkpoint_id": loaded.policy_checkpoint_id,
        "policies": ["current", "rts_rl_explicit"],
        "machine_id": str(machine_id),
        "zone_ids": list(zone_ids) if zone_ids else ["auto"],
        "seed_pack_id": seed_pack["seed_pack_id"],
        "netlogo_steps_per_run": int(seed_pack["netlogo_steps_per_run"]),
        "replications_per_policy": len(seeds),
        "nearest_policy_action_mode": "sample",
        "nearest_rts_rollout_enabled": False,
        "rts_rl_policy_action_mode": policy_action_mode,
        "rts_rl_rollout_enabled": True,
        "feature_ablation": ablation.name,
        "feature_ablation_hash": ablation.hash,
        "charging_mode": charging_mode,
        "charging_enabled": charging_enabled,
        "charging_config_path": str(charging_config_path),
        "charging_placement_source": "legacy_union",
        "state_capture_mode": state_capture_mode,
        "kpi_schema_version": FULL_KPI_V3_SCHEMA_VERSION,
        "allocation_patch_id": allocation_patch_id,
        "simulation_semantics_id": simulation_semantics_id,
        "rts_torch_threads": rts_torch_threads,
        "rts_torch_interop_threads": rts_torch_interop_threads,
        "run_profile": "training",
        "order_generation_mode": "shuffled_historical_cycle",
        "full_raw_order_replay": False,
        "order_rate_per_hour": DEFAULT_RTS_ORDER_RATE_PER_HOUR,
        "tick_to_second": 0.15,
        "max_workers": int(max_workers),
        "pairs_per_wave": int(pairs_per_wave),
        "schedule": "balanced_terminal_wave_barrier.v1",
    }
    computed_campaign_id = f"paired_eval_{short_hash(config)}"
    campaign_id = str(resume_campaign_id).strip() if resume_campaign_id is not None else computed_campaign_id
    if not campaign_id:
        raise ValueError("resume_campaign_id must be nonblank when supplied")
    run_root = Path(output_root) / campaign_id
    resumed = resume_campaign_id is not None
    existing_config: dict[str, Any] | None = None
    if resumed:
        config_path = run_root / "campaign_config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"cannot resume missing paired campaign: {config_path}")
        with config_path.open(encoding="utf-8") as fh:
            existing_config = json.load(fh)
        _validate_resume_scientific_identity(existing_config, config)
    run_root.mkdir(parents=True, exist_ok=True)
    outcomes_path = run_root / "run_outcomes.jsonl"
    scenario_id = f"paired_eval_scenario_{short_hash(config)}"

    all_specs: list[RunSpec] = []
    wave_by_run_id: dict[str, int] = {}
    wave_totals: dict[int, int] = {}
    created_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
    for wave_index, start in enumerate(range(0, len(seeds), int(pairs_per_wave))):
        wave_seeds = seeds[start:start + int(pairs_per_wave)]
        wave_totals[wave_index] = 2 * len(wave_seeds)
        for seed in wave_seeds:
            replication = int(seed["replication"])
            random_seed = int(seed["seed"])
            for arm_index, (arm_label, policy_mode, arm_checkpoint, checkpoint_id, action_mode, rollout_enabled) in enumerate((
                # These arm settings reproduce the original 60-pair comparison:
                # Nearest has no RTS rollout; the explicit policy is greedy and
                # records only in-memory state needed for its policy actor.
                ("nearest", "current", None, None, "sample", False),
                ("rts_rl", "rts_rl_explicit", checkpoint, loaded.policy_checkpoint_id, policy_action_mode, True),
            )):
                run_id = f"{arm_label}_{replication:03d}"
                all_specs.append(
                    RunSpec(
                        run_id=run_id,
                        ticks=int(seed_pack["netlogo_steps_per_run"]),
                        runtime_root=run_root / "workers" / run_id,
                        repo_root=repo_root,
                        branch=branch,
                        commit=commit,
                        python_executable=sys.executable,
                        timestamp=created_at,
                        rts_policy_mode=policy_mode,
                        rts_rollout_enabled=rollout_enabled,
                        rts_rollout_write_disk=False,
                        rts_zone_ids=list(zone_ids) if zone_ids else ["auto"],
                        rts_seed_base=int(seed_pack["seed_base"]),
                        rts_random_seed=random_seed,
                        rts_policy_checkpoint_dir=str(arm_checkpoint) if arm_checkpoint is not None else None,
                        rts_policy_checkpoint_id=checkpoint_id,
                        rts_policy_action_mode=action_mode,
                        rts_policy_device="cpu",
                        rts_feature_ablation=ablation.name,
                        rts_feature_ablation_hash=ablation.hash,
                        rts_state_capture_mode=state_capture_mode,
                        rts_charging_mode=charging_mode,
                        charging_enabled=charging_enabled,
                        charging_config_path=str(charging_config_path),
                        charging_placement_source="legacy_union",
                        committed_next_reservations_enabled=True,
                        robot_count=20,
                        pps_mode="heuristic",
                        experiment_id=campaign_id,
                        scenario_id=scenario_id,
                        artifact_label=campaign_id,
                        worker_id=(2 * replication) + arm_index,
                        campaign_id=campaign_id,
                        allocation_patch_id=allocation_patch_id,
                        simulation_semantics_id=simulation_semantics_id,
                        machine_id=str(machine_id),
                        stage_first_requested=wave_index + 1,
                        kpi_schema_version=FULL_KPI_V3_SCHEMA_VERSION,
                        policy_configuration=policy_mode,
                        replication=replication,
                        campaign_seed=random_seed,
                        pps_model_sha256="none",
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
                        pod_location_seed=random_seed,
                    )
                )
                wave_by_run_id[run_id] = wave_index

    worker_specs_name = "worker_specs_resume_latest.json" if resumed else "worker_specs.json"
    if not resumed:
        atomic_write_json(run_root / "campaign_config.json", config)
    else:
        append_jsonl(
            run_root / "resume_operations.jsonl",
            {
                "resumed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "campaign_id": campaign_id,
                "previous_max_workers": existing_config.get("max_workers") if existing_config else None,
                "previous_pairs_per_wave": existing_config.get("pairs_per_wave") if existing_config else None,
                "current_max_workers": int(max_workers),
                "current_pairs_per_wave": int(pairs_per_wave),
                "current_rts_torch_threads": rts_torch_threads,
                "current_rts_torch_interop_threads": rts_torch_interop_threads,
                "current_machine_id": str(machine_id),
                "previous_repo_commit": existing_config.get("repo_commit") if existing_config else None,
                "current_repo_commit": config["repo_commit"],
                "repo_commit_changed": (
                    existing_config.get("repo_commit") != config["repo_commit"] if existing_config else False
                ),
            },
        )
    atomic_write_json(run_root / worker_specs_name, [spec.to_json_dict() for spec in all_specs])

    if dry_run:
        summary = {
            "status": "dry_run",
            "valid": False,
            "campaign_id": campaign_id,
            "created_at": created_at,
            **config,
            "worker_count": len(all_specs),
            "wave_count": len(wave_totals),
            "worker_artifacts_persisted": False,
            "resumed": resumed,
            "worker_specs_path": str(run_root / worker_specs_name),
        }
        atomic_write_json(run_root / "campaign_summary.json", summary)
        shutil.rmtree(run_root / "workers", ignore_errors=True)
        return summary

    latest_outcomes = _read_latest_outcomes(outcomes_path)
    previous_successes = {
        run_id: outcome for run_id, outcome in latest_outcomes.items() if _outcome_is_success(outcome)
    }
    terminal_by_wave = {wave_index: 0 for wave_index in wave_totals}
    for run_id in previous_successes:
        terminal_by_wave[wave_by_run_id[run_id]] += 1
    specs_to_run = [spec for spec in all_specs if spec.run_id not in previous_successes]
    skipped_completed_workers = len(previous_successes)
    executor_error: dict[str, str] | None = None

    def before_launch(spec: RunSpec, _active_count: int) -> bool:
        wave_index = wave_by_run_id[spec.run_id]
        return all(
            terminal_by_wave[previous_wave] >= wave_totals[previous_wave]
            for previous_wave in range(wave_index)
        )

    def record_and_cleanup(spec: RunSpec, return_code: int) -> None:
        worker_summary = load_worker_summary(spec.runtime_root)
        outcome = {
            "run_id": spec.run_id,
            "replication": spec.replication,
            "seed": spec.campaign_seed,
            "policy_configuration": spec.policy_configuration,
            "wave_index": wave_by_run_id[spec.run_id],
            "return_code": int(return_code),
            "status": worker_summary.get("status", "failure"),
            "error_type": worker_summary.get("error_type"),
            "error_message": worker_summary.get("error_message"),
            "kpi_row": _full_kpi_row(worker_summary),
            "kpi_sidecar_row": _full_kpi_sidecar_row(worker_summary),
            "recorded_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
        append_jsonl(outcomes_path, outcome)
        latest_outcomes[spec.run_id] = outcome
        terminal_by_wave[wave_by_run_id[spec.run_id]] += 1
        shutil.rmtree(spec.runtime_root, ignore_errors=True)

    try:
        run_specs(
            specs_to_run,
            max_workers=int(max_workers),
            progress=True,
            before_launch=before_launch,
            on_run_complete=record_and_cleanup,
        )
    except Exception as exc:
        executor_error = {"error_type": type(exc).__name__, "error_message": str(exc)}
    finally:
        # Covers failures before a completion callback, as well as normal runs.
        shutil.rmtree(run_root / "workers", ignore_errors=True)

    latest_outcomes = _read_latest_outcomes(outcomes_path)
    successful_outcomes = {
        run_id: outcome for run_id, outcome in latest_outcomes.items() if _outcome_is_success(outcome)
    }
    full_kpi_rows = [
        successful_outcomes[spec.run_id]["kpi_row"]
        for spec in all_specs
        if spec.run_id in successful_outcomes
    ]
    full_kpi_rows = _write_kpi_rows(run_root, full_kpi_rows)
    full_kpi_sidecar_rows = [
        _sidecar_row_from_outcome(successful_outcomes[spec.run_id])
        for spec in all_specs
        if spec.run_id in successful_outcomes
    ]
    full_kpi_sidecar_rows = _write_kpi_sidecar_rows(run_root, full_kpi_sidecar_rows)
    completed_per_policy = {
        policy: sum(1 for row in full_kpi_rows if row.get("policy_configuration") == policy)
        for policy in ("current", "rts_rl_explicit")
    }
    valid = (
        executor_error is None
        and len(full_kpi_rows) == len(all_specs)
        and all(count == len(seeds) for count in completed_per_policy.values())
    )
    summary = {
        "status": "valid" if valid else "invalid",
        "valid": valid,
        "campaign_id": campaign_id,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        **config,
        "worker_count": len(all_specs),
        "resumed": resumed,
        "skipped_completed_workers": skipped_completed_workers,
        "completed_per_policy": completed_per_policy,
        "failed_or_incomplete_workers": len(all_specs) - len(full_kpi_rows),
        "wave_count": len(wave_totals),
        "wave_pair_counts": [wave_totals[index] // 2 for index in sorted(wave_totals)],
        "worker_artifacts_persisted": False,
        "resume_ledger_path": str(outcomes_path),
        "worker_specs_path": str(run_root / worker_specs_name),
        "executor_error": executor_error,
        "full_kpi_summary": {
            "schema_version": FULL_KPI_V3_SCHEMA_VERSION,
            "row_count": len(full_kpi_rows),
            "expected_row_count": len(all_specs),
            "json_path": str(run_root / "full_kpi_summary.json"),
            "csv_path": str(run_root / "full_kpi_summary.csv"),
            "fields": [*KPI_ID_FIELDS, *KPI_STATUS_FIELDS, *FULL_KPI_V3_FIELDS],
        },
        "full_kpi_sidecars": {
            "row_count": len(full_kpi_sidecar_rows),
            "expected_row_count": len(all_specs),
            "json_path": str(run_root / "full_kpi_sidecars.json"),
            "fields": [*KPI_ID_FIELDS, *FULL_KPI_V3_SIDECAR_FIELDS],
        },
    }
    atomic_write_json(run_root / "campaign_summary.json", summary)
    return summary


def _full_kpi_row(summary: dict[str, Any]) -> dict[str, Any] | None:
    if summary.get("status") != "success":
        return None
    payload = summary.get("kpi")
    if not isinstance(payload, dict):
        return None
    row = {field: summary.get(field, payload.get(field)) for field in KPI_ID_FIELDS}
    row.update({field: payload.get(field) for field in KPI_STATUS_FIELDS})
    row.update({field: payload.get(field) for field in FULL_KPI_V3_FIELDS})
    return row


def _write_kpi_rows(run_root: Path, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = [dict(row) for row in rows if isinstance(row, dict)]
    rows.sort(key=lambda row: (int(row.get("replication") or 0), str(row.get("run_id") or "")))
    atomic_write_json(run_root / "full_kpi_summary.json", rows)
    output_fields = (*KPI_ID_FIELDS, *KPI_STATUS_FIELDS, *FULL_KPI_V3_FIELDS)
    with (run_root / "full_kpi_summary.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(output_fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return rows


def _full_kpi_sidecar_row(summary: dict[str, Any]) -> dict[str, Any] | None:
    if summary.get("status") != "success":
        return None
    payload = summary.get("kpi")
    if not isinstance(payload, dict):
        return None
    row = {field: summary.get(field, payload.get(field)) for field in KPI_ID_FIELDS}
    row.update({field: payload.get(field) for field in FULL_KPI_V3_SIDECAR_FIELDS})
    return row


def _sidecar_row_from_outcome(outcome: dict[str, Any]) -> dict[str, Any] | None:
    row = outcome.get("kpi_sidecar_row")
    if isinstance(row, dict):
        return row
    # Outcomes recorded before sidecar persistence retain the flat KPI identity,
    # but their structured station detail cannot be reconstructed after worker
    # cleanup. Keep an explicit null placeholder rather than silently dropping
    # the paired observation from the sidecar file.
    flat_row = outcome.get("kpi_row")
    if not isinstance(flat_row, dict):
        return None
    row = {field: flat_row.get(field) for field in KPI_ID_FIELDS}
    row.update({field: None for field in FULL_KPI_V3_SIDECAR_FIELDS})
    return row


def _write_kpi_sidecar_rows(run_root: Path, rows: list[dict[str, Any] | None]) -> list[dict[str, Any]]:
    rows = [dict(row) for row in rows if isinstance(row, dict)]
    rows.sort(key=lambda row: (int(row.get("replication") or 0), str(row.get("run_id") or "")))
    atomic_write_json(run_root / "full_kpi_sidecars.json", rows)
    return rows


def _read_latest_outcomes(path: Path) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return latest
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                outcome = json.loads(line)
            except json.JSONDecodeError:
                continue
            run_id = str(outcome.get("run_id") or "")
            if run_id:
                latest[run_id] = outcome
    return latest


def _outcome_is_success(outcome: dict[str, Any]) -> bool:
    try:
        return_code = int(outcome.get("return_code", -1))
    except (TypeError, ValueError):
        return_code = -1
    return (
        return_code == 0
        and outcome.get("status") == "success"
        and isinstance(outcome.get("kpi_row"), dict)
    )


def _validate_resume_scientific_identity(existing: dict[str, Any], requested: dict[str, Any]) -> None:
    """Permit operational retuning while rejecting a scientific identity change.

    Repository revision is retained as execution provenance, not as a resume
    identity constraint.  Semantic identifiers and all experimental inputs
    below remain strict.
    """
    keys = (
        "campaign_kind",
        "checkpoint_dir",
        "policy_checkpoint_id",
        "policies",
        "zone_ids",
        "seed_pack_id",
        "netlogo_steps_per_run",
        "replications_per_policy",
        "feature_ablation",
        "feature_ablation_hash",
        "charging_mode",
        "charging_enabled",
        "charging_config_path",
        "charging_placement_source",
        "state_capture_mode",
        "kpi_schema_version",
        "allocation_patch_id",
        "simulation_semantics_id",
        "run_profile",
        "order_generation_mode",
        "full_raw_order_replay",
        "order_rate_per_hour",
        "tick_to_second",
    )
    differences = [key for key in keys if existing.get(key) != requested.get(key)]
    existing_action_mode = existing.get("rts_rl_policy_action_mode", existing.get("policy_action_mode"))
    if existing_action_mode != requested.get("rts_rl_policy_action_mode"):
        differences.append("rts_rl_policy_action_mode")
    if differences:
        detail = ", ".join(differences)
        raise ValueError(
            "resume campaign scientific identity mismatch; create a new campaign instead: "
            f"{detail}"
        )
