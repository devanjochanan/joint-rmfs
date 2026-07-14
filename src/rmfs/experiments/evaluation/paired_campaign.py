"""Queued, resumable local campaign for RTS--RL versus Nearest evaluation."""

from __future__ import annotations

import csv
import datetime
import json
import os
from pathlib import Path
import shutil
import socket
import sys
from typing import Any
import uuid

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
    machine_id: str = "local",
    resume_campaign_id: str | None = None,
) -> dict[str, Any]:
    """Run both policy arms through one continuously filled local queue.

    Specs are interleaved Nearest, RTS--RL for every seed.  The local executor
    fills every available slot from that queue, so launch counts between arms
    differ by at most one while workers remain pending.

    A compact outcome JSONL is appended before every worker root is removed.
    It permits resume after interruption without retaining rollout, log, or
    worker-summary artifacts.
    """
    repo_root = Path(repo_root)
    checkpoint = Path(checkpoint_dir)
    if int(max_workers) < 1:
        raise ValueError("max_workers must be >= 1")
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
        "schedule": "rolling_interleaved_queue.v1",
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
    created_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
    for seed in seeds:
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
                    stage_first_requested=1,
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
                "current_max_workers": int(max_workers),
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
            "worker_artifacts_persisted": False,
            "resumed": resumed,
            "worker_specs_path": str(run_root / worker_specs_name),
        }
        atomic_write_json(run_root / "campaign_summary.json", summary)
        return summary

    latest_outcomes = _read_latest_outcomes(outcomes_path)
    previous_successes = {
        run_id: outcome for run_id, outcome in latest_outcomes.items() if _outcome_is_success(outcome)
    }
    specs_to_run = [spec for spec in all_specs if spec.run_id not in previous_successes]
    skipped_completed_workers = len(previous_successes)
    executor_error: dict[str, str] | None = None

    def record_and_cleanup(spec: RunSpec, return_code: int) -> None:
        worker_summary = load_worker_summary(spec.runtime_root)
        outcome = {
            "run_id": spec.run_id,
            "replication": spec.replication,
            "seed": spec.campaign_seed,
            "policy_configuration": spec.policy_configuration,
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
        shutil.rmtree(spec.runtime_root, ignore_errors=True)

    execution_lease = _acquire_campaign_execution_lease(run_root)
    try:
        run_specs(
            specs_to_run,
            max_workers=int(max_workers),
            progress=True,
            on_run_complete=record_and_cleanup,
        )
    except Exception as exc:
        executor_error = {"error_type": type(exc).__name__, "error_message": str(exc)}
    finally:
        # Reclaim only this invocation's worker roots. Never delete the shared
        # workers directory wholesale: another controller may still own roots
        # created by an older runner that predates the execution lease.
        for spec in specs_to_run:
            shutil.rmtree(spec.runtime_root, ignore_errors=True)
        workers_root = run_root / "workers"
        try:
            workers_root.rmdir()
        except OSError:
            pass
        _release_campaign_execution_lease(execution_lease)

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


def _process_is_alive(pid: int) -> bool:
    if int(pid) <= 0:
        return False
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _acquire_campaign_execution_lease(run_root: Path) -> dict[str, Any]:
    """Atomically prevent two executing controllers from sharing worker roots."""
    run_root = Path(run_root)
    run_root.mkdir(parents=True, exist_ok=True)
    lock_path = run_root / "controller_execution.lock"
    host = socket.gethostname()
    payload = {
        "pid": os.getpid(),
        "host": host,
        "token": uuid.uuid4().hex,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    for _attempt in range(2):
        try:
            descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                existing = json.loads(lock_path.read_text(encoding="utf-8"))
            except (OSError, ValueError, TypeError):
                existing = {}
            existing_host = str(existing.get("host") or "")
            try:
                existing_pid = int(existing.get("pid") or 0)
            except (TypeError, ValueError):
                existing_pid = 0
            if existing_host and existing_host != host:
                raise RuntimeError(
                    f"paired campaign already has an execution lease on host {existing_host}: {lock_path}"
                )
            if existing_host == host and _process_is_alive(existing_pid):
                raise RuntimeError(
                    f"paired campaign is already executing in PID {existing_pid}: {lock_path}"
                )
            try:
                lock_path.unlink()
            except FileNotFoundError:
                pass
            continue
        with os.fdopen(descriptor, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, sort_keys=True)
            fh.flush()
            os.fsync(fh.fileno())
        return {"path": lock_path, **payload}
    raise RuntimeError(f"could not acquire paired campaign execution lease: {lock_path}")


def _release_campaign_execution_lease(lease: dict[str, Any]) -> None:
    lock_path = Path(lease["path"])
    try:
        existing = json.loads(lock_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError, TypeError):
        return
    if existing.get("token") != lease.get("token"):
        return
    try:
        lock_path.unlink()
    except FileNotFoundError:
        pass


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
