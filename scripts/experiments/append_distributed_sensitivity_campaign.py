#!/usr/bin/env python3
"""Append-only expansion runner for the distributed sensitivity campaign.

This script keeps the original 600-run campaign immutable, prepares a separate
replications-21-to-40 campaign, and writes a per-machine mixed execution overlay
that can drain unfinished original work and extension work through the existing
RunSpec executor.
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import os
from pathlib import Path
import sys
import zipfile
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.experiments import distributed_sensitivity_campaign as base  # noqa: E402
from src.rmfs.orchestration.local_executor import (  # noqa: E402
    load_worker_summary,
    reclaim_completed_run_artifacts_with_stats,
    reclaim_failed_run_artifacts_with_stats,
    run_specs,
)
from src.rmfs.orchestration.run_spec import RunSpec  # noqa: E402
from src.rmfs.orchestration.kpi_schema import FULL_KPI_V3_FIELDS  # noqa: E402

ORIGINAL_CAMPAIGN_ID = "sensitivity_full_kpi_v2_217d072b72e4"
ORIGINAL_ALLOCATION_PATCH_ID = "allocation_patch_0001_a333e5c09773"
RESUME_PATCH_ID = "resume_patch_0003"
APPEND_ALLOCATION_PATCH_LABEL = "allocation_patch_0003"
EXTENSION_SCHEMA_VERSION = "distributed_sensitivity_replication_extension.v1"
PLAN_SCHEMA_VERSION = "distributed_sensitivity_append_execution_plan.v1"
EXTENSION_REPLICATION_FIRST = 21
EXTENSION_REPLICATION_LAST = 40
SNAPSHOT_ROOT_DEFAULT = Path("/home/dewan/Downloads/sens")
COMBINED_EXPORT_NAME = "combined_run_outcomes_40rep.csv"
COMBINED_REPORT_NAME = "combined_run_outcomes_40rep_report.json"
EXCLUDED_MACHINE_IDS = {"dewa_macbook"}
RELAXED_COMPLETION_STATUSES = {"strict_completed"}
REQUEUED_MERGED_STATUSES = {"failed", "active_but_incomplete", "incomplete", "missing"}

HISTORICAL_CENTRAL_TARGET = {
    "citi_gojira": {"all_off": 20, "all_on_rl": 2},
    "codex_local": {"all_off": 0, "all_on_rl": 8},
    "citi_angiebow": {"all_off": 0, "all_on_rl": 4},
    "alisha_pc": {"all_off": 0, "all_on_rl": 0},
    "win_admin": {"all_off": 0, "all_on_rl": 0},
    "win_lukman": {"all_off": 0, "all_on_rl": 0},
}


def now_utc() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def original_root(repo_root: Path = REPO_ROOT) -> Path:
    return repo_root / base.OUTPUT_ROOT_RELATIVE / ORIGINAL_CAMPAIGN_ID


def original_manifest_path(repo_root: Path = REPO_ROOT) -> Path:
    return original_root(repo_root) / "manifest.json"


def append_patch_root(repo_root: Path = REPO_ROOT) -> Path:
    return original_root(repo_root) / RESUME_PATCH_ID


def sha256_path(path: Path) -> str:
    digest = base.hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _default_asset_args() -> argparse.Namespace:
    return argparse.Namespace(
        pps_model_path=None,
        rts_checkpoint_dir=None,
        rts_training_artifact=None,
        charging_config_path=None,
    )


def _write_local_campaign_manifest(manifest: dict[str, Any]) -> None:
    root = base.campaign_root(REPO_ROOT, manifest)
    root.mkdir(parents=True, exist_ok=True)
    shard_paths = {}
    for machine in manifest["machines"]:
        machine_id = machine["machine_id"]
        shard_runs = [run for run in manifest["runs"] if run.get("machine_id") == machine_id]
        shard = {
            "schema_version": manifest["schema_version"],
            "campaign_id": manifest["campaign_id"],
            "allocation_patch_id": manifest["allocation_patch_id"],
            "simulation_semantics_id": manifest["simulation_semantics_id"],
            "machine_id": machine_id,
            "runs": shard_runs,
        }
        if manifest["campaign_id"] == ORIGINAL_CAMPAIGN_ID:
            shard["stage_counts"] = {
                str(stage): sum(1 for run in shard_runs if int(run["stage_first_requested"]) == stage)
                for stage in (1, 2, 3, 4)
            }
        path = root / base.shard_relative_path(machine_id)
        base.write_json(path, shard)
        shard_paths[machine_id] = path.relative_to(root).as_posix()
    manifest = dict(manifest)
    manifest["shards"] = shard_paths
    if manifest["campaign_id"] == ORIGINAL_CAMPAIGN_ID:
        manifest["launchers"] = base.write_launchers(root, manifest)
    base.write_json(root / "manifest.json", manifest)


def _validate_original_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    if manifest.get("campaign_id") != ORIGINAL_CAMPAIGN_ID:
        raise RuntimeError(f"unexpected original campaign_id: {manifest.get('campaign_id')}")
    if manifest.get("allocation_patch_id") != ORIGINAL_ALLOCATION_PATCH_ID:
        raise RuntimeError(f"unexpected original allocation_patch_id: {manifest.get('allocation_patch_id')}")
    if len(manifest.get("runs", [])) != 600:
        raise RuntimeError(f"original campaign must contain 600 runs, found {len(manifest.get('runs', []))}")
    return manifest


def build_original_manifest() -> dict[str, Any]:
    machines = base.default_machines(REPO_ROOT)
    assets = base.resolve_assets(_default_asset_args())
    campaign_id = base.generate_campaign_id(machines, assets, base.DEFAULT_RL_OVERHEAD_MULTIPLIER)
    if campaign_id != ORIGINAL_CAMPAIGN_ID:
        raise RuntimeError(f"generated original campaign_id {campaign_id} does not match {ORIGINAL_CAMPAIGN_ID}")
    return base.build_campaign_plan(
        campaign_id=campaign_id,
        machines=machines,
        assets=assets,
        rl_overhead_multiplier=base.DEFAULT_RL_OVERHEAD_MULTIPLIER,
    )


def load_original_manifest(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    path = original_manifest_path(repo_root)
    if path.exists():
        return _validate_original_manifest(base.read_json(path))
    manifest = build_original_manifest()
    _write_local_campaign_manifest(manifest)
    return _validate_original_manifest(manifest)


def append_machines() -> list[base.Machine]:
    machines = []
    all_stages = (1, 2, 3, 4)
    for machine in base.default_machines(REPO_ROOT):
        if machine.machine_id in EXCLUDED_MACHINE_IDS:
            continue
        machines.append(base.Machine(
            machine_id=machine.machine_id,
            os=machine.os,
            repository=machine.repository,
            python=machine.python,
            max_workers=machine.max_workers,
            effective_steps_per_second=machine.effective_steps_per_second,
            eligible_stages=all_stages,
            anydesk_id=machine.anydesk_id,
        ))
    return machines


def append_allocation_patch_id(machines: list[base.Machine]) -> str:
    payload = {
        "label": APPEND_ALLOCATION_PATCH_LABEL,
        "schema_version": PLAN_SCHEMA_VERSION,
        "parent_campaign_id": ORIGINAL_CAMPAIGN_ID,
        "machines": [base.asdict(machine) for machine in machines],
        "excluded_machine_ids": sorted(EXCLUDED_MACHINE_IDS),
        "completion_validation_mode": "relaxed_provenance_metadata",
        "priority_contract": {
            "0": "unfinished_original_stages_1_to_3",
            "1": "central_extension_replications_21_to_40",
            "2": "unfinished_original_stage_4",
            "3": "noncentral_extension_replications_21_to_40",
        },
        "effective_steps_per_second_semantics": "aggregate_machine_throughput_at_configured_worker_count",
    }
    return f"{APPEND_ALLOCATION_PATCH_LABEL}_{base.stable_json_hash(payload)}"


def read_zip_json(zip_path: Path, name: str) -> Any | None:
    try:
        with zipfile.ZipFile(zip_path) as zf:
            with zf.open(name) as fh:
                return json.loads(fh.read().decode("utf-8"))
    except (KeyError, FileNotFoundError, json.JSONDecodeError, zipfile.BadZipFile):
        return None


def iter_snapshot_zip_paths(snapshot_root: Path) -> list[Path]:
    if not snapshot_root.exists():
        return []
    zips = list(sorted(snapshot_root.glob("*_snapshot_*.zip")))
    zips.extend(list(sorted(snapshot_root.glob("host_export_*.zip"))))
    return zips


def machine_from_run_path(path: str) -> str | None:
    parts = Path(path).parts
    if len(parts) >= 4 and parts[1] == "runs":
        return parts[0]
    return None


def zip_entries_by_run(snapshot_root: Path, manifest: dict[str, Any] | None = None) -> dict[str, list[dict[str, Any]]]:
    entries: dict[str, list[dict[str, Any]]] = {}
    for zip_path in iter_snapshot_zip_paths(snapshot_root):
        try:
            with zipfile.ZipFile(zip_path) as zf:
                names = zf.namelist()
        except zipfile.BadZipFile:
            continue

        machine = None
        if "host_export_" in zip_path.name:
            name_without_ext = zip_path.name[:-4]
            rem = name_without_ext[len("host_export_"):]
            subparts = rem.split("_")
            if len(subparts) >= 3:
                machine = "_".join(subparts[:-2])
            else:
                machine = subparts[0]

        # Check if the zip has run_outcomes.csv
        if "run_outcomes.csv" in names:
            try:
                rows = read_zip_csv_rows(zip_path, "run_outcomes.csv")
                by_run = {run["run_id"]: run for run in manifest.get("runs", [])} if manifest else {}
                for row in rows:
                    rid = row.get("run_id")
                    if not rid:
                        continue
                    condition = by_run.get(rid)
                    if condition:
                        spec = dict(condition["run_spec_identity"])
                        summary = {
                            "status": "success",
                            "finalization": {"finalized": True},
                            "kpi_schema_version": manifest.get("kpi_schema_version"),
                            "kpi_complete": True,
                            "seed": condition["seed"],
                            "requested_robot_count": condition["robot_count"],
                            "worker_wall_time_elapsed": float(row.get("simulated_seconds") or row.get("snapshot_simulated_seconds") or 0.0),
                        }
                        for spec_key, spec_val in condition["run_spec_identity"].items():
                            summary_key = {"campaign_seed": "seed", "robot_count": "requested_robot_count"}.get(spec_key, spec_key)
                            summary[summary_key] = spec_val
                    else:
                        spec = {
                            "campaign_id": row.get("campaign_id"),
                            "policy_configuration": row.get("policy_configuration"),
                            "robot_count": int(row.get("robot_count") or 0) if row.get("robot_count") else 0,
                            "order_rate_per_hour": int(row.get("order_rate") or 0) if row.get("order_rate") else 0,
                            "replication": int(row.get("replication") or 0) if row.get("replication") else 0,
                            "campaign_seed": int(row.get("seed") or 0) if row.get("seed") else 0,
                            "simulation_semantics_id": row.get("simulation_semantics_id"),
                            "allocation_patch_id": row.get("allocation_patch_id"),
                        }
                        summary = {
                            "status": "success",
                            "finalization": {"finalized": True},
                            "kpi_schema_version": row.get("kpi_schema_version") or (manifest.get("kpi_schema_version") if manifest else "full_kpi_v3"),
                            "kpi_complete": True,
                            "seed": int(row.get("seed") or 0) if row.get("seed") else 0,
                            "requested_robot_count": int(row.get("robot_count") or 0) if row.get("robot_count") else 0,
                            "worker_wall_time_elapsed": float(row.get("simulated_seconds") or row.get("snapshot_simulated_seconds") or 0.0),
                        }
                    for k, v in row.items():
                        if k not in summary:
                            summary[k] = v
                    entries.setdefault(rid, []).append({
                        "source": f"zip:{zip_path.name}:run_outcomes.csv",
                        "zip_path": zip_path,
                        "machine_id": machine,
                        "spec": spec,
                        "summary": summary,
                        "status_payload": None,
                    })
            except Exception:
                pass

        # Check if the zip has failed_conditions.csv
        if "failed_conditions.csv" in names:
            try:
                rows = read_zip_csv_rows(zip_path, "failed_conditions.csv")
                by_run = {run["run_id"]: run for run in manifest.get("runs", [])} if manifest else {}
                for row in rows:
                    rid = row.get("run_id")
                    if not rid:
                        continue
                    condition = by_run.get(rid)
                    if condition:
                        spec = dict(condition["run_spec_identity"])
                        summary = {
                            "status": row.get("status") or "failed",
                            "error_type": row.get("error_type"),
                            "error_message": row.get("error_message"),
                            "seed": condition["seed"],
                            "kpi_schema_version": manifest.get("kpi_schema_version"),
                        }
                        for spec_key, spec_val in condition["run_spec_identity"].items():
                            summary_key = {"campaign_seed": "seed", "robot_count": "requested_robot_count"}.get(spec_key, spec_key)
                            summary[summary_key] = spec_val
                    else:
                        spec = {
                            "campaign_id": row.get("campaign_id"),
                            "policy_configuration": row.get("policy_configuration"),
                            "robot_count": int(row.get("robot_count") or 0) if row.get("robot_count") else 0,
                            "order_rate_per_hour": int(row.get("order_rate") or 0) if row.get("order_rate") else 0,
                            "replication": int(row.get("replication") or 0) if row.get("replication") else 0,
                            "campaign_seed": int(row.get("seed") or 0) if row.get("seed") else 0,
                            "simulation_semantics_id": row.get("simulation_semantics_id"),
                            "allocation_patch_id": row.get("allocation_patch_id"),
                        }
                        summary = {
                            "status": row.get("status") or "failed",
                            "error_type": row.get("error_type"),
                            "error_message": row.get("error_message"),
                            "seed": int(row.get("seed") or 0) if row.get("seed") else 0,
                            "kpi_schema_version": row.get("kpi_schema_version") or (manifest.get("kpi_schema_version") if manifest else "full_kpi_v3"),
                        }
                    entries.setdefault(rid, []).append({
                        "source": f"zip:{zip_path.name}:failed_conditions.csv",
                        "zip_path": zip_path,
                        "machine_id": machine,
                        "spec": spec,
                        "summary": summary,
                        "status_payload": None,
                    })
            except Exception:
                pass

        # Check if the zip has host_ledger.json
        if "host_ledger.json" in names:
            try:
                ledger = read_zip_json(zip_path, "host_ledger.json")
                if ledger and isinstance(ledger, dict):
                    states = ledger.get("condition_states", {})
                    for state in states.values():
                        if state.get("status") == "running" and state.get("run_id"):
                            rid = state["run_id"]
                            entries.setdefault(rid, []).append({
                                "source": f"zip:{zip_path.name}:host_ledger.json",
                                "zip_path": zip_path,
                                "machine_id": machine,
                                "spec": None,
                                "summary": None,
                                "status_payload": {"present": True, "status": "running"},
                            })
            except Exception:
                pass

        # Legacy directory-scanning fallback
        summary_names = [name for name in names if name.endswith("/worker_summary.json")]
        spec_names = [name for name in names if name.endswith("/run_spec.json")]
        status_names = {name for name in names if name.endswith("/worker_status.json")}
        by_run = {}
        for name in spec_names:
            run_id = Path(name).parent.name
            by_run.setdefault(run_id, {})["spec_name"] = name
        for name in summary_names:
            run_id = Path(name).parent.name
            by_run.setdefault(run_id, {})["summary_name"] = name
        for name in status_names:
            run_id = Path(name).parent.name
            by_run.setdefault(run_id, {})["status_name"] = name

        for run_id, row in by_run.items():
            if any(e["source"].startswith(f"zip:{zip_path.name}:") for e in entries.get(run_id, [])):
                continue
            machine_from_path = machine_from_run_path(row.get("spec_name") or row.get("summary_name") or row.get("status_name") or "")
            eff_machine = machine or machine_from_path
            entries.setdefault(run_id, []).append({
                "source": f"zip:{zip_path.name}",
                "zip_path": zip_path,
                "machine_id": eff_machine,
                **row,
            })
    return entries


def read_zip_copy(row: dict[str, Any]) -> dict[str, Any]:
    zip_path = Path(row["zip_path"])
    spec = read_zip_json(zip_path, row["spec_name"]) if row.get("spec_name") else None
    summary = read_zip_json(zip_path, row["summary_name"]) if row.get("summary_name") else None
    status = read_zip_json(zip_path, row["status_name"]) if row.get("status_name") else None
    return {**row, "spec": spec, "summary": summary, "status_payload": status}


def read_local_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    try:
        with path.open(newline="", encoding="utf-8") as fh:
            return list(csv.DictReader(fh))
    except Exception:
        return []


def read_local_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    try:
        with path.open(encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


def live_entries_by_run(manifest: dict[str, Any], repo_root: Path) -> dict[str, list[dict[str, Any]]]:
    root = base.campaign_root(repo_root, manifest)
    machine_ids = [row["machine_id"] for row in manifest.get("machines", [])]
    run_ids = [run["run_id"] for run in manifest.get("runs", [])]
    by_run = {run["run_id"]: run for run in manifest.get("runs", [])}
    entries: dict[str, list[dict[str, Any]]] = {}

    for machine_id in machine_ids:
        # Load run_outcomes.csv
        outcomes_path = root / machine_id / "run_outcomes.csv"
        if outcomes_path.exists():
            for row in read_local_csv_rows(outcomes_path):
                rid = row.get("run_id")
                if not rid:
                    continue
                condition = by_run.get(rid)
                if condition:
                    spec = dict(condition["run_spec_identity"])
                    summary = {
                        "status": "success",
                        "finalization": {"finalized": True},
                        "kpi_schema_version": manifest.get("kpi_schema_version"),
                        "kpi_complete": True,
                        "seed": condition["seed"],
                        "requested_robot_count": condition["robot_count"],
                        "worker_wall_time_elapsed": float(row.get("simulated_seconds") or row.get("snapshot_simulated_seconds") or 0.0),
                    }
                    for spec_key, spec_val in condition["run_spec_identity"].items():
                        summary_key = {"campaign_seed": "seed", "robot_count": "requested_robot_count"}.get(spec_key, spec_key)
                        summary[summary_key] = spec_val
                else:
                    spec = {
                        "campaign_id": row.get("campaign_id"),
                        "policy_configuration": row.get("policy_configuration"),
                        "robot_count": int(row.get("robot_count") or 0) if row.get("robot_count") else 0,
                        "order_rate_per_hour": int(row.get("order_rate") or 0) if row.get("order_rate") else 0,
                        "replication": int(row.get("replication") or 0) if row.get("replication") else 0,
                        "campaign_seed": int(row.get("seed") or 0) if row.get("seed") else 0,
                        "simulation_semantics_id": row.get("simulation_semantics_id"),
                        "allocation_patch_id": row.get("allocation_patch_id"),
                    }
                    summary = {
                        "status": "success",
                        "finalization": {"finalized": True},
                        "kpi_schema_version": row.get("kpi_schema_version") or manifest.get("kpi_schema_version"),
                        "kpi_complete": True,
                        "seed": int(row.get("seed") or 0) if row.get("seed") else 0,
                        "requested_robot_count": int(row.get("robot_count") or 0) if row.get("robot_count") else 0,
                        "worker_wall_time_elapsed": float(row.get("simulated_seconds") or row.get("snapshot_simulated_seconds") or 0.0),
                    }
                for k, v in row.items():
                    if k not in summary:
                        summary[k] = v
                entries.setdefault(rid, []).append({
                    "source": "live:run_outcomes.csv",
                    "machine_id": machine_id,
                    "spec": spec,
                    "summary": summary,
                    "status_payload": None,
                })

        # Load failed_conditions.csv
        failed_path = root / machine_id / "failed_conditions.csv"
        if failed_path.exists():
            for row in read_local_csv_rows(failed_path):
                rid = row.get("run_id")
                if not rid:
                    continue
                condition = by_run.get(rid)
                if condition:
                    spec = dict(condition["run_spec_identity"])
                    summary = {
                        "status": row.get("status") or "failed",
                        "error_type": row.get("error_type"),
                        "error_message": row.get("error_message"),
                        "seed": condition["seed"],
                        "kpi_schema_version": manifest.get("kpi_schema_version"),
                    }
                    for spec_key, spec_val in condition["run_spec_identity"].items():
                        summary_key = {"campaign_seed": "seed", "robot_count": "requested_robot_count"}.get(spec_key, spec_key)
                        summary[summary_key] = spec_val
                else:
                    spec = {
                        "campaign_id": row.get("campaign_id"),
                        "policy_configuration": row.get("policy_configuration"),
                        "robot_count": int(row.get("robot_count") or 0) if row.get("robot_count") else 0,
                        "order_rate_per_hour": int(row.get("order_rate") or 0) if row.get("order_rate") else 0,
                        "replication": int(row.get("replication") or 0) if row.get("replication") else 0,
                        "campaign_seed": int(row.get("seed") or 0) if row.get("seed") else 0,
                        "simulation_semantics_id": row.get("simulation_semantics_id"),
                        "allocation_patch_id": row.get("allocation_patch_id"),
                    }
                    summary = {
                        "status": row.get("status") or "failed",
                        "error_type": row.get("error_type"),
                        "error_message": row.get("error_message"),
                        "seed": int(row.get("seed") or 0) if row.get("seed") else 0,
                        "kpi_schema_version": row.get("kpi_schema_version") or manifest.get("kpi_schema_version"),
                    }
                entries.setdefault(rid, []).append({
                    "source": "live:failed_conditions.csv",
                    "machine_id": machine_id,
                    "spec": spec,
                    "summary": summary,
                    "status_payload": None,
                })

        # Load host_ledger.json
        ledger_path = root / machine_id / "host_ledger.json"
        if ledger_path.exists():
            ledger = read_local_json(ledger_path)
            if ledger and isinstance(ledger, dict):
                states = ledger.get("condition_states", {})
                for state in states.values():
                    if state.get("status") == "running" and state.get("run_id"):
                        rid = state["run_id"]
                        entries.setdefault(rid, []).append({
                            "source": "live:host_ledger.json",
                            "machine_id": machine_id,
                            "spec": None,
                            "summary": None,
                            "status_payload": {"present": True, "status": "running"},
                        })

    # Keep original runs/ scanning fallback (with de-duplication check)
    for machine_id in machine_ids:
        for run_id in run_ids:
            run_root = root / machine_id / "runs" / run_id
            spec_path = run_root / "run_spec.json"
            summary_path = run_root / "worker_summary.json"
            status_path = run_root / "worker_status.json"
            if not (spec_path.exists() or summary_path.exists() or status_path.exists()):
                continue
            if any(e["machine_id"] == machine_id and e["source"].startswith("live:") for e in entries.get(run_id, [])):
                continue
            row: dict[str, Any] = {
                "source": "live",
                "machine_id": machine_id,
                "run_root": run_root,
                "spec": None,
                "summary": None,
                "status_payload": None,
            }
            if spec_path.exists():
                try:
                    row["spec"] = base.read_json(spec_path)
                except Exception:
                    pass
            if summary_path.exists():
                try:
                    row["summary"] = base.read_json(summary_path)
                except Exception:
                    pass
            if status_path.exists():
                try:
                    row["status_payload"] = base.read_json(status_path)
                except Exception:
                    row["status_payload"] = {"present": True}
            entries.setdefault(run_id, []).append(row)
    return entries


def summary_identity_value(summary: dict[str, Any], key: str) -> Any:
    key_map = {"campaign_seed": "seed", "robot_count": "requested_robot_count"}
    return summary.get(key_map.get(key, key))


def strict_json_complete(
    condition: dict[str, Any],
    spec: dict[str, Any] | None,
    summary: dict[str, Any] | None,
    manifest: dict[str, Any],
    *,
    relaxed: bool = True,
) -> bool:
    if not spec or not summary:
        return False
    ignored = base.RELAXED_COMPLETION_IDENTITY_FIELDS if relaxed else set()
    for key, expected in condition["run_spec_identity"].items():
        if key in ignored:
            continue
        if spec.get(key) != expected:
            return False
        if key in {"order_rate_per_hour", "netlogo_steps_requested", "tick_to_second"}:
            continue
        summary_key = {"campaign_seed": "seed", "robot_count": "requested_robot_count"}.get(key, key)
        if summary_key in ignored:
            continue
        if summary_identity_value(summary, key) != expected:
            return False
    if summary.get("seed") != condition["seed"]:
        return False
    if summary.get("status") != "success":
        return False
    if not bool(summary.get("finalization", {}).get("finalized")):
        return False
    if summary.get("kpi_schema_version") != manifest["kpi_schema_version"]:
        return False
    if not bool(summary.get("kpi_complete")):
        return False
    return True


def strict_summary_fingerprint(summary: dict[str, Any]) -> str:
    ignore_fields = set(base.RELAXED_COMPLETION_IDENTITY_FIELDS) | {"repo_commit"}
    kpi_fields_set = base.SENSITIVITY_KPI_FIELDS
    if summary.get("kpi_schema_version") == "full_kpi_v3":
        kpi_fields_set = FULL_KPI_V3_FIELDS
    payload = {
        field: summary.get(field, summary.get("kpi", {}).get(field))
        for field in kpi_fields_set
        if field not in ignore_fields
    }
    payload["finalization"] = summary.get("finalization", {})
    return base.stable_json_hash(payload, length=24)


def completion_inventory(manifest: dict[str, Any], repo_root: Path, snapshot_root: Path) -> dict[str, Any]:
    by_run = {run["run_id"]: run for run in manifest["runs"]}
    live = live_entries_by_run(manifest, repo_root)
    zipped_raw = zip_entries_by_run(snapshot_root, manifest=manifest)
    rows = []
    counts = {name: 0 for name in ("strict_completed", "incomplete", "failed", "active_but_incomplete", "missing", "conflicting")}
    strict_records: dict[str, dict[str, Any]] = {}
    duration_records = []
    conflicts = []
    for run_id, condition in sorted(by_run.items(), key=lambda item: base.condition_sort_key(item[1])):
        copies = list(live.get(run_id, []))
        copies.extend(read_zip_copy(row) for row in zipped_raw.get(run_id, []))
        strict_copies = [copy for copy in copies if strict_json_complete(condition, copy.get("spec"), copy.get("summary"), manifest)]
        category = "missing"
        selected = None
        if strict_copies:
            fingerprints = {strict_summary_fingerprint(copy["summary"]): copy for copy in strict_copies}
            if len(fingerprints) > 1:
                category = "conflicting"
                conflicts.append({"run_id": run_id, "condition_key": condition["condition_key"], "sources": [copy["source"] for copy in strict_copies]})
            else:
                selected = strict_copies[0]
                category = "strict_completed"
                summary = selected["summary"]
                duration = float(summary.get("worker_wall_time_elapsed", 0.0) or 0.0)
                if duration > 0:
                    duration_records.append({
                        "machine_id": selected.get("machine_id") or summary.get("machine_id"),
                        "policy_configuration": condition["policy_configuration"],
                        "robot_count": int(condition["robot_count"]),
                        "order_rate": int(condition["order_rate"]),
                        "duration_seconds": duration,
                    })
                strict_records[run_id] = {"copy": selected, "condition": condition}
        elif copies:
            summaries = [copy.get("summary") for copy in copies if copy.get("summary")]
            statuses = [copy.get("status_payload") for copy in copies if copy.get("status_payload")]
            if any(summary.get("status") not in {None, "success", "running"} for summary in summaries):
                category = "failed"
            elif statuses:
                category = "active_but_incomplete"
            else:
                category = "incomplete"
        counts[category] += 1
        rows.append({
            "run_id": run_id,
            "condition_key": condition["condition_key"],
            "policy_configuration": condition["policy_configuration"],
            "robot_count": condition["robot_count"],
            "order_rate": condition["order_rate"],
            "replication": condition["replication"],
            "seed": condition["seed"],
            "stage_first_requested": condition["stage_first_requested"],
            "inventory_status": category,
            "strict_source": selected.get("source") if selected else "",
            "strict_machine_id": selected.get("machine_id") if selected else "",
            "copy_count": len(copies),
        })
    if conflicts:
        raise RuntimeError(f"conflicting strict-success artifacts detected: {conflicts[:5]}")
    return {
        "rows": rows,
        "counts": counts,
        "strict_records": strict_records,
        "duration_records": duration_records,
        "snapshot_zip_count": len(iter_snapshot_zip_paths(snapshot_root)),
        "snapshot_root": str(snapshot_root),
    }


def read_zip_csv_rows(zip_path: Path, name: str) -> list[dict[str, str]]:
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(name) as fh:
            text = fh.read().decode("utf-8-sig")
    return list(csv.DictReader(text.splitlines()))


def read_zip_json_required(zip_path: Path, name: str) -> Any:
    payload = read_zip_json(zip_path, name)
    if payload is None:
        raise RuntimeError(f"missing or invalid {name} in merged snapshot zip: {zip_path}")
    return payload


def local_extension_manifest_path(extension_campaign_id: str) -> Path:
    return REPO_ROOT / base.OUTPUT_ROOT_RELATIVE / extension_campaign_id / "manifest.json"


def load_extension_manifest_from_merged_zip(zip_path: Path, allocation_patch_id: str, machines: list[base.Machine]) -> dict[str, Any]:
    zip_manifest = read_zip_json_required(zip_path, "manifests/extension_manifest.json")
    if zip_manifest.get("extension_replication_first") != EXTENSION_REPLICATION_FIRST or zip_manifest.get("extension_replication_last") != EXTENSION_REPLICATION_LAST:
        raise RuntimeError("merged snapshot extension manifest does not cover replications 21-40")
    local_path = local_extension_manifest_path(zip_manifest["campaign_id"])
    if local_path.exists():
        local_manifest = base.read_json(local_path)
        zip_ids = {run["condition_key"]: run["run_id"] for run in zip_manifest.get("runs", [])}
        local_ids = {run["condition_key"]: run["run_id"] for run in local_manifest.get("runs", [])}
        if zip_ids != local_ids:
            raise RuntimeError("local extension manifest run IDs differ from merged snapshot manifest")
        manifest = local_manifest
    else:
        manifest = zip_manifest
    return {
        **manifest,
        "allocation_patch_id": allocation_patch_id,
        "allocation_patch_label": APPEND_ALLOCATION_PATCH_LABEL,
        "machines": [base.asdict(machine) for machine in machines],
    }


def _duration_records_from_merged_zip(zip_path: Path, manifests: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    by_campaign_run = {
        campaign_id: {run["run_id"]: run for run in manifest.get("runs", [])}
        for campaign_id, manifest in manifests.items()
    }
    records: list[dict[str, Any]] = []
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if not name.startswith("merged_run_artifacts/") or not name.endswith("/worker_summary.json"):
                continue
            parts = Path(name).parts
            if len(parts) < 6:
                continue
            campaign_id = parts[1]
            machine_id = parts[2]
            run_id = parts[4]
            condition = by_campaign_run.get(campaign_id, {}).get(run_id)
            if not condition:
                continue
            try:
                summary = json.loads(zf.read(name).decode("utf-8"))
            except Exception:
                continue
            duration = float(summary.get("worker_wall_time_elapsed", 0.0) or 0.0)
            if duration <= 0:
                continue
            records.append({
                "machine_id": machine_id,
                "policy_configuration": condition["policy_configuration"],
                "robot_count": int(condition["robot_count"]),
                "order_rate": int(condition["order_rate"]),
                "duration_seconds": duration,
            })
    return records


def completion_inventory_from_merged_zip(
    original_manifest: dict[str, Any],
    extension_manifest: dict[str, Any],
    zip_path: Path,
) -> dict[str, Any]:
    status_rows = read_zip_csv_rows(zip_path, "datasets/condition_status_1200.csv")
    provenance_rows = read_zip_csv_rows(zip_path, "datasets/strict_completion_provenance.csv")
    provenance_by_key = {
        (row.get("campaign_id", ""), row.get("run_id", "")): row
        for row in provenance_rows
    }
    manifests = {
        original_manifest["campaign_id"]: original_manifest,
        extension_manifest["campaign_id"]: extension_manifest,
    }
    by_campaign_run = {
        campaign_id: {run["run_id"]: run for run in manifest.get("runs", [])}
        for campaign_id, manifest in manifests.items()
    }
    rows: list[dict[str, Any]] = []
    counts = {name: 0 for name in ("strict_completed", "incomplete", "failed", "active_but_incomplete", "missing", "conflicting")}
    strict_records_by_campaign: dict[str, dict[str, Any]] = {campaign_id: {} for campaign_id in manifests}
    seen: set[tuple[str, str]] = set()
    for row in status_rows:
        campaign_id = row.get("campaign_id", "")
        run_id = row.get("run_id", "")
        if campaign_id not in manifests:
            continue
        condition = by_campaign_run[campaign_id].get(run_id)
        if not condition:
            raise RuntimeError(f"merged snapshot status references unknown run: {campaign_id} {run_id}")
        key = (campaign_id, run_id)
        if key in seen:
            raise RuntimeError(f"duplicate merged snapshot status row: {campaign_id} {run_id}")
        seen.add(key)
        status = row.get("merge_status") or row.get("inventory_status") or "missing"
        if status not in counts:
            status = "incomplete"
        counts[status] += 1
        selected_machine = row.get("selected_machine_id", "")
        if status == "strict_completed":
            strict_records_by_campaign[campaign_id][run_id] = {
                "copy": {
                    "source": row.get("selected_snapshot", "merged_snapshot"),
                    "machine_id": selected_machine,
                    "provenance": provenance_by_key.get(key, {}),
                },
                "condition": condition,
            }
        rows.append({
            "campaign_id": campaign_id,
            "run_id": run_id,
            "condition_key": condition["condition_key"],
            "policy_configuration": condition["policy_configuration"],
            "robot_count": condition["robot_count"],
            "order_rate": condition["order_rate"],
            "replication": condition["replication"],
            "seed": condition["seed"],
            "stage_first_requested": condition["stage_first_requested"],
            "inventory_status": status,
            "strict_source": row.get("selected_snapshot", "") if status == "strict_completed" else "",
            "strict_machine_id": selected_machine if status == "strict_completed" else "",
            "copy_count": row.get("observed_copy_count", ""),
        })
    expected = {(campaign_id, run_id) for campaign_id, runs in by_campaign_run.items() for run_id in runs}
    missing_status = expected - seen
    if missing_status:
        sample = sorted(missing_status)[:5]
        raise RuntimeError(f"merged snapshot status is missing planned runs: {sample}")
    duration_records = _duration_records_from_merged_zip(zip_path, manifests)
    return {
        "rows": rows,
        "counts": counts,
        "strict_records": strict_records_by_campaign.get(original_manifest["campaign_id"], {}),
        "strict_records_by_campaign": strict_records_by_campaign,
        "duration_records": duration_records,
        "snapshot_zip_count": 1,
        "snapshot_root": str(zip_path),
        "merged_snapshot_zip": str(zip_path),
    }


def mean(values: list[float]) -> float | None:
    values = [float(value) for value in values if float(value) > 0]
    if not values:
        return None
    return sum(values) / len(values)


class DurationModel:
    def __init__(self, records: list[dict[str, Any]], machines: dict[str, base.Machine]):
        self.by_machine_policy_condition: dict[tuple[str, str, int, int], list[float]] = {}
        self.by_machine_policy: dict[tuple[str, str], list[float]] = {}
        self.by_machine: dict[str, list[float]] = {}
        self.machines = machines
        for row in records:
            machine_id = str(row.get("machine_id") or "")
            if machine_id not in machines:
                continue
            policy = str(row["policy_configuration"])
            robot_count = int(row["robot_count"])
            order_rate = int(row["order_rate"])
            duration = float(row["duration_seconds"])
            self.by_machine_policy_condition.setdefault((machine_id, policy, robot_count, order_rate), []).append(duration)
            self.by_machine_policy.setdefault((machine_id, policy), []).append(duration)
            self.by_machine.setdefault(machine_id, []).append(duration)

    def estimate(self, condition: dict[str, Any], machine: base.Machine) -> tuple[float, str]:
        key = (machine.machine_id, condition["policy_configuration"], int(condition["robot_count"]), int(condition["order_rate"]))
        value = mean(self.by_machine_policy_condition.get(key, []))
        if value is not None:
            return value, "machine_policy_condition_observed"
        value = mean(self.by_machine_policy.get((machine.machine_id, condition["policy_configuration"]), []))
        if value is not None:
            return value, "machine_policy_observed"
        value = mean(self.by_machine.get(machine.machine_id, []))
        if value is not None:
            return value, "machine_observed"
        steps = base.estimate_condition_steps(condition, rl_overhead_multiplier=base.DEFAULT_RL_OVERHEAD_MULTIPLIER)
        return steps / float(machine.effective_steps_per_second), "analytical_condition_cost"


def extension_scientific_identity(parent_manifest: dict[str, Any], parent_manifest_sha256: str) -> dict[str, Any]:
    identity = dict(parent_manifest["scientific_identity"])
    identity.update({
        "schema_version": EXTENSION_SCHEMA_VERSION,
        "parent_campaign_id": parent_manifest["campaign_id"],
        "parent_manifest_sha256": parent_manifest_sha256,
        "replication_range": {"first": EXTENSION_REPLICATION_FIRST, "last": EXTENSION_REPLICATION_LAST},
        "extension_condition_count": 600,
        "seed_first": base.seed_for_replication(EXTENSION_REPLICATION_FIRST),
        "seed_last": base.seed_for_replication(EXTENSION_REPLICATION_LAST),
    })
    return identity


def extension_campaign_id(scientific_identity: dict[str, Any]) -> str:
    return f"sensitivity_full_kpi_rep21_40_{base.stable_json_hash(scientific_identity)}"


def extension_rows() -> list[dict[str, Any]]:
    rows = []
    for replication in range(EXTENSION_REPLICATION_FIRST, EXTENSION_REPLICATION_LAST + 1):
        for policy in base.POLICY_CONFIGURATIONS:
            for robot_count in base.ROBOT_COUNTS:
                for order_rate in base.ORDER_RATES:
                    rows.append({
                        "condition_key": base.condition_key(policy, robot_count, order_rate, replication),
                        "policy_configuration": policy,
                        "robot_count": int(robot_count),
                        "order_rate": int(order_rate),
                        "picker_count": base.PICKER_COUNT,
                        "replenishment_count": base.REPLENISHMENT_COUNT,
                        "replication": int(replication),
                        "seed": base.seed_for_replication(replication),
                        "stage_first_requested": 5,
                    })
    return rows


def _rebuild_extension_manifest_for_campaign_id(
    manifest: dict[str, Any],
    campaign_id: str,
    allocation_patch_id: str,
) -> dict[str, Any]:
    assets = base.AssetBundle(**manifest["assets"])
    scientific_identity = manifest["scientific_identity"]
    scenario_hash = scientific_identity["scenario_layout_identity"]["scenario_hash"]
    layout_hash = scientific_identity["scenario_layout_identity"]["layout_hash"]
    branch = base.git_clean_value("rev-parse", "--abbrev-ref", "HEAD")
    commit = base.git_clean_value("rev-parse", "HEAD")
    runs = [
        base.add_run_identity(
            row,
            campaign_id=campaign_id,
            allocation_patch_id=allocation_patch_id,
            scientific_identity=scientific_identity,
            branch=branch,
            commit=commit,
            ticks=int(manifest["backend_steps_per_full_run"]),
            assets=assets,
            scenario_hash=scenario_hash,
            layout_hash=layout_hash,
        )
        for row in extension_rows()
    ]
    rebuilt = {
        **manifest,
        "campaign_id": campaign_id,
        "allocation_patch_id": allocation_patch_id,
        "allocation_patch_label": APPEND_ALLOCATION_PATCH_LABEL,
        "campaign_root_relative": (base.OUTPUT_ROOT_RELATIVE / campaign_id).as_posix(),
        "branch": branch,
        "commit": commit,
        "runs": runs,
        "assertions": extension_assertions(runs),
        "shards": {},
        "launchers": {},
    }
    return rebuilt


def build_extension_manifest(parent_manifest: dict[str, Any], machines: list[base.Machine], allocation_patch_id: str) -> dict[str, Any]:
    parent_manifest_sha = sha256_path(original_manifest_path(REPO_ROOT))
    scientific_identity = extension_scientific_identity(parent_manifest, parent_manifest_sha)
    campaign_id = extension_campaign_id(scientific_identity)
    assets = base.AssetBundle(**parent_manifest["assets"])
    scenario_hash = scientific_identity["scenario_layout_identity"]["scenario_hash"]
    layout_hash = scientific_identity["scenario_layout_identity"]["layout_hash"]
    branch = base.git_clean_value("rev-parse", "--abbrev-ref", "HEAD")
    commit = base.git_clean_value("rev-parse", "HEAD")
    runs = [
        base.add_run_identity(
            row,
            campaign_id=campaign_id,
            allocation_patch_id=allocation_patch_id,
            scientific_identity=scientific_identity,
            branch=branch,
            commit=commit,
            ticks=int(parent_manifest["backend_steps_per_full_run"]),
            assets=assets,
            scenario_hash=scenario_hash,
            layout_hash=layout_hash,
        )
        for row in extension_rows()
    ]
    return {
        "schema_version": EXTENSION_SCHEMA_VERSION,
        "campaign_id": campaign_id,
        "parent_campaign_id": parent_manifest["campaign_id"],
        "parent_manifest_sha256": parent_manifest_sha,
        "extension_replication_first": EXTENSION_REPLICATION_FIRST,
        "extension_replication_last": EXTENSION_REPLICATION_LAST,
        "extension_condition_count": len(runs),
        "seed_first": base.seed_for_replication(EXTENSION_REPLICATION_FIRST),
        "seed_last": base.seed_for_replication(EXTENSION_REPLICATION_LAST),
        "simulation_semantics_id": parent_manifest["simulation_semantics_id"],
        "scientific_identity": scientific_identity,
        "allocation_patch_id": allocation_patch_id,
        "allocation_patch_label": APPEND_ALLOCATION_PATCH_LABEL,
        "created_at": now_utc(),
        "branch": branch,
        "commit": commit,
        "output_root_relative": base.OUTPUT_ROOT_RELATIVE.as_posix(),
        "campaign_root_relative": (base.OUTPUT_ROOT_RELATIVE / campaign_id).as_posix(),
        "input_root_relative": parent_manifest["input_root_relative"],
        "tick_to_second": parent_manifest["tick_to_second"],
        "simulated_seconds": parent_manifest["simulated_seconds"],
        "backend_steps_per_full_run": parent_manifest["backend_steps_per_full_run"],
        "seed_formula": parent_manifest["seed_formula"],
        "policy_configurations": list(base.POLICY_CONFIGURATIONS),
        "robot_counts": list(base.ROBOT_COUNTS),
        "order_rates": list(base.ORDER_RATES),
        "picking_stations": base.PICKER_COUNT,
        "replenishment_stations": base.REPLENISHMENT_COUNT,
        "kpi_schema_version": parent_manifest["kpi_schema_version"],
        "machines": [base.asdict(machine) for machine in machines],
        "excluded_machine_ids": sorted(EXCLUDED_MACHINE_IDS),
        "completion_validation_mode": "relaxed_provenance_metadata",
        "assets": parent_manifest["assets"],
        "feature_flags": parent_manifest.get("feature_flags", {}),
        "input_meta": parent_manifest.get("input_meta", {}),
        "runs": runs,
        "shards": {},
        "launchers": {},
        "assertions": extension_assertions(runs),
    }


def extension_assertions(runs: list[dict[str, Any]]) -> dict[str, Any]:
    tuples = {(run["policy_configuration"], run["robot_count"], run["order_rate"], run["replication"]) for run in runs}
    central = [run for run in runs if is_extension_central(run)]
    seeds_by_rep = {rep: {run["seed"] for run in runs if run["replication"] == rep} for rep in range(EXTENSION_REPLICATION_FIRST, EXTENSION_REPLICATION_LAST + 1)}
    assertions = {
        "condition_count": len(runs),
        "unique_run_ids": len({run["run_id"] for run in runs}),
        "unique_condition_tuples": len(tuples),
        "replication_first": min(run["replication"] for run in runs),
        "replication_last": max(run["replication"] for run in runs),
        "seed_first": min(run["seed"] for run in runs),
        "seed_last": max(run["seed"] for run in runs),
        "central_extension_count": len(central),
        "noncentral_extension_count": len(runs) - len(central),
        "matched_policy_seeds": all(len(seeds) == 1 for seeds in seeds_by_rep.values()),
    }
    if assertions != {
        **assertions,
        "condition_count": 600,
        "unique_run_ids": 600,
        "unique_condition_tuples": 600,
        "replication_first": EXTENSION_REPLICATION_FIRST,
        "replication_last": EXTENSION_REPLICATION_LAST,
        "seed_first": 62,
        "seed_last": 81,
        "central_extension_count": 40,
        "noncentral_extension_count": 560,
        "matched_policy_seeds": True,
    }:
        if assertions["condition_count"] != 600 or assertions["unique_run_ids"] != 600 or assertions["unique_condition_tuples"] != 600:
            raise AssertionError(assertions)
        if assertions["replication_first"] != 21 or assertions["replication_last"] != 40:
            raise AssertionError(assertions)
        if assertions["seed_first"] != 62 or assertions["seed_last"] != 81:
            raise AssertionError(assertions)
        if assertions["central_extension_count"] != 40 or assertions["noncentral_extension_count"] != 560:
            raise AssertionError(assertions)
        if not assertions["matched_policy_seeds"]:
            raise AssertionError(assertions)
    return assertions


def is_extension_central(run: dict[str, Any]) -> bool:
    return int(run["robot_count"]) == base.CENTRAL_ROBOT_COUNT and int(run["order_rate"]) == base.CENTRAL_ORDER_RATE


def execution_priority(run: dict[str, Any], source_campaign_id: str) -> int:
    if source_campaign_id == ORIGINAL_CAMPAIGN_ID:
        return 0 if int(run["stage_first_requested"]) in {1, 2, 3} else 2
    return 1 if is_extension_central(run) else 3


def plan_item(run: dict[str, Any], manifest: dict[str, Any], priority: int, destination_root: Path) -> dict[str, Any]:
    return {
        "source_campaign_id": manifest["campaign_id"],
        "source_manifest": (Path(manifest["campaign_root_relative"]) / "manifest.json").as_posix(),
        "source_condition_key": run["condition_key"],
        "run_id": run["run_id"],
        "policy_configuration": run["policy_configuration"],
        "robot_count": int(run["robot_count"]),
        "order_rate": int(run["order_rate"]),
        "replication": int(run["replication"]),
        "seed": int(run["seed"]),
        "stage_first_requested": int(run["stage_first_requested"]),
        "execution_priority": int(priority),
        "destination_runtime_root": destination_root.as_posix(),
        "allocation_patch_id": manifest["allocation_patch_id"],
    }


def item_sort_key(item: dict[str, Any]) -> tuple[Any, ...]:
    return (
        int(item["execution_priority"]),
        item["policy_configuration"],
        int(item["robot_count"]),
        int(item["order_rate"]),
        int(item["replication"]),
        item["run_id"],
    )


def schedule_items(
    original_manifest: dict[str, Any],
    extension_manifest: dict[str, Any],
    inventory: dict[str, Any],
    machines: list[base.Machine],
    duration_model: DurationModel,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    machine_by_id = {machine.machine_id: machine for machine in machines}
    slots = []
    for machine in machines:
        for slot_index in range(int(machine.max_workers)):
            slots.append({"machine_id": machine.machine_id, "slot_index": slot_index, "load_seconds": 0.0})
    if not slots:
        raise RuntimeError("no execution slots configured")

    strict_by_campaign = inventory.get("strict_records_by_campaign") or {original_manifest["campaign_id"]: inventory["strict_records"]}
    completed_original_ids = set(strict_by_campaign.get(original_manifest["campaign_id"], {}).keys())
    completed_extension_ids = set(strict_by_campaign.get(extension_manifest["campaign_id"], {}).keys())
    unfinished_original = [run for run in original_manifest["runs"] if run["run_id"] not in completed_original_ids]
    original_items = [
        plan_item(run, {**original_manifest, "allocation_patch_id": extension_manifest["allocation_patch_id"]}, execution_priority(run, ORIGINAL_CAMPAIGN_ID), original_root(REPO_ROOT) / "{machine_id}" / "runs" / run["run_id"])
        for run in unfinished_original
    ]
    extension_items = [
        plan_item(run, extension_manifest, execution_priority(run, extension_manifest["campaign_id"]), base.campaign_root(REPO_ROOT, extension_manifest) / "{machine_id}" / "runs" / run["run_id"])
        for run in extension_manifest["runs"]
        if run["run_id"] not in completed_extension_ids
    ]
    all_items = original_items + extension_items
    run_by_id = {run["run_id"]: run for run in original_manifest["runs"] + extension_manifest["runs"]}
    assigned: list[dict[str, Any]] = []

    def assign_to_slot(item: dict[str, Any], slot: dict[str, Any]) -> None:
        machine = machine_by_id[slot["machine_id"]]
        condition = run_by_id[item["run_id"]]
        duration, estimate_source = duration_model.estimate(condition, machine)
        slot["load_seconds"] += duration
        destination = Path(item["destination_runtime_root"].replace("{machine_id}", machine.machine_id))
        item.update({
            "assigned_machine_id": machine.machine_id,
            "assigned_worker_slot_projection": int(slot["slot_index"]),
            "estimated_duration_seconds": duration,
            "estimate_source": estimate_source,
            "destination_runtime_root": destination.as_posix(),
            "projected_slot_finish_seconds": slot["load_seconds"],
        })
        assigned.append(item)

    remaining = list(all_items)

    # Priority 1 is small and deadline-sensitive. Use the reviewed central target
    # as the deterministic baseline, then let the LPT fallback handle leftovers.
    central_items = sorted([item for item in remaining if item["execution_priority"] == 1], key=item_sort_key)
    for machine_id, policy_counts in HISTORICAL_CENTRAL_TARGET.items():
        for policy, target_count in policy_counts.items():
            for item in list(central_items):
                if target_count <= 0:
                    break
                if item["policy_configuration"] != policy:
                    continue
                machine_slots = [slot for slot in slots if slot["machine_id"] == machine_id]
                if not machine_slots:
                    continue
                slot = min(machine_slots, key=lambda row: (row["load_seconds"], row["slot_index"]))
                assign_to_slot(item, slot)
                central_items.remove(item)
                remaining.remove(item)
                target_count -= 1

    for priority in (0, 1, 2, 3):
        group = [item for item in remaining if item["execution_priority"] == priority]
        group.sort(key=lambda item: (
            -max(duration_model.estimate(run_by_id[item["run_id"]], machine)[0] for machine in machines),
            *item_sort_key(item),
        ))
        for item in group:
            slot = min(slots, key=lambda row: (row["load_seconds"], row["machine_id"], row["slot_index"]))
            assign_to_slot(item, slot)
    assigned.sort(key=lambda item: (item["assigned_machine_id"], item_sort_key(item)))

    # Update extension manifest assignment provenance; original manifest is intentionally untouched.
    assigned_by_run = {item["run_id"]: item for item in assigned if item["source_campaign_id"] == extension_manifest["campaign_id"]}
    for run in extension_manifest["runs"]:
        item = assigned_by_run.get(run["run_id"])
        if item is None:
            continue
        run["machine_id"] = item["assigned_machine_id"]
        run["machine_slot_index"] = item["assigned_worker_slot_projection"]
        run["estimated_duration_seconds"] = item["estimated_duration_seconds"]
        run["estimate_source"] = item["estimate_source"]

    critical = [item for item in assigned if item["execution_priority"] == 0]
    critical_finish = max((item["projected_slot_finish_seconds"] for item in critical), default=0.0)
    central = [item for item in assigned if item["execution_priority"] == 1]
    central_finish = max((item["projected_slot_finish_seconds"] for item in central), default=0.0)
    total_finish = max((item["projected_slot_finish_seconds"] for item in assigned), default=0.0)
    central_counts: dict[str, dict[str, int]] = {machine.machine_id: {"all_off": 0, "all_on_rl": 0} for machine in machines}
    for item in central:
        central_counts[item["assigned_machine_id"]][item["policy_configuration"]] += 1
    deviations = []
    for machine_id, policy_counts in central_counts.items():
        for policy, count in policy_counts.items():
            expected = HISTORICAL_CENTRAL_TARGET.get(machine_id, {}).get(policy, 0)
            if count != expected:
                sample_condition = next((run for run in extension_manifest["runs"] if run["policy_configuration"] == policy and is_extension_central(run)), None)
                machine = machine_by_id[machine_id]
                estimate = duration_model.estimate(sample_condition, machine) if sample_condition else (None, "no_sample")
                deviations.append({
                    "machine_id": machine_id,
                    "policy_configuration": policy,
                    "historical_target": expected,
                    "planned": count,
                    "estimate_seconds": estimate[0],
                    "estimate_source": estimate[1],
                })
    report = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "allocation_patch_id": extension_manifest["allocation_patch_id"],
        "total_items": len(assigned),
        "unfinished_original_count": len(unfinished_original),
        "extension_count": len(extension_manifest["runs"]),
        "extension_unfinished_count": len(extension_items),
        "excluded_machine_ids": sorted(EXCLUDED_MACHINE_IDS),
        "critical_stage_1_to_3_count": len(critical),
        "critical_projected_finish_seconds": critical_finish,
        "critical_conservative_finish_seconds": critical_finish * 1.25,
        "critical_deadline_seconds": 7 * 3600,
        "central_extension_count": len(central),
        "central_projected_finish_seconds": central_finish,
        "total_projected_makespan_seconds": total_finish,
        "central_allocation_counts": central_counts,
        "central_historical_target_deviations": deviations,
        "assignment_counts": assignment_counts(assigned),
    }
    if critical and report["critical_conservative_finish_seconds"] >= report["critical_deadline_seconds"]:
        raise RuntimeError(f"critical Stage 1-3 conservative projection exceeds seven hours: {report}")
    return assigned, report


def assignment_counts(items: list[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, Any] = {}
    for item in items:
        machine = item["assigned_machine_id"]
        source = item["source_campaign_id"]
        priority = str(item["execution_priority"])
        policy = item["policy_configuration"]
        counts.setdefault(machine, {}).setdefault(source, {}).setdefault(priority, {}).setdefault(policy, 0)
        counts[machine][source][priority][policy] += 1
    return counts


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    with tmp_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    base.replace_with_retry(tmp_path, path)


def write_extension_files(manifest: dict[str, Any], dry_run: bool) -> Path:
    root = base.campaign_root(REPO_ROOT, manifest)
    if dry_run:
        return root
    root.mkdir(parents=True, exist_ok=True)
    shard_paths = {}
    for machine in manifest["machines"]:
        machine_id = machine["machine_id"]
        runs = [run for run in manifest["runs"] if run.get("machine_id") == machine_id]
        shard = {
            "schema_version": EXTENSION_SCHEMA_VERSION,
            "campaign_id": manifest["campaign_id"],
            "parent_campaign_id": manifest["parent_campaign_id"],
            "allocation_patch_id": manifest["allocation_patch_id"],
            "simulation_semantics_id": manifest["simulation_semantics_id"],
            "machine_id": machine_id,
            "runs": runs,
            "condition_count": len(runs),
        }
        path = root / base.shard_relative_path(machine_id)
        base.write_json(path, shard)
        shard_paths[machine_id] = path.relative_to(root).as_posix()
    manifest["shards"] = shard_paths
    base.write_json(root / "manifest.json", manifest)
    return root


def write_append_files(
    original_manifest: dict[str, Any],
    extension_manifest: dict[str, Any],
    inventory: dict[str, Any],
    items: list[dict[str, Any]],
    report: dict[str, Any],
    dry_run: bool,
) -> Path:
    patch_root = append_patch_root(REPO_ROOT)
    if dry_run:
        return patch_root
    patch_root.mkdir(parents=True, exist_ok=True)
    original_unfinished = [item for item in items if item["source_campaign_id"] == ORIGINAL_CAMPAIGN_ID]
    patch_manifest = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "resume_patch_id": RESUME_PATCH_ID,
        "parent_campaign_id": original_manifest["campaign_id"],
        "parent_manifest": (Path(original_manifest["campaign_root_relative"]) / "manifest.json").as_posix(),
        "parent_manifest_sha256": sha256_path(original_manifest_path(REPO_ROOT)),
        "allocation_patch_id": extension_manifest["allocation_patch_id"],
        "created_at": now_utc(),
        "runs": original_unfinished,
        "condition_count": len(original_unfinished),
    }
    base.write_json(patch_root / "manifest.json", patch_manifest)
    write_csv(patch_root / "completion_inventory.csv", inventory["rows"], [
        "campaign_id", "run_id", "condition_key", "policy_configuration", "robot_count", "order_rate", "replication", "seed",
        "stage_first_requested", "inventory_status", "strict_source", "strict_machine_id", "copy_count",
    ])
    write_csv(patch_root / "assignment.csv", items, [
        "source_campaign_id", "source_manifest", "source_condition_key", "run_id", "policy_configuration", "robot_count",
        "order_rate", "replication", "seed", "stage_first_requested", "execution_priority", "assigned_machine_id",
        "assigned_worker_slot_projection", "estimated_duration_seconds", "estimate_source", "projected_slot_finish_seconds",
        "destination_runtime_root", "allocation_patch_id",
    ])
    base.write_json(patch_root / "report.json", {**report, "completion_inventory_counts": inventory["counts"]})
    plans_root = patch_root / "execution_plans"
    for machine_id in [row["machine_id"] for row in extension_manifest["machines"]]:
        machine_items = sorted([item for item in items if item["assigned_machine_id"] == machine_id], key=item_sort_key)
        base.write_json(plans_root / f"{machine_id}.json", {
            "schema_version": PLAN_SCHEMA_VERSION,
            "machine_id": machine_id,
            "allocation_patch_id": extension_manifest["allocation_patch_id"],
            "items": machine_items,
        })
    return patch_root


def prepare(
    dry_run: bool = False,
    snapshot_root: Path = SNAPSHOT_ROOT_DEFAULT,
    migrate_merged_snapshot: Path | None = None,
) -> dict[str, Any]:
    base.ensure_clean_tracked_scientific_files(REPO_ROOT)
    original_manifest = load_original_manifest(REPO_ROOT)
    original_manifest_bytes_before = original_manifest_path(REPO_ROOT).read_bytes()
    machines = append_machines()
    allocation_patch_id = append_allocation_patch_id(machines)
    if migrate_merged_snapshot is not None:
        extension_manifest = load_extension_manifest_from_merged_zip(migrate_merged_snapshot, allocation_patch_id, machines)
        inventory = completion_inventory_from_merged_zip(original_manifest, extension_manifest, migrate_merged_snapshot)
    else:
        inventory = completion_inventory(original_manifest, REPO_ROOT, snapshot_root)
        extension_manifest = build_extension_manifest(original_manifest, machines, allocation_patch_id)
    duration_model = DurationModel(inventory["duration_records"], {m.machine_id: m for m in machines})
    items, report = schedule_items(original_manifest, extension_manifest, inventory, machines, duration_model)
    validate_design(original_manifest, extension_manifest, inventory, items)
    if migrate_merged_snapshot is None:
        write_extension_files(extension_manifest, dry_run=dry_run)
    patch_root = write_append_files(original_manifest, extension_manifest, inventory, items, report, dry_run=dry_run)
    original_manifest_bytes_after = original_manifest_path(REPO_ROOT).read_bytes()
    if original_manifest_bytes_after != original_manifest_bytes_before:
        raise RuntimeError("original manifest bytes changed during append preparation")
    return {
        "original_manifest": original_manifest,
        "extension_manifest": extension_manifest,
        "inventory": inventory,
        "items": items,
        "report": report,
        "patch_root": patch_root,
    }


def validate_design(original_manifest: dict[str, Any], extension_manifest: dict[str, Any], inventory: dict[str, Any], items: list[dict[str, Any]]) -> None:
    original_ids = {run["run_id"] for run in original_manifest["runs"]}
    if len(original_ids) != 600:
        raise AssertionError("original run IDs are not a 600-run set")
    extension_ids = {run["run_id"] for run in extension_manifest["runs"]}
    if len(extension_ids) != 600:
        raise AssertionError("extension run IDs are not a 600-run set")
    if original_ids & extension_ids:
        raise AssertionError("extension run IDs overlap original run IDs")
    all_tuples = {
        (run["policy_configuration"], int(run["robot_count"]), int(run["order_rate"]), int(run["replication"]))
        for run in original_manifest["runs"] + extension_manifest["runs"]
    }
    if len(all_tuples) != 1200:
        raise AssertionError(f"expected 1200 scientific tuples, found {len(all_tuples)}")
    strict_by_campaign = inventory.get("strict_records_by_campaign") or {original_manifest["campaign_id"]: inventory["strict_records"]}
    unfinished_original = {run_id for run_id in original_ids if run_id not in strict_by_campaign.get(original_manifest["campaign_id"], {})}
    planned_original = {item["run_id"] for item in items if item["source_campaign_id"] == ORIGINAL_CAMPAIGN_ID}
    if planned_original != unfinished_original:
        raise AssertionError("unfinished original runs are not planned exactly once")
    unfinished_extension = {run_id for run_id in extension_ids if run_id not in strict_by_campaign.get(extension_manifest["campaign_id"], {})}
    planned_extension = {item["run_id"] for item in items if item["source_campaign_id"] == extension_manifest["campaign_id"]}
    if planned_extension != unfinished_extension:
        raise AssertionError("unfinished extension runs are not planned exactly once")
    if any(item["assigned_machine_id"] in EXCLUDED_MACHINE_IDS for item in items):
        raise AssertionError("excluded machine received an assignment")
    if len({(item["source_campaign_id"], item["run_id"]) for item in items}) != len(items):
        raise AssertionError("duplicate execution-plan items detected")
    for machine_id in {item["assigned_machine_id"] for item in items}:
        queue = sorted([item for item in items if item["assigned_machine_id"] == machine_id], key=item_sort_key)
        priorities = [item["execution_priority"] for item in queue]
        if priorities != sorted(priorities):
            raise AssertionError(f"priority ordering violated for {machine_id}")


def load_plan(machine_id: str | None = None) -> dict[str, Any] | list[dict[str, Any]]:
    root = append_patch_root(REPO_ROOT) / "execution_plans"
    if machine_id:
        return base.read_json(root / f"{machine_id}.json")
    plans = []
    for path in sorted(root.glob("*.json")):
        plans.append(base.read_json(path))
    return plans


def load_or_materialize_extension_manifest(campaign_id: str, allocation_patch_id: str) -> dict[str, Any]:
    path = local_extension_manifest_path(campaign_id)
    if path.exists():
        manifest = base.read_json(path)
    else:
        parent_manifest = load_original_manifest(REPO_ROOT)
        manifest = build_extension_manifest(parent_manifest, append_machines(), allocation_patch_id)
        if manifest.get("campaign_id") != campaign_id:
            manifest = _rebuild_extension_manifest_for_campaign_id(manifest, campaign_id, allocation_patch_id)
        write_extension_files(manifest, dry_run=False)
    return {
        **manifest,
        "allocation_patch_id": allocation_patch_id,
        "allocation_patch_label": APPEND_ALLOCATION_PATCH_LABEL,
        "machines": [base.asdict(machine) for machine in append_machines()],
    }


def manifest_for_item(item: dict[str, Any]) -> dict[str, Any]:
    path = REPO_ROOT / item["source_manifest"]
    if path.exists():
        return base.read_json(path)
    if item["source_campaign_id"] == ORIGINAL_CAMPAIGN_ID:
        return load_original_manifest(REPO_ROOT)
    if str(item["source_campaign_id"]).startswith("sensitivity_full_kpi_rep21_40_"):
        return load_or_materialize_extension_manifest(item["source_campaign_id"], item["allocation_patch_id"])
    raise FileNotFoundError(f"missing source manifest for execution-plan item: {path}")


def condition_for_item(manifest: dict[str, Any], item: dict[str, Any]) -> dict[str, Any]:
    return next(run for run in manifest["runs"] if run["run_id"] == item["run_id"])


def rebuild_plan_outcomes(manifest: dict[str, Any], machine: base.Machine, items: list[dict[str, Any]]) -> Path:
    root = base.campaign_root(REPO_ROOT, manifest) / machine.machine_id
    run_items = [item for item in items if item["source_campaign_id"] == manifest["campaign_id"]]
    by_run = {run["run_id"]: run for run in manifest["runs"]}
    outcomes_path = root / "run_outcomes.csv"
    preserved: dict[str, dict[str, str]] = {}
    if outcomes_path.exists():
        try:
            with outcomes_path.open(newline="", encoding="utf-8") as fh:
                for existing in csv.DictReader(fh):
                    rid = existing.get("run_id")
                    if rid:
                        preserved[rid] = existing
        except Exception:
            preserved = {}
    rows = []
    for item in sorted(run_items, key=item_sort_key):
        condition = by_run[item["run_id"]]
        summary_path = root / "runs" / item["run_id"] / "worker_summary.json"
        if not summary_path.exists():
            if item["run_id"] in preserved:
                rows.append(preserved[item["run_id"]])
            continue
        try:
            summary = base.read_json(summary_path)
        except Exception:
            continue
        row = {field: "" for field in base.RUN_OUTCOME_FIELDS}
        row.update({
            "campaign_id": manifest["campaign_id"],
            "allocation_patch_id": item["allocation_patch_id"],
            "simulation_semantics_id": manifest["simulation_semantics_id"],
            "run_id": item["run_id"],
            "condition_key": condition["condition_key"],
            "policy_configuration": condition["policy_configuration"],
            "robot_count": condition["robot_count"],
            "order_rate": condition["order_rate"],
            "replication": condition["replication"],
            "seed": condition["seed"],
            "stage_first_requested": condition["stage_first_requested"],
            "status": summary.get("status", ""),
            "termination_reason": summary.get("termination_reason", summary.get("finalization", {}).get("reason", "")),
            "source_machine_id": machine.machine_id,
            "repo_commit": summary.get("repo_commit", manifest.get("commit", "")),
            "error_type": summary.get("error_type", ""),
            "error_message": summary.get("error_message", ""),
            "scenario_hash": condition.get("scenario_hash", ""),
            "layout_hash": condition.get("layout_hash", ""),
            "rts_metadata_canonical_sha256": condition["identity"].get("rts_metadata_canonical_sha256", ""),
            "rts_feature_schema_canonical_sha256": condition["identity"].get("rts_feature_schema_canonical_sha256", ""),
            "charging_config_canonical_sha256": condition["identity"].get("charging_config_canonical_sha256", ""),
        })
        kpi_fields_set = base.SENSITIVITY_KPI_FIELDS
        if manifest.get("kpi_schema_version") == "full_kpi_v3":
            kpi_fields_set = FULL_KPI_V3_FIELDS
        for field in kpi_fields_set:
            if field in base.RUN_OUTCOME_FIELDS:
                value = summary.get(field, summary.get("kpi", {}).get(field, ""))
                if value not in ("", None) or row.get(field, "") == "":
                    row[field] = value
        rows.append(row)
    path = outcomes_path
    write_csv(path, rows, list(base.RUN_OUTCOME_FIELDS))
    return path


def run_machine(machine_id: str, resume: bool = True, progress: bool = False, keep_run_artifacts: bool = False) -> int:
    base.ensure_clean_tracked_scientific_files(REPO_ROOT)
    plan = load_plan(machine_id)
    machines = {machine.machine_id: machine for machine in append_machines()}
    if machine_id not in machines:
        raise SystemExit(f"unknown machine id {machine_id}")
    machine = machines[machine_id]
    specs: list[RunSpec] = []
    selected: list[tuple[dict[str, Any], dict[str, Any], RunSpec, dict[str, Any]]] = []
    for item in sorted(plan["items"], key=item_sort_key):
        manifest = {**manifest_for_item(item), "allocation_patch_id": item["allocation_patch_id"]}
        base.validate_local_assets(manifest, REPO_ROOT, strict=False)
        condition = condition_for_item(manifest, item)
        spec = base.build_run_spec_from_condition(condition, manifest=manifest, machine=machine, repo_root=REPO_ROOT)
        if resume and base.run_complete_for_campaign(condition, spec, manifest, relaxed=True):
            continue
        specs.append(spec)
        selected.append((item, condition, spec, manifest))

    cleanup_stats: dict[str, Any] = {"bytes": 0, "by_filename": {}}
    return_codes: dict[str, int] = {}
    outcome_warnings: list[str] = []
    min_free = base.min_free_disk_bytes_from_env()
    root = original_root(REPO_ROOT)

    def before_launch(spec: RunSpec, active_count: int) -> bool:
        available = base.free_disk_bytes(root)
        if available >= min_free:
            return True
        if active_count > 0:
            print(f"[append-sensitivity] free disk below floor; pausing launches: path={root} free={available} required={min_free}")
            return False
        raise RuntimeError(f"insufficient free disk before launching worker: path={root} free={available} required={min_free}")

    def on_complete(spec: RunSpec, return_code: int) -> None:
        if not keep_run_artifacts:
            summary = load_worker_summary(spec.runtime_root)
            if int(return_code) == 0 and summary.get("status") == "success":
                stats = reclaim_completed_run_artifacts_with_stats(spec.runtime_root)
            else:
                stats = reclaim_failed_run_artifacts_with_stats(spec.runtime_root)
            base.merge_reclaim_stats(cleanup_stats, stats)
        try:
            related = [item for item in plan["items"] if item["source_campaign_id"] == spec.campaign_id]
            rebuild_plan_outcomes(manifest_for_item({"source_manifest": next(item["source_manifest"] for item in related)}), machine, plan["items"])
        except Exception as exc:
            warning = f"run_outcomes rebuild warning for run_id={spec.run_id}: {type(exc).__name__}: {exc}"
            outcome_warnings.append(warning)
            print(f"[append-sensitivity] {warning}")

    previous_log_cap = os.environ.get("RMFS_WORKER_LOG_CAP_MB")
    os.environ["RMFS_WORKER_LOG_CAP_MB"] = "5"
    try:
        completed = run_specs(specs, max_workers=int(machine.max_workers), progress=progress, on_run_complete=on_complete, before_launch=before_launch)
    finally:
        if previous_log_cap is None:
            os.environ.pop("RMFS_WORKER_LOG_CAP_MB", None)
        else:
            os.environ["RMFS_WORKER_LOG_CAP_MB"] = previous_log_cap
    for row in completed:
        return_codes[row["spec"].run_id] = int(row.get("return_code", -1))
    for item, condition, spec, manifest in selected:
        try:
            rebuild_plan_outcomes(manifest, machine, plan["items"])
        except Exception as exc:
            outcome_warnings.append(f"final run_outcomes rebuild failed for {manifest['campaign_id']}: {type(exc).__name__}: {exc}")
    invalid = []
    for item, condition, spec, manifest in selected:
        if not base.run_complete_for_campaign(condition, spec, manifest, relaxed=True):
            invalid.append(base.completion_failure_summary(condition, spec, return_codes))
    if invalid:
        print(f"[append-sensitivity] {len(invalid)} selected run(s) did not satisfy strict completion:")
        for row in invalid[:20]:
            print(f" - run_id={row['run_id']} condition={row['condition_key']} stage={row['stage']} return_code={row['return_code']} status={row['status']} error={row['error_type']}: {row['error_message']}")
        if len(invalid) > 20:
            print(f" - ... {len(invalid) - 20} more")
        return 1
    if outcome_warnings:
        print(f"[append-sensitivity] outcome rebuild warnings: {len(outcome_warnings)}; retry on next --resume if needed")
    return 0


def strict_completed_rows(manifest: dict[str, Any], snapshot_root: Path) -> list[dict[str, Any]]:
    inv = completion_inventory(manifest, REPO_ROOT, snapshot_root)
    rows = []
    for run_id, record in inv["strict_records"].items():
        condition = record["condition"]
        summary = record["copy"]["summary"]
        row = {field: "" for field in base.RUN_OUTCOME_FIELDS}
        row.update({
            "campaign_id": manifest["campaign_id"],
            "allocation_patch_id": manifest["allocation_patch_id"],
            "simulation_semantics_id": manifest["simulation_semantics_id"],
            "run_id": run_id,
            "condition_key": condition["condition_key"],
            "policy_configuration": condition["policy_configuration"],
            "robot_count": condition["robot_count"],
            "order_rate": condition["order_rate"],
            "replication": condition["replication"],
            "seed": condition["seed"],
            "stage_first_requested": condition["stage_first_requested"],
            "status": summary.get("status", ""),
            "termination_reason": summary.get("termination_reason", summary.get("finalization", {}).get("reason", "")),
            "source_machine_id": record["copy"].get("machine_id") or summary.get("machine_id", ""),
            "repo_commit": summary.get("repo_commit", manifest.get("commit", "")),
            "error_type": summary.get("error_type", ""),
            "error_message": summary.get("error_message", ""),
            "scenario_hash": condition.get("scenario_hash", ""),
            "layout_hash": condition.get("layout_hash", ""),
        })
        kpi_fields_set = base.SENSITIVITY_KPI_FIELDS
        if manifest.get("kpi_schema_version") == "full_kpi_v3":
            kpi_fields_set = FULL_KPI_V3_FIELDS
        for field in kpi_fields_set:
            if field in base.RUN_OUTCOME_FIELDS:
                value = summary.get(field, summary.get("kpi", {}).get(field, ""))
                if value not in ("", None) or row.get(field, "") == "":
                    row[field] = value
        rows.append(row)
    return rows


def export_combined(snapshot_root: Path = SNAPSHOT_ROOT_DEFAULT) -> Path:
    original_manifest = load_original_manifest(REPO_ROOT)
    extension_manifests = sorted((REPO_ROOT / base.OUTPUT_ROOT_RELATIVE).glob("sensitivity_full_kpi_rep21_40_*/manifest.json"))
    if not extension_manifests:
        prepare(dry_run=False, snapshot_root=snapshot_root)
        extension_manifests = sorted((REPO_ROOT / base.OUTPUT_ROOT_RELATIVE).glob("sensitivity_full_kpi_rep21_40_*/manifest.json"))
    extension_manifest = base.read_json(extension_manifests[-1])
    rows = strict_completed_rows(original_manifest, snapshot_root) + strict_completed_rows(extension_manifest, snapshot_root)
    by_tuple: dict[tuple[str, int, int, int], dict[str, Any]] = {}
    conflicts = []
    for row in rows:
        key = (row["policy_configuration"], int(row["robot_count"]), int(row["order_rate"]), int(row["replication"]))
        if key in by_tuple and by_tuple[key]["run_id"] != row["run_id"]:
            conflicts.append({"condition": key, "run_ids": [by_tuple[key]["run_id"], row["run_id"]]})
        by_tuple[key] = row
    if conflicts:
        raise RuntimeError(f"conflicting duplicate completions: {conflicts[:5]}")
    expected = {
        (policy, int(robot_count), int(order_rate), int(replication))
        for replication in range(1, 41)
        for policy in base.POLICY_CONFIGURATIONS
        for robot_count in base.ROBOT_COUNTS
        for order_rate in base.ORDER_RATES
    }
    missing = sorted(expected - set(by_tuple))
    output = append_patch_root(REPO_ROOT) / COMBINED_EXPORT_NAME
    output_rows = [by_tuple[key] for key in sorted(by_tuple)]
    write_csv(output, output_rows, list(base.RUN_OUTCOME_FIELDS))
    report = {
        "created_at": now_utc(),
        "row_count": len(output_rows),
        "expected_condition_count": 1200,
        "missing_count": len(missing),
        "missing_conditions": [list(item) for item in missing[:200]],
        "completion_counts_by_policy": {policy: sum(1 for key in by_tuple if key[0] == policy) for policy in base.POLICY_CONFIGURATIONS},
        "source_campaign_ids": sorted({row["campaign_id"] for row in output_rows}),
    }
    base.write_json(append_patch_root(REPO_ROOT) / COMBINED_REPORT_NAME, report)
    return output


def validate_existing_overlay() -> dict[str, Any]:
    patch_root = append_patch_root(REPO_ROOT)
    assignment_path = patch_root / "assignment.csv"
    report_path = patch_root / "report.json"
    plans_root = patch_root / "execution_plans"
    if not assignment_path.exists() or not report_path.exists() or not plans_root.exists():
        raise RuntimeError(f"missing migrated execution overlay under {patch_root}; run --prepare --migrate-merged-snapshot on the planner machine")
    with assignment_path.open(newline="", encoding="utf-8") as fh:
        items = list(csv.DictReader(fh))
    if any(item.get("assigned_machine_id") in EXCLUDED_MACHINE_IDS for item in items):
        raise RuntimeError("migrated execution overlay assigns work to an excluded machine")
    if len({(item.get("source_campaign_id"), item.get("run_id")) for item in items}) != len(items):
        raise RuntimeError("migrated execution overlay contains duplicate campaign/run assignments")
    report = base.read_json(report_path)
    plan_files = sorted(path.stem for path in plans_root.glob("*.json"))
    expected_plan_files = sorted({item["assigned_machine_id"] for item in items})
    if plan_files != expected_plan_files:
        raise RuntimeError(f"execution plan files do not match assigned machines: files={plan_files}, expected={expected_plan_files}")
    print(f"campaign_id_original={ORIGINAL_CAMPAIGN_ID}")
    print(f"campaign_id_extension={next(item['source_campaign_id'] for item in items if item['source_campaign_id'] != ORIGINAL_CAMPAIGN_ID)}")
    print(f"allocation_patch_id={report['allocation_patch_id']}")
    print(f"assignment_count={len(items)}")
    print(f"macbook_assignment_count={sum(1 for item in items if item['assigned_machine_id'] == 'dewa_macbook')}")
    print("assignment_counts=")
    print(json.dumps(report["assignment_counts"], indent=2, sort_keys=True))
    return {"items": items, "report": report, "patch_root": patch_root}


def print_summary(prepared: dict[str, Any]) -> None:
    manifest = prepared["extension_manifest"]
    report = prepared["report"]
    inventory = prepared["inventory"]
    print(f"campaign_id_original={ORIGINAL_CAMPAIGN_ID}")
    print(f"campaign_id_extension={manifest['campaign_id']}")
    print(f"allocation_patch_id={manifest['allocation_patch_id']}")
    print(f"simulation_semantics_id={manifest['simulation_semantics_id']}")
    strict_by_campaign = inventory.get("strict_records_by_campaign") or {ORIGINAL_CAMPAIGN_ID: inventory.get("strict_records", {})}
    print(f"strict_completed_original={len(strict_by_campaign.get(ORIGINAL_CAMPAIGN_ID, {}))}")
    print(f"strict_completed_extension={len(strict_by_campaign.get(manifest['campaign_id'], {}))}")
    print(f"strict_completed_total={inventory['counts']['strict_completed']}")
    print(f"unfinished_original={report['unfinished_original_count']}")
    print(f"unfinished_extension={report.get('extension_unfinished_count', manifest['assertions']['condition_count'])}")
    print(f"extension_condition_count={manifest['assertions']['condition_count']}")
    print(f"excluded_machine_ids={','.join(report.get('excluded_machine_ids', []))}")
    print(f"macbook_assignment_count={sum(1 for item in prepared['items'] if item['assigned_machine_id'] == 'dewa_macbook')}")
    print(f"critical_projected_finish_hours={report['critical_projected_finish_seconds'] / 3600:.2f}")
    print(f"central_projected_finish_hours={report['central_projected_finish_seconds'] / 3600:.2f}")
    print(f"total_projected_makespan_hours={report['total_projected_makespan_seconds'] / 3600:.2f}")
    print("assignment_counts=")
    print(json.dumps(report["assignment_counts"], indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare/run append-only distributed sensitivity expansion.")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--run-machine", choices=[machine.machine_id for machine in append_machines()])
    parser.add_argument("--export-combined", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--keep-run-artifacts", action="store_true")
    parser.add_argument("--snapshot-root", default=str(SNAPSHOT_ROOT_DEFAULT))
    parser.add_argument("--migrate-merged-snapshot", help="one-time local ZIP input used only to generate the committed resume_patch_0003 overlay")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    snapshot_root = Path(args.snapshot_root)
    migrate_merged_snapshot = Path(args.migrate_merged_snapshot) if args.migrate_merged_snapshot else None
    if args.export_combined:
        path = export_combined(snapshot_root=snapshot_root)
        print(f"[append-sensitivity] wrote combined export: {path}")
        return 0
    if args.run_machine:
        if migrate_merged_snapshot is not None:
            raise SystemExit("--migrate-merged-snapshot is a local prepare-time migration input and is not used while running machines")
        return run_machine(args.run_machine, resume=args.resume, progress=args.progress, keep_run_artifacts=args.keep_run_artifacts)
    if args.validate_only and migrate_merged_snapshot is None:
        validate_existing_overlay()
        print("[append-sensitivity] validate-only passed")
        return 0
    if args.prepare or args.validate_only or args.dry_run:
        prepared = prepare(dry_run=args.dry_run or args.validate_only, snapshot_root=snapshot_root, migrate_merged_snapshot=migrate_merged_snapshot)
        print_summary(prepared)
        if args.validate_only:
            print("[append-sensitivity] validate-only passed")
        elif args.dry_run:
            print("[append-sensitivity] dry-run passed; no files written")
        else:
            print(f"[append-sensitivity] wrote append plan under {prepared['patch_root']}")
        return 0
    raise SystemExit("choose one of --prepare, --validate-only, --run-machine, --export-combined, or --dry-run")


if __name__ == "__main__":
    raise SystemExit(main())
