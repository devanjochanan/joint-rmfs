"""Minimal subprocess executor for isolated local RMFS smoke runs."""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import traceback

from src.rmfs.orchestration.input_snapshot import create_input_snapshot
from src.rmfs.orchestration.run_manifest import write_run_manifest
from src.rmfs.orchestration.run_spec import RunSpec
from src.rmfs.decisions.pps.modes import normalize_pps_mode
from src.rmfs.runtime_io.run_profiles import available_profiles, resolve_run_profile
from src.rmfs.runtime_io.timing import configure_timing, timed, write_timing_summary


ROOT_SENSITIVE_FILES = [
    "netlogo.state",
    "warehouse.db",
    "assign_order.csv",
    "pod_info.csv",
    "skus_data.csv",
    "sorted_skus_data.csv",
    "generated_backlog.csv",
    "generated_database_order.csv",
    "generated_order.csv",
    "generated_pod.csv",
    "pods.csv",
]

BASE_EXPECTED_WORKER_FILES = [
    "assign_order.csv",
    "pod_info.csv",
    "skus_data.csv",
    "sorted_skus_data.csv",
    "worker_summary.json",
]

DEFERRED_ROOT_READ_ONLY_INPUTS = [
    "generated_pod.csv",
    "pods.csv",
    "items.csv",
    "raw_order.csv",
]

SENSITIVITY_KPI_SCHEMA_VERSION = "sensitivity_full_kpi.v2"
SENSITIVITY_KPI_FIELDS = (
    "run_id",
    "seed",
    "replication",
    "policy_configuration",
    "simulated_ticks",
    "simulated_seconds",
    "termination_reason",
    "orders_generated",
    "orders_completed",
    "order_lines_completed",
    "order_throughput",
    "average_order_cycle_time",
    "maximum_order_cycle_time",
    "pile_on",
    "pod_visit_picking",
    "pod_visit_replenishment",
    "pod_utilization",
    "total_robot_distance",
    "loaded_robot_distance",
    "empty_robot_distance",
    "robot_distance_per_order",
    "robot_idle_time",
    "robot_cycle_time",
    "stop_and_go_count",
    "turning_count",
    "average_job_queue",
    "peak_job_queue",
    "congestion_detected",
    "replenishment_count",
    "replenishment_trips",
    "replenishment_action_completed_count",
    "replenishment_pick_ratio",
    "total_energy_kj",
    "average_energy",
    "energy_efficiency",
    "total_fixed_load_energy",
    "energy_efficiency_fixed",
    "max_station_assignment_share",
    "station_imbalance_ratio",
    "campaign_id",
    "allocation_patch_id",
    "simulation_semantics_id",
    "machine_id",
    "stage_first_requested",
    "kpi_schema_version",
    "kpi_complete",
    "repo_commit",
    "rts_checkpoint_id",
    "rts_checkpoint_sha256",
    "pps_model_sha256",
)

SENSITIVITY_NUMERICAL_THREAD_DEFAULTS = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "RMFS_RTS_TORCH_THREADS": "1",
    "RMFS_RTS_TORCH_INTEROP_THREADS": "1",
}


def expected_worker_files(detail_db: bool = False, persist_final_state: bool = False) -> list[str]:
    files = list(BASE_EXPECTED_WORKER_FILES)
    if persist_final_state:
        files.insert(0, "netlogo.state")
    if detail_db:
        files.insert(1 if persist_final_state else 0, "warehouse.db")
    return files


# Per-run simulation scratch that is fully reproducible from a run's pinned seed.
# Worker stdout/stderr are diagnostic only -- all real results are written to
# JSON/CSV files. Cap the captured console logs so a stray print() in a
# simulation hot loop can never fill the disk mid-run (the reclaim-on-success
# path below cannot help an in-flight run whose log is still growing). Override
# the 25 MB default with RMFS_WORKER_LOG_CAP_MB (0 = unbounded, legacy behavior).
def _worker_log_cap_bytes() -> int:
    raw = os.environ.get("RMFS_WORKER_LOG_CAP_MB", "25")
    try:
        mb = int(raw)
    except (TypeError, ValueError):
        mb = 25
    return max(0, mb) * 1024 * 1024


def _pump_bounded(src, dst_path: Path, cap_bytes: int) -> None:
    """Copy a subprocess pipe to dst_path, writing at most cap_bytes.

    Once the cap is reached the pipe is still drained (so the child never blocks
    on a full pipe buffer) but further bytes are discarded after a one-time
    truncation marker. cap_bytes <= 0 means unbounded.
    """
    written = 0
    marked = False
    try:
        with open(dst_path, "wb") as dst:
            while True:
                chunk = src.read(65536)
                if not chunk:
                    break
                if cap_bytes <= 0:
                    dst.write(chunk)
                    dst.flush()
                elif written < cap_bytes:
                    room = cap_bytes - written
                    dst.write(chunk[:room])
                    written += min(len(chunk), room)
                    if written >= cap_bytes and not marked:
                        dst.write(b"\n[log truncated: exceeded RMFS_WORKER_LOG_CAP_MB; further output discarded]\n")
                        marked = True
                    dst.flush()
                # else: cap reached -- keep draining, discard the rest
    except Exception:
        pass
    finally:
        try:
            src.close()
        except Exception:
            pass


# Reclaimed after a sensitivity run terminates. Successful runs are reduced to
# run_spec.json + worker_summary.json; failed runs keep bounded logs and status.
SUCCESS_RECLAIMABLE_RUN_ARTIFACTS = (
    "netlogo.state",
    "warehouse.db",
    "assign_order.csv",
    "generated_order.csv",
    "generated_backlog.csv",
    "generated_database_order.csv",
    "generated_order_meta.json",
    "pod_info.csv",
    "skus_data.csv",
    "sorted_skus_data.csv",
    "rts_rollout.jsonl",
    "debug_trace.jsonl",
    "timing_summary.json",
    "worker_status.json",
    "worker_stdout.log",
    "worker_stderr.log",
    "rts_rollout_summary.json",
)

FAILED_RECLAIMABLE_RUN_ARTIFACTS = tuple(
    name
    for name in SUCCESS_RECLAIMABLE_RUN_ARTIFACTS
    if name not in {"worker_status.json", "worker_stdout.log", "worker_stderr.log", "rts_rollout_summary.json"}
)


def directory_size_bytes(root: Path) -> int:
    total = 0
    root = Path(root)
    if not root.exists():
        return 0
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        try:
            total += path.stat().st_size
        except OSError:
            continue
    return total


def _unlink_run_artifacts(run_root: Path, artifact_names: tuple[str, ...]) -> dict[str, object]:
    stats: dict[str, object] = {"bytes": 0, "by_filename": {}}
    by_filename: dict[str, int] = {}
    for name in artifact_names:
        artifact = run_root / name
        try:
            size = artifact.stat().st_size
        except OSError:
            continue
        try:
            artifact.unlink(missing_ok=True)
        except OSError:
            continue
        stats["bytes"] = int(stats["bytes"]) + int(size)
        by_filename[name] = by_filename.get(name, 0) + int(size)
    stats["by_filename"] = by_filename
    return stats


def _merge_reclaim_stats(target: dict[str, object], addition: dict[str, object]) -> dict[str, object]:
    target["bytes"] = int(target.get("bytes", 0) or 0) + int(addition.get("bytes", 0) or 0)
    target_by_filename = dict(target.get("by_filename", {}) or {})
    for name, size in dict(addition.get("by_filename", {}) or {}).items():
        target_by_filename[str(name)] = target_by_filename.get(str(name), 0) + int(size)
    target["by_filename"] = target_by_filename
    return target


def _embed_rollout_summary(run_root: Path) -> None:
    summary_path = run_root / "worker_summary.json"
    rollout_path = run_root / "rts_rollout_summary.json"
    if not summary_path.exists() or not rollout_path.exists():
        return
    try:
        with summary_path.open() as fh:
            summary = json.load(fh)
        with rollout_path.open() as fh:
            rollout_summary = json.load(fh)
    except Exception:
        return
    summary["rts_rollout_summary"] = rollout_summary
    summary["rts_rollout_summary_embedded"] = True
    write_json(summary_path, summary)


def reclaim_completed_run_artifacts_with_stats(run_root: Path) -> dict[str, object]:
    """Delete regenerable simulation scratch from a successful run."""
    run_root = Path(run_root)
    summary_path = run_root / "worker_summary.json"
    if not summary_path.exists():
        return {"bytes": 0, "by_filename": {}}
    try:
        with summary_path.open() as fh:
            summary = json.load(fh)
    except Exception:
        return {"bytes": 0, "by_filename": {}}
    if summary.get("status") != "success":
        return {"bytes": 0, "by_filename": {}}
    _embed_rollout_summary(run_root)
    return _unlink_run_artifacts(run_root, SUCCESS_RECLAIMABLE_RUN_ARTIFACTS)


def reclaim_failed_run_artifacts_with_stats(run_root: Path) -> dict[str, object]:
    """Delete large regenerable scratch while preserving failure diagnostics."""
    run_root = Path(run_root)
    return _unlink_run_artifacts(run_root, FAILED_RECLAIMABLE_RUN_ARTIFACTS)


def reclaim_interrupted_run_artifacts_with_stats(run_root: Path) -> dict[str, object]:
    """Delete stale scratch for a run that will restart from step zero."""
    run_root = Path(run_root)
    summary_path = run_root / "worker_summary.json"
    if summary_path.exists():
        try:
            with summary_path.open() as fh:
                summary = json.load(fh)
            if summary.get("status") == "success":
                return {"bytes": 0, "by_filename": {}}
        except Exception:
            pass
    return reclaim_failed_run_artifacts_with_stats(run_root)


def reclaim_completed_run_artifacts(run_root: Path) -> int:
    """Delete regenerable simulation scratch from a successful run, returning bytes freed.

    Keeps result files needed for aggregation and strict resume
    (worker_summary.json, run_spec.json). Failed or unfinished runs are left to
    the failed/interrupted cleanup helpers so diagnostics are preserved.
    Idempotent: already missing files are ignored.
    """
    return int(reclaim_completed_run_artifacts_with_stats(run_root).get("bytes", 0) or 0)


def worker_environment_overrides(spec: RunSpec) -> dict[str, str]:
    pps_mode = normalize_pps_mode(spec.pps_mode)
    env = {
        "RMFS_RUN_PROFILE": spec.run_profile,
        "RMFS_NUM_ROBOTS": str(int(spec.robot_count)),
        "PPS_MODE": pps_mode,
        "PPS_RL_ENABLED": "1" if pps_mode == "ppo" else "0",
        "RMFS_ORDER_GENERATION_MODE": spec.order_generation_mode,
        "RMFS_FULL_RAW_ORDER_REPLAY": "1" if spec.full_raw_order_replay else "0",
        "RMFS_DETAIL_DB": "1" if spec.detail_db else "0",
        "RMFS_ROBOT_TASK_ALLOCATOR": spec.robot_task_allocator,
        "RMFS_TASK_ALLOCATOR_SCOPE": spec.task_allocator_scope,
        "RMFS_COMMITTED_NEXT_RESERVATIONS": "1" if spec.committed_next_reservations_enabled else "0",
        "RMFS_POD_LOCATION_MODE": spec.pod_location_mode,
    }
    if spec.pps_model_path:
        env["PPS_RL_MODEL_PATH"] = spec.pps_model_path
    if spec.charging_enabled is not None:
        env["RMFS_CHARGING_ENABLED"] = "1" if spec.charging_enabled else "0"
    if spec.charging_config_path:
        env["RMFS_CHARGING_CONFIG"] = spec.charging_config_path
    if spec.rts_policy_mode == "rts_rl_explicit":
        torch_threads = spec.rts_torch_threads if spec.rts_torch_threads is not None else 1
        torch_interop_threads = spec.rts_torch_interop_threads if spec.rts_torch_interop_threads is not None else 1
        env["RMFS_RTS_TORCH_THREADS"] = str(torch_threads)
        env["RMFS_RTS_TORCH_INTEROP_THREADS"] = str(torch_interop_threads)
        env["OMP_NUM_THREADS"] = str(torch_threads)
        env["MKL_NUM_THREADS"] = str(torch_threads)
        env["OPENBLAS_NUM_THREADS"] = str(torch_threads)
    else:
        if spec.rts_torch_threads is not None:
            env["RMFS_RTS_TORCH_THREADS"] = str(spec.rts_torch_threads)
            env["OMP_NUM_THREADS"] = str(spec.rts_torch_threads)
            env["MKL_NUM_THREADS"] = str(spec.rts_torch_threads)
            env["OPENBLAS_NUM_THREADS"] = str(spec.rts_torch_threads)
        if spec.rts_torch_interop_threads is not None:
            env["RMFS_RTS_TORCH_INTEROP_THREADS"] = str(spec.rts_torch_interop_threads)
    if spec.regret_k is not None:
        env["RMFS_REGRET_K"] = str(spec.regret_k)
    if spec.rts_random_seed is not None:
        env["RMFS_SIM_SEED"] = str(spec.rts_random_seed)
    if spec.run_horizon_ticks is not None:
        env["RMFS_RUN_HORIZON_TICKS"] = str(spec.run_horizon_ticks)
    if spec.bootstrap_n_orders is not None:
        env["RMFS_BOOTSTRAP_N_ORDERS"] = str(spec.bootstrap_n_orders)
    if spec.demand_horizon_ticks is not None:
        env["RMFS_DEMAND_HORIZON_TICKS"] = str(spec.demand_horizon_ticks)
    if spec.demand_buffer_ticks is not None:
        env["RMFS_DEMAND_BUFFER_TICKS"] = str(spec.demand_buffer_ticks)
    if spec.order_rate_per_hour is not None:
        env["RMFS_ORDER_CYCLE_TIME"] = str(int(spec.order_rate_per_hour))
    if spec.pod_location_seed is not None:
        env["RMFS_POD_LOCATION_SEED"] = str(spec.pod_location_seed)
    return env


def worker_child_environment(spec: RunSpec) -> dict[str, str]:
    child_env = os.environ.copy()
    child_env.update(worker_environment_overrides(spec))
    if spec.kpi_schema_version == SENSITIVITY_KPI_SCHEMA_VERSION:
        for key, value in SENSITIVITY_NUMERICAL_THREAD_DEFAULTS.items():
            child_env[key] = value
    return child_env


def file_digest(path: Path):
    if not path.exists():
        return {"exists": False, "size": 0, "sha256": None}
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            digest.update(chunk)
    return {"exists": True, "size": path.stat().st_size, "sha256": digest.hexdigest()}


def snapshot_root(repo_root: Path):
    return {name: file_digest(repo_root / name) for name in ROOT_SENSITIVE_FILES}


def hash_snapshot_files(snapshot_root_dir: Path):
    if not snapshot_root_dir.exists():
        return {}
    digests = {}
    for path in snapshot_root_dir.rglob("*"):
        if path.is_file():
            rel_path = path.relative_to(snapshot_root_dir).as_posix()
            digests[rel_path] = file_digest(path)
    return digests


def changed_root_files(before, after):
    return [name for name in ROOT_SENSITIVE_FILES if before[name] != after[name]]


def stable_digest(payload):
    def sanitize(obj):
        if obj is None or isinstance(obj, (int, str, bool)):
            return obj
        if isinstance(obj, float):
            return f"{obj:.6f}"
        if isinstance(obj, dict):
            return {str(k): sanitize(v) for k, v in sorted(obj.items())}
        if isinstance(obj, set):
            return sorted(
                (sanitize(item) for item in obj),
                key=lambda x: json.dumps(x, sort_keys=True, default=str),
            )
        if isinstance(obj, (list, tuple)):
            return [sanitize(item) for item in obj]
        if hasattr(obj, "__dict__"):
            return sanitize(obj.__dict__)
        return str(obj)

    serialized = json.dumps(sanitize(payload), sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def return_signature(payload):
    if hasattr(payload, "__len__"):
        length = len(payload)
    else:
        length = None
    return {"type": type(payload).__name__, "length": length}


def _safe_float(value, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if result != result or result in (float("inf"), float("-inf")):
        return default
    return result


def _safe_div(numerator, denominator) -> float:
    denom = _safe_float(denominator)
    if denom <= 0.0:
        return 0.0
    return _safe_float(numerator) / denom


def _sensitivity_rts_checkpoint_id(spec: RunSpec) -> str | None:
    if spec.rts_policy_checkpoint_id:
        return spec.rts_policy_checkpoint_id
    if str(spec.policy_configuration or "").strip() == "all_off":
        return "not_applicable"
    return None


def _robot_objects(warehouse) -> list[object]:
    if warehouse is None:
        return []
    return [
        obj
        for obj in getattr(warehouse, "_objects", []) or []
        if getattr(obj, "object_type", None) == "robot"
    ]


def _completed_orders(warehouse) -> list[object]:
    if warehouse is None or not hasattr(warehouse, "order_manager"):
        return []
    return [
        order
        for order in getattr(warehouse.order_manager, "orders", []) or []
        if order.is_order_completed()
    ]


def derive_sensitivity_kpi_payload(
    spec: RunSpec,
    warehouse,
    finalization: dict | None,
    generated_order_contract: dict | None = None,
) -> dict[str, object]:
    finalization = dict(finalization or {})
    generated_order_contract = dict(generated_order_contract or {})
    robots = _robot_objects(warehouse)
    orders = list(getattr(getattr(warehouse, "order_manager", None), "orders", []) or [])
    completed_orders = _completed_orders(warehouse)
    cycle_times = [
        _safe_float(order.order_complete_time) - _safe_float(order.process_start_time)
        for order in completed_orders
        if _safe_float(order.order_complete_time, -1.0) >= 0.0
        and _safe_float(order.process_start_time, -1.0) >= 0.0
    ]
    station_counts: dict[str, int] = {}
    for order in completed_orders:
        station_id = getattr(order, "station_id", None)
        if station_id is not None:
            station_counts[str(station_id)] = station_counts.get(str(station_id), 0) + 1
    station_total = sum(station_counts.values())
    max_station = max(station_counts.values()) if station_counts else 0
    min_station = min(station_counts.values()) if station_counts else 0
    total_robot_distance = sum(_safe_float(getattr(robot, "travel_distances", 0.0)) for robot in robots)
    loaded_robot_distance = sum(_safe_float(getattr(robot, "loaded_travel_distance", 0.0)) for robot in robots)
    empty_robot_distance = sum(_safe_float(getattr(robot, "empty_travel_distance", 0.0)) for robot in robots)
    completed_cycle_count = sum(int(getattr(robot, "completed_cycle_count", 0) or 0) for robot in robots)
    completed_cycle_duration = sum(_safe_float(getattr(robot, "completed_cycle_duration_sum", 0.0)) for robot in robots)
    pod_count = len(getattr(getattr(warehouse, "pod_manager", None), "pods", []) or [])
    pod_visit_picking = int(getattr(warehouse, "pps_pod_visits", getattr(warehouse, "pps_rl_pod_visits", 0)) or 0)
    picked_quantity = int(getattr(warehouse, "pps_picked_quantity", getattr(warehouse, "pps_rl_picked_quantity", 0)) or 0)
    total_energy_kj = _safe_float(getattr(warehouse, "total_energy", 0.0)) / 1000.0
    total_fixed_load_energy = sum(_safe_float(getattr(robot, "fixed_load_energy_consumption", 0.0)) for robot in robots) / 1000.0
    simulated_seconds = _safe_float(spec.simulated_horizon_seconds)
    generated_orders = generated_order_contract.get("generated_unique_orders")
    if generated_orders is None:
        generated_orders = len(orders)
    order_lines_completed = 0
    for order in completed_orders:
        for details in getattr(order, "skus", {}).values():
            if _safe_float(details.get("quantity_delivered")) >= _safe_float(details.get("total_quantity")):
                order_lines_completed += 1
    job_queue_sample_count = int(getattr(warehouse, "job_queue_sample_count", 0) or 0)
    job_queue_cumulative = _safe_float(getattr(warehouse, "job_queue_cumulative_sum", 0.0))
    peak_job_queue = int(getattr(warehouse, "peak_job_queue", len(getattr(warehouse, "job_queue", []) or [])) or 0)
    replenishment_trips = int(getattr(warehouse, "replenishment_trips", 0) or 0)
    replenishment_completed = (
        int(getattr(warehouse, "completed_post_pick_replenishment_actions", 0) or 0)
        + int(getattr(warehouse, "completed_rts_replenishment_actions", 0) or 0)
        + int(getattr(warehouse, "completed_proactive_replenishment_actions", 0) or 0)
    )
    if replenishment_completed <= 0:
        replenishment_completed = replenishment_trips
    reason = finalization.get("reason")
    runtime_invariants = finalization.get("runtime_invariants", {}) or {}
    payload = {
        "run_id": spec.run_id,
        "seed": spec.campaign_seed if spec.campaign_seed is not None else spec.rts_random_seed,
        "replication": spec.replication,
        "policy_configuration": spec.policy_configuration or ("all_on_rl" if spec.rts_policy_mode == "rts_rl_explicit" else "all_off"),
        "simulated_ticks": int(spec.ticks),
        "simulated_seconds": simulated_seconds,
        "termination_reason": reason,
        "orders_generated": int(generated_orders or 0),
        "orders_completed": len(completed_orders),
        "order_lines_completed": int(order_lines_completed),
        "order_throughput": _safe_div(len(completed_orders), simulated_seconds / 3600.0),
        "average_order_cycle_time": sum(cycle_times) / len(cycle_times) if cycle_times else 0.0,
        "maximum_order_cycle_time": max(cycle_times) if cycle_times else 0.0,
        "pile_on": _safe_div(picked_quantity, pod_visit_picking),
        "pod_visit_picking": pod_visit_picking,
        "pod_visit_replenishment": replenishment_trips,
        "pod_utilization": _safe_div(pod_visit_picking + replenishment_trips, pod_count),
        "total_robot_distance": total_robot_distance,
        "loaded_robot_distance": loaded_robot_distance,
        "empty_robot_distance": empty_robot_distance,
        "robot_distance_per_order": _safe_div(total_robot_distance, len(completed_orders)),
        "robot_idle_time": _safe_float(getattr(warehouse, "total_robot_idle", 0.0)),
        "robot_cycle_time": _safe_div(completed_cycle_duration, completed_cycle_count),
        "stop_and_go_count": int(getattr(warehouse, "stop_and_go", 0) or 0),
        "turning_count": int(getattr(warehouse, "total_turning", 0) or 0),
        "average_job_queue": _safe_div(job_queue_cumulative, job_queue_sample_count),
        "peak_job_queue": peak_job_queue,
        "congestion_detected": bool(reason == "congestion" or runtime_invariants.get("hard_violation_count", 0)),
        "replenishment_count": int(getattr(warehouse, "replenishment_count", 0) or 0),
        "replenishment_trips": replenishment_trips,
        "replenishment_action_completed_count": replenishment_completed,
        "post_pick_store_action_completed_count": int(getattr(warehouse, "completed_post_pick_store_actions", 0) or 0),
        "replenishment_pick_ratio": _safe_div(replenishment_trips, pod_visit_picking),
        "total_energy_kj": total_energy_kj,
        "average_energy": _safe_div(total_energy_kj, len(robots)),
        "energy_efficiency": _safe_div(len(completed_orders), total_energy_kj),
        "total_fixed_load_energy": total_fixed_load_energy,
        "energy_efficiency_fixed": _safe_div(len(completed_orders), total_fixed_load_energy),
        "max_station_assignment_share": _safe_div(max_station, station_total),
        "station_imbalance_ratio": _safe_div(max_station, min_station),
        "campaign_id": spec.campaign_id,
        "allocation_patch_id": spec.allocation_patch_id,
        "simulation_semantics_id": spec.simulation_semantics_id,
        "machine_id": spec.machine_id,
        "stage_first_requested": spec.stage_first_requested,
        "kpi_schema_version": spec.kpi_schema_version,
        "repo_commit": spec.commit,
        "rts_checkpoint_id": _sensitivity_rts_checkpoint_id(spec),
        "rts_checkpoint_sha256": spec.rts_checkpoint_sha256,
        "pps_model_sha256": spec.pps_model_sha256,
    }
    payload["kpi_complete"] = all(field in payload and payload[field] is not None for field in SENSITIVITY_KPI_FIELDS if field != "kpi_complete")
    return payload


def git_value(repo_root: Path, *args):
    try:
        return subprocess.check_output(["git", *args], cwd=repo_root, text=True).strip()
    except Exception:
        return None


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    with tmp_path.open("w") as fh:
        json.dump(payload, fh, indent=2)
    _atomic_replace_with_retry(tmp_path, path)


def _atomic_replace_with_retry(tmp_path: Path, path: Path, attempts: int = 12) -> None:
    """Replace path with tmp_path, retrying transient Windows share violations.

    On Windows os.replace/Path.replace raises PermissionError (WinError 5) when
    another process momentarily holds the destination open -- e.g. the
    controller polling worker_status.json for the progress bar. That window is
    tiny, so retry with short backoff instead of letting the worker die at
    ticks_completed=0. On POSIX the first attempt always succeeds.
    """
    last_exc: PermissionError | None = None
    for attempt in range(attempts):
        try:
            tmp_path.replace(path)
            return
        except PermissionError as exc:
            last_exc = exc
            time.sleep(min(0.05 * (attempt + 1), 0.5))
    try:
        tmp_path.unlink()
    except OSError:
        pass
    raise last_exc if last_exc is not None else RuntimeError(f"failed to replace {path}")


def validate_generated_order_contract(spec: RunSpec) -> dict[str, object]:
    meta_path = spec.runtime_root / "generated_order_meta.json"
    if not meta_path.exists():
        raise RuntimeError(f"generated_order_meta.json missing in worker runtime: {meta_path}")
    with meta_path.open() as fh:
        meta = json.load(fh)
    checks = {
        "profile": (spec.run_profile, meta.get("profile")),
        "order_generation_mode": (spec.order_generation_mode, meta.get("order_generation_mode")),
        "full_raw_order_replay": (bool(spec.full_raw_order_replay), bool(meta.get("full_raw_order_replay"))),
        "seed": (spec.rts_random_seed, meta.get("seed")),
        "order_rate_per_hour": (spec.order_rate_per_hour, meta.get("order_rate_per_hour", meta.get("order_cycle_time"))),
    }
    mismatches = {
        name: {"requested": requested, "generated": generated}
        for name, (requested, generated) in checks.items()
        if requested is not None and requested != generated
    }
    if mismatches:
        raise RuntimeError(f"generated order metadata mismatch: {mismatches}")
    if meta.get("arrival_time_unit") != "simulated_seconds":
        raise RuntimeError("generated order metadata must declare arrival_time_unit=simulated_seconds")
    if meta.get("order_rate_unit") not in (None, "orders_per_simulated_hour"):
        raise RuntimeError("generated order metadata order_rate_unit must be orders_per_simulated_hour")
    if spec.order_generation_mode == "legacy_compat":
        raise RuntimeError("headless workers reject generated legacy_compat order generation")
    csv_path = spec.runtime_root / "generated_order.csv"
    measured = _measure_generated_order_rate(csv_path, int(spec.order_rate_per_hour or 0))
    return {
        "metadata_path": str(meta_path),
        "requested_profile": spec.run_profile,
        "generated_profile": meta.get("profile"),
        "requested_mode": spec.order_generation_mode,
        "generated_mode": meta.get("order_generation_mode"),
        "requested_order_rate_per_hour": spec.order_rate_per_hour,
        "generated_order_rate_per_hour": meta.get("order_rate_per_hour", meta.get("order_cycle_time")),
        "order_rate_unit": meta.get("order_rate_unit"),
        "seed": meta.get("seed"),
        "full_raw_order_replay": bool(meta.get("full_raw_order_replay")),
        "arrival_time_unit": meta.get("arrival_time_unit"),
        "generated_unique_orders": meta.get("generated_unique_orders"),
        "generated_order_lines": meta.get("generated_order_lines"),
        "generated_max_arrival": meta.get("generated_max_arrival"),
        "measured_complete_hour_order_counts": measured.get("complete_hour_order_counts", []),
        "measured_complete_hour_rate_ok": measured.get("complete_hour_rate_ok"),
    }


def _measure_generated_order_rate(csv_path: Path, expected_rate: int) -> dict[str, object]:
    if expected_rate <= 0 or not csv_path.exists():
        return {"complete_hour_order_counts": [], "complete_hour_rate_ok": None}
    import csv

    order_arrivals: dict[str, int] = {}
    with csv_path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            order_id = str(row.get("order_id", ""))
            if order_id not in order_arrivals:
                order_arrivals[order_id] = int(float(row.get("order_arrival", 0)))
    if not order_arrivals:
        return {"complete_hour_order_counts": [], "complete_hour_rate_ok": False}
    max_arrival = max(order_arrivals.values())
    complete_hours = max_arrival // 3600
    counts = []
    for hour in range(int(complete_hours)):
        start = hour * 3600
        end = start + 3600
        counts.append(sum(1 for arrival in order_arrivals.values() if start <= arrival < end))
    return {
        "complete_hour_order_counts": counts,
        "complete_hour_rate_ok": all(count == expected_rate for count in counts),
    }


def finalize_rts_static_runtime_after_setup(warehouse, requested_zone_ids) -> dict[str, object] | None:
    if warehouse is None:
        return None
    runtime = getattr(warehouse, "rts_rollout_runtime", None)
    config = getattr(runtime, "config", None)
    policy = getattr(warehouse, "rts_policy", None)
    policy_mode = getattr(config, "policy_mode", None)
    if policy_mode not in {"current", "current_probe", "random_valid", "rts_rl_explicit", "vrsla_teacher"} or not bool(getattr(config, "rollout_enabled", False)):
        return None
    from dataclasses import replace
    from src.rmfs.rl.rts.static_runtime_index import (
        install_static_runtime_index,
        resolve_runtime_zone_ids,
        static_runtime_index_diagnostics,
    )
    from src.rmfs.rl.rts.runtime_config import validate_rts_runtime_config

    index = install_static_runtime_index(warehouse)
    requested = tuple(requested_zone_ids or getattr(config, "zone_ids", ()) or ())
    resolved_zone_ids = resolve_runtime_zone_ids(warehouse, requested)
    if policy is not None and hasattr(policy, "zone_ids"):
        policy.zone_ids = tuple(resolved_zone_ids)
    if config is not None and tuple(getattr(config, "zone_ids", ()) or ()) != tuple(resolved_zone_ids):
        runtime.config = replace(config, zone_ids=tuple(resolved_zone_ids))
        validate_rts_runtime_config(runtime.config)
    setattr(runtime, "static_runtime_index_summary", index.to_summary_dict())
    return {
        "runtime_zone_manifest": index.manifest.to_json_dict(),
        "rts_static_runtime_index": index.to_summary_dict(),
        "rts_static_runtime_index_lifecycle": static_runtime_index_diagnostics(),
        "resolved_zone_ids": list(resolved_zone_ids),
    }


def run_worker(spec: RunSpec):
    worker_start = time.perf_counter()
    original_cwd = Path.cwd()
    netlogo_module = None
    session = None
    status_path = spec.runtime_root / "worker_status.json"
    ticks_done = 0
    last_status_tick = -1

    def write_worker_status(status: str, force: bool = False) -> None:
        nonlocal last_status_tick
        cadence = max(1, int(getattr(spec, "worker_status_cadence", 100) or 100))
        should_write = (
            force
            or spec.debug_trace
            or ticks_done == 0
            or ticks_done == spec.ticks
            or ticks_done - last_status_tick >= cadence
        )
        if not should_write:
            return
        with timed("worker_status_write"):
            write_json(status_path, {
                "current_progress_steps": ticks_done,
                "progress_target_steps": spec.ticks,
                "status": status,
            })
        last_status_tick = ticks_done

    configure_timing(spec.timing, spec.runtime_root / "timing_summary.json")
    write_worker_status("running", force=True)
    summary = {
        "run_id": spec.run_id,
        "status": "failure",
        "ticks_requested": spec.ticks,
        "netlogo_steps_requested": spec.ticks,
        "ticks_completed": 0,
        "netlogo_steps_completed": 0,
        "setup_digest": None,
        "setup_signature": None,
        "first_tick_digest": None,
        "final_tick_digest": None,
        "first_tick_signature": None,
        "final_tick_signature": None,
        "final_metrics": None,
        "runtime_root": str(spec.runtime_root),
        "repo_root": str(spec.repo_root),
        "persist_final_state": bool(spec.persist_final_state),
        "requested_robot_count": int(spec.robot_count),
        "realized_robot_count": None,
        "expected_picking_station_count": spec.expected_picking_station_count,
        "realized_picking_station_count": None,
        "expected_replenishment_station_count": spec.expected_replenishment_station_count,
        "realized_replenishment_station_count": None,
    }
    debug_rows = []

    try:
        spec.validate_runtime_semantics()
        sys.path.insert(0, str(spec.repo_root))
        os.chdir(spec.repo_root)

        from src.rmfs.runtime_io import RunContext
        from src.rmfs.runtime_io.detail_db import configure_detail_db
        from src.rmfs.rl.rts.runtime_config import RTSRuntimeConfig
        from src.rmfs.rl.rts.runtime_registry import configure_rts_runtime, reset_rts_runtime
        from src.rmfs.rl.rts.static_runtime_index import reset_static_runtime_index_diagnostics
        import netlogo

        netlogo_module = netlogo
        ctx = RunContext.isolated(spec.runtime_root, repo_root=spec.repo_root, input_root=spec.input_root)
        ctx.ensure_runtime_dirs()
        configure_detail_db(enabled=spec.detail_db, db_path=ctx.sqlite_db)
        netlogo.configure_run_context(ctx)
        env_overrides = worker_environment_overrides(spec)
        previous_env = {key: os.environ.get(key) for key in env_overrides}
        os.environ.update(env_overrides)
        if spec.rts_random_seed is not None:
            netlogo.set_sim_seed(spec.rts_random_seed)
        configure_rts_runtime(
            RTSRuntimeConfig(
                policy_mode=spec.rts_policy_mode,
                rollout_enabled=spec.rts_rollout_enabled,
                rollout_write_disk=spec.rts_rollout_write_disk,
                zone_ids=tuple(spec.rts_zone_ids or ()),
                reward_reference_path=spec.rts_reward_reference_path,
                random_seed=spec.rts_random_seed,
                max_events=spec.rts_max_events,
                policy_checkpoint_dir=spec.rts_policy_checkpoint_dir,
                policy_checkpoint_id=spec.rts_policy_checkpoint_id,
                policy_action_mode=spec.rts_policy_action_mode,
                policy_device=spec.rts_policy_device,
                feature_ablation=spec.rts_feature_ablation,
                feature_ablation_hash=spec.rts_feature_ablation_hash,
                state_capture_mode=spec.rts_state_capture_mode,
                committed_next_reservations_enabled=spec.committed_next_reservations_enabled,
                run_id=spec.run_id,
                batch_id=spec.batch_id,
                worker_id=spec.worker_id,
                artifact_label=spec.artifact_label,
            ),
            runtime_root=spec.runtime_root,
        )
        reset_static_runtime_index_diagnostics()

        if spec.rts_policy_mode == "rts_rl_explicit":
            import torch
            torch_threads = spec.rts_torch_threads if spec.rts_torch_threads is not None else 1
            torch_interop_threads = spec.rts_torch_interop_threads if spec.rts_torch_interop_threads is not None else 1
            torch.set_num_threads(torch_threads)
            torch.set_num_interop_threads(torch_interop_threads)


        with timed("setup"):
            session = netlogo.HeadlessSimulationSession(persist_final_state=spec.persist_final_state)
            setup_result = session.setup()
        if isinstance(setup_result, str) and "An error occurred" in setup_result:
            raise RuntimeError(setup_result)
        summary["generated_order_contract"] = validate_generated_order_contract(spec)
        summary["setup_digest"] = stable_digest(setup_result)
        summary["setup_signature"] = return_signature(setup_result)

        warehouse = session.warehouse
        if warehouse is not None:
            robots = [
                obj
                for obj in warehouse.get_movable_objects()
                if getattr(obj, "object_type", None) == "robot"
            ]
            station_manager = getattr(warehouse, "station_manager", None)
            picking_stations = list(getattr(station_manager, "picking_stations", []) or [])
            replenishment_stations = list(getattr(station_manager, "replenishment_stations", []) or [])
            summary["realized_robot_count"] = len(robots)
            summary["realized_picking_station_count"] = len(picking_stations)
            summary["realized_replenishment_station_count"] = len(replenishment_stations)
            if spec.expected_picking_station_count is not None and len(picking_stations) != spec.expected_picking_station_count:
                raise RuntimeError(
                    "picking station count mismatch: "
                    f"expected {spec.expected_picking_station_count}, realized {len(picking_stations)}"
                )
            if (
                spec.expected_replenishment_station_count is not None
                and len(replenishment_stations) != spec.expected_replenishment_station_count
            ):
                raise RuntimeError(
                    "replenishment station count mismatch: "
                    f"expected {spec.expected_replenishment_station_count}, realized {len(replenishment_stations)}"
                )
            if len(robots) != int(spec.robot_count):
                raise RuntimeError(
                    f"robot count mismatch: requested {int(spec.robot_count)}, realized {len(robots)}"
                )
        if warehouse is not None:
            try:
                finalized_rts = finalize_rts_static_runtime_after_setup(warehouse, spec.rts_zone_ids)
                if finalized_rts is not None:
                    summary.update(finalized_rts)
            except Exception as exc:
                summary["rts_static_runtime_index_error"] = f"{type(exc).__name__}: {exc}"
                raise
        tick_to_second = getattr(warehouse, "tick_to_second", 1.0)
        summary["warehouse_time_start"] = float(getattr(warehouse, "_tick", 0.0)) if warehouse is not None else 0.0
        summary["tick_to_second"] = tick_to_second
        summary["run_time_unit_contract"] = {
            "backend_step": "one warehouse.tick() / NetLogo simulation step",
            "netlogo_steps_requested": spec.netlogo_steps_requested,
            "tick_to_second": spec.tick_to_second,
            "simulated_horizon_seconds": spec.simulated_horizon_seconds,
            "demand_horizon_steps": spec.demand_horizon_steps,
            "demand_horizon_simulated_seconds": spec.demand_horizon_simulated_seconds,
            "demand_buffer_steps": spec.demand_buffer_steps,
            "demand_buffer_simulated_seconds": spec.demand_buffer_simulated_seconds,
            "order_rate_per_hour": spec.order_rate_per_hour,
            "order_rate_unit": "orders_per_simulated_hour" if spec.order_rate_per_hour is not None else None,
            "order_arrival_unit": "simulated_seconds",
        }

        first_result = None
        final_result = None
        for index in range(spec.ticks):
            with timed("tick"):
                step_result = session.step()
            tick_result = step_result.payload
            if tick_result is None:
                raise RuntimeError(f"worker stopped without a tick payload: {step_result.status}")

            ticks_done += step_result.steps_executed
            write_worker_status("running")

            is_first = index == 0
            is_final = index == spec.ticks - 1

            trace_selected = False
            if spec.debug_trace:
                if spec.trace_first_n > 0 and index < spec.trace_first_n:
                    trace_selected = True
                if spec.trace_cadence > 0 and (index + 1) % spec.trace_cadence == 0:
                    trace_selected = True
                if is_final:
                    trace_selected = True

            if is_first or is_final or trace_selected:
                digest = stable_digest(tick_result)
                sig = return_signature(tick_result)

            if is_first:
                first_result = (digest, sig)
            if is_final:
                final_result = (digest, sig, tick_result)

            if trace_selected:
                metrics = {}
                if isinstance(tick_result, list) and len(tick_result) >= 6:
                    metrics = {
                        "total_energy": tick_result[1],
                        "job_queue_len": tick_result[2],
                        "stop_and_go": tick_result[3],
                        "total_turning": tick_result[4],
                    }
                debug_rows.append({
                    "tick_index": index + 1,
                    "digest": digest,
                    "signature": sig,
                    "metrics": metrics,
                })

        summary["ticks_completed"] = ticks_done
        summary["netlogo_steps_completed"] = ticks_done
        if first_result:
            summary["first_tick_digest"] = first_result[0]
            summary["first_tick_signature"] = first_result[1]
        if final_result:
            summary["final_tick_digest"] = final_result[0]
            summary["final_tick_signature"] = final_result[1]
            tick_res = final_result[2]
            if isinstance(tick_res, list) and len(tick_res) >= 6:
                summary["final_metrics"] = {
                    "total_energy": tick_res[1],
                    "job_queue_len": tick_res[2],
                    "stop_and_go": tick_res[3],
                    "total_turning": tick_res[4],
                    "warehouse_orders_completed": None,
                    "warehouse_orders_completed_available": False,
                    "warehouse_average_order_cycle_time": None,
                    "warehouse_average_order_cycle_time_available": False,
                }

        finalization = session.finalize(
            reason=netlogo.SimulationTermination.MAXIMUM_HORIZON,
            success=True,
        )
        summary["finalization"] = finalization
        final_warehouse = session.warehouse
        final_warehouse_time = 0.0
        if final_warehouse is not None:
            final_warehouse_time = getattr(final_warehouse, "_tick", 0.0)
            tick_to_second = getattr(final_warehouse, "tick_to_second", tick_to_second)

        summary["warehouse_time_end"] = float(final_warehouse_time)
        summary["warehouse_time_elapsed"] = summary["warehouse_time_end"] - summary["warehouse_time_start"]
        summary["worker_wall_time_elapsed"] = time.perf_counter() - worker_start

        # ── Derive authoritative order metrics from OrderManager ──
        if final_warehouse is not None and hasattr(final_warehouse, "order_manager"):
            om = final_warehouse.order_manager
            completed_orders = [
                o for o in getattr(om, "orders", [])
                if o.is_order_completed()
            ]
            orders_completed = len(completed_orders)
            if orders_completed > 0:
                cycle_times = [
                    o.order_complete_time - o.process_start_time
                    for o in completed_orders
                    if o.order_complete_time >= 0 and o.process_start_time >= 0
                ]
                avg_cycle_time = (
                    sum(cycle_times) / len(cycle_times) if cycle_times else 0.0
                )
            else:
                avg_cycle_time = 0.0

            if summary.get("final_metrics") is None:
                summary["final_metrics"] = {}
            summary["final_metrics"]["warehouse_orders_completed"] = orders_completed
            summary["final_metrics"]["warehouse_orders_completed_available"] = True
            summary["final_metrics"]["warehouse_average_order_cycle_time"] = avg_cycle_time
            summary["final_metrics"]["warehouse_average_order_cycle_time_available"] = True

        if spec.kpi_schema_version == SENSITIVITY_KPI_SCHEMA_VERSION:
            kpi_payload = derive_sensitivity_kpi_payload(
                spec,
                final_warehouse,
                summary.get("finalization", {}),
                summary.get("generated_order_contract", {}),
            )
            summary["kpi"] = kpi_payload
            summary.update(kpi_payload)
            if not kpi_payload.get("kpi_complete"):
                missing = [
                    field
                    for field in SENSITIVITY_KPI_FIELDS
                    if field != "kpi_complete" and kpi_payload.get(field) is None
                ]
                raise RuntimeError(f"sensitivity KPI schema incomplete: missing {missing}")

        summary["status"] = "success"

        if spec.debug_trace and debug_rows:
            trace_path = spec.runtime_root / "debug_trace.jsonl"
            with trace_path.open("w") as fh:
                for row in debug_rows:
                    fh.write(json.dumps(row) + "\n")

        return 0
    except Exception as exc:
        finalization_error = None
        if session is not None and not getattr(session, "finalized", False):
            try:
                finalization = session.finalize(
                    reason=getattr(netlogo_module, "SimulationTermination").WORKER_EXCEPTION,
                    success=False,
                )
                summary["finalization"] = finalization
            except Exception as finalize_exc:
                finalization_error = {
                    "error_type": type(finalize_exc).__name__,
                    "error_message": str(finalize_exc),
                    "traceback": traceback.format_exc(),
                }
        summary.update(
            {
                "status": "failure",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        if finalization_error is not None:
            summary["finalization_failure"] = finalization_error
        return 1
    finally:
        if "previous_env" in locals():
            for key, value in previous_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
        try:
            from src.rmfs.rl.rts.runtime_registry import reset_rts_runtime

            reset_rts_runtime()
        except Exception:
            pass
        try:
            if netlogo_module is not None:
                netlogo_module.reset_run_context()
        except Exception:
            pass
        rts_summary_path = spec.runtime_root / "rts_rollout_summary.json"
        if rts_summary_path.exists():
            summary["rts_rollout_summary_path"] = str(rts_summary_path)
            try:
                with rts_summary_path.open() as fh:
                    summary["rts_rollout_summary"] = json.load(fh)
                summary["rts_rollout_summary_embedded"] = True
            except Exception:
                summary["rts_rollout_summary_embedded"] = False
        try:
            from src.rmfs.rl.rts.static_runtime_index import static_runtime_index_diagnostics

            summary["rts_static_runtime_index_lifecycle"] = static_runtime_index_diagnostics()
        except Exception:
            pass
        if spec.timing:
            timing_summary_path = write_timing_summary()
            if timing_summary_path is not None:
                summary["timing_summary_path"] = str(timing_summary_path)
        summary.update({
            "seed_base": spec.rts_seed_base,
            "derived_seed": spec.rts_random_seed,
            "repo_branch": spec.branch,
            "repo_commit": spec.commit,
            "python_executable": spec.python_executable,
            "experiment_id": spec.experiment_id,
            "scenario_id": spec.scenario_id,
            "artifact_label": spec.artifact_label,
            "batch_id": spec.batch_id,
            "worker_id": spec.worker_id,
            "robot_task_allocator": spec.robot_task_allocator,
            "regret_k": spec.regret_k,
            "task_allocator_scope": spec.task_allocator_scope,
            "committed_next_reservations_enabled": spec.committed_next_reservations_enabled,
            "rts_torch_threads": spec.rts_torch_threads if spec.rts_torch_threads is not None else (1 if spec.rts_policy_mode == "rts_rl_explicit" else None),
            "rts_torch_interop_threads": spec.rts_torch_interop_threads if spec.rts_torch_interop_threads is not None else (1 if spec.rts_policy_mode == "rts_rl_explicit" else None),
            "rts_policy_mode": spec.rts_policy_mode,
            "rts_rollout_enabled": spec.rts_rollout_enabled,
            "rts_state_capture_mode": spec.rts_state_capture_mode,
            "detail_db": spec.detail_db,
            "timing": spec.timing,
            "worker_status_cadence": spec.worker_status_cadence,
            "pps_mode": normalize_pps_mode(spec.pps_mode),
            "pps_model_path": spec.pps_model_path,
            "pps_rl_enabled": normalize_pps_mode(spec.pps_mode) == "ppo",
            "charging_enabled": spec.charging_enabled,
            "charging_config_path": spec.charging_config_path,
            "persist_final_state": bool(spec.persist_final_state),
            "campaign_id": spec.campaign_id,
            "allocation_patch_id": spec.allocation_patch_id,
            "simulation_semantics_id": spec.simulation_semantics_id,
            "machine_id": spec.machine_id,
            "stage_first_requested": spec.stage_first_requested,
            "kpi_schema_version": spec.kpi_schema_version,
            "policy_configuration": spec.policy_configuration,
            "replication": spec.replication,
            "seed": spec.campaign_seed if spec.campaign_seed is not None else spec.rts_random_seed,
            "rts_checkpoint_id": _sensitivity_rts_checkpoint_id(spec),
            "rts_checkpoint_sha256": spec.rts_checkpoint_sha256,
            "pps_model_sha256": spec.pps_model_sha256,
        })
        write_json(spec.runtime_root / "worker_summary.json", summary)
        write_worker_status("success" if summary["status"] == "success" else "failure", force=True)
        os.chdir(original_cwd)


def load_worker_summary(runtime_root: Path):
    path = runtime_root / "worker_summary.json"
    if not path.exists():
        return {"status": "failure", "error_message": "worker_summary.json missing"}
    with path.open() as fh:
        return json.load(fh)


def _read_worker_progress(specs: list[RunSpec]) -> int:
    completed_steps = 0
    for spec in specs:
        status_path = spec.runtime_root / "worker_status.json"
        if not status_path.exists():
            continue
        try:
            with status_path.open() as fh:
                status = json.load(fh)
        except (OSError, json.JSONDecodeError):
            continue
        current = status.get("current_progress_steps", 0)
        try:
            completed_steps += max(0, int(current))
        except (TypeError, ValueError):
            continue
    return completed_steps


def _terminal_worker_steps(spec: RunSpec) -> int:
    summary_path = spec.runtime_root / "worker_summary.json"
    if summary_path.exists():
        try:
            with summary_path.open() as fh:
                summary = json.load(fh)
            return max(0, int(summary.get("netlogo_steps_completed", summary.get("ticks_completed", 0)) or 0))
        except Exception:
            pass
    status_path = spec.runtime_root / "worker_status.json"
    if status_path.exists():
        try:
            with status_path.open() as fh:
                status = json.load(fh)
            return max(0, int(status.get("current_progress_steps", 0) or 0))
        except Exception:
            pass
    return 0


def _read_active_worker_progress(processes: list[dict[str, object]]) -> int:
    return _read_worker_progress([item["spec"] for item in processes])


def _make_controller_progress_bar(enabled: bool, total_steps: int):
    if not enabled:
        return None
    try:
        from tqdm import tqdm
    except ImportError:
        print("warning: tqdm is not installed; controller progress bar disabled", file=sys.stderr)
        return None
    return tqdm(
        total=max(0, int(total_steps)),
        desc="RMFS backend steps",
        unit="step",
        dynamic_ncols=True,
        leave=True,
        smoothing=0,
    )


def _annotate_worker_summary(spec: RunSpec, extra: dict[str, object]) -> None:
    summary_path = spec.runtime_root / "worker_summary.json"
    if not summary_path.exists():
        return
    try:
        with summary_path.open() as fh:
            summary = json.load(fh)
    except Exception:
        return
    summary.update(extra)
    write_json(summary_path, summary)


def run_specs(
    specs: list[RunSpec],
    max_workers: int,
    progress: bool = False,
    on_run_complete=None,
    before_launch=None,
):
    """Run worker specs concurrently.

    on_run_complete(spec, return_code), if given, is invoked as each run
    finishes -- callers can use it to reclaim per-run artifacts immediately
    instead of accumulating them until the batch ends. Not passed by default,
    so training/eval callers keep their rollout data intact.

    before_launch(spec, active_count), if given, can return False to pause
    launches while active workers finish and cleanup frees space.
    """
    if max_workers < 1:
        raise ValueError("max_workers must be >= 1")
    if not specs:
        return []

    for spec in specs:
        spec.runtime_root.mkdir(parents=True, exist_ok=True)
        write_json(spec.runtime_root / "run_spec.json", spec.to_json_dict())
        # Clear stale progress from a prior (crashed/resumed) attempt so the
        # controller rate/ETA starts from this invocation's real progress
        # rather than jumping off a leftover worker_status.json (which showed
        # an absurdly high step/s on resume).
        status_path = spec.runtime_root / "worker_status.json"
        if status_path.exists():
            try:
                status_path.unlink()
            except OSError:
                pass

    processes = []
    completed = []
    total_steps = sum(int(spec.ticks) for spec in specs)
    progress_bar = _make_controller_progress_bar(progress, total_steps)
    progress_last = 0
    # Refresh the display on a coarse cadence (~500 steps/worker) or whenever a
    # worker starts/finishes, so the bar stays readable instead of bouncing.
    display_step_threshold = 500 * max(1, int(max_workers))
    last_display_progress = 0
    last_display_time = time.monotonic()

    cap_bytes = _worker_log_cap_bytes()

    def launch(spec: RunSpec):
        stdout_path = spec.runtime_root / "worker_stdout.log"
        stderr_path = spec.runtime_root / "worker_stderr.log"
        child_env = worker_child_environment(spec)
        proc = subprocess.Popen(
            [
                spec.python_executable or sys.executable,
                "-m",
                "src.rmfs.orchestration.local_executor",
                "worker",
                "--spec",
                str(spec.runtime_root / "run_spec.json"),
            ],
            cwd=spec.repo_root,
            env=child_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        pumps = (
            threading.Thread(target=_pump_bounded, args=(proc.stdout, stdout_path, cap_bytes), daemon=True),
            threading.Thread(target=_pump_bounded, args=(proc.stderr, stderr_path, cap_bytes), daemon=True),
        )
        for pump in pumps:
            pump.start()
        return {"spec": spec, "proc": proc, "pumps": pumps}

    pending = list(specs)
    terminal_completed_steps = 0
    callback_error: RuntimeError | None = None
    try:
        while pending or processes:
            counts_changed = False
            launch_blocked = False
            while pending and len(processes) < max_workers and callback_error is None:
                if before_launch is not None:
                    should_launch = before_launch(pending[0], len(processes))
                    if should_launch is False:
                        launch_blocked = True
                        break
                processes.append(launch(pending.pop(0)))
                counts_changed = True

            still_running = []
            for item in processes:
                peak = max(
                    int(item.get("peak_runtime_directory_bytes", 0) or 0),
                    directory_size_bytes(item["spec"].runtime_root),
                )
                item["peak_runtime_directory_bytes"] = peak
                return_code = item["proc"].poll()
                if return_code is None:
                    still_running.append(item)
                    continue
                for pump in item["pumps"]:
                    pump.join(timeout=10)
                _annotate_worker_summary(
                    item["spec"],
                    {"peak_runtime_directory_bytes": int(item.get("peak_runtime_directory_bytes", 0) or 0)},
                )
                completed.append({
                    "spec": item["spec"],
                    "return_code": return_code,
                    "peak_runtime_directory_bytes": int(item.get("peak_runtime_directory_bytes", 0) or 0),
                })
                counts_changed = True
                if on_run_complete is not None:
                    try:
                        on_run_complete(item["spec"], return_code)
                    except Exception as exc:
                        if callback_error is None:
                            callback_error = RuntimeError(
                                "completion callback failed: "
                                f"run_id={item['spec'].run_id} operation=on_run_complete "
                                f"error={type(exc).__name__}: {exc}"
                            )
                            callback_error.__cause__ = exc
                terminal_completed_steps += _terminal_worker_steps(item["spec"])
            processes = still_running

            if progress_bar is not None:
                current_progress = min(total_steps, terminal_completed_steps + _read_active_worker_progress(processes))
                now = time.monotonic()
                # Coarse refresh: enough new steps, a worker started/finished, or
                # a few seconds elapsed (keeps the elapsed clock alive without
                # bouncing -- smoothing=0 keeps the rate/ETA an accurate average).
                if (
                    current_progress - last_display_progress >= display_step_threshold
                    or counts_changed
                    or now - last_display_time >= 3.0
                ):
                    if current_progress > progress_last:
                        progress_bar.update(current_progress - progress_last)
                        progress_last = current_progress
                    progress_bar.set_postfix(
                        running=len(processes),
                        pending=len(pending),
                        completed=len(completed),
                        refresh=False,
                    )
                    progress_bar.refresh()
                    last_display_progress = current_progress
                    last_display_time = now

            if processes:
                time.sleep(0.5 if progress_bar is not None else 0.25)
            elif pending and not processes and launch_blocked and not counts_changed:
                raise RuntimeError("worker launch blocked before any active worker could free resources")
            elif callback_error is not None and not processes:
                raise callback_error
    finally:
        if progress_bar is not None:
            current_progress = min(total_steps, terminal_completed_steps + _read_active_worker_progress(processes))
            if current_progress > progress_last:
                progress_bar.update(current_progress - progress_last)
            progress_bar.close()
        for item in processes:
            for pump in item.get("pumps", ()):
                pump.join(timeout=5)

    return completed


def run_controller(
    repo_root: Path,
    output_root: Path,
    runs: int,
    ticks: int,
    max_workers: int,
    python_executable: str,
    debug_trace: bool = False,
    trace_cadence: int = 1000,
    trace_first_n: int = 0,
    snapshot_inputs: bool = False,
    rts_policy_mode: str = "current",
    rts_rollout_enabled: bool = False,
    rts_zone_ids: list[str] | None = None,
    rts_reward_reference_path: str | None = None,
    rts_random_seed: int | None = None,
    rts_max_events: int | None = None,
    rts_policy_checkpoint_dir: str | None = None,
    rts_policy_checkpoint_id: str | None = None,
    rts_policy_action_mode: str = "sample",
    rts_policy_device: str = "cpu",
    rts_state_capture_mode: str = "auto",
    keep_runtime_artifacts: bool = False,
    persist_final_state: bool = False,
    detail_db: bool | None = None,
    timing: bool = False,
    worker_status_cadence: int = 100,
    profile: str = "smoke",
    bootstrap_n_orders: int | None = None,
    demand_horizon_ticks: int | None = None,
    demand_buffer_ticks: int | None = None,
    order_rate_per_hour: int | None = None,
    pod_location_mode: str | None = None,
    pod_location_seed: int | None = None,
    full_raw_order_replay: bool = False,
    rts_torch_threads: int | None = None,
    rts_torch_interop_threads: int | None = None,
    progress: bool = False,
):
    output_root.mkdir(parents=True, exist_ok=True)
    profile_cfg = resolve_run_profile(
        profile,
        run_horizon_ticks=ticks,
        bootstrap_n_orders=bootstrap_n_orders,
        demand_horizon_ticks=demand_horizon_ticks,
        demand_buffer_ticks=demand_buffer_ticks,
        order_rate_per_hour=order_rate_per_hour,
        full_raw_order_replay=full_raw_order_replay,
        detail_db=detail_db,
        debug_trace=debug_trace or None,
        keep_runtime_artifacts=keep_runtime_artifacts or None,
        pod_location_mode=pod_location_mode,
        pod_location_seed=pod_location_seed,
        seed=rts_random_seed,
    )
    ticks = int(profile_cfg.run_horizon_ticks or 0)
    debug_trace = bool(debug_trace or profile_cfg.debug_trace)
    keep_runtime_artifacts = bool(keep_runtime_artifacts or profile_cfg.keep_runtime_artifacts)
    detail_db = bool(profile_cfg.detail_db)
    branch = git_value(repo_root, "rev-parse", "--abbrev-ref", "HEAD")
    commit = git_value(repo_root, "rev-parse", "HEAD")
    timestamp = datetime.datetime.now().isoformat()
    input_snapshot_root = output_root / "input_snapshot" if snapshot_inputs else None
    input_manifest_path = None
    input_manifest = None
    if snapshot_inputs:
        input_manifest_path, input_manifest = create_input_snapshot(repo_root, input_snapshot_root)

    if snapshot_inputs:
        deferred_root_read_only_inputs = []
        snapshot_copied_inputs = [r["repo_path"] for r in input_manifest.get("copied_inputs", [])] if input_manifest else []
    else:
        deferred_root_read_only_inputs = DEFERRED_ROOT_READ_ONLY_INPUTS
        snapshot_copied_inputs = []
    committed_next_reservations_enabled = bool(rts_policy_mode in {"rts_rl_explicit", "vrsla_teacher"})

    manifest = {
        "status": "started",
        "runs": runs,
        "ticks": ticks,
        "run_profile": profile_cfg.profile,
        "run_horizon_ticks": profile_cfg.run_horizon_ticks,
        "bootstrap_n_orders": profile_cfg.bootstrap_n_orders,
        "demand_horizon_ticks": profile_cfg.demand_horizon_ticks,
        "demand_buffer_ticks": profile_cfg.demand_buffer_ticks,
        "demand_buffer_simulated_seconds": profile_cfg.demand_buffer_simulated_seconds,
        "order_generation_mode": profile_cfg.order_generation_mode,
        "full_raw_order_replay": profile_cfg.full_raw_order_replay,
        "order_rate_per_hour": profile_cfg.order_rate_per_hour,
        "order_rate_unit": "orders_per_simulated_hour" if profile_cfg.order_rate_per_hour is not None else None,
        "tick_to_second": profile_cfg.tick_to_second,
        "simulated_horizon_seconds": profile_cfg.simulated_horizon_seconds,
        "demand_horizon_simulated_seconds": profile_cfg.demand_horizon_simulated_seconds,
        "pod_location_mode": profile_cfg.pod_location_mode,
        "pod_location_seed": profile_cfg.pod_location_seed,
        "max_workers": max_workers,
        "repo_root": str(repo_root),
        "output_root": str(output_root),
        "branch": branch,
        "commit": commit,
        "python_executable": python_executable,
        "timestamp": timestamp,
        "deferred_root_read_only_inputs": deferred_root_read_only_inputs,
        "snapshot_copied_inputs": snapshot_copied_inputs,
        "debug_trace": debug_trace,
        "trace_cadence": trace_cadence,
        "trace_first_n": trace_first_n,
        "snapshot_inputs": snapshot_inputs,
        "input_snapshot_root": str(input_snapshot_root) if input_snapshot_root is not None else None,
        "input_manifest_path": str(input_manifest_path) if input_manifest_path is not None else None,
        "rts_policy_mode": rts_policy_mode,
        "rts_rollout_enabled": rts_rollout_enabled,
        "rts_zone_ids": rts_zone_ids,
        "rts_reward_reference_path": rts_reward_reference_path,
        "rts_random_seed": rts_random_seed,
        "rts_max_events": rts_max_events,
        "rts_policy_checkpoint_dir": rts_policy_checkpoint_dir,
        "rts_policy_checkpoint_id": rts_policy_checkpoint_id,
        "rts_policy_action_mode": rts_policy_action_mode,
        "rts_policy_device": rts_policy_device,
        "rts_state_capture_mode": rts_state_capture_mode,
        "committed_next_reservations_enabled": committed_next_reservations_enabled,
        "keep_runtime_artifacts": keep_runtime_artifacts,
        "persist_final_state": bool(persist_final_state),
        "detail_db": detail_db,
        "timing": timing,
        "worker_status_cadence": worker_status_cadence,
        "artifact_policy": {
            "debug_trace": "enabled" if debug_trace else "disabled",
            "successful_workers": "preserved" if keep_runtime_artifacts else "cleanup_eligible",
            "failed_workers": "preserved",
            "persist_final_state": bool(persist_final_state),
            "detail_db": "enabled" if detail_db else "disabled",
            "worker_status_cadence": worker_status_cadence,
        },
    }
    write_json(output_root / "manifest.json", manifest)
    policy_config = None
    if rts_policy_mode != "current" or rts_rollout_enabled:
        policy_config = {
            "poa": "future_aware",
            "pps": "station_match",
            "rts": rts_policy_mode,
            "charging": "disabled",
            "rts_rollout_enabled": rts_rollout_enabled,
            "rts_policy_checkpoint_id": rts_policy_checkpoint_id,
            "rts_policy_action_mode": rts_policy_action_mode,
            "rts_state_capture_mode": rts_state_capture_mode,
        }

    write_run_manifest(
        output_root / "run_manifest.json",
        created_at=timestamp,
        branch=branch,
        commit=commit,
        python_executable=python_executable,
        repo_root=repo_root,
        output_root=output_root,
        input_snapshot_root=input_snapshot_root,
        input_manifest_path=input_manifest_path,
        input_manifest=input_manifest,
        runs=runs,
        ticks=ticks,
        max_workers=max_workers,
        debug_trace=debug_trace,
        trace_cadence=trace_cadence,
        trace_first_n=trace_first_n,
        root_sensitive_files=ROOT_SENSITIVE_FILES,
        policy_config=policy_config,
    )

    specs = []
    for index in range(runs):
        run_id = f"run_{index + 1:03d}"
        spec = RunSpec(
            run_id=run_id,
            ticks=ticks,
            runtime_root=output_root / run_id,
            repo_root=repo_root,
            input_root=input_snapshot_root,
            branch=branch,
            commit=commit,
            python_executable=python_executable,
            timestamp=timestamp,
            debug_trace=debug_trace,
            trace_cadence=trace_cadence,
            trace_first_n=trace_first_n,
            rts_policy_mode=rts_policy_mode,
            rts_rollout_enabled=rts_rollout_enabled,
            rts_zone_ids=rts_zone_ids,
            rts_reward_reference_path=rts_reward_reference_path,
            rts_random_seed=rts_random_seed,
            rts_max_events=rts_max_events,
            rts_policy_checkpoint_dir=rts_policy_checkpoint_dir,
            rts_policy_checkpoint_id=rts_policy_checkpoint_id,
            rts_policy_action_mode=rts_policy_action_mode,
            rts_policy_device=rts_policy_device,
            rts_state_capture_mode=rts_state_capture_mode,
            committed_next_reservations_enabled=committed_next_reservations_enabled,
            persist_final_state=bool(persist_final_state),
            keep_runtime_artifacts=keep_runtime_artifacts,
            detail_db=detail_db,
            timing=timing,
            worker_status_cadence=worker_status_cadence,
            run_profile=profile_cfg.profile,
            run_horizon_ticks=profile_cfg.run_horizon_ticks,
            bootstrap_n_orders=profile_cfg.bootstrap_n_orders,
            demand_horizon_ticks=profile_cfg.demand_horizon_ticks,
            demand_buffer_ticks=profile_cfg.demand_buffer_ticks,
            order_generation_mode=profile_cfg.order_generation_mode,
            full_raw_order_replay=profile_cfg.full_raw_order_replay,
            order_rate_per_hour=profile_cfg.order_rate_per_hour,
            pod_location_mode=profile_cfg.pod_location_mode,
            pod_location_seed=profile_cfg.pod_location_seed if profile_cfg.pod_location_seed is not None else index,
            rts_torch_threads=rts_torch_threads,
            rts_torch_interop_threads=rts_torch_interop_threads,
        )
        specs.append(spec)
        write_json(spec.runtime_root / "run_spec.json", spec.to_json_dict())

    root_before = snapshot_root(repo_root)
    snapshot_before = hash_snapshot_files(input_snapshot_root) if snapshot_inputs else {}
    completed = run_specs(specs, max_workers=max_workers, progress=progress)

    root_after = snapshot_root(repo_root)
    root_changed = changed_root_files(root_before, root_after)

    snapshot_after = hash_snapshot_files(input_snapshot_root) if snapshot_inputs else {}
    snapshot_changed = []
    if snapshot_inputs:
        for rel_path, digest_before in snapshot_before.items():
            digest_after = snapshot_after.get(rel_path)
            if digest_after is None or digest_before != digest_after:
                snapshot_changed.append(rel_path)
        for rel_path in snapshot_after:
            if rel_path not in snapshot_before:
                snapshot_changed.append(rel_path)

    worker_results = []
    collisions = []
    for item in completed:
        spec = item["spec"]
        worker_summary = load_worker_summary(spec.runtime_root)
        expected_files = {
            name: file_digest(spec.runtime_root / name)
            for name in expected_worker_files(
                detail_db=spec.detail_db,
                persist_final_state=spec.persist_final_state,
            )
        }
        missing = [name for name, info in expected_files.items() if not info["exists"]]
        if missing:
            worker_summary["status"] = "failure"
            worker_summary["missing_runtime_files"] = missing

        runtime_files = {
            "state_file": str(spec.runtime_root / "netlogo.state"),
            "sqlite_db": str(spec.runtime_root / "warehouse.db"),
            "assign_order_csv": str(spec.runtime_root / "assign_order.csv"),
            "pod_info_csv": str(spec.runtime_root / "pod_info.csv"),
        }
        worker_results.append(
            {
                "run_id": spec.run_id,
                "return_code": item["return_code"],
                "summary": worker_summary,
                "expected_runtime_files": expected_files,
                "runtime_files": runtime_files,
            }
        )

    for field in ("state_file", "sqlite_db", "assign_order_csv", "pod_info_csv"):
        seen = {}
        for result in worker_results:
            path = result["runtime_files"][field]
            if path in seen:
                collisions.append({"field": field, "path": path, "runs": [seen[path], result["run_id"]]})
            seen[path] = result["run_id"]

    failed_workers = [
        result["run_id"]
        for result in worker_results
        if result["return_code"] != 0 or result["summary"].get("status") != "success"
    ]
    reclaimed_workers = []
    reclaimed_run_artifact_bytes = 0
    if not keep_runtime_artifacts:
        for result in worker_results:
            if result["return_code"] == 0 and result["summary"].get("status") == "success":
                run_root = Path(result["runtime_files"]["state_file"]).parent
                freed = reclaim_completed_run_artifacts(run_root)
                if freed > 0:
                    reclaimed_run_artifact_bytes += freed
                    reclaimed_workers.append(result["run_id"])

    controller_status = "success"
    failure_reasons = []
    if root_changed:
        controller_status = "failure"
        failure_reasons.append("root_sensitive_files_changed")
    if snapshot_changed:
        controller_status = "failure"
        failure_reasons.append("snapshot_files_changed")
    if failed_workers:
        controller_status = "failure"
        failure_reasons.append("worker_failures")
    if collisions:
        controller_status = "failure"
        failure_reasons.append("runtime_path_collisions")

    workers_succeeded = sum(
        1
        for r in worker_results
        if r["return_code"] == 0 and r["summary"].get("status") == "success"
    )
    workers_failed = len(worker_results) - workers_succeeded

    compact_worker_list = []
    for r in worker_results:
        compact_worker_list.append(
            {
                "run_id": r["run_id"],
                "return_code": r["return_code"],
                "status": r["summary"].get("status"),
                "ticks_completed": r["summary"].get("ticks_completed", 0),
                "final_metrics": r["summary"].get("final_metrics"),
            }
        )

    summary = {
        "status": controller_status,
        "failure_reasons": failure_reasons,
        "runs_requested": runs,
        "ticks": ticks,
        "run_profile": profile_cfg.profile,
        "run_horizon_ticks": profile_cfg.run_horizon_ticks,
        "bootstrap_n_orders": profile_cfg.bootstrap_n_orders,
        "demand_horizon_ticks": profile_cfg.demand_horizon_ticks,
        "demand_buffer_ticks": profile_cfg.demand_buffer_ticks,
        "demand_buffer_simulated_seconds": profile_cfg.demand_buffer_simulated_seconds,
        "order_generation_mode": profile_cfg.order_generation_mode,
        "full_raw_order_replay": profile_cfg.full_raw_order_replay,
        "order_rate_per_hour": profile_cfg.order_rate_per_hour,
        "order_rate_unit": "orders_per_simulated_hour" if profile_cfg.order_rate_per_hour is not None else None,
        "tick_to_second": profile_cfg.tick_to_second,
        "simulated_horizon_seconds": profile_cfg.simulated_horizon_seconds,
        "demand_horizon_simulated_seconds": profile_cfg.demand_horizon_simulated_seconds,
        "pod_location_mode": profile_cfg.pod_location_mode,
        "pod_location_seed": profile_cfg.pod_location_seed,
        "max_workers": max_workers,
        "workers_succeeded": workers_succeeded,
        "workers_failed": workers_failed,
        "output_root": str(output_root),
        "branch": branch,
        "commit": commit,
        "root_changed_files": root_changed,
        "snapshot_changed_files": snapshot_changed,
        "deferred_root_read_only_inputs": deferred_root_read_only_inputs,
        "snapshot_copied_inputs": snapshot_copied_inputs,
        "snapshot_inputs": snapshot_inputs,
        "input_snapshot_root": str(input_snapshot_root) if input_snapshot_root is not None else None,
        "input_manifest_path": str(input_manifest_path) if input_manifest_path is not None else None,
        "debug_trace_enabled": debug_trace,
        "trace_cadence": trace_cadence,
        "trace_first_n": trace_first_n,
        "worker_statuses": compact_worker_list,
        "runtime_path_collisions": collisions,
        "rts_policy_mode": rts_policy_mode,
        "rts_rollout_enabled": rts_rollout_enabled,
        "rts_zone_ids": rts_zone_ids,
        "rts_policy_checkpoint_dir": rts_policy_checkpoint_dir,
        "rts_policy_checkpoint_id": rts_policy_checkpoint_id,
        "rts_policy_action_mode": rts_policy_action_mode,
        "rts_policy_device": rts_policy_device,
        "rts_state_capture_mode": rts_state_capture_mode,
        "keep_runtime_artifacts": keep_runtime_artifacts,
        "persist_final_state": bool(persist_final_state),
        "detail_db": detail_db,
        "timing": timing,
        "worker_status_cadence": worker_status_cadence,
        "reclaimed_workers": reclaimed_workers,
        "reclaimed_run_artifact_bytes": reclaimed_run_artifact_bytes,
    }
    write_json(output_root / "controller_summary.json", summary)

    if controller_status != "success":
        raise RuntimeError(f"local executor smoke failed: {failure_reasons}")

    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description="Local isolated RMFS executor smoke.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    worker_parser = subparsers.add_parser("worker")
    worker_parser.add_argument("--spec", required=True)

    controller_parser = subparsers.add_parser("controller")
    controller_parser.add_argument("--runs", type=int, default=4, help="Number of worker runs to execute.")
    controller_parser.add_argument("--profile", choices=available_profiles(), default="smoke")
    controller_parser.add_argument("--ticks", type=int, default=None, help="Number of NetLogo steps per worker run. Defaults to profile horizon.")
    controller_parser.add_argument("--max-workers", type=int, default=2)
    controller_parser.add_argument("--output-root", required=True)
    controller_parser.add_argument("--repo-root", default=None)
    controller_parser.add_argument("--debug-trace", action="store_true", default=False)
    controller_parser.add_argument("--keep-runtime-artifacts", action="store_true", default=False)
    controller_parser.add_argument("--persist-final-state", action="store_true", default=False)
    controller_parser.add_argument("--detail-db", action="store_true", default=None, help="Enable detail SQLite DB writes in workers. Disabled by default for headless runs.")
    controller_parser.add_argument("--timing", action="store_true", default=False, help="Collect compact worker timing summaries.")
    controller_parser.add_argument("--progress", action="store_true", default=False, help="Show a tqdm progress bar for completed backend steps.")
    controller_parser.add_argument("--worker-status-cadence", type=int, default=100, help="Write worker_status.json every N ticks, plus start/final status.")
    controller_parser.add_argument("--bootstrap-n-orders", type=int, default=None)
    controller_parser.add_argument("--demand-horizon-ticks", type=int, default=None)
    controller_parser.add_argument("--demand-buffer-ticks", type=int, default=None)
    controller_parser.add_argument("--order-rate-per-hour", type=int, default=None, help="Orders per simulated hour for shuffled historical-cycle demand.")
    controller_parser.add_argument("--pod-location-mode", choices=("fixed", "randomize_slots"), default=None)
    controller_parser.add_argument("--pod-location-seed", type=int, default=None)
    controller_parser.add_argument("--full-raw-order-replay", action="store_true", default=False)
    controller_parser.add_argument("--trace-cadence", type=int, default=1000)
    controller_parser.add_argument("--trace-first-n", type=int, default=0)
    controller_parser.add_argument("--snapshot-inputs", action="store_true", default=False)
    controller_parser.add_argument("--rts-policy-mode", choices=("current", "current_probe", "random_valid", "rts_rl_explicit", "vrsla_teacher"), default="current")
    controller_parser.add_argument("--rts-rollout", action="store_true", default=False)
    controller_parser.add_argument("--rts-zone-ids", default=None)
    controller_parser.add_argument("--rts-reward-reference", default=None)
    controller_parser.add_argument("--rts-random-seed", type=int, default=None)
    controller_parser.add_argument("--rts-max-events", type=int, default=None)
    controller_parser.add_argument("--rts-policy-checkpoint-dir", default=None)
    controller_parser.add_argument("--rts-policy-checkpoint-id", default=None)
    controller_parser.add_argument("--rts-policy-action-mode", choices=("sample", "greedy"), default="sample")
    controller_parser.add_argument("--rts-policy-device", choices=("cpu", "cuda", "auto"), default="cpu")
    controller_parser.add_argument("--rts-state-capture-mode", choices=("auto", "full", "minimal"), default="auto")
    controller_parser.add_argument("--rts-torch-threads", type=int, default=None)
    controller_parser.add_argument("--rts-torch-interop-threads", type=int, default=None)

    args = parser.parse_args(argv)

    if args.command == "worker":
        with Path(args.spec).open() as fh:
            spec = RunSpec.from_json_dict(json.load(fh))
        return run_worker(spec)

    if args.runs < 1:
        parser.error("--runs must be >= 1")
    if args.ticks is not None and args.ticks < 0:
        parser.error("--ticks must be >= 0")
    if args.max_workers < 1:
        parser.error("--max-workers must be >= 1")
    if args.trace_cadence < 0:
        parser.error("--trace-cadence must be >= 0")
    if args.trace_first_n < 0:
        parser.error("--trace-first-n must be >= 0")
    if args.worker_status_cadence < 1:
        parser.error("--worker-status-cadence must be >= 1")
    if args.rts_max_events is not None and args.rts_max_events <= 0:
        parser.error("--rts-max-events must be positive")
    rts_zone_ids = None
    if getattr(args, "rts_zone_ids", None):
        rts_zone_ids = [zone.strip() for zone in args.rts_zone_ids.split(",") if zone.strip()]

    repo_root = Path(args.repo_root).resolve() if args.repo_root else Path.cwd().resolve()
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = repo_root / output_root

    run_controller(
        repo_root=repo_root,
        output_root=output_root.resolve(),
        runs=args.runs,
        ticks=args.ticks,
        max_workers=args.max_workers,
        python_executable=sys.executable,
        debug_trace=args.debug_trace,
        trace_cadence=args.trace_cadence,
        trace_first_n=args.trace_first_n,
        snapshot_inputs=args.snapshot_inputs,
        rts_policy_mode=args.rts_policy_mode,
        rts_rollout_enabled=args.rts_rollout,
        rts_zone_ids=rts_zone_ids,
        rts_reward_reference_path=args.rts_reward_reference,
        rts_random_seed=args.rts_random_seed,
        rts_max_events=args.rts_max_events,
        rts_policy_checkpoint_dir=args.rts_policy_checkpoint_dir,
        rts_policy_checkpoint_id=args.rts_policy_checkpoint_id,
        rts_policy_action_mode=args.rts_policy_action_mode,
        rts_policy_device=args.rts_policy_device,
        rts_state_capture_mode=args.rts_state_capture_mode,
        keep_runtime_artifacts=args.keep_runtime_artifacts,
        persist_final_state=args.persist_final_state,
        detail_db=args.detail_db,
        timing=args.timing,
        progress=args.progress,
        worker_status_cadence=args.worker_status_cadence,
        profile=args.profile,
        bootstrap_n_orders=args.bootstrap_n_orders,
        demand_horizon_ticks=args.demand_horizon_ticks,
        demand_buffer_ticks=args.demand_buffer_ticks,
        order_rate_per_hour=args.order_rate_per_hour,
        pod_location_mode=args.pod_location_mode,
        pod_location_seed=args.pod_location_seed,
        full_raw_order_replay=args.full_raw_order_replay,
        rts_torch_threads=args.rts_torch_threads,
        rts_torch_interop_threads=args.rts_torch_interop_threads,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
