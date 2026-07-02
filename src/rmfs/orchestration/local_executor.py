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
import time
import traceback

from src.rmfs.orchestration.input_snapshot import create_input_snapshot
from src.rmfs.orchestration.run_manifest import write_run_manifest
from src.rmfs.orchestration.run_spec import RunSpec
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
    "netlogo.state",
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


def expected_worker_files(detail_db: bool = False) -> list[str]:
    files = list(BASE_EXPECTED_WORKER_FILES)
    if detail_db:
        files.insert(1, "warehouse.db")
    return files


def worker_environment_overrides(spec: RunSpec) -> dict[str, str]:
    env = {
        "RMFS_RUN_PROFILE": spec.run_profile,
        "RMFS_ORDER_GENERATION_MODE": spec.order_generation_mode,
        "RMFS_FULL_RAW_ORDER_REPLAY": "1" if spec.full_raw_order_replay else "0",
        "RMFS_DETAIL_DB": "1" if spec.detail_db else "0",
        "RMFS_ROBOT_TASK_ALLOCATOR": spec.robot_task_allocator,
        "RMFS_TASK_ALLOCATOR_SCOPE": spec.task_allocator_scope,
        "RMFS_COMMITTED_NEXT_RESERVATIONS": "1" if spec.committed_next_reservations_enabled else "0",
        "RMFS_POD_LOCATION_MODE": spec.pod_location_mode,
    }
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
    tmp_path.replace(path)


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
    if policy_mode not in {"current", "current_probe", "random_valid", "rts_rl_explicit"} or not bool(getattr(config, "rollout_enabled", False)):
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
            session = netlogo.HeadlessSimulationSession(persist_final_state=True)
            setup_result = session.setup()
        if isinstance(setup_result, str) and "An error occurred" in setup_result:
            raise RuntimeError(setup_result)
        summary["generated_order_contract"] = validate_generated_order_contract(spec)
        summary["setup_digest"] = stable_digest(setup_result)
        summary["setup_signature"] = return_signature(setup_result)

        warehouse = session.warehouse
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
            digest = stable_digest(tick_result)
            sig = return_signature(tick_result)
            
            write_worker_status("running")
            
            if index == 0:
                first_result = (digest, sig)
            final_result = (digest, sig, tick_result)

            if spec.debug_trace:
                record = False
                if spec.trace_first_n > 0 and index < spec.trace_first_n:
                    record = True
                if spec.trace_cadence > 0 and (index + 1) % spec.trace_cadence == 0:
                    record = True
                if index == spec.ticks - 1:
                    record = True

                if record:
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
            "rts_state_capture_mode": spec.rts_state_capture_mode,
            "detail_db": spec.detail_db,
            "timing": spec.timing,
            "worker_status_cadence": spec.worker_status_cadence,
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
    committed_next_reservations_enabled = bool(rts_policy_mode == "rts_rl_explicit")

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
        "detail_db": detail_db,
        "timing": timing,
        "worker_status_cadence": worker_status_cadence,
        "artifact_policy": {
            "debug_trace": "enabled" if debug_trace else "disabled",
            "successful_workers": "preserved" if keep_runtime_artifacts else "cleanup_eligible",
            "failed_workers": "preserved",
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
    processes = []
    completed = []

    def launch(spec: RunSpec):
        stdout_path = spec.runtime_root / "worker_stdout.log"
        stderr_path = spec.runtime_root / "worker_stderr.log"
        stdout_fh = stdout_path.open("w")
        stderr_fh = stderr_path.open("w")
        proc = subprocess.Popen(
            [
                python_executable,
                "-m",
                "src.rmfs.orchestration.local_executor",
                "worker",
                "--spec",
                str(spec.runtime_root / "run_spec.json"),
            ],
            cwd=repo_root,
            stdout=stdout_fh,
            stderr=stderr_fh,
            text=True,
        )
        return {"spec": spec, "proc": proc, "stdout": stdout_fh, "stderr": stderr_fh}

    pending = list(specs)
    while pending or processes:
        while pending and len(processes) < max_workers:
            processes.append(launch(pending.pop(0)))

        still_running = []
        for item in processes:
            return_code = item["proc"].poll()
            if return_code is None:
                still_running.append(item)
                continue
            item["stdout"].close()
            item["stderr"].close()
            completed.append({"spec": item["spec"], "return_code": return_code})
        processes = still_running

        if processes and (pending or len(processes) >= max_workers):
            processes[0]["proc"].wait(timeout=None)

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
            for name in expected_worker_files(detail_db=spec.detail_db)
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
    cleanup_eligible_workers = []
    if not keep_runtime_artifacts:
        for result in worker_results:
            if result["return_code"] == 0 and result["summary"].get("status") == "success":
                cleanup_marker = Path(result["runtime_files"]["state_file"]).parent / ".rmfs_cleanup_eligible"
                cleanup_marker.write_text("successful worker; eligible for cleanup\n", encoding="utf-8")
                cleanup_eligible_workers.append(result["run_id"])

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
        "detail_db": detail_db,
        "timing": timing,
        "worker_status_cadence": worker_status_cadence,
        "cleanup_eligible_workers": cleanup_eligible_workers,
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
    controller_parser.add_argument("--detail-db", action="store_true", default=None, help="Enable detail SQLite DB writes in workers. Disabled by default for headless runs.")
    controller_parser.add_argument("--timing", action="store_true", default=False, help="Collect compact worker timing summaries.")
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
    controller_parser.add_argument("--rts-policy-mode", choices=("current", "current_probe", "random_valid", "rts_rl_explicit"), default="current")
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
        detail_db=args.detail_db,
        timing=args.timing,
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
