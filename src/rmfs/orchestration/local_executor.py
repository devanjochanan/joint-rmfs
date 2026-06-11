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
import traceback

from src.rmfs.orchestration.input_snapshot import create_input_snapshot
from src.rmfs.orchestration.run_manifest import write_run_manifest
from src.rmfs.orchestration.run_spec import RunSpec


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

EXPECTED_WORKER_FILES = [
    "netlogo.state",
    "warehouse.db",
    "assign_order.csv",
    "pod_info.csv",
    "skus_data.csv",
    "sorted_skus_data.csv",
    "worker_summary.json",
]

DEFERRED_ROOT_READ_ONLY_INPUTS = [
    "generated_order.csv",
    "generated_pod.csv",
    "pods.csv",
]


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
    with path.open("w") as fh:
        json.dump(payload, fh, indent=2)


def run_worker(spec: RunSpec):
    original_cwd = Path.cwd()
    netlogo_module = None
    summary = {
        "run_id": spec.run_id,
        "status": "failure",
        "ticks_requested": spec.ticks,
        "ticks_completed": 0,
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
        sys.path.insert(0, str(spec.repo_root))
        os.chdir(spec.repo_root)

        from src.rmfs.runtime_io import RunContext
        from src.rmfs.rl.rts.runtime_config import RTSRuntimeConfig
        from src.rmfs.rl.rts.runtime_registry import configure_rts_runtime, reset_rts_runtime
        import netlogo

        netlogo_module = netlogo
        ctx = RunContext.isolated(spec.runtime_root, repo_root=spec.repo_root, input_root=spec.input_root)
        ctx.ensure_runtime_dirs()
        netlogo.configure_run_context(ctx)
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
            ),
            runtime_root=spec.runtime_root,
        )

        setup_result = netlogo.setup()
        if isinstance(setup_result, str) and "An error occurred" in setup_result:
            raise RuntimeError(setup_result)
        summary["setup_digest"] = stable_digest(setup_result)
        summary["setup_signature"] = return_signature(setup_result)

        first_result = None
        final_result = None
        ticks_done = 0

        for index in range(spec.ticks):
            tick_result = netlogo.tick()
            if isinstance(tick_result, str) and "An error occurred" in tick_result:
                raise RuntimeError(tick_result)
            
            ticks_done += 1
            digest = stable_digest(tick_result)
            sig = return_signature(tick_result)
            
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

        summary["status"] = "success"

        if spec.debug_trace and debug_rows:
            trace_path = spec.runtime_root / "debug_trace.jsonl"
            with trace_path.open("w") as fh:
                for row in debug_rows:
                    fh.write(json.dumps(row) + "\n")

        return 0
    except Exception as exc:
        summary.update(
            {
                "status": "failure",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        return 1
    finally:
        try:
            runtime = None
            if netlogo_module is not None:
                universe = getattr(netlogo_module, "universe", None)
                runtime = getattr(universe, "rts_rollout_runtime", None)
            if runtime is not None:
                runtime.close()
        except Exception:
            pass
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
        write_json(spec.runtime_root / "worker_summary.json", summary)
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
):
    output_root.mkdir(parents=True, exist_ok=True)
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

    manifest = {
        "status": "started",
        "runs": runs,
        "ticks": ticks,
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
            for name in EXPECTED_WORKER_FILES
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
    controller_parser.add_argument("--runs", type=int, default=4)
    controller_parser.add_argument("--ticks", type=int, default=3)
    controller_parser.add_argument("--max-workers", type=int, default=2)
    controller_parser.add_argument("--output-root", required=True)
    controller_parser.add_argument("--repo-root", default=None)
    controller_parser.add_argument("--debug-trace", action="store_true", default=False)
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

    args = parser.parse_args(argv)

    if args.command == "worker":
        with Path(args.spec).open() as fh:
            spec = RunSpec.from_json_dict(json.load(fh))
        return run_worker(spec)

    if args.runs < 1:
        parser.error("--runs must be >= 1")
    if args.ticks < 0:
        parser.error("--ticks must be >= 0")
    if args.max_workers < 1:
        parser.error("--max-workers must be >= 1")
    if args.trace_cadence < 0:
        parser.error("--trace-cadence must be >= 0")
    if args.trace_first_n < 0:
        parser.error("--trace-first-n must be >= 0")
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
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
