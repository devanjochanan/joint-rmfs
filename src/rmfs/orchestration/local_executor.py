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
            return sorted(sanitize(item) for item in obj)
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
        "runtime_root": str(spec.runtime_root),
        "repo_root": str(spec.repo_root),
        "setup_digest": None,
        "setup_signature": None,
        "tick_digests": [],
        "tick_signatures": [],
    }

    try:
        sys.path.insert(0, str(spec.repo_root))
        os.chdir(spec.repo_root)

        from src.rmfs.runtime_io import RunContext
        import netlogo

        netlogo_module = netlogo
        ctx = RunContext.isolated(spec.runtime_root, repo_root=spec.repo_root)
        ctx.ensure_runtime_dirs()
        netlogo.configure_run_context(ctx)

        setup_result = netlogo.setup()
        if isinstance(setup_result, str) and "An error occurred" in setup_result:
            raise RuntimeError(setup_result)
        summary["setup_digest"] = stable_digest(setup_result)
        summary["setup_signature"] = return_signature(setup_result)

        for index in range(spec.ticks):
            tick_result = netlogo.tick()
            if isinstance(tick_result, str) and "An error occurred" in tick_result:
                raise RuntimeError(tick_result)
            summary["tick_digests"].append(stable_digest(tick_result))
            signature = return_signature(tick_result)
            signature["tick_index"] = index + 1
            summary["tick_signatures"].append(signature)

        summary.update({"status": "success", "ticks_completed": spec.ticks})
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
            if netlogo_module is not None:
                netlogo_module.reset_run_context()
        except Exception:
            pass
        write_json(spec.runtime_root / "worker_summary.json", summary)
        os.chdir(original_cwd)


def load_worker_summary(runtime_root: Path):
    path = runtime_root / "worker_summary.json"
    if not path.exists():
        return {"status": "failure", "error_message": "worker_summary.json missing"}
    with path.open() as fh:
        return json.load(fh)


def run_controller(repo_root: Path, output_root: Path, runs: int, ticks: int, max_workers: int, python_executable: str):
    output_root.mkdir(parents=True, exist_ok=True)
    branch = git_value(repo_root, "rev-parse", "--abbrev-ref", "HEAD")
    commit = git_value(repo_root, "rev-parse", "HEAD")
    timestamp = datetime.datetime.now().isoformat()

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
        "deferred_root_read_only_inputs": DEFERRED_ROOT_READ_ONLY_INPUTS,
    }
    write_json(output_root / "manifest.json", manifest)

    specs = []
    for index in range(runs):
        run_id = f"run_{index + 1:03d}"
        spec = RunSpec(
            run_id=run_id,
            ticks=ticks,
            runtime_root=output_root / run_id,
            repo_root=repo_root,
            branch=branch,
            commit=commit,
            python_executable=python_executable,
            timestamp=timestamp,
        )
        specs.append(spec)
        write_json(spec.runtime_root / "run_spec.json", spec.to_json_dict())

    root_before = snapshot_root(repo_root)
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
    if failed_workers:
        controller_status = "failure"
        failure_reasons.append("worker_failures")
    if collisions:
        controller_status = "failure"
        failure_reasons.append("runtime_path_collisions")

    summary = {
        "status": controller_status,
        "failure_reasons": failure_reasons,
        "runs": runs,
        "ticks": ticks,
        "max_workers": max_workers,
        "output_root": str(output_root),
        "branch": branch,
        "commit": commit,
        "root_changed_files": root_changed,
        "deferred_root_read_only_inputs": DEFERRED_ROOT_READ_ONLY_INPUTS,
        "worker_results": worker_results,
        "runtime_path_collisions": collisions,
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
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
