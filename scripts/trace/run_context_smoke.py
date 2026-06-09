#!/usr/bin/env python3
"""Smoke test isolated RMFS runtime paths."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback


ROOT_MUTABLE_FILES = [
    "assign_order.csv",
    "pod_info.csv",
    "netlogo.state",
    "warehouse.db",
    "skus_data.csv",
    "sorted_skus_data.csv",
    "generated_backlog.csv",
    "generated_database_order.csv",
    "generated_order.csv",
    "generated_pod.csv",
]

EXPECTED_RUNTIME_FILES = [
    "netlogo.state",
    "warehouse.db",
    "assign_order.csv",
    "pod_info.csv",
    "skus_data.csv",
    "sorted_skus_data.csv",
]


def file_digest(path: Path):
    if not path.exists():
        return {"exists": False, "size": 0, "sha256": None}

    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            digest.update(chunk)
    return {"exists": True, "size": path.stat().st_size, "sha256": digest.hexdigest()}


def snapshot(repo_root: Path):
    return {name: file_digest(repo_root / name) for name in ROOT_MUTABLE_FILES}


def changed_files(before, after):
    changed = []
    for name in ROOT_MUTABLE_FILES:
        if before[name] != after[name]:
            changed.append(name)
    return changed


def main():
    parser = argparse.ArgumentParser(description="Run a 3-tick isolated RunContext smoke.")
    parser.add_argument("--ticks", type=int, default=3)
    parser.add_argument("--runtime-root", required=True)
    args = parser.parse_args()

    if args.ticks < 0:
        parser.error("Tick count must be non-negative.")

    repo_root = Path(__file__).resolve().parents[2]
    runtime_root = Path(args.runtime_root)
    if not runtime_root.is_absolute():
        runtime_root = repo_root / runtime_root
    runtime_root = runtime_root.resolve()
    runtime_root.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(repo_root))
    original_cwd = Path.cwd()
    os.chdir(repo_root)

    summary = {
        "status": "failure",
        "ticks_requested": args.ticks,
        "runtime_root": str(runtime_root),
        "root_changed_files": [],
        "expected_runtime_files": {},
    }

    netlogo_module = None
    before = snapshot(repo_root)
    try:
        import netlogo
        netlogo_module = netlogo
        from src.rmfs.runtime_io import RunContext

        ctx = RunContext.isolated(runtime_root, repo_root=repo_root)
        netlogo.configure_run_context(ctx)

        setup_result = netlogo.setup()
        if isinstance(setup_result, str) and "An error occurred" in setup_result:
            raise RuntimeError(setup_result)

        for _ in range(args.ticks):
            tick_result = netlogo.tick()
            if isinstance(tick_result, str) and "An error occurred" in tick_result:
                raise RuntimeError(tick_result)

        after = snapshot(repo_root)
        root_changed = changed_files(before, after)
        expected_files = {
            name: file_digest(runtime_root / name)
            for name in EXPECTED_RUNTIME_FILES
        }
        missing = [name for name, info in expected_files.items() if not info["exists"]]

        summary.update(
            {
                "status": "success",
                "ticks_completed": args.ticks,
                "root_changed_files": root_changed,
                "expected_runtime_files": expected_files,
                "missing_runtime_files": missing,
            }
        )

        if root_changed:
            raise RuntimeError(f"Isolated smoke changed root files: {root_changed}")
        if missing:
            raise RuntimeError(f"Isolated smoke missed runtime files: {missing}")

    except Exception as exc:
        summary.update(
            {
                "status": "failure",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        raise
    finally:
        try:
            summary_path = runtime_root / "run_context_smoke_summary.json"
            with summary_path.open("w") as fh:
                json.dump(summary, fh, indent=2)
        finally:
            try:
                if netlogo_module is not None:
                    netlogo_module.reset_run_context()
            except Exception:
                pass
            os.chdir(original_cwd)


if __name__ == "__main__":
    main()
