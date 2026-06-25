"""Run manifest writer for local RMFS executor runs."""

from __future__ import annotations

import json
import platform
import sys
from pathlib import Path


DEFAULT_POLICY_CONFIG = {
    "poa": "future_aware",
    "pps": "station_match",
    "rts": "nearest",
    "charging": "disabled",
}


def write_run_manifest(
    path: Path,
    *,
    created_at: str,
    branch: str | None,
    commit: str | None,
    python_executable: str,
    repo_root: Path,
    output_root: Path,
    runs: int,
    ticks: int,
    max_workers: int,
    debug_trace: bool,
    trace_cadence: int,
    trace_first_n: int,
    root_sensitive_files: list[str],
    input_snapshot_root: Path | None = None,
    input_manifest_path: Path | None = None,
    input_manifest: dict | None = None,
    policy_config: dict | None = None,
    purpose: str = "local_executor_smoke",
):
    input_manifest = input_manifest or {}
    copied_inputs = input_manifest.get("copied_inputs", [])
    hash_only_inputs = input_manifest.get("hash_only_inputs", [])
    deferred_inputs = input_manifest.get("deferred_inputs", [])
    excluded_inputs = input_manifest.get("excluded_inputs", [])

    manifest = {
        "schema_version": "1.0",
        "created_at": created_at,
        "branch": branch,
        "commit": commit,
        "python_executable": python_executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "repo_root": str(repo_root),
        "output_root": str(output_root),
        "input_snapshot_root": str(input_snapshot_root) if input_snapshot_root is not None else None,
        "input_manifest_path": str(input_manifest_path) if input_manifest_path is not None else None,
        "runs": runs,
        "ticks": ticks,
        "max_workers": max_workers,
        "debug_trace": debug_trace,
        "trace_cadence": trace_cadence,
        "trace_first_n": trace_first_n,
        "summary_mode": "aggregate",
        "purpose": purpose,
        "policy_config": policy_config if policy_config is not None else DEFAULT_POLICY_CONFIG,
        "snapshot_inputs": input_manifest.get("inputs", []),
        "copied_inputs": copied_inputs,
        "hash_only_inputs": hash_only_inputs,
        "deferred_inputs": deferred_inputs,
        "excluded_inputs": excluded_inputs,
        "root_sensitive_files": root_sensitive_files,
        "runtime_output_contract": {
            "worker_summary": "aggregate",
            "debug_trace": "opt_in_jsonl",
            "runtime_files": [
                "netlogo.state",
                "warehouse.db",
                "assign_order.csv",
                "pod_info.csv",
                "skus_data.csv",
                "sorted_skus_data.csv",
            ],
        },
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        json.dump(manifest, fh, indent=2)
    return manifest
