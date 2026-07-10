#!/usr/bin/env python3
"""Create a Git-free, KPI-only export for one autonomous sensitivity host."""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import hashlib
import json
import os
import platform
import sys
import zipfile
from dataclasses import asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.rmfs.orchestration.host_ledger import HostLedger
from src.rmfs.orchestration.source_identity import SourceIdentity


def _rows(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as fh:
        return {str(row.get("run_id", "")): row for row in csv.DictReader(fh) if row.get("run_id")}


def _validate_terminal_rows(ledger: HostLedger, outcomes: dict[str, dict[str, str]], failures: dict[str, dict[str, str]]) -> None:
    for condition in ledger.assigned_conditions:
        key = str(condition.get("condition_key", ""))
        state = ledger.state_for(key)
        status = state.get("status")
        run_id = str(condition.get("run_id", ""))
        if status in {"completed_strict", "completed_with_warnings"}:
            row = outcomes.get(run_id)
            expected = ledger.local_result_index.get(key, {}).get("row_sha256")
            if not row or not expected or row.get("row_sha256") != expected:
                raise RuntimeError(f"completed condition lacks matching KPI row: {key}")
        elif status in {"failed_final", "quarantined"}:
            row = failures.get(run_id)
            expected = state.get("terminal_failure_row_sha256")
            if not row or not expected or row.get("row_sha256") != expected:
                raise RuntimeError(f"final failed condition lacks matching failure row: {key}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--host-id", required=True)
    parser.add_argument("--host-data-root")
    parser.add_argument("--output-dir")
    return parser


def export_host(manifest_path: Path, host_id: str, host_data_root: Path, output_dir: Path | None = None) -> Path:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest.get("campaign_id"):
        raise RuntimeError("manifest does not contain campaign_id")
    ledger_path = host_data_root / "host_ledger.json"
    ledger = HostLedger.load(ledger_path)
    if ledger.host_id != host_id or ledger.campaign_id != manifest["campaign_id"]:
        raise RuntimeError("host ledger identity does not match requested host/campaign")

    outcomes_path = host_data_root / "run_outcomes.csv"
    failures_path = host_data_root / "failed_conditions.csv"
    outcomes, failures = _rows(outcomes_path), _rows(failures_path)
    _validate_terminal_rows(ledger, outcomes, failures)

    timestamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    export_name = f"host_export_{host_id}_{timestamp}.zip"
    for condition in ledger.assigned_conditions:
        state = ledger.state_for(str(condition.get("condition_key", "")))
        if state.get("status") in {"completed_strict", "completed_with_warnings", "failed_final", "quarantined"}:
            ledger.mark_exported(str(condition.get("condition_key", "")), export_name)
    ledger.save(ledger_path)

    assignment = {
        "campaign_id": ledger.campaign_id,
        "host_id": ledger.host_id,
        "manifest_sha256": ledger.manifest_sha256,
        "assigned_conditions": ledger.assigned_conditions,
    }
    source = SourceIdentity.compute(REPO_ROOT)
    metadata = {
        "host_id": host_id,
        "campaign_id": ledger.campaign_id,
        "manifest_sha256": ledger.manifest_sha256,
        "kpi_schema_version": ledger.kpi_schema_version or manifest.get("kpi_schema_version"),
        "source_tree_hash": source.source_tree_hash,
        "git_commit": source.git_commit,
        "timestamp": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "system_info": {"platform": platform.platform(), "python_version": platform.python_version(), "cpu_count": os.cpu_count()},
    }
    output_dir = output_dir or host_data_root / "exports"
    output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = output_dir / export_name
    members = {
        "run_outcomes.csv": outcomes_path.read_bytes() if outcomes_path.exists() else b"",
        "failed_conditions.csv": failures_path.read_bytes() if failures_path.exists() else b"",
        "host_assignment.json": json.dumps(assignment, indent=2, sort_keys=True).encode("utf-8"),
        "host_ledger.json": json.dumps(asdict(ledger), indent=2, sort_keys=True, default=str).encode("utf-8"),
        "export_metadata.json": json.dumps(metadata, indent=2, sort_keys=True, default=str).encode("utf-8"),
    }
    hashes = {name: hashlib.sha256(content).hexdigest() for name, content in members.items()}
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, content in members.items():
            zf.writestr(name, content)
        zf.writestr("sha256_manifest.json", json.dumps({"files": hashes}, indent=2, sort_keys=True).encode("utf-8"))
    return archive_path


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest_path = Path(args.manifest).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    root = Path(args.host_data_root).resolve() if args.host_data_root else REPO_ROOT / "data/runtime/distributed_sensitivity" / manifest["campaign_id"] / args.host_id
    try:
        result = export_host(manifest_path, args.host_id, root, Path(args.output_dir).resolve() if args.output_dir else None)
    except Exception as exc:
        print(f"host export failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
