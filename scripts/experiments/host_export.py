#!/usr/bin/env python3
"""Export package creator for autonomous host results.

Packages completed local outcomes, run specs, summaries, interval snapshots,
and host metadata into a single zip file for Git-free ingestion.
"""

from __future__ import annotations

import argparse
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export completed local results for RMFS campaign.")
    parser.add_argument("--manifest", required=True, help="Path to the campaign manifest.json")
    parser.add_argument("--host-id", required=True, help="ID of this host (e.g. codex_local)")
    parser.add_argument("--host-data-root", help="Root directory of local runs/ledger")
    parser.add_argument("--output-dir", help="Directory where the export ZIP will be saved")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    manifest_path = Path(args.manifest).resolve()
    if not manifest_path.exists():
        print(f"Error: Manifest not found at {manifest_path}", file=sys.stderr)
        return 1

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"Error reading manifest: {exc}", file=sys.stderr)
        return 1

    campaign_id = manifest.get("campaign_id")
    if not campaign_id:
        print("Error: manifest does not contain campaign_id", file=sys.stderr)
        return 1

    # Resolve host data root
    if args.host_data_root:
        host_data_root = Path(args.host_data_root).resolve()
    else:
        # Default: repo_root / data/runtime/distributed_sensitivity / campaign_id / host_id
        host_data_root = REPO_ROOT / "data" / "runtime" / "distributed_sensitivity" / campaign_id / args.host_id

    ledger_path = host_data_root / "host_ledger.json"
    if not ledger_path.exists():
        print(f"Error: Ledger not found at {ledger_path}", file=sys.stderr)
        return 1

    try:
        ledger = HostLedger.load(ledger_path)
    except Exception as exc:
        print(f"Error loading ledger: {exc}", file=sys.stderr)
        return 1

    # Verify ledger consistency
    if ledger.host_id != args.host_id:
        print(f"Warning: Ledger host_id ({ledger.host_id}) does not match CLI host-id ({args.host_id})", file=sys.stderr)
    if ledger.campaign_id != campaign_id:
        print(f"Warning: Ledger campaign_id ({ledger.campaign_id}) does not match manifest campaign_id ({campaign_id})", file=sys.stderr)

    # Compute source identity
    print("Computing source tree identity (Git-free)...")
    source_ident = SourceIdentity.compute(REPO_ROOT)
    print(f"Source Tree Hash: {source_ident.source_tree_hash}")

    # Build export metadata
    timestamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    export_filename = f"host_export_{args.host_id}_{timestamp}.zip"

    output_dir = Path(args.output_dir or host_data_root / "exports").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    export_zip_path = output_dir / export_filename

    # Gather manifest assets hashes
    assets = manifest.get("assets", {})
    checkpoint_hashes = {
        "pps_model_sha256": assets.get("pps_model_sha256"),
        "rts_model_sha256": assets.get("rts_model_sha256"),
        "rts_metadata_canonical_sha256": assets.get("rts_metadata_canonical_sha256"),
        "rts_feature_schema_canonical_sha256": assets.get("rts_feature_schema_canonical_sha256"),
        "charging_config_canonical_sha256": assets.get("charging_config_canonical_sha256"),
    }

    metadata = {
        "host_id": args.host_id,
        "campaign_id": campaign_id,
        "manifest_sha256": ledger.manifest_sha256,
        "kpi_schema_version": ledger.kpi_schema_version or manifest.get("kpi_schema_version"),
        "timestamp": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "source_tree_hash": source_ident.source_tree_hash,
        "git_commit": source_ident.git_commit,
        "git_branch": source_ident.git_branch,
        "git_dirty": source_ident.git_dirty,
        "checkpoint_hashes": checkpoint_hashes,
        "completed_count": len(ledger.completed_conditions),
        "failed_count": len(ledger.failed_conditions),
        "retry_counts": ledger.retry_counts,
        "system_info": {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "cpu_count": os.cpu_count(),
        }
    }

    # Create zip file
    print(f"Creating export package: {export_zip_path}")
    runs_dir = host_data_root / "runs"
    outcomes_csv_path = host_data_root / "run_outcomes.csv"

    archive_hashes: dict[str, str] = {}
    def add_bytes(zf, arcname: str, content: bytes) -> None:
        zf.writestr(arcname, content)
        archive_hashes[arcname] = hashlib.sha256(content).hexdigest()

    def add_file(zf, path: Path, arcname: str) -> None:
        content = path.read_bytes()
        add_bytes(zf, arcname, content)

    with zipfile.ZipFile(export_zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        # 1. Add ledger and export metadata
        add_bytes(zf, "host_ledger.json", json.dumps(asdict(ledger), indent=2, default=str).encode("utf-8"))
        add_bytes(zf, "export_metadata.json", json.dumps(metadata, indent=2, default=str).encode("utf-8"))

        # 2. Add run_outcomes.csv if it exists
        if outcomes_csv_path.exists():
            add_file(zf, outcomes_csv_path, "run_outcomes.csv")
            print("Included run_outcomes.csv")

        # 3. Add run artifacts for completed conditions
        added_runs = 0
        for cond in ledger.assigned_conditions:
            run_id = cond.get("run_id")
            key = cond.get("condition_key")
            state = ledger.state_for(str(key)) if key else {}
            if not run_id or state.get("status") not in {"completed_strict", "completed_with_warnings"}:
                continue

            run_dir = runs_dir / run_id
            spec_path = run_dir / "run_spec.json"
            summary_path = run_dir / "worker_summary.json"
            snapshots_path = run_dir / "kpi_snapshots.jsonl"

            # A result counts only when both its immutable spec and terminal
            # summary can actually be transferred.
            if not spec_path.exists() or not summary_path.exists():
                continue
            add_file(zf, spec_path, f"runs/{run_id}/run_spec.json")
            add_file(zf, summary_path, f"runs/{run_id}/worker_summary.json")
            if snapshots_path.exists():
                add_file(zf, snapshots_path, f"runs/{run_id}/kpi_snapshots.jsonl")

            added_runs += 1
            ledger.mark_exported(str(key), export_filename)

        # Preserve all failed attempt summaries in the ledger and attach a
        # file-level manifest for Git-free ingestion.
        add_bytes(zf, "sha256_manifest.json", json.dumps({"files": archive_hashes}, indent=2, sort_keys=True).encode("utf-8"))

    ledger.save(ledger_path)

    print(f"Included run spec & summary for {added_runs} completed runs.")

    print(f"Export complete. File size: {export_zip_path.stat().st_size / 1024:.1f} KB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
