#!/usr/bin/env python3
"""Git-free importer for autonomous RMFS host export archives."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import zipfile
from pathlib import Path
from typing import Any


def _read_json(zf: zipfile.ZipFile, name: str) -> dict[str, Any]:
    return json.loads(zf.read(name).decode("utf-8"))


def _validate_hashes(zf: zipfile.ZipFile) -> None:
    hashes = _read_json(zf, "sha256_manifest.json").get("files", {})
    for name, expected in hashes.items():
        if name not in zf.namelist():
            raise RuntimeError(f"archive hash manifest references missing member: {name}")
        actual = hashlib.sha256(zf.read(name)).hexdigest()
        if actual != expected:
            raise RuntimeError(f"archive member hash mismatch: {name}")


def _condition_key(spec: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(spec.get(key) for key in (
        "policy_configuration", "rts_policy_mode", "pps_mode", "charging_placement_source",
        "charging_config_sha256", "charging_realized_layout_sha256", "rts_checkpoint_sha256",
        "pps_model_sha256", "campaign_id", "robot_count", "order_rate_per_hour", "replication",
        "campaign_seed",
    )) + (spec.get("source_tree_hash"),)


def import_exports(archives: list[Path], output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outcomes: dict[tuple[Any, ...], dict[str, Any]] = {}
    failures: list[dict[str, Any]] = []
    quarantined: list[dict[str, Any]] = []
    for archive in archives:
        with zipfile.ZipFile(archive) as zf:
            _validate_hashes(zf)
            ledger = _read_json(zf, "host_ledger.json")
            states = ledger.get("condition_states", {})
            for condition in ledger.get("assigned_conditions", []):
                key = str(condition.get("condition_key", ""))
                state = states.get(key, {})
                status = state.get("status")
                if status not in {"completed_strict", "completed_with_warnings"}:
                    for attempt in state.get("attempts", []):
                        if attempt.get("status", "").startswith(("failed", "quarantined")):
                            failures.append({"archive": archive.name, "condition_key": key, **attempt})
                    continue
                run_id = str(condition.get("run_id", ""))
                spec_name = f"runs/{run_id}/run_spec.json"
                summary_name = f"runs/{run_id}/worker_summary.json"
                if spec_name not in zf.namelist() or summary_name not in zf.namelist():
                    quarantined.append({"archive": archive.name, "condition_key": key, "reason": "missing_required_result_member"})
                    continue
                spec, summary = _read_json(zf, spec_name), _read_json(zf, summary_name)
                if summary.get("status") != "success" or not summary.get("finalization", {}).get("finalized", False):
                    quarantined.append({"archive": archive.name, "condition_key": key, "reason": "nonterminal_success"})
                    continue
                scientific_key = _condition_key(spec)
                record = {"archive": archive.name, "source_campaign_id": spec.get("campaign_id"), "run_id": run_id,
                          "condition": condition, "spec": spec, "summary": summary, "status": status}
                prior = outcomes.get(scientific_key)
                if prior is None:
                    outcomes[scientific_key] = record
                elif prior["summary"].get("kpi", prior["summary"]) == record["summary"].get("kpi", record["summary"]):
                    continue
                else:
                    quarantined.append({"condition": scientific_key, "reason": "conflicting_successful_duplicate", "archives": [prior["archive"], archive.name]})
                    outcomes.pop(scientific_key, None)
    rows = []
    for key, record in sorted(outcomes.items(), key=lambda item: tuple(str(v) for v in item[0])):
        row = dict(record["condition"])
        row.update(record["summary"].get("kpi", {}))
        row.update({"source_campaign_id": record["source_campaign_id"], "run_id": record["run_id"], "completion_status": record["status"]})
        rows.append(row)
    fields = sorted({field for row in rows for field in row})
    with (output_dir / "imported_outcomes.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    report = {"accepted": len(rows), "strict_or_warning_results": len(rows), "failures": failures, "quarantined": quarantined}
    (output_dir / "import_report.json").write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archives", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = import_exports(args.archives, args.output_dir)
    except Exception as exc:
        print(f"host export import failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({key: len(value) if isinstance(value, list) else value for key, value in report.items()}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
