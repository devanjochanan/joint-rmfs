#!/usr/bin/env python3
"""Import hash-verified, KPI-only RMFS host exports without Git metadata."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import zipfile
from pathlib import Path
from typing import Any


IDENTITY_FIELDS = (
    "campaign_id", "run_id", "paired_group_id", "policy_configuration", "robot_count",
    "order_rate", "replication", "seed", "source_tree_hash", "layout_hash",
    "charging_placement_source", "charging_config_sha256", "effective_charger_coordinate_hash",
    "rts_checkpoint_sha256", "pps_model_sha256", "kpi_schema_version", "simulation_semantics_id",
)


def _json(zf: zipfile.ZipFile, name: str) -> dict[str, Any]:
    return json.loads(zf.read(name).decode("utf-8"))


def _csv(zf: zipfile.ZipFile, name: str) -> list[dict[str, str]]:
    if name not in zf.namelist():
        return []
    return list(csv.DictReader(zf.read(name).decode("utf-8").splitlines()))


def _validate_hashes(zf: zipfile.ZipFile) -> None:
    for name, expected in _json(zf, "sha256_manifest.json").get("files", {}).items():
        if name not in zf.namelist() or hashlib.sha256(zf.read(name)).hexdigest() != expected:
            raise RuntimeError(f"archive member hash mismatch: {name}")


def _identity(row: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(row.get(field, "") for field in IDENTITY_FIELDS)


def _canonical_row(row: dict[str, Any]) -> str:
    return json.dumps(row, sort_keys=True, separators=(",", ":"), default=str)


def import_exports(archives: list[Path], output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    accepted: dict[tuple[Any, ...], dict[str, str]] = {}
    conflicted: set[tuple[Any, ...]] = set()
    failures: list[dict[str, Any]] = []
    quarantined: list[dict[str, Any]] = []
    for archive in archives:
        with zipfile.ZipFile(archive) as zf:
            _validate_hashes(zf)
            ledger = _json(zf, "host_ledger.json")
            states = ledger.get("condition_states", {})
            for state_key, state in states.items():
                for attempt in state.get("attempts", []):
                    if str(attempt.get("status", "")).startswith(("failed", "quarantined")):
                        failures.append({"archive": archive.name, "condition_key": state_key, **attempt})
            for row in _csv(zf, "failed_conditions.csv"):
                failures.append({"archive": archive.name, "terminal_failure": True, **row})
            for row in _csv(zf, "run_outcomes.csv"):
                if row.get("status") not in {"completed_strict", "completed_with_warnings"}:
                    quarantined.append({"archive": archive.name, "run_id": row.get("run_id"), "reason": "non_completed_outcome_row"})
                    continue
                key = _identity(row)
                if key in conflicted:
                    continue
                previous = accepted.get(key)
                if previous is None:
                    accepted[key] = row
                elif _canonical_row(previous) == _canonical_row(row):
                    continue
                else:
                    quarantined.append({"identity": key, "reason": "conflicting_successful_duplicate", "archives": [previous.get("source_machine_id"), archive.name]})
                    accepted.pop(key, None)
                    conflicted.add(key)
    rows = [accepted[key] for key in sorted(accepted, key=lambda key: tuple(str(v) for v in key))]
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
