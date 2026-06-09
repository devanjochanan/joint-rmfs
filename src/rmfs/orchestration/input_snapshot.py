"""Input snapshot utilities for local RMFS executor runs."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class InputFileSpec:
    logical_name: str
    repo_path: str
    classification: str
    owner_boundary: str
    mode: str
    reason: str


INPUT_FILE_SPECS = [
    InputFileSpec("generated_order.csv", "generated_order.csv", "generated input", "Lukman boundary", "copied", "active generated order stream"),
    InputFileSpec("generated_pod.csv", "generated_pod.csv", "generated input", "Devan/shared layout boundary", "copied", "active generated layout grid"),
    InputFileSpec("pods.csv", "pods.csv", "generated input", "Devan/Lukman boundary", "copied", "active pod-SKU allocation input"),
    InputFileSpec("items.csv", "items.csv", "generated input", "Lukman/Devan boundary", "copied", "active item catalog input"),
    InputFileSpec("generated_backlog.csv", "generated_backlog.csv", "generated input", "Lukman boundary", "copied", "active generated backlog input"),
    InputFileSpec("generated_database_order.csv", "generated_database_order.csv", "generated input", "Lukman boundary", "copied", "active generated order database input"),
    InputFileSpec("items_dictionary.csv", "items_dictionary.csv", "canonical input", "Lukman/shared boundary", "hash_only", "optional canonical dictionary; hash without copying for Phase 4B"),
    InputFileSpec("items_slots_configuration.csv", "items_slots_configuration.csv", "canonical input", "Lukman/Devan boundary", "hash_only", "optional slot configuration; hash without copying for Phase 4B"),
    InputFileSpec("pods_dictionary.csv", "pods_dictionary.csv", "canonical input", "Devan/shared boundary", "hash_only", "optional pod dictionary; hash without copying for Phase 4B"),
    InputFileSpec("raw_order.csv", "raw_order.csv", "legacy input", "Lukman boundary", "excluded", "legacy-only reader via stock_out_probability.py; not active executor/setup path"),
]


def sha256_file(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(repo_root: Path, snapshot_root: Path, spec: InputFileSpec):
    source_path = repo_root / spec.repo_path
    snapshot_path = snapshot_root / spec.repo_path if spec.mode == "copied" else None
    exists = source_path.exists()
    return {
        "logical_name": spec.logical_name,
        "repo_path": spec.repo_path,
        "source_path": str(source_path),
        "snapshot_path": str(snapshot_path) if snapshot_path is not None else None,
        "exists": exists,
        "size_bytes": source_path.stat().st_size if exists and source_path.is_file() else None,
        "sha256": sha256_file(source_path) if exists and source_path.is_file() else None,
        "classification": spec.classification,
        "owner_boundary": spec.owner_boundary,
        "mode": spec.mode,
        "reason": spec.reason,
    }


def create_input_snapshot(repo_root: Path, snapshot_root: Path):
    snapshot_root.mkdir(parents=True, exist_ok=True)
    records = []
    missing_required = []

    for spec in INPUT_FILE_SPECS:
        record = file_record(repo_root, snapshot_root, spec)
        if spec.mode == "copied":
            if not record["exists"]:
                missing_required.append(spec.repo_path)
            else:
                target = snapshot_root / spec.repo_path
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(repo_root / spec.repo_path, target)
                record["snapshot_path"] = str(target)
        records.append(record)

    manifest = {
        "schema_version": "1.0",
        "snapshot_root": str(snapshot_root),
        "inputs": records,
        "copied_inputs": [r for r in records if r["mode"] == "copied"],
        "hash_only_inputs": [r for r in records if r["mode"] == "hash_only"],
        "deferred_inputs": [r for r in records if r["mode"] == "deferred"],
        "excluded_inputs": [r for r in records if r["mode"] == "excluded"],
        "missing_required_inputs": missing_required,
    }

    manifest_path = snapshot_root / "input_manifest.json"
    with manifest_path.open("w") as fh:
        json.dump(manifest, fh, indent=2)

    if missing_required:
        raise FileNotFoundError(f"missing required snapshot inputs: {missing_required}")

    return manifest_path, manifest
