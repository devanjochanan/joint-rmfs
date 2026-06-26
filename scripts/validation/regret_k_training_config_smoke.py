#!/usr/bin/env python3
"""Verify regret-k scheduler metadata in RTS training dry-run artifacts."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import sqlite3
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.training.rts_train_controller import main as controller_main
from src.rmfs.experiments.ledger.ingest_phase9 import ingest_phase9_run



EXPECTED = {
    "robot_task_allocator": "regret_k",
    "regret_k": 2,
    "task_allocator_scope": "active_job_queue",
    "committed_next_reservations_enabled": True,
}


def read_json(path: Path):
    with path.open() as fh:
        return json.load(fh)


def assert_scheduler(payload):
    for key, value in EXPECTED.items():
        assert payload.get(key) == value, f"{key}: expected {value!r}, got {payload.get(key)!r}"


def main():
    output_root = REPO_ROOT / "data" / "runtime" / "regret_k_training_config_smoke"
    shutil.rmtree(output_root, ignore_errors=True)
    output_root.mkdir(parents=True, exist_ok=True)

    try:
        controller_main(
            [
                "--artifact-label",
                "regret_k_training_config_smoke",
                "--output-root",
                str(output_root),
                "--batches",
                "1",
                "--workers",
                "2",
                "--netlogo-steps-per-run",
                "3",
                "--seed",
                "42",

                "--no-progress",
                "--no-tensorboard",
                "--dry-run",
            ]
        )

        run_root = output_root / "regret_k_training_config_smoke"
        config = read_json(run_root / "training_config.json")
        controller = read_json(run_root / "controller_summary.json")
        worker_specs = read_json(run_root / "batch_000001" / "worker_specs.json")
        run_spec = read_json(run_root / "batch_000001" / "workers" / "run_001" / "run_spec.json")
        batch_summary = read_json(run_root / "batch_000001" / "batch_summary.json")

        assert_scheduler(config)
        assert_scheduler(controller)
        assert_scheduler(batch_summary)
        assert len(worker_specs) == 2
        assert_scheduler(worker_specs[0])
        assert_scheduler(run_spec)

        db_path = output_root / "ledger.sqlite"
        ingest_phase9_run(run_root, db_path)
        with sqlite3.connect(db_path) as conn:
            row = conn.execute("SELECT config_json FROM experiments").fetchone()
            assert row is not None
            ingested_config = json.loads(row[0])
        assert_scheduler(ingested_config)

        print("regret-k training config smoke ok")
    finally:
        shutil.rmtree(output_root, ignore_errors=True)


if __name__ == "__main__":
    main()
