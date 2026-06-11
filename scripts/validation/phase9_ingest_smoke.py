#!/usr/bin/env python3
"""Smoke test for Phase 9 ingestion into SQLite."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import sqlite3
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.rmfs.experiments.ledger.ingest_phase9 import ingest_phase9_run


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        json.dump(payload, fh)


def main():
    root = REPO_ROOT / "data" / "runtime" / "phase10_ingest_smoke"
    shutil.rmtree(root, ignore_errors=True)

    # Path 1
    run_root = root / "rts_training" / "smoke_1"
    config = {"artifact_label": "smoke", "seed": 42, "netlogo_steps_per_run": 3, "workers": 1, "commit": "abc"}
    write_json(run_root / "training_config.json", config)
    write_json(run_root / "latest.json", {"checkpoint_dir": str(run_root / "batch_000001" / "checkpoint")})
    write_json(run_root / "batch_000001" / "batch_summary.json", {"status": "updated", "trainable_step_count": 2, "latest_updated": True, "dataset_summary": {"avg_reward": 1.0}, "ppo_update": {"total_loss_mean": 0.5}})
    write_json(run_root / "batch_000001" / "checkpoint" / "metadata.json", {"ok": True})
    write_json(run_root / "batch_000001" / "checkpoint" / "feature_schema.json", {"action_feature_dim": 2})

    # Include alias fields, legacy fallback, and warehouse times
    write_json(
        run_root / "batch_000001" / "workers" / "run_001" / "worker_summary.json",
        {
            "status": "success",
            "ticks_completed": 3,
            "netlogo_steps_completed": 5,  # Alias (should win over ticks_completed)
            "netlogo_steps_requested": 4,  # Alias
            "warehouse_time_start": 100.0,
            "warehouse_time_end": 101.2,
            "warehouse_time_elapsed": 1.2,
            "tick_to_second": 0.25,
        }
    )
    write_json(run_root / "batch_000001" / "workers" / "run_001" / "rts_rollout_summary.json", {"decision_count": 2})

    db_path = root / "ledger.sqlite"
    result_1 = ingest_phase9_run(run_root, db_path)

    # Path 2 (same config but different filesystem path)
    run_root_2 = root / "rts_training" / "smoke_2"
    write_json(run_root_2 / "training_config.json", config)
    write_json(run_root_2 / "latest.json", {"checkpoint_dir": str(run_root_2 / "batch_000001" / "checkpoint")})
    write_json(run_root_2 / "batch_000001" / "batch_summary.json", {"status": "updated", "trainable_step_count": 2, "latest_updated": True, "dataset_summary": {"avg_reward": 1.0}, "ppo_update": {"total_loss_mean": 0.5}})
    write_json(run_root_2 / "batch_000001" / "checkpoint" / "metadata.json", {"ok": True})
    write_json(run_root_2 / "batch_000001" / "checkpoint" / "feature_schema.json", {"action_feature_dim": 2})
    write_json(
        run_root_2 / "batch_000001" / "workers" / "run_001" / "worker_summary.json",
        {
            "status": "success",
            "ticks_completed": 3,
            "netlogo_steps_completed": 5,
            "netlogo_steps_requested": 4,
            "warehouse_time_start": 100.0,
            "warehouse_time_end": 101.2,
            "warehouse_time_elapsed": 1.2,
            "tick_to_second": 0.25,
        }
    )
    write_json(run_root_2 / "batch_000001" / "workers" / "run_001" / "rts_rollout_summary.json", {"decision_count": 2})

    result_2 = ingest_phase9_run(run_root_2, db_path)

    # Assert experiment_id is stable (path independent)
    assert result_1["experiment_id"] == result_2["experiment_id"], "experiment_id should be stable across path moves"

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        assert conn.execute("SELECT COUNT(*) FROM experiments").fetchone()[0] == 1  # Upsert replaced the row
        assert conn.execute("SELECT COUNT(*) FROM training_batches").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM worker_rollouts").fetchone()[0] == 1

        row = conn.execute("SELECT * FROM worker_rollouts").fetchone()
        assert row["netlogo_steps_completed"] == 5, f"Expected 5 (alias), got {row['netlogo_steps_completed']}"
        assert row["netlogo_steps_requested"] == 4, f"Expected 4 (alias), got {row['netlogo_steps_requested']}"
        assert row["warehouse_time_start"] == 100.0
        assert row["warehouse_time_end"] == 101.2
        assert row["warehouse_time_elapsed"] == 1.2
        assert row["tick_to_second"] == 0.25

    shutil.rmtree(root)
    print("phase9 ingest smoke ok")


if __name__ == "__main__":
    main()
