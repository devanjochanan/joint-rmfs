#!/usr/bin/env python3
"""Smoke test for evaluation summary ingestion into SQLite."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import sqlite3
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.rmfs.experiments.ledger.ingest_evaluation import ingest_evaluation_summary


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        json.dump(payload, fh, indent=2)


def main():
    root = REPO_ROOT / "data" / "runtime" / "phase10_eval_ingest_smoke"
    shutil.rmtree(root, ignore_errors=True)

    summary_path = root / "eval_summary.json"
    eval_run_id = "eval_smoke_test_run"
    status = "completed"
    metrics = {
        "avg_order_cycle_time": 42.5,
        "orders_completed": 100,
        "congestion_rate": 0.12,
        "energy_per_order": 5.0
    }

    summary_data = {
        "eval_run_id": eval_run_id,
        "policy_checkpoint_id": "batch_000005",
        "seed_pack_id": "pack_123",
        "netlogo_steps_per_run": 100,
        "replications": 5,
        "status": status,
        "created_at": "2026-06-11T12:00:00Z",
        "metrics": metrics
    }

    write_json(summary_path, summary_data)
    db_path = root / "ledger.sqlite"

    row = ingest_evaluation_summary(summary_path, db_path)

    assert row["eval_run_id"] == eval_run_id
    assert row["status"] == status

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.execute("SELECT * FROM evaluations")
        rows = cur.fetchall()
        assert len(rows) == 1
        db_row = rows[0]
        assert db_row["eval_run_id"] == eval_run_id
        assert db_row["status"] == status
        assert db_row["policy_checkpoint_id"] == "batch_000005"
        assert db_row["seed_pack_id"] == "pack_123"
        assert db_row["netlogo_steps_per_run"] == 100
        assert db_row["replications"] == 5
        assert db_row["policy_mode"] == "rts_rl_explicit"
        assert db_row["policy_action_mode"] == "greedy"

        # Verify metrics roundtrip
        db_metrics = json.loads(db_row["metrics_json"])
        assert db_metrics["avg_order_cycle_time"] == 42.5
        assert db_metrics["orders_completed"] == 100

    shutil.rmtree(root)
    print("eval ingest smoke ok")


if __name__ == "__main__":
    main()
