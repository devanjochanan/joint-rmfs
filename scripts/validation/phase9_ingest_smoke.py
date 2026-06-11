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
    run_root = root / "rts_training" / "smoke"
    write_json(run_root / "training_config.json", {"artifact_label": "smoke", "seed": 42, "netlogo_steps_per_run": 3, "workers": 1})
    write_json(run_root / "latest.json", {"checkpoint_dir": str(run_root / "batch_000001" / "checkpoint")})
    write_json(run_root / "batch_000001" / "batch_summary.json", {"status": "updated", "trainable_step_count": 2, "latest_updated": True, "dataset_summary": {"avg_reward": 1.0}, "ppo_update": {"total_loss_mean": 0.5}})
    write_json(run_root / "batch_000001" / "checkpoint" / "metadata.json", {"ok": True})
    write_json(run_root / "batch_000001" / "checkpoint" / "feature_schema.json", {"action_feature_dim": 2})
    write_json(run_root / "batch_000001" / "workers" / "run_001" / "worker_summary.json", {"status": "success", "ticks_completed": 3})
    write_json(run_root / "batch_000001" / "workers" / "run_001" / "rts_rollout_summary.json", {"decision_count": 2})
    db_path = root / "ledger.sqlite"
    result = ingest_phase9_run(run_root, db_path)
    with sqlite3.connect(db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM experiments").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM training_batches").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM worker_rollouts").fetchone()[0] == 1
    assert result["experiment_id"].startswith("rtsrl_")
    shutil.rmtree(root)
    print("phase9 ingest smoke ok")


if __name__ == "__main__":
    main()

