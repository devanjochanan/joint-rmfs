#!/usr/bin/env python3
"""Smoke test SQLite CSV exports."""

from __future__ import annotations

from pathlib import Path
import shutil
import sqlite3
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.rmfs.experiments.ledger.schema import init_schema
from src.rmfs.experiments.ledger.writer import upsert_evaluation, upsert_training_batch
from scripts.experiments.export_training_summary import main as export_training
from scripts.experiments.export_eval_summary import main as export_eval


def main():
    root = REPO_ROOT / "data" / "runtime" / "phase10_export_smoke"
    shutil.rmtree(root, ignore_errors=True)
    db = root / "ledger.sqlite"
    init_schema(db)
    upsert_training_batch(db, {"experiment_id": "e", "batch_id": 1, "status": "updated", "policy_checkpoint_id_before": "b0", "policy_checkpoint_id_after": "b1", "netlogo_steps_per_run": 3, "workers": 1, "trainable_step_count": 1, "avg_reward": 1, "ppo_total_loss": 0.1, "policy_loss": 0.1, "value_loss": 0.1, "entropy": 0.1, "approx_kl": 0.0, "clip_fraction": 0.0, "latest_updated": 1, "batch_summary_path": "p", "dataset_summary_json": "{}", "ppo_update_json": "{}"})
    upsert_evaluation(db, {"eval_run_id": "ev", "experiment_id": "e", "policy_checkpoint_id": "b1", "policy_mode": "rts_rl_explicit", "policy_action_mode": "greedy", "seed_pack_id": "s", "netlogo_steps_per_run": 3, "replications": 1, "status": "dry_run", "created_at": "now", "summary_path": "p", "metrics_json": "{}"})
    out = root / "exports"
    export_training(["--db-path", str(db), "--export-dir", str(out)])
    export_eval(["--db-path", str(db), "--export-dir", str(out)])
    assert (out / "training_batches.csv").exists()
    assert (out / "checkpoint_index.csv").exists()
    assert (out / "eval_checkpoint_summary.csv").exists()
    assert (out / "eval_metrics_long.csv").exists()
    shutil.rmtree(root)
    print("export summary smoke ok")


if __name__ == "__main__":
    main()

