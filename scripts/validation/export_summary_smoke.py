#!/usr/bin/env python3
"""Smoke test SQLite CSV exports."""

from __future__ import annotations

import csv
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
    
    # Ingest with a populated metrics_json dict
    upsert_evaluation(db, {
        "eval_run_id": "ev",
        "experiment_id": "e",
        "policy_checkpoint_id": "b1",
        "policy_mode": "rts_rl_explicit",
        "policy_action_mode": "greedy",
        "seed_pack_id": "s",
        "netlogo_steps_per_run": 3,
        "replications": 1,
        "status": "dry_run",
        "created_at": "now",
        "summary_path": "p",
        "metrics_json": '{"avg_order_cycle_time": 4.5, "orders_completed": 10}'
    })
    
    out = root / "exports"
    export_training(["--db-path", str(db), "--export-dir", str(out)])
    export_eval(["--db-path", str(db), "--export-dir", str(out)])
    
    assert (out / "training_batches.csv").exists()
    assert (out / "checkpoint_index.csv").exists()
    assert (out / "eval_checkpoint_summary.csv").exists()
    
    # Verify eval_metrics_long.csv format and content
    long_csv_path = out / "eval_metrics_long.csv"
    assert long_csv_path.exists()
    
    with long_csv_path.open() as fh:
        reader = csv.reader(fh)
        header = next(reader)
        assert header == ["eval_run_id", "metric_name", "metric_value"]
        
        rows = list(reader)
        assert len(rows) == 2
        # Assert specific keys were flattened
        metric_names = {row[1] for row in rows}
        assert "avg_order_cycle_time" in metric_names
        assert "orders_completed" in metric_names
        
        # Verify value mapping
        for row in rows:
            assert row[0] == "ev"
            if row[1] == "avg_order_cycle_time":
                assert float(row[2]) == 4.5
            elif row[1] == "orders_completed":
                assert int(row[2]) == 10
                
    shutil.rmtree(root)
    print("export summary smoke ok")


if __name__ == "__main__":
    main()
