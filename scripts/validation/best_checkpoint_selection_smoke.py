#!/usr/bin/env python3
"""Smoke test best-checkpoint metadata selection."""

from __future__ import annotations

from pathlib import Path
import json
import shutil
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.rmfs.experiments.evaluation.selection import select_best_checkpoint, write_best_checkpoint


def main():
    root = REPO_ROOT / "data" / "runtime" / "phase10_best_smoke"
    shutil.rmtree(root, ignore_errors=True)
    candidates = [
        {"valid": True, "policy_checkpoint_id": "batch_000001", "checkpoint_dir": "c1", "eval_run_id": "e1", "metrics": {"avg_order_cycle_time": 5, "orders_completed": 10, "congestion_rate": 0.2, "energy_per_order": 3}},
        {"valid": True, "policy_checkpoint_id": "batch_000002", "checkpoint_dir": "c2", "eval_run_id": "e2", "metrics": {"avg_order_cycle_time": 4, "orders_completed": 8, "congestion_rate": 0.3, "energy_per_order": 4}},
    ]
    pointer = select_best_checkpoint(candidates)
    assert pointer["selected_checkpoint_id"] == "batch_000002"
    path = root / "best_checkpoint.json"
    write_best_checkpoint(path, pointer)
    assert json.load(path.open())["selected_checkpoint_id"] == "batch_000002"
    shutil.rmtree(root)
    print("best checkpoint selection smoke ok")


if __name__ == "__main__":
    main()

