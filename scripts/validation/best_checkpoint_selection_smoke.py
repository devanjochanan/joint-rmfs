#!/usr/bin/env python3
"""Smoke test best-checkpoint metadata selection."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.rmfs.experiments.evaluation.selection import select_best_checkpoint, write_best_checkpoint


def main():
    root = REPO_ROOT / "data" / "runtime" / "phase10_best_smoke"
    shutil.rmtree(root, ignore_errors=True)

    # 1. Base functionality
    candidates = [
        {"valid": True, "policy_checkpoint_id": "batch_000001", "checkpoint_dir": "c1", "eval_run_id": "e1", "seed_pack_id": "pack", "charging_mode": "inherit", "metrics": {"mean_completed_paper_cycle_duration": 5, "completed_cycle_count": 10, "completion_rate": 1.0, "orders_completed": 10, "congestion_rate": 0.2, "energy_per_order": 3, "worker_failures": 0, "runtime_invariant_violations": 0}},
        {"valid": True, "policy_checkpoint_id": "batch_000002", "checkpoint_dir": "c2", "eval_run_id": "e2", "seed_pack_id": "pack", "charging_mode": "inherit", "metrics": {"mean_completed_paper_cycle_duration": 4, "completed_cycle_count": 8, "completion_rate": 1.0, "orders_completed": 8, "congestion_rate": 0.3, "energy_per_order": 4, "worker_failures": 0, "runtime_invariant_violations": 0}},
    ]
    pointer = select_best_checkpoint(candidates)
    assert pointer["selected_checkpoint_id"] == "batch_000002"

    # 2. Tie-breaker where batch_000074 beats batch_000020 on equal metrics
    tied_candidates = [
        {"valid": True, "policy_checkpoint_id": "batch_000020", "checkpoint_dir": "c20", "eval_run_id": "e20", "metrics": {"mean_completed_paper_cycle_duration": 4, "completed_cycle_count": 8, "completion_rate": 1.0, "orders_completed": 8, "congestion_rate": 0.3, "energy_per_order": 4, "worker_failures": 0, "runtime_invariant_violations": 0}},
        {"valid": True, "policy_checkpoint_id": "batch_000074", "checkpoint_dir": "c74", "eval_run_id": "e74", "metrics": {"mean_completed_paper_cycle_duration": 4, "completed_cycle_count": 8, "completion_rate": 1.0, "orders_completed": 8, "congestion_rate": 0.3, "energy_per_order": 4, "worker_failures": 0, "runtime_invariant_violations": 0}},
    ]
    pointer_tied = select_best_checkpoint(tied_candidates)
    assert pointer_tied["selected_checkpoint_id"] == "batch_000074", f"Expected batch_000074, got {pointer_tied['selected_checkpoint_id']}"

    # 3. Mean-style metric alias normalization
    alias_candidates = [
        {"valid": True, "policy_checkpoint_id": "batch_000003", "checkpoint_dir": "c3", "eval_run_id": "e3", "metrics": {"avg_order_cycle_time_mean": 6, "completed_cycle_count": 12, "orders_completed_mean": 12, "congestion_rate_mean": 0.1, "energy_per_order_mean": 2, "worker_failures": 0, "runtime_invariant_violations": 0}},
        {"valid": True, "policy_checkpoint_id": "batch_000004", "checkpoint_dir": "c4", "eval_run_id": "e4", "metrics": {"avg_order_cycle_time_mean": 3, "completed_cycle_count": 12, "orders_completed_mean": 12, "congestion_rate_mean": 0.1, "energy_per_order_mean": 2, "worker_failures": 0, "runtime_invariant_violations": 0}},
    ]
    pointer_alias = select_best_checkpoint(alias_candidates)
    assert pointer_alias["selected_checkpoint_id"] == "batch_000004"

    invalid_candidates = [
        {"valid": False, "policy_checkpoint_id": "batch_000099", "checkpoint_dir": "bad", "metrics": {"mean_completed_paper_cycle_duration": 1, "completed_cycle_count": 10}},
        {"valid": True, "policy_checkpoint_id": "batch_000098", "checkpoint_dir": "bad2", "metrics": {"mean_completed_paper_cycle_duration": 1, "completed_cycle_count": 0}},
        candidates[0],
    ]
    assert select_best_checkpoint(invalid_candidates)["selected_checkpoint_id"] == "batch_000001"

    # 4. Write pointer and reload
    path = root / "best_checkpoint.json"
    write_best_checkpoint(path, pointer)
    assert json.load(path.open())["selected_checkpoint_id"] == "batch_000002"

    shutil.rmtree(root)
    print("best checkpoint selection smoke ok")


if __name__ == "__main__":
    main()
