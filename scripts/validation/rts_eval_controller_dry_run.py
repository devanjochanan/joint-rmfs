#!/usr/bin/env python3
"""Smoke test dry-run RTS evaluation controller."""

from __future__ import annotations

from pathlib import Path
import shutil
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.rmfs.experiments.evaluation.controller import write_eval_dry_run
from src.rmfs.experiments.evaluation.seed_pack import build_seed_pack, write_seed_pack


def main():
    root = REPO_ROOT / "data" / "runtime" / "phase10_eval_smoke"
    shutil.rmtree(root, ignore_errors=True)
    pack = build_seed_pack(seed_base=42, replications=2, netlogo_steps_per_run=3, purpose="eval_smoke")
    pack_path = root / "pack.json"
    write_seed_pack(pack_path, pack)
    summary = write_eval_dry_run(
        checkpoint_dir=root / "fake_checkpoint",
        policy_checkpoint_id="batch_000001",
        zone_ids=("A", "B"),
        seed_pack_path=pack_path,
        output_root=root / "evals",
    )
    run_root = root / "evals" / summary["eval_run_id"]
    assert (run_root / "eval_config.json").exists()
    assert (run_root / "worker_specs.json").exists()
    assert (run_root / "eval_summary.json").exists()
    assert summary["status"] == "dry_run"
    shutil.rmtree(root)
    print("rts eval controller dry run ok")


if __name__ == "__main__":
    main()

