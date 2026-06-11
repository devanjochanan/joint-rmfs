#!/usr/bin/env python3
"""Dry-run smoke for the RTS on-policy training controller."""

from __future__ import annotations

from pathlib import Path
import shutil
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.rmfs.rl.rts.training.references import write_synthetic_cycle_reference
from scripts.training.rts_train_controller import main as controller_main


def main():
    output_root = REPO_ROOT / "data" / "runtime" / "rts_training" / "phase9_dry_run_smoke"
    shutil.rmtree(output_root, ignore_errors=True)
    output_root.mkdir(parents=True, exist_ok=True)
    reference_path = output_root / "cycle_reference.json"
    write_synthetic_cycle_reference(reference_path)
    controller_main(
        [
            "--artifact-label",
            "phase9_dry_run_smoke",
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
            "--cycle-reference",
            str(reference_path),
            "--no-progress",
            "--no-tensorboard",
            "--dry-run",
        ]
    )
    run_root = output_root / "phase9_dry_run_smoke"
    assert (run_root / "training_config.json").exists()
    assert (run_root / "batch_000001" / "rollout_input" / "active_checkpoint_ref.json").exists()
    assert (run_root / "batch_000001" / "workers" / "run_001" / "run_spec.json").exists()
    assert (run_root / "batch_000001" / "workers" / "run_002" / "run_spec.json").exists()
    shutil.rmtree(output_root, ignore_errors=True)
    print("rts training controller dry run ok")


if __name__ == "__main__":
    main()

