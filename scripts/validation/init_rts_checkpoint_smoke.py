#!/usr/bin/env python3
"""Smoke test for the bootstrap checkpoint initialization tool."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.rmfs.rl.rts.training.policy_loader import load_policy_from_checkpoint


def main():
    tmp_dir = REPO_ROOT / "data" / "runtime" / "phase_bootstrap_smoke"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)

    try:
        # Run init_rts_checkpoint.py
        subprocess.check_call(
            [
                sys.executable,
                "scripts/training/init_rts_checkpoint.py",
                "--checkpoint-dir",
                str(tmp_dir),
                "--zone-ids",
                "A,B",
                "--policy-checkpoint-id",
                "bootstrap_smoke",
            ],
            cwd=REPO_ROOT,
        )

        # Verify files exist
        for filename in (
            "model.pt",
            "optimizer.pt",
            "metadata.json",
            "feature_schema.json",
            "cycle_reference.json",
            "zone_ids",
            "policy_checkpoint_id",
        ):
            path = tmp_dir / filename
            assert path.exists(), f"Missing expected checkpoint file: {filename}"

        # Load policy
        loaded = load_policy_from_checkpoint(tmp_dir, device="cpu")
        assert loaded.policy_checkpoint_id == "bootstrap_smoke"
        assert loaded.model.training is False

        # Run dummy forward pass
        action_dim = loaded.feature_schema["action_feature_dim"]
        stock_dim = loaded.feature_schema["stock_feature_dim"]

        X_actions = torch.zeros((1, 4, action_dim), dtype=torch.float32)
        M_actions = torch.ones((1, 4), dtype=torch.int64)
        X_stock = torch.zeros((1, 2, stock_dim), dtype=torch.float32)
        M_stock = torch.ones((1, 2), dtype=torch.int64)

        logits, values = loaded.model(X_actions, M_actions, X_stock, M_stock)
        assert logits.shape == (1, 4)
        assert values.shape == (1,)

        print("init_rts_checkpoint smoke test passed successfully")
    finally:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    main()
