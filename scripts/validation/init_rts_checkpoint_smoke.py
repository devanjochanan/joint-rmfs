#!/usr/bin/env python3
"""Smoke test for the bootstrap checkpoint initialization tool."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
import torch
import hashlib

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.rmfs.rl.rts.training.policy_loader import load_policy_from_checkpoint


def _model_digest(path: Path) -> str:
    state = torch.load(path, map_location="cpu", weights_only=True)
    digest = hashlib.sha256()
    for key in sorted(state.keys()):
        tensor = state[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


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
            "zone_ids",
            "policy_checkpoint_id",
        ):
            path = tmp_dir / filename
            assert path.exists(), f"Missing expected checkpoint file: {filename}"
        assert not (tmp_dir / "cycle_reference.json").exists()

        # Load policy
        loaded = load_policy_from_checkpoint(tmp_dir, device="cpu")
        assert loaded.policy_checkpoint_id == "bootstrap_smoke"
        assert loaded.model.training is False
        reward_metadata = loaded.metadata.get("reward_normalizer") or {}
        assert reward_metadata.get("reward_mode") == "cold_start_paper_cycle_duration"
        assert reward_metadata.get("reward_horizon") == "paper_cycle_duration"
        assert reward_metadata.get("reward_reference_required") is False
        assert reward_metadata.get("cycle_reference_enabled") is False
        assert reward_metadata.get("alpha_enabled") is False
        assert loaded.feature_schema["action_feature_dim"] == 18
        assert loaded.feature_schema["stock_feature_dim"] == 4
        assert loaded.feature_schema["action_feature_schema_version"] == "rts_action_features.v4"
        assert loaded.metadata["checkpoint_kind"] == "initial_untrained"
        assert loaded.metadata["initialization_seed"] == 42
        assert loaded.metadata["optimizer"]["type"] == "Adam"

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

        legacy_dir = tmp_dir.parent / "phase_bootstrap_legacy_smoke"
        if legacy_dir.exists():
            shutil.rmtree(legacy_dir)
        subprocess.check_call(
            [
                sys.executable,
                "scripts/training/init_rts_checkpoint.py",
                "--checkpoint-dir",
                str(legacy_dir),
                "--zone-ids",
                "A,B",
                "--policy-checkpoint-id",
                "bootstrap_legacy_smoke",
                "--write-legacy-cycle-reference",
            ],
            cwd=REPO_ROOT,
        )
        assert (legacy_dir / "cycle_reference.json").exists()
        shutil.rmtree(legacy_dir)

        same_seed_dir = tmp_dir.parent / "phase_bootstrap_same_seed_smoke"
        other_seed_dir = tmp_dir.parent / "phase_bootstrap_other_seed_smoke"
        for path in (same_seed_dir, other_seed_dir):
            shutil.rmtree(path, ignore_errors=True)
        subprocess.check_call(
            [
                sys.executable,
                "scripts/training/init_rts_checkpoint.py",
                "--checkpoint-dir",
                str(same_seed_dir),
                "--zone-ids",
                "A,B",
                "--policy-checkpoint-id",
                "bootstrap_same_seed_smoke",
                "--seed",
                "42",
            ],
            cwd=REPO_ROOT,
        )
        subprocess.check_call(
            [
                sys.executable,
                "scripts/training/init_rts_checkpoint.py",
                "--checkpoint-dir",
                str(other_seed_dir),
                "--zone-ids",
                "A,B",
                "--policy-checkpoint-id",
                "bootstrap_other_seed_smoke",
                "--seed",
                "43",
            ],
            cwd=REPO_ROOT,
        )
        assert _model_digest(tmp_dir / "model.pt") == _model_digest(same_seed_dir / "model.pt")
        assert _model_digest(tmp_dir / "model.pt") != _model_digest(other_seed_dir / "model.pt")
        shutil.rmtree(same_seed_dir)
        shutil.rmtree(other_seed_dir)

        print("init_rts_checkpoint smoke test passed successfully")
    finally:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    main()
