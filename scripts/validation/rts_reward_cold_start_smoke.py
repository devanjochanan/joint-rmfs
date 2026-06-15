#!/usr/bin/env python3
"""Smoke tests for reference-free RTS reward cold-start training."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.validation.rts_on_policy_dataset_smoke import decision, outcome
from src.rmfs.rl.rts.action_space import STORE
from src.rmfs.rl.rts.reward import build_reward_components_from_realized_cycle, compute_cold_start_reward
from src.rmfs.rl.rts.training.checkpoint import save_training_checkpoint
from src.rmfs.rl.rts.training.config import RTSTrainingConfig
from src.rmfs.rl.rts.training.on_policy_config import RTSOnPolicyTrainingConfig, validate_on_policy_training_config
from src.rmfs.rl.rts.training.on_policy_dataset import build_on_policy_training_steps
from src.rmfs.rl.rts.training.policy_loader import load_policy_from_checkpoint
from src.rmfs.rl.rts.training.reward_normalizer import (
    derive_reward_normalizer_from_events,
    load_reward_normalizer_metadata,
)
from src.rmfs.rl.rts.model import RTSMaskedActorCritic
from src.rmfs.rl.rts.training.rollout_dataset import build_feature_tensors_from_steps


def main():
    checkpoint_dir = REPO_ROOT / "data" / "runtime" / "reward_cold_start_bootstrap_smoke"
    shutil.rmtree(checkpoint_dir, ignore_errors=True)
    try:
        subprocess.check_call(
            [
                sys.executable,
                "scripts/training/init_rts_checkpoint.py",
                "--checkpoint-dir",
                str(checkpoint_dir),
                "--zone-ids",
                "A,B",
                "--policy-checkpoint-id",
                "bootstrap_cold_start",
            ],
            cwd=REPO_ROOT,
        )
        assert not (checkpoint_dir / "cycle_reference.json").exists()
        loaded = load_policy_from_checkpoint(checkpoint_dir, device="cpu")
        assert loaded.policy_checkpoint_id == "bootstrap_cold_start"
        assert loaded.metadata["reward_normalizer"]["reward_reference_required"] is False

        config = RTSOnPolicyTrainingConfig(
            artifact_label="cold_start_config",
            output_root=REPO_ROOT / "data" / "runtime" / "cold_start_config",
            batches=1,
            workers=1,
            netlogo_steps_per_run=1,
            seed=7,
        )
        validate_on_policy_training_config(config)
        assert config.cycle_reference_path is None

        reward = compute_cold_start_reward(
            build_reward_components_from_realized_cycle(
                selected_action_branch=STORE,
                realized_cycle_time=12.0,
            ),
            reward_time_scale=6.0,
            reward_time_scale_source="smoke",
            reward_valid_cycle_count=2,
        )
        assert reward["reward_computed"] is True
        assert reward["reward_value"] == -2.0

        events = [
            decision("cold_start_1", "rts_rl_explicit"),
            outcome("cold_start_1"),
            decision("cold_start_2", "rts_rl_explicit"),
            outcome("cold_start_2"),
        ]
        events[1]["reward_json"] = {"reward_computed": False, "reward_value": None}
        events[1]["realized_cycle_time"] = 10.0
        events[3]["reward_json"] = None
        events[3]["realized_cycle_time"] = 20.0
        metadata = derive_reward_normalizer_from_events(events, batch_id=1)
        assert metadata["reward_time_scale"] == 15.0
        assert metadata["reward_valid_cycle_count"] == 2
        dataset = build_on_policy_training_steps(events, required_policy_checkpoint_id="batch_000001")
        assert dataset.summary["trainable_step_count"] == 2
        assert dataset.summary["rejected_reward_uncomputed_count"] == 0
        assert dataset.summary["reward_time_scale"] == 15.0

        padded = build_feature_tensors_from_steps(dataset.steps)
        model = RTSMaskedActorCritic(
            action_feature_dim=padded.X_actions.shape[-1],
            stock_feature_dim=padded.X_stock.shape[-1],
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        out_root = checkpoint_dir.parent / "reward_cold_start_checkpoint_smoke"
        shutil.rmtree(out_root, ignore_errors=True)
        checkpoint = save_training_checkpoint(
            model=model,
            optimizer=optimizer,
            config=RTSTrainingConfig(artifact_label="cold_start_ckpt", output_root=out_root),
            batch_id=1,
            dataset_summary=dataset.summary,
            ppo_update_result={"optimizer_steps": 0},
            action_feature_names=padded.action_feature_names,
            stock_feature_names=padded.stock_feature_names,
            reward_normalizer_metadata=metadata,
        )
        with (checkpoint / "metadata.json").open() as fh:
            saved = json.load(fh)
        assert saved["reward_normalizer"]["reward_time_scale"] == 15.0
        assert load_reward_normalizer_metadata(checkpoint)["reward_time_scale"] == 15.0
        shutil.rmtree(out_root, ignore_errors=True)

        print("rts reward cold start smoke ok")
    finally:
        shutil.rmtree(checkpoint_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
