#!/usr/bin/env python3
"""Validate Adam optimizer continuity across RTS checkpoint save/reload."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.validation.rts_ppo_update_smoke import synthetic_events
from src.rmfs.rl.rts.model import RTSMaskedActorCritic
from src.rmfs.rl.rts.training.checkpoint import (
    load_training_checkpoint,
    optimizer_state_fingerprint,
    save_training_checkpoint,
)
from src.rmfs.rl.rts.training.config import RTSTrainingConfig
from src.rmfs.rl.rts.training.policy_loader import load_policy_from_checkpoint
from src.rmfs.rl.rts.training.ppo import build_synthetic_ppo_smoke_batch, run_ppo_update
from src.rmfs.rl.rts.training.rollout_dataset import build_feature_tensors_from_steps, build_smoke_training_steps


def main() -> None:
    root = REPO_ROOT / "data" / "runtime" / "rts_optimizer_continuity_smoke"
    shutil.rmtree(root, ignore_errors=True)
    try:
        dataset = build_smoke_training_steps(synthetic_events())
        padded = build_feature_tensors_from_steps(dataset.steps)
        model = RTSMaskedActorCritic(
            action_feature_dim=padded.X_actions.shape[-1],
            stock_feature_dim=padded.X_stock.shape[-1],
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        config = RTSTrainingConfig(
            artifact_label="adam_continuity",
            output_root=root,
            learning_rate=1e-3,
            ppo_epochs=2,
            minibatch_size=1,
            zone_ids=("A", "B"),
            tensorboard_enabled=False,
        )
        batch = build_synthetic_ppo_smoke_batch(model, padded, "cpu", config.gamma, config.gae_lambda)
        first_update = run_ppo_update(model, optimizer, batch, config, "cpu")
        first_fingerprint = optimizer_state_fingerprint(optimizer.state_dict())
        assert first_fingerprint["state_parameter_count"] > 0
        assert first_fingerprint["max_step"] >= 1.0
        checkpoint_dir = save_training_checkpoint(
            model=model,
            optimizer=optimizer,
            config=config,
            batch_id=1,
            dataset_summary={"trainable_step_count": int(padded.rewards.shape[0]), "avg_reward": 1.0},
            ppo_update_result=first_update,
            action_feature_names=padded.action_feature_names,
            stock_feature_names=padded.stock_feature_names,
            reward_normalizer_metadata={"reward_time_scale": 1.0, "reward_time_scale_source": "smoke", "reward_valid_cycle_count": 2},
        )

        fresh = subprocess.check_output(
            [
                sys.executable,
                "-c",
                (
                    "import json, torch, sys; "
                    f"sys.path.insert(0, {str(REPO_ROOT)!r}); "
                    "from pathlib import Path; "
                    "from src.rmfs.rl.rts.training.policy_loader import load_policy_from_checkpoint; "
                    "from src.rmfs.rl.rts.training.checkpoint import load_training_checkpoint, optimizer_state_fingerprint; "
                    f"loaded=load_policy_from_checkpoint(Path({str(checkpoint_dir)!r}), device='cpu'); "
                    "opt=torch.optim.Adam(loaded.model.parameters(), lr=1e-3); "
                    f"load_training_checkpoint(Path({str(checkpoint_dir)!r}), model=loaded.model, optimizer=opt, device='cpu'); "
                    "print(json.dumps(optimizer_state_fingerprint(opt.state_dict()), sort_keys=True))"
                ),
            ],
            cwd=REPO_ROOT,
            text=True,
        )
        reloaded_fingerprint = json.loads(fresh)
        assert reloaded_fingerprint == first_fingerprint

        loaded = load_policy_from_checkpoint(checkpoint_dir, device="cpu")
        reloaded_optimizer = torch.optim.Adam(loaded.model.parameters(), lr=1e-3)
        load_training_checkpoint(checkpoint_dir, model=loaded.model, optimizer=reloaded_optimizer, device="cpu")
        second_update = run_ppo_update(loaded.model, reloaded_optimizer, batch, config, "cpu")
        second_fingerprint = optimizer_state_fingerprint(reloaded_optimizer.state_dict())
        assert second_update.optimizer_steps > 0
        assert second_fingerprint["min_step"] > first_fingerprint["min_step"]
        assert second_fingerprint["sha256"] != first_fingerprint["sha256"]
        print("rts optimizer continuity smoke ok")
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()
