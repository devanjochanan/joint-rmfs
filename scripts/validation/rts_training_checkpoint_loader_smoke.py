#!/usr/bin/env python3
"""Pure smoke for RTS policy checkpoint loader."""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.validation.rts_ppo_update_smoke import synthetic_events
from src.rmfs.rl.rts.model import RTSMaskedActorCritic
from src.rmfs.rl.rts.training.checkpoint import save_training_checkpoint
from src.rmfs.rl.rts.training.config import RTSTrainingConfig
from src.rmfs.rl.rts.training.policy_loader import load_policy_from_checkpoint
from src.rmfs.rl.rts.training.rollout_dataset import build_feature_tensors_from_steps, build_smoke_training_steps


def main():
    dataset = build_smoke_training_steps(synthetic_events())
    padded = build_feature_tensors_from_steps(dataset.steps)
    with tempfile.TemporaryDirectory() as tmp:
        config = RTSTrainingConfig(
            artifact_label="loader_smoke",
            output_root=Path(tmp),
            tensorboard_enabled=False,
        )
        model = RTSMaskedActorCritic(
            action_feature_dim=padded.X_actions.shape[-1],
            stock_feature_dim=padded.X_stock.shape[-1],
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        checkpoint_dir = save_training_checkpoint(
            model=model,
            optimizer=optimizer,
            config=config,
            batch_id=1,
            dataset_summary=dataset.summary,
            ppo_update_result={"optimizer_steps": 0},
            action_feature_names=padded.action_feature_names,
            stock_feature_names=padded.stock_feature_names,
        )
        loaded = load_policy_from_checkpoint(checkpoint_dir, device="cpu")
        assert loaded.policy_checkpoint_id == "batch_000001"
        assert loaded.model.training is False
        assert loaded.feature_schema["action_feature_dim"] == padded.X_actions.shape[-1]
    print("rts training checkpoint loader smoke ok")


if __name__ == "__main__":
    main()

