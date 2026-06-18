#!/usr/bin/env python3
"""Pure smoke for RTS policy checkpoint ID accounting."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.rmfs.rl.rts.features import build_action_feature_names, build_stock_feature_names
from src.rmfs.rl.rts.model import RTSMaskedActorCritic
from src.rmfs.rl.rts.training.checkpoint import (
    atomic_torch_save,
    resolve_policy_checkpoint_id,
    save_training_checkpoint,
    write_feature_schema,
)
from src.rmfs.rl.rts.training.config import RTSTrainingConfig
from src.rmfs.rl.rts.training.controller import _checkpoint_id
from src.rmfs.rl.rts.training.policy_loader import load_policy_from_checkpoint


def make_model(action_feature_names: tuple[str, ...], stock_feature_names: tuple[str, ...]) -> RTSMaskedActorCritic:
    return RTSMaskedActorCritic(
        action_feature_dim=len(action_feature_names),
        stock_feature_dim=len(stock_feature_names),
    )


def write_bootstrap_checkpoint(checkpoint_dir: Path, *, policy_checkpoint_id: str) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    action_feature_names = build_action_feature_names(("A", "B"))
    stock_feature_names = build_stock_feature_names()
    model = make_model(action_feature_names, stock_feature_names)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    atomic_torch_save(model.state_dict(), checkpoint_dir / "model.pt")
    atomic_torch_save(optimizer.state_dict(), checkpoint_dir / "optimizer.pt")
    feature_schema = write_feature_schema(
        checkpoint_dir / "feature_schema.json",
        action_feature_names=action_feature_names,
        stock_feature_names=stock_feature_names,
    )
    with (checkpoint_dir / "metadata.json").open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "policy_checkpoint_id": policy_checkpoint_id,
                "training_config": {"zone_ids": ["A", "B"]},
                "feature_schema": feature_schema,
            },
            fh,
            indent=2,
        )


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        bootstrap_dir = tmp_path / "level4_bootstrap"
        write_bootstrap_checkpoint(bootstrap_dir, policy_checkpoint_id="bootstrap_000000")
        assert resolve_policy_checkpoint_id(bootstrap_dir) == "bootstrap_000000"
        assert _checkpoint_id(bootstrap_dir) == "bootstrap_000000"
        assert load_policy_from_checkpoint(bootstrap_dir, device="cpu").policy_checkpoint_id == "bootstrap_000000"

        action_feature_names = build_action_feature_names(("A", "B"))
        stock_feature_names = build_stock_feature_names()
        model = make_model(action_feature_names, stock_feature_names)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        config = RTSTrainingConfig(
            artifact_label="checkpoint_id_smoke",
            output_root=tmp_path / "trained",
            tensorboard_enabled=False,
        )
        checkpoint_dir = save_training_checkpoint(
            model=model,
            optimizer=optimizer,
            config=config,
            batch_id=1,
            dataset_summary={"trainable_step_count": 0, "avg_reward": 0.0},
            ppo_update_result={"optimizer_steps": 0},
            action_feature_names=action_feature_names,
            stock_feature_names=stock_feature_names,
            checkpoint_id_before="bootstrap_000000",
        )

        metadata = read_json(checkpoint_dir / "metadata.json")
        assert metadata["policy_checkpoint_id"] == "batch_000001"
        assert resolve_policy_checkpoint_id(checkpoint_dir) == "batch_000001"
        assert _checkpoint_id(checkpoint_dir) == "batch_000001"
        assert load_policy_from_checkpoint(checkpoint_dir, device="cpu").policy_checkpoint_id == "batch_000001"

        run_root = Path(config.output_root) / config.artifact_label
        latest = read_json(run_root / "latest.json")
        assert latest["policy_checkpoint_id"] == "batch_000001"
        assert latest["checkpoint_dir"] == str(checkpoint_dir)

        with (run_root / "checkpoint_history.jsonl").open(encoding="utf-8") as fh:
            history_rows = [json.loads(line) for line in fh if line.strip()]
        assert history_rows[-1]["checkpoint_id_after"] == "batch_000001"
        assert history_rows[-1]["policy_checkpoint_id"] == "batch_000001"

    print("rts checkpoint id accounting smoke ok")


if __name__ == "__main__":
    main()
