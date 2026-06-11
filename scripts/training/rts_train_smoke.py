#!/usr/bin/env python3
"""Synthetic RTS PPO/checkpoint smoke. This is not a training run."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.validation.rts_ppo_update_smoke import synthetic_events
from src.rmfs.rl.rts.model import RTSMaskedActorCritic
from src.rmfs.rl.rts.training.checkpoint import load_training_checkpoint, save_training_checkpoint, write_batch_summary
from src.rmfs.rl.rts.training.config import RTSTrainingConfig, validate_training_config
from src.rmfs.rl.rts.training.lineage import build_batch_summary, build_lineage_metadata, write_lineage_json
from src.rmfs.rl.rts.training.ppo import build_synthetic_ppo_smoke_batch, compute_log_probs_values, run_ppo_update
from src.rmfs.rl.rts.training.references import write_synthetic_cycle_reference
from src.rmfs.rl.rts.training.rollout_dataset import build_feature_tensors_from_steps, build_smoke_training_steps


def main(argv=None):
    parser = argparse.ArgumentParser(description="Synthetic RTS training smoke.")
    parser.add_argument("--artifact-label", default="phase8_synthetic_smoke")
    parser.add_argument("--output-root", default="data/runtime/rts_training_smoke")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = REPO_ROOT / output_root
    try:
        output_root.relative_to(REPO_ROOT / "data" / "runtime")
    except ValueError as exc:
        raise RuntimeError("synthetic training smoke outputs must stay under data/runtime") from exc

    config = RTSTrainingConfig(
        artifact_label=args.artifact_label,
        output_root=output_root,
        seed=args.seed,
        learning_rate=1e-3,
        ppo_epochs=2,
        minibatch_size=1,
        tensorboard_enabled=False,
    )
    validate_training_config(config)
    torch.manual_seed(config.seed)
    run_root = output_root / args.artifact_label
    run_root.mkdir(parents=True, exist_ok=True)
    reference_path = run_root / "cycle_reference.json"
    reference = write_synthetic_cycle_reference(reference_path)
    dataset = build_smoke_training_steps(synthetic_events())
    padded = build_feature_tensors_from_steps(dataset.steps)
    model = RTSMaskedActorCritic(
        action_feature_dim=padded.X_actions.shape[-1],
        stock_feature_dim=padded.X_stock.shape[-1],
        hidden_sizes=config.hidden_sizes,
        stock_hidden_sizes=config.stock_hidden_sizes,
        stock_embedding_dim=config.stock_embedding_dim,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    ppo_batch = build_synthetic_ppo_smoke_batch(model, padded, "cpu", config.gamma, config.gae_lambda)
    result = run_ppo_update(model, optimizer, ppo_batch, config, "cpu")
    feature_schema = {
        "action_feature_names": list(padded.action_feature_names),
        "stock_feature_names": list(padded.stock_feature_names),
        "action_feature_dim": len(padded.action_feature_names),
        "stock_feature_dim": len(padded.stock_feature_names),
    }
    lineage = build_lineage_metadata(
        artifact_label=config.artifact_label,
        batch_id=1,
        config=config,
        dataset_summary=dataset.summary,
        ppo_update_result=result,
        feature_schema=feature_schema,
        cycle_reference_path=str(reference_path),
        cycle_reference_source=reference.source,
        device="cpu",
    )
    checkpoint_dir = save_training_checkpoint(
        model=model,
        optimizer=optimizer,
        config=config,
        batch_id=1,
        dataset_summary=dataset.summary,
        ppo_update_result=result,
        action_feature_names=padded.action_feature_names,
        stock_feature_names=padded.stock_feature_names,
        cycle_reference_path=reference_path,
        lineage_metadata=lineage,
    )
    batch_summary = build_batch_summary(
        artifact_label=config.artifact_label,
        batch_id=1,
        checkpoint_dir=checkpoint_dir,
        dataset_summary=dataset.summary,
        ppo_update_result=result,
    )
    write_batch_summary(checkpoint_dir.parent / "batch_summary.json", batch_summary)
    write_lineage_json(checkpoint_dir / "lineage.json", lineage)
    assert (checkpoint_dir / "model.pt").exists()
    assert (checkpoint_dir / "optimizer.pt").exists()
    assert (checkpoint_dir / "metadata.json").exists()
    assert (checkpoint_dir / "feature_schema.json").exists()
    assert (checkpoint_dir / "cycle_reference.json").exists()
    assert (checkpoint_dir.parent.parent / "latest.json").exists()
    assert (checkpoint_dir.parent.parent / "checkpoint_history.jsonl").exists()
    loaded_model = RTSMaskedActorCritic(
        action_feature_dim=padded.X_actions.shape[-1],
        stock_feature_dim=padded.X_stock.shape[-1],
        hidden_sizes=config.hidden_sizes,
        stock_hidden_sizes=config.stock_hidden_sizes,
        stock_embedding_dim=config.stock_embedding_dim,
    )
    load_training_checkpoint(checkpoint_dir, model=loaded_model, device="cpu")
    with torch.no_grad():
        compute_log_probs_values(loaded_model, padded, "cpu")
    print("rts train smoke ok")


if __name__ == "__main__":
    main()

