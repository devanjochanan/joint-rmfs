"""Checkpoint helpers for synthetic RTS PPO training smokes."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch

from .metrics import append_jsonl, atomic_write_json, json_safe, write_json
from .references import copy_cycle_reference_to_checkpoint


def checkpoint_root(output_root: Path, artifact_label: str) -> Path:
    return Path(output_root) / str(artifact_label)


def batch_checkpoint_dir(output_root: Path, artifact_label: str, batch_id: int) -> Path:
    return checkpoint_root(output_root, artifact_label) / f"batch_{int(batch_id):06d}" / "checkpoint"


def write_feature_schema(path: Path, *, action_feature_names: tuple[str, ...], stock_feature_names: tuple[str, ...]) -> dict[str, Any]:
    schema = {
        "action_feature_names": list(action_feature_names),
        "stock_feature_names": list(stock_feature_names),
        "action_feature_dim": len(action_feature_names),
        "stock_feature_dim": len(stock_feature_names),
    }
    atomic_write_json(path, schema)
    return schema


def write_latest_pointer(root: Path, *, batch_id: int, checkpoint_dir: Path) -> None:
    atomic_write_json(
        Path(root) / "latest.json",
        {
            "batch_id": int(batch_id),
            "checkpoint_dir": str(checkpoint_dir),
        },
    )


def append_checkpoint_history(root: Path, payload: Mapping[str, Any]) -> None:
    append_jsonl(Path(root) / "checkpoint_history.jsonl", payload)


def save_training_checkpoint(
    *,
    model,
    optimizer,
    config,
    batch_id: int,
    dataset_summary: Mapping[str, Any],
    ppo_update_result: Any,
    action_feature_names: tuple[str, ...],
    stock_feature_names: tuple[str, ...],
    cycle_reference_path: Path | None = None,
    lineage_metadata: Mapping[str, Any] | None = None,
) -> Path:
    root = checkpoint_root(config.output_root, config.artifact_label)
    checkpoint_dir = batch_checkpoint_dir(config.output_root, config.artifact_label, batch_id)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_dir / "model.pt")
    torch.save(optimizer.state_dict(), checkpoint_dir / "optimizer.pt")
    feature_schema = write_feature_schema(
        checkpoint_dir / "feature_schema.json",
        action_feature_names=action_feature_names,
        stock_feature_names=stock_feature_names,
    )
    copied_reference = None
    if cycle_reference_path is not None:
        copied_reference = copy_cycle_reference_to_checkpoint(cycle_reference_path, checkpoint_dir)
    metadata = json_safe(
        {
            "batch_id": int(batch_id),
            "training_config": config,
            "dataset_summary": dataset_summary,
            "ppo_update_result": ppo_update_result,
            "feature_schema": feature_schema,
            "cycle_reference_path": str(copied_reference) if copied_reference else None,
            "lineage": lineage_metadata,
        }
    )
    atomic_write_json(checkpoint_dir / "metadata.json", metadata)
    write_latest_pointer(root, batch_id=batch_id, checkpoint_dir=checkpoint_dir)
    append_checkpoint_history(
        root,
        {
            "batch_id": int(batch_id),
            "checkpoint_dir": str(checkpoint_dir),
            "ppo_update_result": ppo_update_result,
        },
    )
    return checkpoint_dir


def load_training_checkpoint(checkpoint_dir: Path, *, model, optimizer=None, device: str | torch.device = "cpu") -> dict[str, Any]:
    checkpoint_dir = Path(checkpoint_dir)
    model_state = torch.load(checkpoint_dir / "model.pt", map_location=device)
    model.load_state_dict(model_state)
    if optimizer is not None:
        optimizer_state = torch.load(checkpoint_dir / "optimizer.pt", map_location=device)
        optimizer.load_state_dict(optimizer_state)
    with (checkpoint_dir / "metadata.json").open() as fh:
        import json

        metadata = json.load(fh)
    return metadata


def write_batch_summary(path: Path, payload: Mapping[str, Any]) -> None:
    write_json(path, payload)

