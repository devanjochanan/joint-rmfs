"""Explicit RTS policy checkpoint loader for rollout workers."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import torch

from src.rmfs.rl.rts.model import RTSMaskedActorCritic


@dataclass(frozen=True)
class LoadedRTSPolicy:
    model: RTSMaskedActorCritic
    checkpoint_dir: Path
    policy_checkpoint_id: str
    feature_schema: dict
    metadata: dict


def load_policy_from_checkpoint(checkpoint_dir: Path, *, device: str = "cpu") -> LoadedRTSPolicy:
    checkpoint = Path(checkpoint_dir)
    metadata_path = checkpoint / "metadata.json"
    schema_path = checkpoint / "feature_schema.json"
    model_path = checkpoint / "model.pt"
    for path in (metadata_path, schema_path, model_path):
        if not path.exists():
            raise FileNotFoundError(f"missing RTS policy checkpoint file: {path}")
    with metadata_path.open() as fh:
        metadata = json.load(fh)
    with schema_path.open() as fh:
        feature_schema = json.load(fh)
    training_config = dict(metadata.get("training_config", {}) or {})
    model = RTSMaskedActorCritic(
        action_feature_dim=int(feature_schema["action_feature_dim"]),
        stock_feature_dim=int(feature_schema["stock_feature_dim"]),
        hidden_sizes=tuple(training_config.get("hidden_sizes", (64, 64))),
        stock_hidden_sizes=tuple(training_config.get("stock_hidden_sizes", (32, 32))),
        stock_embedding_dim=int(training_config.get("stock_embedding_dim", 16)),
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    policy_checkpoint_id = str(metadata.get("policy_checkpoint_id") or _checkpoint_id_from_path(checkpoint))
    if not policy_checkpoint_id.strip():
        raise ValueError("policy_checkpoint_id must be nonblank")
    return LoadedRTSPolicy(
        model=model,
        checkpoint_dir=checkpoint,
        policy_checkpoint_id=policy_checkpoint_id,
        feature_schema=feature_schema,
        metadata=metadata,
    )


def _checkpoint_id_from_path(checkpoint_dir: Path) -> str:
    parent = checkpoint_dir.parent
    if parent.name.startswith("batch_"):
        return parent.name
    return checkpoint_dir.name

