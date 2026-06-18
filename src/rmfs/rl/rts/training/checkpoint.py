"""Checkpoint helpers for synthetic RTS PPO training smokes."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

import torch

from .metrics import append_jsonl, atomic_write_json, json_safe, write_json
from .references import copy_cycle_reference_to_checkpoint
from ..graph_distance import DISTANCE_SEMANTICS_VERSION
from ..reward import REWARD_HORIZON
from ..zone_registry import schema_metadata_for_zone_ids, zone_ids_from_action_feature_names


def atomic_torch_save(payload: Any, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def checkpoint_root(output_root: Path, artifact_label: str) -> Path:
    return Path(output_root) / str(artifact_label)


def batch_checkpoint_dir(output_root: Path, artifact_label: str, batch_id: int) -> Path:
    return checkpoint_root(output_root, artifact_label) / f"batch_{int(batch_id):06d}" / "checkpoint"


def checkpoint_id_from_path(checkpoint_dir: Path) -> str:
    checkpoint = Path(checkpoint_dir)
    parent = checkpoint.parent
    return parent.name if parent.name.startswith("batch_") else checkpoint.name


def resolve_policy_checkpoint_id(checkpoint_dir: Path | None) -> str:
    if checkpoint_dir is None:
        return "dry_run_uninitialized"
    checkpoint = Path(checkpoint_dir)
    metadata_id = _policy_checkpoint_id_from_metadata(checkpoint)
    if metadata_id:
        return metadata_id
    sidecar_id = _policy_checkpoint_id_from_sidecar(checkpoint)
    if sidecar_id:
        return sidecar_id
    return checkpoint_id_from_path(checkpoint)


def write_feature_schema(
    path: Path,
    *,
    action_feature_names: tuple[str, ...],
    stock_feature_names: tuple[str, ...],
    schema_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    zone_ids = zone_ids_from_action_feature_names(action_feature_names)
    metadata = dict(schema_metadata or {})
    if zone_ids:
        metadata = {**schema_metadata_for_zone_ids(zone_ids), **metadata}
    metadata.setdefault("reward_horizon", REWARD_HORIZON)
    metadata.setdefault("distance_semantics_version", DISTANCE_SEMANTICS_VERSION)
    schema = {
        "action_feature_names": list(action_feature_names),
        "stock_feature_names": list(stock_feature_names),
        "action_feature_dim": len(action_feature_names),
        "stock_feature_dim": len(stock_feature_names),
        **metadata,
    }
    atomic_write_json(path, schema)
    return schema


def write_latest_pointer(root: Path, *, batch_id: int, checkpoint_dir: Path, policy_checkpoint_id: str | None = None) -> None:
    resolved_checkpoint_id = str(policy_checkpoint_id or resolve_policy_checkpoint_id(checkpoint_dir))
    atomic_write_json(
        Path(root) / "latest.json",
        {
            "batch_id": int(batch_id),
            "checkpoint_dir": str(checkpoint_dir),
            "policy_checkpoint_id": resolved_checkpoint_id,
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
    reward_normalizer_metadata: Mapping[str, Any] | None = None,
    lineage_metadata: Mapping[str, Any] | None = None,
    checkpoint_id_before: str | None = None,
) -> Path:
    import datetime
    root = checkpoint_root(config.output_root, config.artifact_label)
    checkpoint_dir = batch_checkpoint_dir(config.output_root, config.artifact_label, batch_id)
    checkpoint_id_after = checkpoint_id_from_path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    atomic_torch_save(model.state_dict(), checkpoint_dir / "model.pt")
    atomic_torch_save(optimizer.state_dict(), checkpoint_dir / "optimizer.pt")
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
            "policy_checkpoint_id": checkpoint_id_after,
            "training_config": config,
            "dataset_summary": dataset_summary,
            "ppo_update_result": ppo_update_result,
            "feature_schema": feature_schema,
            "cycle_reference_path": str(copied_reference) if copied_reference else None,
            "reward_normalizer": reward_normalizer_metadata,
            "lineage": lineage_metadata,
        }
    )
    atomic_write_json(checkpoint_dir / "metadata.json", metadata)
    write_latest_pointer(root, batch_id=batch_id, checkpoint_dir=checkpoint_dir, policy_checkpoint_id=checkpoint_id_after)
    
    append_checkpoint_history(
        root,
        {
            "batch_id": int(batch_id),
            "checkpoint_id_before": checkpoint_id_before,
            "checkpoint_id_after": checkpoint_id_after,
            "policy_checkpoint_id": checkpoint_id_after,
            "checkpoint_dir": str(checkpoint_dir),
            "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "dataset_summary": dict(dataset_summary),
            "ppo_update_result": ppo_update_result.to_json_dict() if hasattr(ppo_update_result, "to_json_dict") else ppo_update_result,
            "trainable_step_count": int(dataset_summary.get("trainable_step_count", 0)),
            "avg_reward": float(dataset_summary.get("avg_reward", 0.0)),
            "cycle_reference_path": str(copied_reference) if copied_reference else None,
            "reward_normalizer": dict(reward_normalizer_metadata or {}),
            "feature_schema_path": str(checkpoint_dir / "feature_schema.json"),
            "latest_updated": True,
        },
    )
    return checkpoint_dir


def load_training_checkpoint(checkpoint_dir: Path, *, model, optimizer=None, device: str | torch.device = "cpu") -> dict[str, Any]:
    checkpoint_dir = Path(checkpoint_dir)
    model_state = torch.load(
        checkpoint_dir / "model.pt",
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(model_state)
    if optimizer is not None:
        optimizer_state = torch.load(
            checkpoint_dir / "optimizer.pt",
            map_location=device,
            weights_only=True,
        )
        optimizer.load_state_dict(optimizer_state)
    with (checkpoint_dir / "metadata.json").open() as fh:
        import json

        metadata = json.load(fh)
    return metadata


def write_batch_summary(path: Path, payload: Mapping[str, Any]) -> None:
    write_json(path, payload)


def _policy_checkpoint_id_from_metadata(checkpoint_dir: Path) -> str | None:
    try:
        with (checkpoint_dir / "metadata.json").open() as fh:
            import json

            metadata = json.load(fh)
    except Exception:
        return None
    value = metadata.get("policy_checkpoint_id")
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _policy_checkpoint_id_from_sidecar(checkpoint_dir: Path) -> str | None:
    try:
        value = (checkpoint_dir / "policy_checkpoint_id").read_text(encoding="utf-8")
    except Exception:
        return None
    normalized = value.strip()
    return normalized or None
