"""Lineage metadata helpers for synthetic RTS training artifacts."""

from __future__ import annotations

import datetime
import sys
from pathlib import Path
from typing import Any, Mapping

import torch

from .metrics import atomic_write_json, json_safe


def build_lineage_metadata(
    *,
    artifact_label: str,
    batch_id: int,
    config: Any,
    dataset_summary: Mapping[str, Any],
    ppo_update_result: Any,
    feature_schema: Mapping[str, Any],
    cycle_reference_path: str | None,
    cycle_reference_source: str | None,
    branch: str | None = None,
    commit: str | None = None,
    python_executable: str | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    return json_safe(
        {
            "artifact_label": artifact_label,
            "batch_id": int(batch_id),
            "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "branch": branch,
            "commit": commit,
            "python_executable": python_executable or sys.executable,
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device": device,
            "training_config": config,
            "dataset_summary": dataset_summary,
            "ppo_update_result": ppo_update_result,
            "feature_schema": feature_schema,
            "cycle_reference_path": cycle_reference_path,
            "cycle_reference_source": cycle_reference_source,
        }
    )


def write_lineage_json(path: Path, metadata: Mapping[str, Any]) -> None:
    atomic_write_json(path, metadata)


def build_batch_summary(
    *,
    artifact_label: str,
    batch_id: int,
    checkpoint_dir: Path,
    dataset_summary: Mapping[str, Any],
    ppo_update_result: Any,
) -> dict[str, Any]:
    return json_safe(
        {
            "artifact_label": artifact_label,
            "batch_id": int(batch_id),
            "checkpoint_dir": str(checkpoint_dir),
            "dataset_summary": dataset_summary,
            "ppo_update_result": ppo_update_result,
        }
    )

