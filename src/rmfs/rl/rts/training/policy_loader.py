"""Explicit RTS policy checkpoint loader for rollout workers."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import torch

from src.rmfs.rl.rts.training.device import resolve_rts_torch_device
from src.rmfs.rl.rts.training.checkpoint import resolve_policy_checkpoint_id
from src.rmfs.rl.rts.model import RTSMaskedActorCritic
from src.rmfs.rl.rts.features import ACTION_FEATURE_SCHEMA_VERSION, build_action_feature_names, feature_schema_identity
from src.rmfs.rl.rts.action_space import ACTION_BRANCHES, ACTION_BRANCH_ORDER_VERSION
from src.rmfs.rl.rts.cycle_estimator import CYCLE_ESTIMATE_VERSION, SEMANTICS_HOST_STRUCTURAL
from src.rmfs.rl.rts.graph_distance import DISTANCE_SEMANTICS_VERSION
from src.rmfs.rl.rts.macro_region import MACRO_REGION_SEMANTIC_VERSION, TRAFFIC_PRESSURE_SEMANTIC_VERSION
from src.rmfs.rl.rts.reward import REWARD_HORIZON
from src.rmfs.rl.rts.travel_time import TIME_CONVERSION_VERSION, TRAVEL_TIME_VERSION
from src.rmfs.rl.rts.static_state_context import (
    DISTANCE_NORMALIZATION_VERSION,
    HISTORICAL_POD_RANK_VERSION,
    LAYOUT_NORMALIZATION_VERSION,
)
from src.rmfs.rl.rts.stock_features import STOCK_FEATURE_SCHEMA_VERSION, STOCK_SOURCE_VERSION, STOCK_FEATURE_NAMES
from src.rmfs.rl.rts.zone_features import SKU_SIMILARITY_VERSION
from src.rmfs.rl.rts.zone_registry import ZONE_GEOMETRY_VERSION


@dataclass(frozen=True)
class LoadedRTSPolicy:
    model: RTSMaskedActorCritic
    checkpoint_dir: Path
    policy_checkpoint_id: str
    feature_schema: dict
    metadata: dict


def load_policy_from_checkpoint(checkpoint_dir: Path, *, device: str = "cpu") -> LoadedRTSPolicy:
    checkpoint = Path(checkpoint_dir)
    resolved_device = resolve_rts_torch_device(device)
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
    expected_action_dim = len(build_action_feature_names(()))
    if int(feature_schema.get("action_feature_dim") or 0) != expected_action_dim:
        raise ValueError(
            "unsupported RTS checkpoint action_feature_schema_version/action_feature_dim; "
            f"runtime expects {ACTION_FEATURE_SCHEMA_VERSION} width {expected_action_dim}; "
            "fresh training or an explicitly authorized migration is required"
        )
    if int(feature_schema.get("stock_feature_dim") or 0) != 4:
        raise ValueError("RTS checkpoints must have stock_feature_dim=4")
    training_config = dict(metadata.get("training_config", {}) or {})
    model = RTSMaskedActorCritic(
        action_feature_dim=int(feature_schema["action_feature_dim"]),
        stock_feature_dim=int(feature_schema["stock_feature_dim"]),
        hidden_sizes=tuple(training_config.get("hidden_sizes", (64, 64))),
        stock_hidden_sizes=tuple(training_config.get("stock_hidden_sizes", (32, 32))),
        stock_embedding_dim=int(training_config.get("stock_embedding_dim", 16)),
    )
    model.load_state_dict(
        torch.load(model_path, map_location=resolved_device, weights_only=True)
    )
    model.to(resolved_device)
    model.eval()
    policy_checkpoint_id = resolve_policy_checkpoint_id(checkpoint)
    if not policy_checkpoint_id.strip():
        raise ValueError("policy_checkpoint_id must be nonblank")
    _validate_schema_semantics(feature_schema)
    expected_schema_id = feature_schema_identity(feature_schema)
    actual_schema_id = feature_schema.get("feature_schema_id")
    if actual_schema_id is not None and actual_schema_id != expected_schema_id:
        raise ValueError("incompatible RTS checkpoint feature_schema_id")
    feature_schema["feature_schema_id"] = expected_schema_id
    return LoadedRTSPolicy(
        model=model,
        checkpoint_dir=checkpoint,
        policy_checkpoint_id=policy_checkpoint_id,
        feature_schema=feature_schema,
        metadata=metadata,
    )

def _validate_schema_semantics(feature_schema: dict[str, Any]) -> None:
    reward_horizon = feature_schema.get("reward_horizon")
    if reward_horizon is not None and reward_horizon != REWARD_HORIZON:
        raise ValueError(
            f"unsupported RTS checkpoint reward_horizon: {reward_horizon!r}; expected {REWARD_HORIZON!r}"
        )
    distance_semantics = feature_schema.get("distance_semantics_version")
    if distance_semantics is not None and distance_semantics != DISTANCE_SEMANTICS_VERSION:
        raise ValueError(
            "unsupported RTS checkpoint distance_semantics_version: "
            f"{distance_semantics!r}; expected {DISTANCE_SEMANTICS_VERSION!r}"
        )
    required = {
        "action_feature_schema_version": ACTION_FEATURE_SCHEMA_VERSION,
        "stock_feature_schema_version": STOCK_FEATURE_SCHEMA_VERSION,
        "stock_source_version": STOCK_SOURCE_VERSION,
        "cycle_estimator_version": CYCLE_ESTIMATE_VERSION,
        "cycle_estimate_semantics": SEMANTICS_HOST_STRUCTURAL,
        "travel_time_version": TRAVEL_TIME_VERSION,
        "time_conversion_version": TIME_CONVERSION_VERSION,
        "layout_normalization_version": LAYOUT_NORMALIZATION_VERSION,
        "historical_pod_rank_version": HISTORICAL_POD_RANK_VERSION,
        "sku_similarity_version": SKU_SIMILARITY_VERSION,
        "distance_normalization_version": DISTANCE_NORMALIZATION_VERSION,
        "action_branch_order_version": ACTION_BRANCH_ORDER_VERSION,
        "zone_geometry_version": ZONE_GEOMETRY_VERSION,
        "macro_region_semantic_version": MACRO_REGION_SEMANTIC_VERSION,
        "traffic_pressure_semantic_version": TRAFFIC_PRESSURE_SEMANTIC_VERSION,
    }
    for key, expected in required.items():
        actual = feature_schema.get(key)
        if actual != expected:
            raise ValueError(f"incompatible RTS checkpoint {key}: {actual!r}; expected {expected!r}")
    if tuple(feature_schema.get("action_branch_order", []) or ()) != tuple(ACTION_BRANCHES):
        raise ValueError(
            "incompatible RTS checkpoint action_branch_order: "
            f"{feature_schema.get('action_branch_order')!r}; expected {list(ACTION_BRANCHES)!r}"
        )
    expected_action_names = build_action_feature_names(())
    actual_action_names = tuple(str(name) for name in feature_schema.get("action_feature_names", []) or ())
    if actual_action_names != expected_action_names:
        raise ValueError("incompatible RTS checkpoint action_feature_names order")
    actual_stock_names = tuple(str(name) for name in feature_schema.get("stock_feature_names", []) or ())
    if actual_stock_names != tuple(STOCK_FEATURE_NAMES):
        raise ValueError("incompatible RTS checkpoint stock_feature_names order")
