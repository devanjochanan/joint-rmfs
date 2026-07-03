"""Name-based RTS-RL feature and branch ablations."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping, Sequence

import numpy as np

from .action_space import STORE, decode_action


SUPPORTED_ABLATIONS: tuple[str, ...] = (
    "full",
    "no_next_job_distance",
    "no_macro_region_pressure",
    "no_zone_pressure",
    "no_source_picker_coordinates",
    "no_sku_similarity",
    "no_historical_request_rank",
    "no_stock_encoder",
    "store_only",
)

ACTION_FEATURE_ZERO_MAP: dict[str, tuple[str, ...]] = {
    "no_next_job_distance": ("candidate_to_proposed_next_pod_distance",),
    "no_macro_region_pressure": (
        "macro_region_present_robot_pressure",
        "macro_region_destination_robot_pressure",
    ),
    "no_zone_pressure": (
        "zone_present_robot_pressure",
        "zone_destination_robot_pressure",
        "selected_replenishment_station_destination_pressure",
    ),
    "no_source_picker_coordinates": ("source_picker_x_norm", "source_picker_y_norm"),
    "no_sku_similarity": ("sku_similarity_fraction",),
    "no_historical_request_rank": ("historical_pod_request_rank",),
}


@dataclass(frozen=True)
class AblationSpec:
    name: str
    action_zero_features: tuple[str, ...]
    zero_stock_encoder: bool
    store_only: bool
    hash: str

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "rts_feature_ablation.v1",
            "ablation_name": self.name,
            "action_zero_features": list(self.action_zero_features),
            "zero_stock_encoder": self.zero_stock_encoder,
            "store_only": self.store_only,
            "ablation_hash": self.hash,
        }


def resolve_ablation(name: str | None) -> AblationSpec:
    normalized = str(name or "full").strip() or "full"
    if normalized not in SUPPORTED_ABLATIONS:
        raise ValueError(f"unsupported RTS feature ablation: {normalized!r}")
    zero_features = ACTION_FEATURE_ZERO_MAP.get(normalized, ())
    zero_stock = normalized == "no_stock_encoder"
    store_only = normalized == "store_only"
    payload = {
        "schema_version": "rts_feature_ablation.v1",
        "ablation_name": normalized,
        "action_zero_features": list(zero_features),
        "zero_stock_encoder": zero_stock,
        "store_only": store_only,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    return AblationSpec(
        name=normalized,
        action_zero_features=tuple(zero_features),
        zero_stock_encoder=zero_stock,
        store_only=store_only,
        hash="rts_ablation_" + digest[:16],
    )


def apply_ablation_to_arrays(
    *,
    X_actions: np.ndarray,
    M_actions: np.ndarray,
    X_stock: np.ndarray,
    M_stock: np.ndarray,
    action_feature_names: Sequence[str],
    zone_ids: Sequence[str],
    ablation: AblationSpec,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    actions = np.array(X_actions, copy=True)
    action_mask = np.array(M_actions, copy=True)
    stocks = np.array(X_stock, copy=True)
    stock_mask = np.array(M_stock, copy=True)

    name_to_index = {str(name): idx for idx, name in enumerate(action_feature_names)}
    for feature_name in ablation.action_zero_features:
        if feature_name not in name_to_index:
            raise ValueError(f"ablation {ablation.name!r} references unknown feature {feature_name!r}")
        actions[..., name_to_index[feature_name]] = 0.0

    if ablation.zero_stock_encoder:
        stocks[...] = 0.0
        stock_mask[...] = 0

    if ablation.store_only:
        zones = tuple(str(zone) for zone in zone_ids)
        if action_mask.ndim == 1:
            _mask_store_only_row(action_mask, zones)
        elif action_mask.ndim == 2:
            for row in action_mask:
                _mask_store_only_row(row, zones)
        else:
            raise ValueError("store_only ablation expects 1D or 2D action mask")

    return actions, action_mask, stocks, stock_mask


def _mask_store_only_row(mask_row: np.ndarray, zone_ids: tuple[str, ...]) -> None:
    for idx in range(mask_row.shape[0]):
        if decode_action(idx, zone_ids).branch != STORE:
            mask_row[idx] = 0
    if not np.any(mask_row > 0):
        raise ValueError("store_only ablation left no valid store action")
