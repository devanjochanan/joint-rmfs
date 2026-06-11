"""Explicit on-policy RTS actor used only by training/evaluation workers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch

from engine.netlogo_coordinate import NetLogoCoordinate
from src.rmfs.decisions.rts.types import RTSDecision
from src.rmfs.rl.rts.action_space import build_action_mask, decode_action
from src.rmfs.rl.rts.features import build_feature_bundle
from src.rmfs.rl.rts.state import build_state
from src.rmfs.rl.rts.storage_resolver import find_free_storage_in_zone
from src.rmfs.rl.rts.training.device import resolve_rts_torch_device
from src.rmfs.rl.rts.training.ppo import masked_categorical_from_logits


@dataclass(frozen=True)
class RTSOnPolicyActorConfig:
    policy_checkpoint_id: str
    policy_action_mode: str = "sample"
    policy_device: str = "cpu"
    feature_schema_id: str | None = None


class RTSOnPolicyActor:
    def __init__(self, *, model, zone_ids: Sequence[str], config: RTSOnPolicyActorConfig):
        if config.policy_action_mode not in {"sample", "greedy"}:
            raise ValueError("policy_action_mode must be sample or greedy")
        if not config.policy_checkpoint_id.strip():
            raise ValueError("policy_checkpoint_id must be nonblank")
        self.model = model
        self.zone_ids = tuple(str(zone_id) for zone_id in zone_ids)
        if not self.zone_ids:
            raise ValueError("rts_rl_explicit actor requires zone_ids")
        self.config = config

    def select_destination(self, context: Any) -> RTSDecision:
        zones = self.zone_ids
        state = build_state(context, zones)
        store_valid = {row["zone_id"]: bool(row["store_action_valid"]) for row in state.state_json["zone_rows"]}
        repl_valid = {row["zone_id"]: bool(row["replenish_store_action_valid"]) for row in state.state_json["zone_rows"]}
        action_mask = build_action_mask(zones, store_valid_by_zone=store_valid, replenish_valid_by_zone=repl_valid)
        features = build_feature_bundle(zones, action_mask, state.state_json)
        device_str = resolve_rts_torch_device(self.config.policy_device)
        device = torch.device(device_str)
        with torch.no_grad():
            X_actions = torch.as_tensor(features.X_actions, dtype=torch.float32, device=device).unsqueeze(0)
            M_actions = torch.as_tensor(features.M_actions, dtype=torch.int64, device=device).unsqueeze(0)
            X_stock = torch.as_tensor(features.X_stock, dtype=torch.float32, device=device).unsqueeze(0)
            M_stock = torch.as_tensor(features.M_stock, dtype=torch.int64, device=device).unsqueeze(0)
            logits, values = self.model(X_actions, M_actions, X_stock, M_stock)
            dist = masked_categorical_from_logits(logits, M_actions)
            if self.config.policy_action_mode == "greedy":
                selected = int(logits.masked_fill(M_actions <= 0, torch.finfo(logits.dtype).min).argmax(dim=-1).item())
            else:
                selected = int(dist.sample().item())
            selected_tensor = torch.as_tensor([selected], dtype=torch.int64, device=device)
            old_log_prob = float(dist.log_prob(selected_tensor).item())
            old_value = float(values.squeeze(0).item())
            policy_entropy = float(dist.entropy().squeeze(0).item())
        action = decode_action(selected, zones)
        storage = find_free_storage_in_zone(context, action.zone_id, action.branch)
        if storage is None:
            raise RuntimeError(f"rts_rl_explicit selected {action.branch}:{action.zone_id}, but no storage resolved")
        return RTSDecision(
            storage=storage,
            destination=NetLogoCoordinate(storage.pos_x, storage.pos_y),
            policy_name="rts_rl_explicit",
            mode="rl",
            reason="explicit RTS-RL on-policy actor",
            metadata={
                "actor_kind": "rts_rl_explicit",
                "policy_checkpoint_id": self.config.policy_checkpoint_id,
                "policy_mode": self.config.policy_action_mode,
                "old_log_prob": old_log_prob,
                "old_value": old_value,
                "policy_entropy": policy_entropy,
                "selected_action_index": selected,
                "selected_action_branch": action.branch,
                "selected_zone_id": action.zone_id,
                "feature_schema_id": self.config.feature_schema_id or self.config.policy_checkpoint_id,
            },
        )

