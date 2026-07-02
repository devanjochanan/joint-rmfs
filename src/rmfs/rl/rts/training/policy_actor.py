"""Explicit on-policy RTS actor used only by training/evaluation workers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch

from engine.netlogo_coordinate import NetLogoCoordinate
from src.rmfs.decisions.rts.types import RTSDecision
from src.rmfs.rl.rts.action_context import revalidate_selected_context, selected_context_by_index
from src.rmfs.rl.rts.action_space import build_action_mask_from_contexts, decode_action, validate_action_mask
from src.rmfs.rl.rts.ablation import AblationSpec, resolve_ablation, apply_ablation_to_arrays
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
    feature_ablation: str = "full"
    feature_ablation_hash: str | None = None


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
        self.ablation: AblationSpec = resolve_ablation(config.feature_ablation)
        if config.feature_ablation_hash is not None and config.feature_ablation_hash != self.ablation.hash:
            raise ValueError("feature_ablation_hash does not match feature_ablation")
        self.config = config

    def select_destination(self, context: Any) -> RTSDecision:
        import time
        t_total_start = time.perf_counter()

        zones = self.zone_ids

        # 1. build_state_ms
        t_state_start = time.perf_counter()
        state = build_state(context, zones)
        t_state_end = time.perf_counter()
        build_state_ms = max(0.0, (t_state_end - t_state_start) * 1000.0)

        # 2. build_feature_bundle_ms
        t_feature_start = time.perf_counter()
        action_mask = build_action_mask_from_contexts(zones, state.action_contexts)
        validate_action_mask(zones, action_mask, require_valid=True)
        features = build_feature_bundle(zones, action_mask, state.state_json)
        X_actions, M_actions, X_stock, M_stock = apply_ablation_to_arrays(
            X_actions=features.X_actions,
            M_actions=features.M_actions,
            X_stock=features.X_stock,
            M_stock=features.M_stock,
            action_feature_names=features.action_feature_names,
            zone_ids=zones,
            ablation=self.ablation,
        )
        t_feature_end = time.perf_counter()
        build_feature_bundle_ms = max(0.0, (t_feature_end - t_feature_start) * 1000.0)

        # 3. tensor_and_forward_ms
        t_forward_start = time.perf_counter()
        device_str = resolve_rts_torch_device(self.config.policy_device)
        device = torch.device(device_str)
        with torch.no_grad():
            X_actions_tensor = torch.as_tensor(X_actions, dtype=torch.float32, device=device).unsqueeze(0)
            M_actions_tensor = torch.as_tensor(M_actions, dtype=torch.int64, device=device).unsqueeze(0)
            X_stock_tensor = torch.as_tensor(X_stock, dtype=torch.float32, device=device).unsqueeze(0)
            M_stock_tensor = torch.as_tensor(M_stock, dtype=torch.int64, device=device).unsqueeze(0)
            logits, values = self.model(X_actions_tensor, M_actions_tensor, X_stock_tensor, M_stock_tensor)
            dist = masked_categorical_from_logits(logits, M_actions_tensor)
            if self.config.policy_action_mode == "greedy":
                selected = int(logits.masked_fill(M_actions_tensor <= 0, torch.finfo(logits.dtype).min).argmax(dim=-1).item())
            else:
                selected = int(dist.sample().item())
            selected_tensor = torch.as_tensor([selected], dtype=torch.int64, device=device)
            old_log_prob = float(dist.log_prob(selected_tensor).item())
            old_value = float(values.squeeze(0).item())
            policy_entropy = float(dist.entropy().squeeze(0).item())
        t_forward_end = time.perf_counter()
        tensor_and_forward_ms = max(0.0, (t_forward_end - t_forward_start) * 1000.0)

        # 4. selected_context_revalidation_ms
        t_reval_start = time.perf_counter()
        action = decode_action(selected, zones)
        action_context = revalidate_selected_context(context, selected_context_by_index(state.action_contexts, selected))
        storage = action_context.candidate_storage
        if storage is None:
            raise RuntimeError(f"rts_rl_explicit selected {action.branch}:{action.zone_id}, but no storage resolved")
        proposal = action_context.next_job_proposal
        proposal_has_next_job = bool(getattr(proposal, "has_next_job", False))
        cycle_estimate = (
            action_context.cycle_estimate.to_json_dict()
            if getattr(action_context, "cycle_estimate", None) is not None
            else None
        )
        t_reval_end = time.perf_counter()
        selected_context_revalidation_ms = max(0.0, (t_reval_end - t_reval_start) * 1000.0)

        t_total_end = time.perf_counter()
        total_select_destination_ms = max(0.0, (t_total_end - t_total_start) * 1000.0)

        timing = dict(state.timing or {})
        timing["build_state_ms"] = build_state_ms
        timing["feature_bundle_ms"] = build_feature_bundle_ms
        timing["build_feature_bundle_ms"] = build_feature_bundle_ms
        timing["tensor_and_forward_ms"] = tensor_and_forward_ms
        timing["selected_revalidation_ms"] = selected_context_revalidation_ms
        timing["selected_context_revalidation_ms"] = selected_context_revalidation_ms
        timing["total_select_destination_ms"] = total_select_destination_ms

        return RTSDecision(
            storage=storage,
            destination=NetLogoCoordinate(storage.pos_x, storage.pos_y),
            policy_name="rts_rl_explicit",
            mode="rl",
            action_index=selected,
            branch=action.branch,
            zone_id=action.zone_id,
            storage_id=action_context.candidate_storage_id,
            replenishment_station=action_context.replenishment_station,
            replenishment_station_id=action_context.replenishment_station_id,
            action_context_id=action_context.context_id,
            action_context_version=action_context.context_version,
            next_job_proposal_id=getattr(proposal, "proposal_id", None),
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
                "raw_action_mask": list(action_mask),
                "action_mask": [int(value) for value in M_actions.tolist()],
                "action_context_id": action_context.context_id,
                "action_context_version": action_context.context_version,
                "candidate_storage_id": action_context.candidate_storage_id,
                "selected_replenishment_station_id": action_context.replenishment_station_id,
                "validity_reason_codes": list(action_context.invalid_reason_codes),
                "next_job_proposal_id": getattr(proposal, "proposal_id", None),
                "selected_proposal_id": getattr(proposal, "proposal_id", None),
                "selected_proposal_has_next_job": proposal_has_next_job,
                "selected_proposal_job_id": getattr(proposal, "job_id", None) if proposal_has_next_job else None,
                "selected_proposal_pod_id": getattr(proposal, "pod_id", None) if proposal_has_next_job else None,
                "selected_proposal_picker_id": getattr(proposal, "picking_station_id", None) if proposal_has_next_job else None,
                "selected_proposal_candidate_count": int(getattr(proposal, "candidate_count", 0) or 0),
                "selected_cycle_estimate": cycle_estimate,
                "state_json": state.state_json,
                "state_capture_timing": "actor_decision_state",
                "feature_schema_id": self.config.feature_schema_id or self.config.policy_checkpoint_id,
                "feature_ablation": self.ablation.name,
                "feature_ablation_hash": self.ablation.hash,
                "feature_ablation_specification": self.ablation.to_json_dict(),
                "rts_decision_timing": timing,
            },
        )
