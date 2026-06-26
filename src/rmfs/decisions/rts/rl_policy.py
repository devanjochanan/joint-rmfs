"""Opt-in RTS-RL policy adapter.

This adapter is intentionally not wired into the simulator default.
It requires an explicit model and a safe zone-to-storage resolver.
"""

from __future__ import annotations

from typing import Callable, Sequence

from engine.netlogo_coordinate import NetLogoCoordinate
from src.rmfs.decisions.rts.types import RTSDestinationContext, RTSDecision
from src.rmfs.rl.rts.action_context import revalidate_selected_context, selected_context_by_index
from src.rmfs.rl.rts.action_space import build_action_mask_from_contexts, validate_action_mask
from src.rmfs.rl.rts.features import build_feature_bundle
from src.rmfs.rl.rts.inference import run_inference
from src.rmfs.rl.rts.state import build_state

ZoneToStorageResolver = Callable[[RTSDestinationContext, str, str], object | None]


class RTSRLPolicy:
    policy_name = "rts_rl_opt_in"

    def __init__(
        self,
        *,
        model=None,
        zone_ids: Sequence[str] | None = None,
        zone_to_storage_resolver: ZoneToStorageResolver | None = None,
        device: str = "auto",
    ) -> None:
        self.model = model
        self.zone_ids = tuple(str(zone_id) for zone_id in (zone_ids or ()))
        self.zone_to_storage_resolver = zone_to_storage_resolver
        self.device = device

    def select_destination(self, context: RTSDestinationContext) -> RTSDecision:
        if self.model is None:
            raise RuntimeError("RTSRLPolicy requires an explicit model; no checkpoints are loaded automatically")
        if not self.zone_ids:
            raise RuntimeError("RTSRLPolicy requires explicit zone_ids")
        if self.zone_to_storage_resolver is None:
            raise RuntimeError("RTSRLPolicy requires a safe zone_to_storage_resolver to map zones to concrete storage")

        state = build_state(context, self.zone_ids)
        mask = build_action_mask_from_contexts(self.zone_ids, state.action_contexts)
        validate_action_mask(self.zone_ids, mask, require_valid=True)
        features = build_feature_bundle(self.zone_ids, mask, state.state_json)
        result = run_inference(
            self.model,
            action_features=features.X_actions,
            action_mask=features.M_actions,
            stock_features=features.X_stock,
            stock_mask=features.M_stock,
            zone_ids=features.zone_ids,
            mode="greedy",
            device=self.device,
        )
        action_context = revalidate_selected_context(
            context,
            selected_context_by_index(state.action_contexts, result.selected_action_index),
        )
        storage = action_context.candidate_storage
        if storage is None and self.zone_to_storage_resolver is not None:
            storage = self.zone_to_storage_resolver(context, result.action.zone_id, result.action.branch)
        if storage is None:
            raise RuntimeError(f"RTSRLPolicy resolver returned no storage for zone {result.action.zone_id!r}")
        proposal = action_context.next_job_proposal
        destination = NetLogoCoordinate(storage.pos_x, storage.pos_y)
        return RTSDecision(
            storage=storage,
            destination=destination,
            policy_name=self.policy_name,
            mode="rl",
            action_index=result.selected_action_index,
            branch=result.action.branch,
            zone_id=result.action.zone_id,
            storage_id=action_context.candidate_storage_id,
            replenishment_station=action_context.replenishment_station,
            replenishment_station_id=action_context.replenishment_station_id,
            action_context_id=action_context.context_id,
            action_context_version=action_context.context_version,
            next_job_proposal_id=getattr(proposal, "proposal_id", None),
            reason="opt-in RTS-RL selected a zone and resolver mapped it to storage",
            metadata={
                "selected_action_index": result.selected_action_index,
                "selected_action_branch": result.action.branch,
                "selected_branch": result.action.branch,
                "selected_zone_id": result.action.zone_id,
                "action_mask": list(mask),
                "action_context_id": action_context.context_id,
                "action_context_version": action_context.context_version,
                "candidate_storage_id": action_context.candidate_storage_id,
                "selected_replenishment_station_id": action_context.replenishment_station_id,
                "validity_reason_codes": list(action_context.invalid_reason_codes),
                "next_job_proposal_id": getattr(proposal, "proposal_id", None),
                "state_json": state.state_json,
                "value_estimate": result.value_estimate,
            },
        )
