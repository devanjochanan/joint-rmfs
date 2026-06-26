"""Opt-in RTS evaluation policies."""

from __future__ import annotations

import random
from typing import Any, Sequence

from engine.netlogo_coordinate import NetLogoCoordinate
from src.rmfs.decisions.rts.types import RTSDecision

from .action_context import selected_context_by_index
from .action_space import build_action_mask_from_contexts, decode_action, validate_action_mask
from .state import build_state
from .storage_resolver import find_free_storage_in_zone
from .zone_registry import build_zone_registry


class RTSRandomValidPolicy:
    def select_action(self, zone_ids: Sequence[str], action_mask: Sequence[int], rng: random.Random) -> Any:
        mask = validate_action_mask(zone_ids, action_mask, require_valid=True)
        valid_indexes = [index for index, value in enumerate(mask) if value == 1]
        return decode_action(rng.choice(valid_indexes), zone_ids)


class RTSRandomValidStoragePolicy:
    def __init__(self, *, zone_ids: Sequence[str] = (), random_seed: int | None = None):
        self.zone_ids = tuple(zone_ids)
        self.rng = random.Random(random_seed)
        self.action_policy = RTSRandomValidPolicy()

    def select_destination(self, context: Any) -> RTSDecision:
        zones = self.zone_ids or infer_zone_ids_from_context(context)
        if not zones:
            raise RuntimeError("random_valid RTS policy requires zone_ids or inferable storage zones")
        state = build_state(context, zones)
        mask = build_action_mask_from_contexts(zones, state.action_contexts)
        for action_context in state.action_contexts:
            if action_context.branch != "store":
                mask[action_context.action_index] = 0
        action = self.action_policy.select_action(zones, mask, self.rng)
        action_context = selected_context_by_index(state.action_contexts, action.action_index)
        storage = action_context.candidate_storage or find_free_storage_in_zone(context, action.zone_id, action.branch)
        if storage is None:
            raise RuntimeError(
                f"random_valid selected {action.branch}:{action.zone_id}, but no free storage resolved"
            )
        cycle_estimate = (
            action_context.cycle_estimate.to_json_dict()
            if getattr(action_context, "cycle_estimate", None) is not None
            else None
        )
        destination = NetLogoCoordinate(storage.pos_x, storage.pos_y)
        return RTSDecision(
            storage=storage,
            destination=destination,
            policy_name="rts_random_valid",
            mode="nearest",
            reason="explicit random_valid RTS rollout evaluation mode",
            metadata={
                "selected_action_index": action.action_index,
                "selected_action_branch": action.branch,
                "selected_zone_id": action.zone_id,
                "action_context_id": action_context.context_id,
                "action_context_version": action_context.context_version,
                "candidate_storage_id": action_context.candidate_storage_id,
                "selected_cycle_estimate": cycle_estimate,
                "action_mask": list(mask),
                "zone_ids": list(zones),
            },
        )


def infer_zone_ids_from_context(context: Any) -> tuple[str, ...]:
    return build_zone_registry(context).zone_ids
