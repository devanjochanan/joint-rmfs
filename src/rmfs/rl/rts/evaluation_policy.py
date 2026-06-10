"""Opt-in RTS evaluation policies."""

from __future__ import annotations

import random
from typing import Any, Sequence

from engine.netlogo_coordinate import NetLogoCoordinate
from src.rmfs.decisions.rts.types import RTSDecision

from .action_space import build_action_mask, decode_action, validate_action_mask
from .state import build_state
from .storage_resolver import find_free_storage_in_zone
from .zone_features import infer_zone_id


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
        store_valid = {row["zone_id"]: bool(row["store_action_valid"]) for row in state.state_json["zone_rows"]}
        repl_valid = {row["zone_id"]: bool(row["replenish_store_action_valid"]) for row in state.state_json["zone_rows"]}
        mask = build_action_mask(zones, store_valid_by_zone=store_valid, replenish_valid_by_zone=repl_valid)
        action = self.action_policy.select_action(zones, mask, self.rng)
        storage = find_free_storage_in_zone(context, action.zone_id, action.branch)
        if storage is None:
            raise RuntimeError(
                f"random_valid selected {action.branch}:{action.zone_id}, but no free storage resolved"
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
                "action_mask": list(mask),
                "zone_ids": list(zones),
            },
        )


def infer_zone_ids_from_context(context: Any) -> tuple[str, ...]:
    storage_manager = getattr(getattr(context, "warehouse", None), "storage_manager", None)
    storages = list(getattr(storage_manager, "storages", []) or [])
    zones = sorted({infer_zone_id(storage) for storage in storages})
    return tuple(zones)

