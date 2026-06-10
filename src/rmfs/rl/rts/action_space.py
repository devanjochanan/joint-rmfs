"""RTS-RL joint action space and masks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence


STORE = "store"
REPLENISH_STORE = "replenish_store"
ACTION_BRANCHES = (STORE, REPLENISH_STORE)


@dataclass(frozen=True)
class RTSAction:
    branch: str
    zone_id: str
    action_index: int


def normalize_zone_ids(zone_ids: Sequence[str]) -> tuple[str, ...]:
    normalized = tuple(str(zone_id).strip() for zone_id in zone_ids)
    if not normalized:
        raise ValueError("RTS action space requires at least one zone")
    if any(not zone_id for zone_id in normalized):
        raise ValueError("RTS zone ids must be nonblank")
    if len(set(normalized)) != len(normalized):
        raise ValueError("RTS zone ids must be unique")
    return normalized


def action_space_size(zone_ids: Sequence[str]) -> int:
    return 2 * len(normalize_zone_ids(zone_ids))


def encode_action(branch: str, zone_id: str, zone_ids: Sequence[str]) -> int:
    zones = normalize_zone_ids(zone_ids)
    normalized_branch = str(branch).strip()
    normalized_zone = str(zone_id).strip()
    if normalized_branch not in ACTION_BRANCHES:
        raise ValueError(f"unknown RTS action branch: {branch!r}")
    if normalized_zone not in zones:
        raise ValueError(f"unknown RTS action zone: {zone_id!r}")
    offset = zones.index(normalized_zone)
    if normalized_branch == STORE:
        return offset
    return len(zones) + offset


def decode_action(action_index: int, zone_ids: Sequence[str]) -> RTSAction:
    zones = normalize_zone_ids(zone_ids)
    index = int(action_index)
    count = 2 * len(zones)
    if index < 0 or index >= count:
        raise ValueError(f"RTS action index out of bounds: {index} for {count} actions")
    if index < len(zones):
        return RTSAction(branch=STORE, zone_id=zones[index], action_index=index)
    return RTSAction(
        branch=REPLENISH_STORE,
        zone_id=zones[index - len(zones)],
        action_index=index,
    )


def build_action_mask(
    zone_ids: Sequence[str],
    *,
    store_valid_by_zone: Mapping[str, bool],
    replenish_valid_by_zone: Mapping[str, bool],
) -> list[int]:
    zones = normalize_zone_ids(zone_ids)
    mask = [1 if bool(store_valid_by_zone.get(zone_id, False)) else 0 for zone_id in zones]
    mask.extend(1 if bool(replenish_valid_by_zone.get(zone_id, False)) else 0 for zone_id in zones)
    return mask


def action_mask_entry(action_index: int, zone_ids: Sequence[str], action_mask: Sequence[int]) -> int:
    validate_action_mask(zone_ids, action_mask, require_valid=False)
    index = int(action_index)
    if index < 0 or index >= len(action_mask):
        raise ValueError(f"RTS action index out of bounds: {index}")
    return int(action_mask[index])


def build_empty_action_mask(zone_ids: Sequence[str]) -> list[int]:
    return [0] * action_space_size(zone_ids)


def validate_action_mask(
    zone_ids: Sequence[str],
    action_mask: Sequence[int],
    *,
    require_valid: bool = True,
) -> tuple[int, ...]:
    expected = action_space_size(zone_ids)
    normalized = tuple(1 if int(value) > 0 else 0 for value in action_mask)
    if len(normalized) != expected:
        raise ValueError(f"RTS action mask length mismatch: {len(normalized)} != {expected}")
    if require_valid and not any(normalized):
        raise ValueError("RTS action mask contains no valid actions")
    return normalized
