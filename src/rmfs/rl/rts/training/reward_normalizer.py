"""Cold-start reward normalization metadata for RTS on-policy training."""

from __future__ import annotations

import math
import statistics
from pathlib import Path
import json
from typing import Any, Mapping, Sequence


REWARD_MODE = "cold_start_realized_cycle_time"
REWARD_REFERENCE_REQUIRED = False
CYCLE_REFERENCE_ENABLED = False
ALPHA_ENABLED = False
DEFAULT_REWARD_TIME_SCALE = 1.0


def default_reward_normalizer_metadata(
    *,
    reward_time_scale: float | None = None,
    reward_time_scale_source: str | None = None,
    reward_valid_cycle_count: int = 0,
) -> dict[str, Any]:
    return {
        "reward_mode": REWARD_MODE,
        "reward_reference_required": REWARD_REFERENCE_REQUIRED,
        "cycle_reference_enabled": CYCLE_REFERENCE_ENABLED,
        "alpha_enabled": ALPHA_ENABLED,
        "reward_time_scale": float(reward_time_scale) if reward_time_scale is not None else None,
        "reward_time_scale_source": reward_time_scale_source,
        "reward_valid_cycle_count": int(reward_valid_cycle_count),
    }


def load_reward_normalizer_metadata(checkpoint_dir: Path | None) -> dict[str, Any]:
    if checkpoint_dir is None:
        return default_reward_normalizer_metadata()
    metadata_path = Path(checkpoint_dir) / "metadata.json"
    if not metadata_path.exists():
        return default_reward_normalizer_metadata()
    try:
        with metadata_path.open() as fh:
            metadata = json.load(fh)
    except Exception:
        return default_reward_normalizer_metadata()
    reward_metadata = metadata.get("reward_normalizer") or metadata
    scale = _finite_positive_or_none(reward_metadata.get("reward_time_scale"))
    return default_reward_normalizer_metadata(
        reward_time_scale=scale,
        reward_time_scale_source=reward_metadata.get("reward_time_scale_source"),
        reward_valid_cycle_count=int(reward_metadata.get("reward_valid_cycle_count") or 0),
    )


def derive_reward_normalizer_from_events(
    events: Sequence[Mapping[str, Any]],
    *,
    batch_id: int | None = None,
) -> dict[str, Any]:
    cycles = valid_realized_cycle_times(events)
    if cycles:
        scale = float(statistics.median(cycles))
        source = f"batch_{int(batch_id):06d}_valid_cycles" if batch_id is not None else "valid_cycles"
        return default_reward_normalizer_metadata(
            reward_time_scale=scale,
            reward_time_scale_source=source,
            reward_valid_cycle_count=len(cycles),
        )
    source = f"batch_{int(batch_id):06d}_fallback_no_valid_cycles" if batch_id is not None else "fallback_no_valid_cycles"
    return default_reward_normalizer_metadata(
        reward_time_scale=DEFAULT_REWARD_TIME_SCALE,
        reward_time_scale_source=source,
        reward_valid_cycle_count=0,
    )


def choose_reward_metadata_for_batch(
    *,
    previous_metadata: Mapping[str, Any] | None,
    derived_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    previous_scale = _finite_positive_or_none((previous_metadata or {}).get("reward_time_scale"))
    if previous_scale is not None:
        return default_reward_normalizer_metadata(
            reward_time_scale=previous_scale,
            reward_time_scale_source=(previous_metadata or {}).get("reward_time_scale_source") or "checkpoint_metadata",
            reward_valid_cycle_count=int((previous_metadata or {}).get("reward_valid_cycle_count") or 0),
        )
    return default_reward_normalizer_metadata(
        reward_time_scale=_finite_positive_or_none(derived_metadata.get("reward_time_scale")) or DEFAULT_REWARD_TIME_SCALE,
        reward_time_scale_source=derived_metadata.get("reward_time_scale_source"),
        reward_valid_cycle_count=int(derived_metadata.get("reward_valid_cycle_count") or 0),
    )


def apply_cold_start_rewards(
    events: Sequence[Mapping[str, Any]],
    *,
    reward_metadata: Mapping[str, Any],
) -> list[dict[str, Any]]:
    scale = _finite_positive_or_none(reward_metadata.get("reward_time_scale")) or DEFAULT_REWARD_TIME_SCALE
    normalized: list[dict[str, Any]] = []
    for row in events:
        item = dict(row)
        if item.get("event_type") == "outcome":
            realized = _finite_positive_or_none(item.get("realized_cycle_time"))
            if realized is not None:
                reward_json = dict(item.get("reward_json") or {})
                if not reward_json.get("reward_computed"):
                    item["reward_json"] = cold_start_reward_json(
                        realized_cycle_time=realized,
                        reward_time_scale=scale,
                        reward_time_scale_source=reward_metadata.get("reward_time_scale_source"),
                        reward_valid_cycle_count=int(reward_metadata.get("reward_valid_cycle_count") or 0),
                    )
        normalized.append(item)
    return normalized


def cold_start_reward_json(
    *,
    realized_cycle_time: float,
    reward_time_scale: float,
    reward_time_scale_source: str | None,
    reward_valid_cycle_count: int,
) -> dict[str, Any]:
    scale = _finite_positive_or_none(reward_time_scale) or DEFAULT_REWARD_TIME_SCALE
    cycle_time = _finite_positive_or_none(realized_cycle_time)
    if cycle_time is None:
        return {
            **default_reward_normalizer_metadata(),
            "reward_computed": False,
            "reward_value": None,
            "normalized_cycle_time": None,
        }
    normalized_cycle_time = float(cycle_time / scale)
    return {
        **default_reward_normalizer_metadata(
            reward_time_scale=scale,
            reward_time_scale_source=reward_time_scale_source,
            reward_valid_cycle_count=reward_valid_cycle_count,
        ),
        "reward_computed": True,
        "reward_value": -normalized_cycle_time,
        "normalized_cycle_time": normalized_cycle_time,
        "realized_cycle_time": float(cycle_time),
    }


def pending_cold_start_reward_json(*, realized_cycle_time: float | None = None) -> dict[str, Any]:
    payload = {
        **default_reward_normalizer_metadata(),
        "reward_computed": False,
        "reward_value": None,
        "normalized_cycle_time": None,
    }
    if realized_cycle_time is not None:
        payload["realized_cycle_time"] = realized_cycle_time
    return payload


def valid_realized_cycle_times(events: Sequence[Mapping[str, Any]]) -> list[float]:
    cycles: list[float] = []
    for row in events:
        if row.get("event_type") != "outcome":
            continue
        cycle_time = _finite_positive_or_none(row.get("realized_cycle_time"))
        if cycle_time is not None:
            cycles.append(cycle_time)
    return cycles


def _finite_positive_or_none(value: Any) -> float | None:
    try:
        result = float(value)
    except Exception:
        return None
    if not math.isfinite(result) or result <= 0.0:
        return None
    return result
