"""Process-local RTS rollout runtime configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping


SUPPORTED_RTS_POLICY_MODES = (
    "current",
    "current_probe",
    "random_valid",
    "rts_rl_explicit",
)


@dataclass(frozen=True)
class RTSRuntimeConfig:
    policy_mode: str = "current"
    rollout_enabled: bool = False
    zone_ids: tuple[str, ...] = ()
    reward_reference_path: str | None = None
    random_seed: int | None = None
    max_events: int | None = None
    rollout_filename: str = "rts_rollout.jsonl"
    summary_filename: str = "rts_rollout_summary.json"

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "RTSRuntimeConfig":
        if data is None:
            return cls()
        payload = dict(data)
        if payload.get("zone_ids") is None:
            payload["zone_ids"] = ()
        else:
            payload["zone_ids"] = tuple(str(zone_id) for zone_id in payload["zone_ids"])
        config = cls(**payload)
        validate_rts_runtime_config(config)
        return config

    def to_json_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["zone_ids"] = list(self.zone_ids)
        return data


def validate_rts_runtime_config(config: RTSRuntimeConfig) -> None:
    if config.policy_mode not in SUPPORTED_RTS_POLICY_MODES:
        raise ValueError(f"unsupported RTS policy mode: {config.policy_mode!r}")
    if config.policy_mode in {"current_probe", "random_valid"} and not config.rollout_enabled:
        raise ValueError(f"{config.policy_mode} requires rollout_enabled=True")
    if config.max_events is not None and int(config.max_events) <= 0:
        raise ValueError("rts max_events must be positive when provided")
    if any(not str(zone_id).strip() for zone_id in config.zone_ids):
        raise ValueError("rts zone_ids must be nonblank")
    if len(set(config.zone_ids)) != len(config.zone_ids):
        raise ValueError("rts zone_ids must be unique")
    if not str(config.rollout_filename).strip():
        raise ValueError("rts rollout_filename must be nonblank")
    if not str(config.summary_filename).strip():
        raise ValueError("rts summary_filename must be nonblank")

