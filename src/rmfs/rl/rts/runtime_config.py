"""Process-local RTS rollout runtime configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

from .zone_registry import validate_no_col_zone_ids
from .cycle_estimator import SUPPORTED_CYCLE_ESTIMATE_SEMANTICS, SEMANTICS_HOST_STRUCTURAL


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
    policy_checkpoint_dir: str | None = None
    policy_checkpoint_id: str | None = None
    policy_action_mode: str = "sample"
    policy_device: str = "cpu"
    feature_ablation: str = "full"
    feature_ablation_hash: str | None = None
    committed_next_reservations_enabled: bool = False
    cycle_estimate_semantics: str = SEMANTICS_HOST_STRUCTURAL
    cycle_estimate_allow_metric_fallback: bool = True

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
    if config.policy_action_mode not in {"sample", "greedy"}:
        raise ValueError("rts policy_action_mode must be sample or greedy")
    if config.policy_device not in {"cpu", "cuda", "auto"}:
        raise ValueError("rts policy_device must be cpu, cuda, or auto")
    from .ablation import resolve_ablation
    ablation = resolve_ablation(config.feature_ablation)
    if config.feature_ablation_hash is not None and config.feature_ablation_hash != ablation.hash:
        raise ValueError("rts feature_ablation_hash does not match feature_ablation")
    if config.cycle_estimate_semantics not in SUPPORTED_CYCLE_ESTIMATE_SEMANTICS:
        raise ValueError(
            "rts cycle_estimate_semantics must be one of "
            f"{sorted(SUPPORTED_CYCLE_ESTIMATE_SEMANTICS)}"
        )
    if config.policy_mode == "rts_rl_explicit":
        if not config.rollout_enabled:
            raise ValueError("rts_rl_explicit requires rollout_enabled=True")
        if not config.committed_next_reservations_enabled:
            raise ValueError("rts_rl_explicit requires committed_next_reservations_enabled=True")
        if not config.policy_checkpoint_dir:
            raise ValueError("rts_rl_explicit requires policy_checkpoint_dir")
        if not config.policy_checkpoint_id:
            raise ValueError("rts_rl_explicit requires policy_checkpoint_id")
        if config.zone_ids and "auto" in config.zone_ids and config.zone_ids != ("auto",):
            raise ValueError("rts zone_ids must be exactly ('auto',) or an explicit zone list")
    if config.max_events is not None and int(config.max_events) <= 0:
        raise ValueError("rts max_events must be positive when provided")
    if any(not str(zone_id).strip() for zone_id in config.zone_ids):
        raise ValueError("rts zone_ids must be nonblank")
    if len(set(config.zone_ids)) != len(config.zone_ids):
        raise ValueError("rts zone_ids must be unique")
    if config.policy_mode == "rts_rl_explicit" and config.zone_ids != ("auto",):
        validate_no_col_zone_ids(config.zone_ids, context="rts_rl_explicit")
    if not str(config.rollout_filename).strip():
        raise ValueError("rts rollout_filename must be nonblank")
    if not str(config.summary_filename).strip():
        raise ValueError("rts summary_filename must be nonblank")
