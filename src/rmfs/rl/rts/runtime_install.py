"""Inventory installer for opt-in RTS rollout/evaluation runtime."""

from __future__ import annotations

from pathlib import Path

from .evaluation_policy import RTSRandomValidStoragePolicy
from .outcome_tracker import NoopRTSRolloutRuntime, RTSRolloutRuntime
from .runtime_config import RTSRuntimeConfig, validate_rts_runtime_config


def install_rts_runtime(inventory, config: RTSRuntimeConfig, runtime_root: Path | None):
    validate_rts_runtime_config(config)
    inventory.rts_rollout_runtime = NoopRTSRolloutRuntime()

    if config.policy_mode == "current":
        if config.rollout_enabled:
            if runtime_root is None:
                raise RuntimeError("RTS rollout logging requires a worker runtime_root")
            inventory.rts_rollout_runtime = RTSRolloutRuntime(config=config, runtime_root=runtime_root)
        return inventory.rts_rollout_runtime

    if config.policy_mode == "current_probe":
        if runtime_root is None:
            raise RuntimeError("current_probe RTS rollout requires a worker runtime_root")
        inventory.rts_rollout_runtime = RTSRolloutRuntime(config=config, runtime_root=runtime_root)
        return inventory.rts_rollout_runtime

    if config.policy_mode == "random_valid":
        if runtime_root is None:
            raise RuntimeError("random_valid RTS rollout requires a worker runtime_root")
        inventory.rts_policy = RTSRandomValidStoragePolicy(
            zone_ids=config.zone_ids,
            random_seed=config.random_seed,
        )
        inventory.rts_rollout_runtime = RTSRolloutRuntime(config=config, runtime_root=runtime_root)
        return inventory.rts_rollout_runtime

    if config.policy_mode == "rts_rl_explicit":
        raise RuntimeError("rts_rl_explicit requires explicit Python model/resolver installation")

    raise RuntimeError(f"unsupported RTS policy mode: {config.policy_mode}")

