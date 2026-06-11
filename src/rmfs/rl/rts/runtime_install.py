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
        if runtime_root is None:
            raise RuntimeError("rts_rl_explicit requires a worker runtime_root")
        from .training.policy_actor import RTSOnPolicyActor, RTSOnPolicyActorConfig
        from .training.policy_loader import load_policy_from_checkpoint

        load_device = "cpu" if config.policy_device == "auto" else config.policy_device
        loaded = load_policy_from_checkpoint(Path(config.policy_checkpoint_dir), device=load_device)
        if loaded.policy_checkpoint_id != config.policy_checkpoint_id:
            raise RuntimeError(
                f"policy checkpoint id mismatch: loaded {loaded.policy_checkpoint_id!r}, expected {config.policy_checkpoint_id!r}"
            )
        inventory.rts_policy = RTSOnPolicyActor(
            model=loaded.model,
            zone_ids=config.zone_ids,
            config=RTSOnPolicyActorConfig(
                policy_checkpoint_id=config.policy_checkpoint_id,
                policy_action_mode=config.policy_action_mode,
                policy_device=load_device,
                feature_schema_id=config.policy_checkpoint_id,
            ),
        )
        inventory.rts_rollout_runtime = RTSRolloutRuntime(config=config, runtime_root=runtime_root)
        return inventory.rts_rollout_runtime

    raise RuntimeError(f"unsupported RTS policy mode: {config.policy_mode}")
