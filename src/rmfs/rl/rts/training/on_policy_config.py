"""On-policy RTS PPO controller configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class RTSOnPolicyTrainingConfig:
    artifact_label: str
    output_root: Path
    batches: int
    workers: int
    netlogo_steps_per_run: int
    seed: int
    cycle_reference_path: Path
    device: str = "auto"
    worker_device: str = "cpu"
    policy_action_mode: str = "sample"
    progress: bool | None = None
    tensorboard_enabled: bool = False
    min_trainable_steps: int = 1
    ppo_epochs: int = 4
    minibatch_size: int = 64
    zone_ids: tuple[str, ...] = ()
    feature_flags: dict[str, Any] | None = None
    scenario_id: str | None = None
    experiment_id: str | None = None
    branch: str | None = None
    commit: str | None = None
    python_executable: str | None = None

    def to_json_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["output_root"] = str(self.output_root)
        data["cycle_reference_path"] = str(self.cycle_reference_path)
        data["zone_ids"] = list(self.zone_ids)
        data["seed_base"] = self.seed
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RTSOnPolicyTrainingConfig":
        import inspect
        payload = dict(data)
        payload["output_root"] = Path(payload["output_root"])
        payload["cycle_reference_path"] = Path(payload["cycle_reference_path"])
        if "zone_ids" in payload:
            payload["zone_ids"] = tuple(payload["zone_ids"])
        
        # Handle seed_base/seed alignment
        if "seed_base" in payload and "seed" not in payload:
            payload["seed"] = payload["seed_base"]
        elif "seed" in payload and "seed_base" not in payload:
            payload["seed_base"] = payload["seed"]

        sig = inspect.signature(cls)
        filtered_payload = {
            k: v for k, v in payload.items() if k in sig.parameters
        }
        config = cls(**filtered_payload)
        validate_on_policy_training_config(config, require_cycle_reference_exists=True)
        return config


def validate_on_policy_training_config(
    config: RTSOnPolicyTrainingConfig,
    *,
    require_cycle_reference_exists: bool = True,
) -> None:
    if not str(config.artifact_label).strip():
        raise ValueError("artifact_label must be nonblank")
    if int(config.batches) < 1:
        raise ValueError("batches must be >= 1")
    if int(config.workers) < 1:
        raise ValueError("workers must be >= 1")
    if int(config.netlogo_steps_per_run) < 1:
        raise ValueError("netlogo_steps_per_run must be >= 1")
    if require_cycle_reference_exists and not Path(config.cycle_reference_path).exists():
        raise ValueError("cycle_reference_path must exist")
    if config.policy_action_mode not in {"sample", "greedy"}:
        raise ValueError("policy_action_mode must be sample or greedy")
    if config.device not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be auto, cpu, or cuda")
    if config.worker_device not in {"auto", "cpu", "cuda"}:
        raise ValueError("worker_device must be auto, cpu, or cuda")
    if int(config.min_trainable_steps) < 1:
        raise ValueError("min_trainable_steps must be >= 1")
    if int(config.ppo_epochs) < 1:
        raise ValueError("ppo_epochs must be >= 1")
    if int(config.minibatch_size) < 1:
        raise ValueError("minibatch_size must be >= 1")
    if config.feature_flags is not None:
        from src.rmfs.experiments.feature_flags import validate_feature_flags
        validate_feature_flags(config.feature_flags)

