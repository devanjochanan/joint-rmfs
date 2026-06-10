"""Configuration for synthetic/offline RTS PPO training infrastructure."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class RTSTrainingConfig:
    artifact_label: str
    output_root: Path
    seed: int = 42
    learning_rate: float = 1e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.1
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    ppo_epochs: int = 4
    minibatch_size: int = 64
    max_grad_norm: float = 1.0
    hidden_sizes: tuple[int, ...] = (64, 64)
    stock_hidden_sizes: tuple[int, ...] = (32, 32)
    stock_embedding_dim: int = 16
    checkpoint_every_batch: bool = True
    keep_all_batch_checkpoints: bool = True
    use_latest_only_for_resume: bool = True
    tensorboard_enabled: bool = True

    def to_json_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["output_root"] = str(self.output_root)
        data["hidden_sizes"] = list(self.hidden_sizes)
        data["stock_hidden_sizes"] = list(self.stock_hidden_sizes)
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RTSTrainingConfig":
        payload = dict(data)
        payload["output_root"] = Path(payload["output_root"])
        payload["hidden_sizes"] = tuple(payload.get("hidden_sizes", (64, 64)))
        payload["stock_hidden_sizes"] = tuple(payload.get("stock_hidden_sizes", (32, 32)))
        config = cls(**payload)
        validate_training_config(config)
        return config


def validate_training_config(config: RTSTrainingConfig) -> None:
    if not str(config.artifact_label).strip():
        raise ValueError("artifact_label must be nonblank")
    if config.output_root is None:
        raise ValueError("output_root is required")
    if float(config.learning_rate) <= 0.0:
        raise ValueError("learning_rate must be positive")
    if not (0.0 < float(config.gamma) <= 1.0):
        raise ValueError("gamma must be in (0, 1]")
    if not (0.0 < float(config.gae_lambda) <= 1.0):
        raise ValueError("gae_lambda must be in (0, 1]")
    if not (0.0 < float(config.clip_epsilon) < 1.0):
        raise ValueError("clip_epsilon must be in (0, 1)")
    if int(config.ppo_epochs) < 1:
        raise ValueError("ppo_epochs must be >= 1")
    if int(config.minibatch_size) < 1:
        raise ValueError("minibatch_size must be >= 1")
    if float(config.max_grad_norm) <= 0.0:
        raise ValueError("max_grad_norm must be positive")
    if not config.hidden_sizes or any(int(size) <= 0 for size in config.hidden_sizes):
        raise ValueError("hidden_sizes must be positive")
    if not config.stock_hidden_sizes or any(int(size) <= 0 for size in config.stock_hidden_sizes):
        raise ValueError("stock_hidden_sizes must be positive")
    if int(config.stock_embedding_dim) <= 0:
        raise ValueError("stock_embedding_dim must be positive")

