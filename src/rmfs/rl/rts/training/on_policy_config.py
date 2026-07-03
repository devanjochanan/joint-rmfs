"""On-policy RTS PPO controller configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Any, Mapping

from src.rmfs.decisions.task_allocation import DEFAULT_REGRET_K, DEFAULT_ROBOT_TASK_ALLOCATOR, TASK_ALLOCATOR_SCOPE
from src.rmfs.rl.rts.zone_registry import validate_no_col_zone_ids


@dataclass(frozen=True)
class RTSOnPolicyTrainingConfig:
    artifact_label: str | None
    output_root: Path
    batches: int
    workers: int
    netlogo_steps_per_run: int
    seed: int
    simulated_seconds_per_run: float | None = None
    cycle_reference_path: Path | None = None
    device: str = "auto"
    worker_device: str = "cpu"
    policy_action_mode: str = "sample"
    progress: bool | None = None
    tensorboard_enabled: bool = False
    min_trainable_steps: int = 1
    ppo_epochs: int = 4
    minibatch_size: int = 64
    learning_rate: float | None = None
    zone_ids: tuple[str, ...] = ()
    feature_ablation: str = "full"
    ledger_path: Path | None = None
    charging_mode: str = "inherit"
    feature_flags: dict[str, Any] | None = None
    scenario_id: str | None = None
    experiment_id: str | None = None
    branch: str | None = None
    commit: str | None = None
    python_executable: str | None = None
    robot_task_allocator: str = DEFAULT_ROBOT_TASK_ALLOCATOR
    regret_k: int | None = DEFAULT_REGRET_K
    task_allocator_scope: str = TASK_ALLOCATOR_SCOPE
    committed_next_reservations_enabled: bool = True
    debug_worker_logs: bool = False
    rts_torch_threads: int | None = None
    rts_torch_interop_threads: int | None = None
    order_rate_per_hour: int = 500
    worker_spawn_delay_seconds: float = 5.0

    def to_json_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["output_root"] = str(self.output_root)
        data["cycle_reference_path"] = str(self.cycle_reference_path) if self.cycle_reference_path is not None else None
        data["ledger_path"] = str(self.ledger_path) if self.ledger_path is not None else None
        data["zone_ids"] = list(self.zone_ids)
        data["seed_base"] = self.seed
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RTSOnPolicyTrainingConfig":
        import inspect
        payload = dict(data)
        payload["output_root"] = Path(payload["output_root"])
        if payload.get("cycle_reference_path"):
            payload["cycle_reference_path"] = Path(payload["cycle_reference_path"])
        else:
            payload["cycle_reference_path"] = None
        if payload.get("ledger_path"):
            payload["ledger_path"] = Path(payload["ledger_path"])
        else:
            payload["ledger_path"] = None
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
    if config.artifact_label is not None and not str(config.artifact_label).strip():
        raise ValueError("artifact_label must be nonblank")
    if int(config.batches) < 1:
        raise ValueError("batches must be >= 1")
    if int(config.workers) < 1:
        raise ValueError("workers must be >= 1")
    if int(config.netlogo_steps_per_run) < 1:
        raise ValueError("netlogo_steps_per_run must be >= 1")
    if config.simulated_seconds_per_run is not None:
        simulated_seconds = float(config.simulated_seconds_per_run)
        if not math.isfinite(simulated_seconds) or simulated_seconds <= 0.0:
            raise ValueError("simulated_seconds_per_run must be finite and positive when supplied")
    if require_cycle_reference_exists and config.cycle_reference_path is not None and not Path(config.cycle_reference_path).exists():
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
    if config.learning_rate is not None and float(config.learning_rate) <= 0.0:
        raise ValueError("learning_rate must be positive")
    if int(config.order_rate_per_hour) <= 0:
        raise ValueError("order_rate_per_hour must be positive")
    if float(config.worker_spawn_delay_seconds) < 0.0:
        raise ValueError("worker_spawn_delay_seconds must be >= 0")
    from src.rmfs.rl.rts.ablation import resolve_ablation
    resolve_ablation(config.feature_ablation)
    if config.charging_mode not in {"inherit", "enabled", "disabled"}:
        raise ValueError("charging_mode must be inherit, enabled, or disabled")
    if config.feature_flags is not None:
        from src.rmfs.experiments.feature_flags import validate_feature_flags
        validate_feature_flags(config.feature_flags)
    if config.robot_task_allocator not in {"regret_k", "legacy_nearest"}:
        raise ValueError("robot_task_allocator must be regret_k or legacy_nearest")
    if config.robot_task_allocator == "regret_k":
        if config.regret_k is None or int(config.regret_k) < 1:
            raise ValueError("regret_k must be >= 1 when robot_task_allocator=regret_k")
    if config.robot_task_allocator == "legacy_nearest" and config.regret_k is not None:
        raise ValueError("regret_k must be None when robot_task_allocator=legacy_nearest")
    if config.task_allocator_scope != TASK_ALLOCATOR_SCOPE:
        raise ValueError(f"task_allocator_scope must be {TASK_ALLOCATOR_SCOPE}")
    if not bool(config.committed_next_reservations_enabled):
        raise ValueError("RTS on-policy training requires committed_next_reservations_enabled=True")
    validate_no_col_zone_ids(config.zone_ids, context="RTS on-policy training")
