"""Cycle reference JSON helpers for RTS-RL reward probes."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RTSCycleReference:
    reference_overall_cycle_time: float
    reference_avg_storage_cycle_time: float
    reference_avg_replenish_cycle_time: float
    store_action_count: int = 0
    replenish_action_count: int = 0
    alpha: float = 0.0
    source: str = "current_repo_ledger_or_reference"
    source_run_id: str | None = None
    semantics: str = "realized_robot_cycle_time"

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json_dict(cls, payload: dict[str, Any]) -> "RTSCycleReference":
        return cls(**dict(payload))


def validate_cycle_reference(reference: RTSCycleReference) -> None:
    if reference.reference_overall_cycle_time <= 0:
        raise ValueError("reference_overall_cycle_time must be positive")
    if reference.reference_avg_storage_cycle_time <= 0:
        raise ValueError("reference_avg_storage_cycle_time must be positive")
    if reference.reference_avg_replenish_cycle_time <= 0:
        raise ValueError("reference_avg_replenish_cycle_time must be positive")
    if reference.store_action_count < 0 or reference.replenish_action_count < 0:
        raise ValueError("RTS cycle reference action counts cannot be negative")
    if reference.alpha < 0:
        raise ValueError("RTS cycle reference alpha cannot be negative")
    if reference.semantics != "realized_robot_cycle_time":
        raise ValueError("RTS cycle reference semantics must be realized_robot_cycle_time")


def write_cycle_reference(path: Path, reference: RTSCycleReference) -> None:
    validate_cycle_reference(reference)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        json.dump(reference.to_json_dict(), fh, indent=2)


def read_cycle_reference(path: Path) -> RTSCycleReference:
    with path.open() as fh:
        reference = RTSCycleReference.from_json_dict(json.load(fh))
    validate_cycle_reference(reference)
    return reference
