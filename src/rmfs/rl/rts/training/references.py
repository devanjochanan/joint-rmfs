"""Cycle-reference helpers for synthetic RTS training smokes."""

from __future__ import annotations

import shutil
from pathlib import Path

from src.rmfs.rl.rts.cycle_reference import RTSCycleReference, read_cycle_reference, write_cycle_reference


def create_synthetic_cycle_reference(
    *,
    overall_cycle_time: float = 10.0,
    storage_cycle_time: float = 8.0,
    replenish_cycle_time: float = 12.0,
    alpha: float = 0.1,
) -> RTSCycleReference:
    return RTSCycleReference(
        reference_overall_cycle_time=overall_cycle_time,
        reference_avg_storage_cycle_time=storage_cycle_time,
        reference_avg_replenish_cycle_time=replenish_cycle_time,
        store_action_count=2,
        replenish_action_count=1,
        alpha=alpha,
        source="synthetic_phase8_smoke",
    )


def write_synthetic_cycle_reference(path: Path) -> RTSCycleReference:
    reference = create_synthetic_cycle_reference()
    write_cycle_reference(path, reference)
    return reference


def copy_cycle_reference_to_checkpoint(source_path: Path, checkpoint_dir: Path) -> Path:
    source = Path(source_path)
    if not source.exists():
        raise FileNotFoundError(source)
    read_cycle_reference(source)
    target = Path(checkpoint_dir) / "cycle_reference.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    return target

