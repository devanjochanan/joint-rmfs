"""Charging module data types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass(frozen=True)
class ChargerPosition:
    """A single charger cell on the warehouse grid (row, col)."""
    row: int
    col: int

    def as_list(self) -> list[int]:
        return [self.row, self.col]

    def as_tuple(self) -> Tuple[int, int]:
        return (self.row, self.col)


@dataclass(frozen=True)
class ChargingThresholdPolicy:
    """Battery threshold policy parameters."""
    battery_low_pct: int
    battery_charged_pct: int
    battery_interrupt_pct: int


@dataclass(frozen=True)
class ChargingConfig:
    """Full charging configuration as loaded from salsa_charging_config.json."""
    pipeline: int
    num_chargers: int
    disable_active_charging: bool
    selective_picker_chargers: bool
    policy: ChargingThresholdPolicy
    charger_positions: List[ChargerPosition] = field(default_factory=list)
    comment: str = ""
