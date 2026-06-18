"""Training-facing RMFS timebase helpers."""

from __future__ import annotations

from dataclasses import dataclass
import math


def validate_tick_to_second(value: float) -> float:
    normalized = float(value)
    if not math.isfinite(normalized) or normalized <= 0.0:
        raise ValueError("tick_to_second must be finite and positive")
    return normalized


def netlogo_steps_to_warehouse_time(netlogo_steps: int, tick_to_second: float) -> float:
    steps = int(netlogo_steps)
    if steps < 0:
        raise ValueError("netlogo_steps must be >= 0")
    return float(steps) * validate_tick_to_second(tick_to_second)


def warehouse_time_to_netlogo_steps(warehouse_time: float, tick_to_second: float) -> int:
    value = float(warehouse_time)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("warehouse_time must be finite and >= 0")
    return int(round(value / validate_tick_to_second(tick_to_second)))


@dataclass(frozen=True)
class RMFSTimebase:
    netlogo_steps_completed: int
    tick_to_second: float
    warehouse_time_start: float
    warehouse_time_end: float

    def __post_init__(self) -> None:
        if int(self.netlogo_steps_completed) < 0:
            raise ValueError("netlogo_steps_completed must be >= 0")
        validate_tick_to_second(self.tick_to_second)
        if self.warehouse_time_end < self.warehouse_time_start:
            raise ValueError("warehouse_time_end must be >= warehouse_time_start")

    @property
    def warehouse_time_elapsed(self) -> float:
        return float(self.warehouse_time_end) - float(self.warehouse_time_start)

    @property
    def expected_warehouse_time_elapsed(self) -> float:
        return netlogo_steps_to_warehouse_time(self.netlogo_steps_completed, self.tick_to_second)

