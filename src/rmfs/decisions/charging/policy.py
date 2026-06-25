"""Charging threshold policy helpers."""

from __future__ import annotations

from .types import ChargingThresholdPolicy


# Default validated policy (Taguchi-robust from Phase-3)
DEFAULT_POLICY = ChargingThresholdPolicy(
    battery_low_pct=18,
    battery_charged_pct=60,
    battery_interrupt_pct=50,
)


def should_start_charging(battery_pct: float, policy: ChargingThresholdPolicy = DEFAULT_POLICY) -> bool:
    """Return True if the robot's battery percentage is at or below the low threshold."""
    return battery_pct <= policy.battery_low_pct


def is_fully_charged(battery_pct: float, policy: ChargingThresholdPolicy = DEFAULT_POLICY) -> bool:
    """Return True if the robot's battery has reached the charged threshold."""
    return battery_pct >= policy.battery_charged_pct


def should_interrupt_charging(
    battery_pct: float,
    jobs_pending: bool,
    no_other_free_robot: bool,
    policy: ChargingThresholdPolicy = DEFAULT_POLICY,
) -> bool:
    """Return True if the robot should leave the charger early."""
    return (
        battery_pct >= policy.battery_interrupt_pct
        and jobs_pending
        and no_other_free_robot
    )
