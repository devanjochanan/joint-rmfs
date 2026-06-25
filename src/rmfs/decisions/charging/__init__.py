"""Charging decisions module — Salsa's config, placement, and policy ownership.

Active charging runtime remains absent/inactive. This module provides:
- Config loading/validation (config.py)
- Placement generation (placement.py)
- Threshold policy helpers (policy.py)
- Data types (types.py)
"""

from .types import ChargingConfig, ChargerPosition, ChargingThresholdPolicy
from .config import (
    canonical_charging_config_path,
    load_charging_config,
    validate_charging_config,
    save_charging_config,
)
from .policy import DEFAULT_POLICY

__all__ = [
    "ChargingConfig",
    "ChargerPosition",
    "ChargingThresholdPolicy",
    "canonical_charging_config_path",
    "load_charging_config",
    "validate_charging_config",
    "save_charging_config",
    "DEFAULT_POLICY",
]
