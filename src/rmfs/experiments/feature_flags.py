"""Feature flag schema for RMFS experiment metadata."""

from __future__ import annotations

from typing import Any, Mapping


def default_rika_rts_rl_feature_flags() -> dict[str, Any]:
    return {
        "poa": "future_aware",
        "pps": "station_match",
        "rts": "rts_rl_explicit",
        "charging": "disabled",
        "pps_rl_enabled": False,
        "rts_rl_enabled": True,
        "charging_learning_enabled": False,
        "charging_enabled": False,
        "advanced_order_generation_enabled": False,
        "pod_sku_allocation_learning_enabled": False,
        "pod_sku_allocation_enabled": False,
    }


def validate_feature_flags(flags: Mapping[str, Any]) -> None:
    if "charging_learning_enabled" in flags and "charging_enabled" in flags:
        if flags["charging_learning_enabled"] != flags["charging_enabled"]:
            raise ValueError("charging_learning_enabled and charging_enabled must match")
    if "pod_sku_allocation_learning_enabled" in flags and "pod_sku_allocation_enabled" in flags:
        if flags["pod_sku_allocation_learning_enabled"] != flags["pod_sku_allocation_enabled"]:
            raise ValueError("pod_sku_allocation_learning_enabled and pod_sku_allocation_enabled must match")

    required = default_rika_rts_rl_feature_flags()
    missing = [key for key in required if key not in flags]
    if missing:
        raise ValueError(f"missing feature flags: {missing}")
    for key in (
        "pps_rl_enabled",
        "rts_rl_enabled",
        "charging_learning_enabled",
        "charging_enabled",
        "advanced_order_generation_enabled",
        "pod_sku_allocation_learning_enabled",
        "pod_sku_allocation_enabled",
    ):
        if not isinstance(flags[key], bool):
            raise ValueError(f"{key} must be boolean")
    for key in ("poa", "pps", "rts", "charging"):
        if not isinstance(flags[key], str) or not flags[key].strip():
            raise ValueError(f"{key} must be a nonblank string")

