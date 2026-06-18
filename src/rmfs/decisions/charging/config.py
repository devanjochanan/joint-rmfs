"""Charging configuration loader and validator."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from .types import ChargingConfig, ChargerPosition, ChargingThresholdPolicy


def canonical_charging_config_path() -> Path:
    """Return the canonical path to the charging config JSON."""
    repo_root = Path(__file__).resolve().parents[4]
    return repo_root / "data" / "input" / "charging" / "salsa_charging_config.json"


def load_charging_config(path: Optional[Path] = None) -> ChargingConfig:
    """Load and parse a ChargingConfig from JSON.

    Parameters
    ----------
    path : Path, optional
        Path to the JSON config file.  Defaults to
        ``data/input/charging/salsa_charging_config.json`` relative to repo root.
    """
    config_path = Path(path) if path is not None else canonical_charging_config_path()
    with open(config_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    positions = [
        ChargerPosition(row=int(pos[0]), col=int(pos[1]))
        for pos in raw.get("charger_positions", [])
    ]
    policy = ChargingThresholdPolicy(
        battery_low_pct=int(raw["battery_low_pct"]),
        battery_charged_pct=int(raw["battery_charged_pct"]),
        battery_interrupt_pct=int(raw["battery_interrupt_pct"]),
    )
    return ChargingConfig(
        pipeline=int(raw.get("pipeline", 0)),
        num_chargers=int(raw.get("num_chargers", len(positions))),
        disable_active_charging=bool(raw.get("disable_active_charging", True)),
        selective_picker_chargers=bool(raw.get("selective_picker_chargers", False)),
        policy=policy,
        charger_positions=positions,
        comment=str(raw.get("_comment", "")),
    )


def validate_charging_config(config: ChargingConfig) -> list[str]:
    """Return a list of validation error strings (empty if valid)."""
    errors: list[str] = []
    if config.num_chargers != len(config.charger_positions):
        errors.append(
            f"num_chargers ({config.num_chargers}) != "
            f"len(charger_positions) ({len(config.charger_positions)})"
        )
    if config.policy.battery_low_pct >= config.policy.battery_charged_pct:
        errors.append(
            f"battery_low_pct ({config.policy.battery_low_pct}) must be < "
            f"battery_charged_pct ({config.policy.battery_charged_pct})"
        )
    for i, pos in enumerate(config.charger_positions):
        if pos.row < 0 or pos.col < 0:
            errors.append(f"charger_positions[{i}] has negative coordinate: {pos}")
    return errors


def save_charging_config(config: ChargingConfig, path: Optional[Path] = None) -> Path:
    """Serialize a ChargingConfig to JSON and write it to disk."""
    out_path = Path(path) if path is not None else canonical_charging_config_path()
    data = {
        "_comment": config.comment,
        "pipeline": config.pipeline,
        "num_chargers": config.num_chargers,
        "disable_active_charging": config.disable_active_charging,
        "selective_picker_chargers": config.selective_picker_chargers,
        "battery_low_pct": config.policy.battery_low_pct,
        "battery_charged_pct": config.policy.battery_charged_pct,
        "battery_interrupt_pct": config.policy.battery_interrupt_pct,
        "charger_positions": [pos.as_list() for pos in config.charger_positions],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, indent=2))
    return out_path
