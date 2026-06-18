#!/usr/bin/env python3
"""Fixture-runtime smoke for the Salsa charging bridge."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.rmfs.app.charging_bridge import (
    apply_charging_runtime_tick,
    charging_runtime_disabled_reason,
    get_charging_runtime_summary,
    install_charging_runtime_if_enabled,
    load_charging_runtime_config,
    set_robot_battery,
)
from src.rmfs.experiments.feature_flags import default_rika_rts_rl_feature_flags


ROOT_ARTIFACTS = ("warehouse.db", "netlogo.state", "assign_order.csv", "pod_info.csv")


def file_digest(path: Path) -> dict[str, object]:
    if not path.exists():
        return {"exists": False, "sha256": None, "size": 0}
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            digest.update(chunk)
    return {"exists": True, "sha256": digest.hexdigest(), "size": path.stat().st_size}


def snapshot_root() -> dict[str, dict[str, object]]:
    return {name: file_digest(REPO_ROOT / name) for name in ROOT_ARTIFACTS}


def robot(robot_id: str, row: int, col: int) -> SimpleNamespace:
    return SimpleNamespace(object_type="robot", id=robot_id, pos_y=row, pos_x=col)


def main() -> int:
    root_before = snapshot_root()
    config = load_charging_runtime_config()
    flags = default_rika_rts_rl_feature_flags()
    flags["charging_enabled"] = True
    flags["charging_learning_enabled"] = True
    flags["charging"] = "salsa_forced_smoke"

    low = robot("robot-low", 7, 2)
    normal = robot("robot-normal", 1, 2)
    high = robot("robot-high", 25, 2)
    non_robot = SimpleNamespace(object_type="pod", id="pod-1")
    universe = SimpleNamespace(_objects=[low, normal, high, non_robot])

    state = install_charging_runtime_if_enabled(universe, flags, config)
    assert state is not None
    assert universe.charging_runtime is state
    assert len(state.chargers) == 12
    assert hasattr(low, "battery_pct")
    assert hasattr(low, "charging_status")
    assert not hasattr(non_robot, "battery_pct")

    set_robot_battery(state, "robot-low", 15)
    set_robot_battery(state, "robot-normal", 50)
    set_robot_battery(state, "robot-high", 80)
    tick_summary = apply_charging_runtime_tick(
        state,
        tick=11,
        robot_positions={
            "robot-low": (7, 2),
            "robot-normal": (1, 2),
            "robot-high": (25, 2),
        },
    )

    assert tick_summary["tick"] == 11
    assert tick_summary["low_battery_robots"] == ["robot-low"]
    assert tick_summary["eligible_low_battery_robots"] == ["robot-low"]
    assert len(tick_summary["new_assignments"]) == 1
    assignment = tick_summary["new_assignments"][0]
    assert assignment["robot_id"] == "robot-low"
    assert assignment["charger_index"] == 1
    assert state.robot_statuses["robot-low"] == "assigned_to_charger"
    assert low.charging_status == "assigned_to_charger"
    assert state.robot_statuses["robot-normal"] == "idle"
    assert state.total_low_battery_detections == 1
    assert state.total_assignments_made == 1

    summary = get_charging_runtime_summary(state)
    assert summary["num_chargers"] == 12
    assert summary["occupied_chargers"] == 1
    assert summary["available_chargers"] == 11
    assert summary["active_assignments"][0]["robot_id"] == "robot-low"
    assert summary["physical_dispatch_implemented"] is False

    disabled_universe = SimpleNamespace(_objects=[robot("disabled", 0, 0)])
    disabled_flags = default_rika_rts_rl_feature_flags()
    assert charging_runtime_disabled_reason(disabled_flags, config) == "feature_flag_disabled"
    assert install_charging_runtime_if_enabled(disabled_universe, disabled_flags, config) is None
    assert not hasattr(disabled_universe, "charging_runtime")

    config_disabled_universe = SimpleNamespace(_objects=[robot("config-disabled", 0, 0)])
    config_disabled = replace(config, disable_active_charging=True)
    assert charging_runtime_disabled_reason(flags, config_disabled) == "config_disable_active_charging"
    assert install_charging_runtime_if_enabled(config_disabled_universe, flags, config_disabled) is None
    assert not hasattr(config_disabled_universe, "charging_runtime")

    second_tick = apply_charging_runtime_tick(
        state,
        tick=12,
        robot_positions={
            "robot-low": (7, 2),
            "robot-normal": (1, 2),
            "robot-high": (25, 2),
        },
    )
    assert second_tick["low_battery_robots"] == ["robot-low"]
    assert second_tick["eligible_low_battery_robots"] == []
    assert second_tick["new_assignments"] == []
    assert state.total_low_battery_detections == 1

    root_after = snapshot_root()
    assert root_before == root_after, "fixture-runtime charging smoke changed root runtime artifacts"

    print("salsa charging runtime smoke ok")
    print("fixture-runtime smoke (no setup/tick path used)")
    print("validated config load, charger registry, low-battery detection, status assignment, metrics, and default-disabled behavior")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
