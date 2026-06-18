#!/usr/bin/env python3
"""Pure smoke for Salsa charging policy and bridge accounting."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.rmfs.app.charging_bridge import (
    assign_charger,
    create_charging_runtime_state,
    get_charging_runtime_summary,
    install_charging_runtime_if_enabled,
    is_charging_enabled,
    load_charging_runtime_config,
    release_charger,
    select_nearest_available_charger,
    set_robot_battery,
)
from src.rmfs.decisions.charging.policy import (
    is_fully_charged,
    should_interrupt_charging,
    should_start_charging,
)
from src.rmfs.experiments.feature_flags import default_rika_rts_rl_feature_flags


def main() -> int:
    config = load_charging_runtime_config()
    policy = config.policy
    assert config.num_chargers == 12
    assert len(config.charger_positions) == 12
    assert policy.battery_low_pct == 18
    assert policy.battery_charged_pct == 60
    assert policy.battery_interrupt_pct == 50

    assert should_start_charging(17, policy)
    assert should_start_charging(18, policy)
    assert not should_start_charging(19, policy)
    assert not is_fully_charged(59, policy)
    assert is_fully_charged(60, policy)
    assert is_fully_charged(61, policy)
    assert not should_interrupt_charging(49, True, True, policy)
    assert should_interrupt_charging(50, True, True, policy)
    assert not should_interrupt_charging(55, False, True, policy)
    assert not should_interrupt_charging(55, True, False, policy)

    disabled_flags = default_rika_rts_rl_feature_flags()
    assert not is_charging_enabled(disabled_flags)
    disabled_universe = SimpleNamespace(_objects=[])
    assert install_charging_runtime_if_enabled(disabled_universe, disabled_flags, config) is None
    assert not hasattr(disabled_universe, "charging_runtime")

    state = create_charging_runtime_state(config)
    set_robot_battery(state, "robot-low", 15)
    set_robot_battery(state, "robot-normal", 50)
    assert select_nearest_available_charger(state, robot_row=7, robot_col=2) == 1

    assignment = assign_charger(state, "robot-low", robot_row=7, robot_col=2, tick=3)
    assert assignment is not None
    assert assignment.charger_index == 1
    assert state.robot_statuses["robot-low"] == "assigned_to_charger"
    assert state.total_assignments_made == 1
    assert not state.chargers[1].available

    release_charger(state, 1)
    assert state.chargers[1].available
    assert state.robot_statuses["robot-low"] == "idle"

    for index, charger in enumerate(state.chargers):
        robot_id = f"capacity-{index}"
        set_robot_battery(state, robot_id, 10)
        assigned = assign_charger(state, robot_id, charger.position.row, charger.position.col, tick=4)
        assert assigned is not None
    assert select_nearest_available_charger(state, 0, 0) is None

    summary = get_charging_runtime_summary(state)
    assert summary["num_chargers"] == 12
    assert summary["available_chargers"] == 0
    assert summary["occupied_chargers"] == 12
    assert summary["total_assignments"] == 13
    assert summary["policy"]["battery_low_pct"] == 18
    assert summary["physical_dispatch_implemented"] is False

    print("salsa charging artificial smoke ok")
    print("validated policy thresholds, deterministic charger selection, capacity/status metrics, and default-disabled behavior")
    print("forced bridge accounting only; physical charging trips and long-run energy performance are not validated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
