from model.robot import (
    DEFAULT_MOVEMENT_STATE_PRIORITY,
    MOVEMENT_STATE_PRIORITY,
    Robot,
)


def _robot_with_state(state: str) -> Robot:
    robot = Robot.__new__(Robot)
    robot.current_state = state
    robot.job = None
    return robot


def test_charging_movement_states_have_priority_entries():
    for state in (
        "going_to_charge",
        "waiting_for_charger",
        "physically_charging",
        "leaving_charger",
    ):
        assert state in MOVEMENT_STATE_PRIORITY


def test_charging_movement_states_do_not_crash_priority_diff():
    for self_state in (
        "waiting_for_charger",
        "physically_charging",
        "leaving_charger",
    ):
        robot = _robot_with_state(self_state)
        for other_state in (
            "waiting_for_charger",
            "physically_charging",
            "leaving_charger",
            "idle",
            "delivering_pod",
        ):
            assert isinstance(robot.get_priority_diff({"state": other_state}), int)


def test_unknown_movement_state_uses_safe_default_priority():
    robot = _robot_with_state("future_state")

    assert robot.get_priority_diff({"state": "idle"}) == DEFAULT_MOVEMENT_STATE_PRIORITY
    assert robot.get_priority_diff({"state": "another_future_state"}) == 0

