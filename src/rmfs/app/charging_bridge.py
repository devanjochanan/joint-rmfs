"""Charging runtime bridge - minimal Stage 3 integration for Salsa charging.

This module provides a narrow runtime bridge that:
- Loads and validates the Salsa charging configuration
- Maintains a lightweight runtime state (charger registry, robot battery tracking)
- Detects low-battery robots using existing Salsa policy helpers
- Assigns chargers deterministically (nearest available)
- Records assignment/status/metrics for validation

Physical charging-trip movement (pathing robots to charger cells) is NOT
implemented here. That requires ``engine/universe.py`` changes and is
documented as future work in ``docs/integration/salsa_charging_pending.md``.

Default baseline behavior is completely unchanged - this bridge is only
activated when ``charging_enabled=True`` in feature flags.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from src.rmfs.decisions.charging.config import (
    load_charging_config,
    validate_charging_config,
)
from src.rmfs.decisions.charging.policy import (
    should_start_charging,
)
from src.rmfs.decisions.charging.types import (
    ChargerPosition,
    ChargingConfig,
    ChargingThresholdPolicy,
)


# ---------------------------------------------------------------------------
# Runtime state
# ---------------------------------------------------------------------------

@dataclass
class ChargerSlot:
    """A single charger in the runtime registry."""

    position: ChargerPosition
    occupied_by: Optional[str] = None  # robot_id or None

    @property
    def available(self) -> bool:
        return self.occupied_by is None


@dataclass
class ChargingAssignment:
    """Record of one charging assignment."""

    robot_id: str
    charger_index: int
    charger_position: Tuple[int, int]
    battery_pct_at_assignment: float
    tick_assigned: int = 0


@dataclass
class ChargingRuntimeState:
    """Lightweight runtime state for the charging bridge.

    This is installed on the universe/warehouse object when charging is
    explicitly enabled.  It does NOT modify Robot class attributes.
    """

    config: ChargingConfig
    policy: ChargingThresholdPolicy
    chargers: List[ChargerSlot] = field(default_factory=list)

    # Robot battery tracking: {robot_id: battery_pct}
    # In real simulation these would live on Robot objects; here they are
    # synthetic values managed by the bridge for smoke validation.
    robot_batteries: Dict[str, float] = field(default_factory=dict)
    robot_statuses: Dict[str, str] = field(default_factory=dict)
    robot_objects: Dict[str, Any] = field(default_factory=dict)

    # Assignment log
    assignments: List[ChargingAssignment] = field(default_factory=list)

    # Metrics
    total_low_battery_detections: int = 0
    total_assignments_made: int = 0
    total_ticks_processed: int = 0


# ---------------------------------------------------------------------------
# Bridge functions
# ---------------------------------------------------------------------------

def load_charging_runtime_config(config_path=None) -> ChargingConfig:
    """Load and validate the Salsa charging config.

    Parameters
    ----------
    config_path : path-like, optional
        Override path to the JSON config.  Defaults to the canonical
        ``data/input/charging/salsa_charging_config.json``.

    Returns
    -------
    ChargingConfig
        The validated configuration.

    Raises
    ------
    ValueError
        If the configuration has validation errors.
    """
    config = load_charging_config(config_path)
    errors = validate_charging_config(config)
    if errors:
        raise ValueError(
            f"Charging config validation failed: {'; '.join(errors)}"
        )
    return config


def create_charging_runtime_state(
    config: ChargingConfig,
) -> ChargingRuntimeState:
    """Create an initialized ``ChargingRuntimeState`` from a config."""
    chargers = [
        ChargerSlot(position=pos) for pos in config.charger_positions
    ]
    return ChargingRuntimeState(
        config=config,
        policy=config.policy,
        chargers=chargers,
    )


def is_charging_enabled(feature_flags: Dict[str, Any]) -> bool:
    """Check whether charging is enabled in the given feature flags."""
    if feature_flags is None:
        return False
    return bool(feature_flags.get("charging_enabled", False))


def _is_robot_like(obj: Any) -> bool:
    return str(getattr(obj, "object_type", "")).lower() == "robot"


def _robot_id(obj: Any, fallback: str) -> str:
    for attr in ("robot_id", "id", "_id", "name", "label"):
        value = getattr(obj, attr, None)
        if value is not None:
            return str(value)
    robot_name = getattr(obj, "robotName", None)
    if callable(robot_name):
        try:
            value = robot_name()
        except Exception:
            value = None
        if value is not None:
            return str(value)
    return fallback


def initialize_robot_charging_fields(
    state: ChargingRuntimeState,
    universe: Any,
    *,
    default_battery_pct: float = 100.0,
) -> None:
    """Register robot-like objects and ensure instance charging fields exist."""
    for index, obj in enumerate(getattr(universe, "_objects", []) or []):
        if not _is_robot_like(obj):
            continue
        robot_id = _robot_id(obj, f"robot_{index}")
        if not hasattr(obj, "battery_pct"):
            setattr(obj, "battery_pct", float(default_battery_pct))
        if not hasattr(obj, "charging_status"):
            setattr(obj, "charging_status", "idle")
        battery_pct = float(getattr(obj, "battery_pct", default_battery_pct))
        state.robot_objects[robot_id] = obj
        state.robot_batteries.setdefault(robot_id, battery_pct)
        state.robot_statuses.setdefault(
            robot_id,
            str(getattr(obj, "charging_status", "idle")),
        )


def install_charging_runtime(
    universe: Any,
    config: ChargingConfig,
) -> ChargingRuntimeState:
    """Install charging runtime state on the universe object.

    The state is attached as ``universe.charging_runtime`` so that
    downstream hooks can access it.

    Parameters
    ----------
    universe : Inventory or similar
        The simulation universe object.
    config : ChargingConfig
        The validated charging configuration.

    Returns
    -------
    ChargingRuntimeState
        The newly created and installed state.
    """
    state = create_charging_runtime_state(config)
    initialize_robot_charging_fields(state, universe)
    universe.charging_runtime = state
    return state


def install_charging_runtime_if_enabled(
    universe: Any,
    feature_flags: Dict[str, Any] | None,
    config: ChargingConfig | None = None,
) -> Optional[ChargingRuntimeState]:
    """Install charging state only when explicit feature flags enable it."""
    if not is_charging_enabled(feature_flags or {}):
        return None
    resolved_config = config if config is not None else load_charging_runtime_config()
    return install_charging_runtime(universe, resolved_config)


def set_robot_battery(
    state: ChargingRuntimeState,
    robot_id: str,
    battery_pct: float,
) -> None:
    """Set or update the battery percentage for a robot."""
    state.robot_batteries[robot_id] = float(battery_pct)
    robot_obj = state.robot_objects.get(robot_id)
    if robot_obj is not None:
        setattr(robot_obj, "battery_pct", float(battery_pct))
    state.robot_statuses.setdefault(robot_id, "idle")


def detect_low_battery_robots(
    state: ChargingRuntimeState,
) -> List[str]:
    """Return robot IDs whose battery is at or below the low threshold.

    Uses ``should_start_charging()`` from the existing Salsa policy module.
    """
    low = []
    for robot_id, battery_pct in state.robot_batteries.items():
        if should_start_charging(battery_pct, state.policy):
            low.append(robot_id)
    return low


def _charger_distance(
    charger: ChargerSlot,
    robot_row: int,
    robot_col: int,
) -> float:
    """Manhattan distance from a robot position to a charger."""
    return abs(charger.position.row - robot_row) + abs(
        charger.position.col - robot_col
    )


def select_nearest_available_charger(
    state: ChargingRuntimeState,
    robot_row: int = 0,
    robot_col: int = 0,
) -> Optional[int]:
    """Select the nearest available charger index.

    Returns None if no charger is available.  Ties are broken by
    charger index (deterministic).
    """
    best_idx: Optional[int] = None
    best_dist = math.inf
    for idx, charger in enumerate(state.chargers):
        if not charger.available:
            continue
        dist = _charger_distance(charger, robot_row, robot_col)
        if dist < best_dist:
            best_dist = dist
            best_idx = idx
    return best_idx


def assign_charger(
    state: ChargingRuntimeState,
    robot_id: str,
    robot_row: int = 0,
    robot_col: int = 0,
    tick: int = 0,
) -> Optional[ChargingAssignment]:
    """Assign the nearest available charger to a robot.

    Records the assignment in ``state.assignments`` and marks the charger
    as occupied.  Returns ``None`` if no charger is available.

    .. note::
        Physical movement of the robot to the charger is NOT performed.
        This only records the assignment/status for validation.
    """
    charger_idx = select_nearest_available_charger(
        state, robot_row, robot_col
    )
    if charger_idx is None:
        return None

    charger = state.chargers[charger_idx]
    charger.occupied_by = robot_id
    battery_pct = state.robot_batteries.get(robot_id, 0.0)
    state.robot_statuses[robot_id] = "assigned_to_charger"
    robot_obj = state.robot_objects.get(robot_id)
    if robot_obj is not None:
        setattr(robot_obj, "charging_status", "assigned_to_charger")
        setattr(robot_obj, "assigned_charger_index", charger_idx)

    assignment = ChargingAssignment(
        robot_id=robot_id,
        charger_index=charger_idx,
        charger_position=charger.position.as_tuple(),
        battery_pct_at_assignment=battery_pct,
        tick_assigned=tick,
    )
    state.assignments.append(assignment)
    state.total_assignments_made += 1
    return assignment


def release_charger(
    state: ChargingRuntimeState,
    charger_index: int,
) -> None:
    """Release a charger slot, making it available again."""
    if 0 <= charger_index < len(state.chargers):
        robot_id = state.chargers[charger_index].occupied_by
        state.chargers[charger_index].occupied_by = None
        if robot_id is not None:
            state.robot_statuses[robot_id] = "idle"
            robot_obj = state.robot_objects.get(robot_id)
            if robot_obj is not None:
                setattr(robot_obj, "charging_status", "idle")
                setattr(robot_obj, "assigned_charger_index", None)


def apply_charging_runtime_tick(
    state: ChargingRuntimeState,
    tick: int = 0,
    robot_positions: Optional[Dict[str, Tuple[int, int]]] = None,
) -> Dict[str, Any]:
    """Per-tick charging hook.

    Detects low-battery robots, assigns chargers, updates metrics.

    Parameters
    ----------
    state : ChargingRuntimeState
        The current charging runtime state.
    tick : int
        Current simulation tick.
    robot_positions : dict, optional
        Mapping of ``{robot_id: (row, col)}``.  Used for nearest-charger
        selection.  Defaults to ``(0, 0)`` for all robots if not provided.

    Returns
    -------
    dict
        Tick summary with keys:
        ``low_battery_robots``, ``new_assignments``, ``tick``.
    """
    if robot_positions is None:
        robot_positions = {}

    state.total_ticks_processed += 1

    low_battery_ids = detect_low_battery_robots(state)
    state.total_low_battery_detections += len(low_battery_ids)

    new_assignments = []
    for robot_id in low_battery_ids:
        # Skip if already assigned
        already_assigned = any(
            a.robot_id == robot_id
            for a in state.assignments
            if state.chargers[a.charger_index].occupied_by == robot_id
        )
        if already_assigned:
            continue

        row, col = robot_positions.get(robot_id, (0, 0))
        assignment = assign_charger(state, robot_id, row, col, tick)
        if assignment is not None:
            new_assignments.append(assignment)

    return {
        "low_battery_robots": low_battery_ids,
        "new_assignments": [
            {
                "robot_id": a.robot_id,
                "charger_index": a.charger_index,
                "charger_position": a.charger_position,
                "battery_pct": a.battery_pct_at_assignment,
            }
            for a in new_assignments
        ],
        "tick": tick,
    }


def get_charging_runtime_summary(
    state: ChargingRuntimeState,
) -> Dict[str, Any]:
    """Return a summary dict of the current charging runtime state.

    Useful for validation output and smoke tests.
    """
    return {
        "num_chargers": len(state.chargers),
        "charger_positions": [
            c.position.as_tuple() for c in state.chargers
        ],
        "available_chargers": sum(1 for c in state.chargers if c.available),
        "occupied_chargers": sum(
            1 for c in state.chargers if not c.available
        ),
        "tracked_robots": len(state.robot_batteries),
        "robot_batteries": dict(state.robot_batteries),
        "robot_statuses": dict(state.robot_statuses),
        "total_assignments": state.total_assignments_made,
        "total_low_battery_detections": state.total_low_battery_detections,
        "total_ticks_processed": state.total_ticks_processed,
        "active_assignments": [
            {
                "robot_id": c.occupied_by,
                "charger_index": idx,
                "charger_position": c.position.as_tuple(),
            }
            for idx, c in enumerate(state.chargers)
            if not c.available
        ],
        "policy": {
            "battery_low_pct": state.policy.battery_low_pct,
            "battery_charged_pct": state.policy.battery_charged_pct,
            "battery_interrupt_pct": state.policy.battery_interrupt_pct,
        },
        "physical_dispatch_implemented": False,
    }
