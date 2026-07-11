"""Deterministic event-driven active-charging dispatch."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


# Canonical charging-counter field names. The warehouse pre-initializes
# ``charging_counters`` to 0 for every name here so that an *applicable* event
# with zero occurrences serializes as ``0`` (not ``None``); a missing key then
# unambiguously means "producer not wired". Every name below has a producer in
# ``dispatch.py`` or ``model/robot.py``. ``robots_died_from_battery`` is
# intentionally NOT listed here — it is pre-initialized alongside its own
# producer (battery-death wiring) so a genuine death is never masked by a 0.
CHARGING_COUNTER_FIELDS: tuple[str, ...] = (
    # Episode / FIFO lifecycle
    "charging_episodes_created",
    "charging_requests",
    "charging_fifo_entries",
    "charging_fifo_reentries",
    "charging_assignment_evaluations",
    "charging_waiting_seconds",
    # Claim / assignment outcomes
    "charger_claims_created",
    "charger_claims_released",
    "charger_assignment_successes",
    "charger_assignment_failures",
    "charger_arrivals",
    # Candidate-rejection diagnostics (instrument-only; no threshold change)
    "charger_candidate_unroutable",
    "charger_candidate_occupied",
    "charger_candidate_insufficient_energy",
    "charger_assignment_no_feasible_candidate",
    # Legacy/compat rejection names (retained; charging-integrity tests use these)
    "charger_route_failures",
    "charger_candidate_route_rejections",
    "charger_energy_infeasible_candidates",
    "charger_unavailable_events",
    "stale_charger_claims_detected",
    # Physical charging sessions
    "charging_sessions_started",
    "charging_sessions_completed",
    "charging_sessions_interrupted",
    "robots_died_from_battery",
    # Energy / drive-by
    "charging_energy_added_j",
    "drive_by_charging_events",
    # Committed-next linkage (charging seam)
    "committed_next_cancelled_for_charging",
)


def initial_charging_counters() -> dict[str, float]:
    """Return a fresh charging-counter dict pre-initialized to 0."""
    return {name: 0 for name in CHARGING_COUNTER_FIELDS}


def stable_robot_id(robot: Any) -> tuple[int, str]:
    raw = getattr(robot, "_id", None)
    if raw is None:
        raw = getattr(robot, "id", "")
    try:
        return (0, f"{int(raw):020d}")
    except (TypeError, ValueError):
        return (1, str(raw))


@dataclass(frozen=True)
class ChargingQueueEntry:
    entered_at: float
    robot_key: tuple[int, str]
    robot: Any


class ChargingDispatcher:
    """One FIFO queue for a warehouse's active-charging episodes.

    Assignment is reconsidered only when this object is explicitly notified of
    a queue/claim change, or when charger-cell physical occupancy changes.
    """

    def __init__(self, warehouse: Any):
        self.warehouse = warehouse
        self._entries: dict[tuple[int, str], ChargingQueueEntry] = {}
        self._dispatching = False
        self._last_occupancy_signature = None

    def _bump(self, name: str, amount: float = 1) -> None:
        counters = getattr(self.warehouse, "charging_counters", None)
        if counters is None:
            counters = self.warehouse.charging_counters = {}
        counters[name] = counters.get(name, 0) + amount

    def enqueue(self, robot: Any, *, preserve_timestamp: bool = False) -> bool:
        key = stable_robot_id(robot)
        if key in self._entries:
            return False
        entered_at = getattr(robot, "_charging_queue_entered_at", None)
        if entered_at is None or not preserve_timestamp:
            entered_at = float(getattr(self.warehouse, "_tick", 0.0))
            robot._charging_queue_entered_at = entered_at
        if getattr(robot, "_charging_episode_id", None) is None:
            robot._charging_episode_id = f"charge:{key[1]}:{entered_at:.9f}"
            self._bump("charging_episodes_created")
            self._bump("charging_requests")
        robot._charging_lifecycle_state = "waiting_fifo"
        robot._enter_waiting_for_charger("waiting_fifo")
        self._entries[key] = ChargingQueueEntry(float(entered_at), key, robot)
        if preserve_timestamp:
            self._bump("charging_fifo_reentries")
        else:
            self._bump("charging_fifo_entries")
        self.dispatch(reason="queue_entry")
        return True

    def remove(self, robot: Any) -> None:
        self._entries.pop(stable_robot_id(robot), None)

    def ordered_entries(self) -> tuple[ChargingQueueEntry, ...]:
        return tuple(sorted(self._entries.values(), key=lambda item: (item.entered_at, item.robot_key)))

    def dispatch(self, *, reason: str) -> None:
        if self._dispatching:
            return
        self._dispatching = True
        try:
            while self._entries:
                entry = self.ordered_entries()[0]
                self._bump("charging_assignment_evaluations")
                if not entry.robot._assign_charger_from_fifo():
                    self._bump("charger_assignment_failures")
                    break
                self._entries.pop(entry.robot_key, None)
                waited = max(0.0, float(getattr(self.warehouse, "_tick", 0.0)) - entry.entered_at)
                self._bump("charging_waiting_seconds", waited)
        finally:
            self._dispatching = False
            self._last_occupancy_signature = self.occupancy_signature()

    def notify_availability_changed(self, reason: str) -> None:
        self.dispatch(reason=reason)

    def occupancy_signature(self):
        chargers = set(getattr(self.warehouse, "active_charger_cells", None) or ())
        if not chargers:
            chargers = set(getattr(self.warehouse, "charger_cells", ()) or ())
        occupied = []
        for obj in getattr(self.warehouse, "get_movable_objects", lambda: ())():
            if getattr(obj, "object_type", None) != "robot":
                continue
            cell = (round(getattr(obj, "pos_x", -1)), round(getattr(obj, "pos_y", -1)))
            if cell in chargers:
                occupied.append((cell, stable_robot_id(obj)))
        claims = tuple(sorted((cell, str(owner)) for cell, owner in getattr(self.warehouse, "occupied_chargers", {}).items()))
        station_use = tuple(sorted(
            (cell, tuple(sorted(str(robot_id) for robot_id in getattr(station, "robot_ids", {}))))
            for cell, station in getattr(self.warehouse, "charger_station_by_cell", {}).items()
        ))
        return (tuple(sorted(occupied)), claims, station_use)

    def reconsider_if_occupancy_changed(self) -> None:
        signature = self.occupancy_signature()
        if signature != self._last_occupancy_signature:
            self._last_occupancy_signature = signature
            self.dispatch(reason="charger_occupancy_changed")

    def terminal_remove(self, robot: Any) -> None:
        self.remove(robot)
        robot._charging_lifecycle_state = "terminal_cleanup"
