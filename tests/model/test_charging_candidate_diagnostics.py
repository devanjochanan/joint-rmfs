"""Part B — energy/route candidate diagnostics (instrument-only).

These pin the four distinct candidate-rejection counters onto the *existing*
dispatch decisions without changing the feasibility threshold or selection
behavior. The no-feasible-charger path must fall back to the existing
authoritative battery-death behavior (a robot dies when its battery reaches 0),
never teleport / grant free energy / silently claim / retry every tick.
"""

from types import SimpleNamespace

from model.robot import Robot
from src.rmfs.decisions.charging.dispatch import (
    CHARGING_COUNTER_FIELDS,
    ChargingDispatcher,
    initial_charging_counters,
)


class _RouteGraph:
    """Two-hop route from (5,5) to (7,5)."""

    def dijkstra(self, start, end):
        return ["5,5", "6,5", "7,5"]


class _NoRouteGraph:
    def dijkstra(self, start, end):
        return None


def _dispatch_warehouse():
    warehouse = SimpleNamespace(
        _tick=10.0,
        charging_counters=initial_charging_counters(),
        charging_enabled=True,
        disable_active_charging=False,
        charger_cells=set(),
        active_charger_cells=set(),
        occupied_chargers={},
        charger_station_by_cell={},
        charger_route_graph_by_cell={},
        zoning=False,
        tick_to_second=0.15,
        landscape=SimpleNamespace(get_neighbor_object=lambda *args: None),
    )
    robots = []
    warehouse.get_movable_objects = lambda: robots
    warehouse.robots = robots
    return warehouse


def _robot_on(warehouse, monkeypatch, *, pos=(5, 5)):
    robot = Robot(warehouse)
    robot.setUniverse(warehouse)
    robot.id = 1
    robot._id = 1
    robot.pos_x, robot.pos_y = pos
    robot.heading = 0
    warehouse.robots.append(robot)
    monkeypatch.setattr(
        robot, "set_move",
        lambda dest, graph: setattr(robot, "route_stop_points", [dest]),
    )
    return robot


def test_initial_charging_counters_zero_initialized():
    counters = initial_charging_counters()
    for field in CHARGING_COUNTER_FIELDS:
        assert counters[field] == 0
    # The four Part B diagnostics are part of the canonical contract.
    for field in (
        "charger_candidate_unroutable",
        "charger_candidate_occupied",
        "charger_candidate_insufficient_energy",
        "charger_assignment_no_feasible_candidate",
    ):
        assert field in CHARGING_COUNTER_FIELDS


def test_insufficient_energy_is_distinct_from_unroutable(monkeypatch):
    warehouse = _dispatch_warehouse()
    warehouse.charger_cells = {(7, 5)}
    warehouse.active_charger_cells = {(7, 5)}
    warehouse.charger_route_graph_by_cell = {(7, 5): "standard"}
    warehouse.graph = _RouteGraph()
    warehouse.graph_pod = _RouteGraph()
    robot = _robot_on(warehouse, monkeypatch)

    robot.battery_level_j = 1.0  # cannot afford the two-hop route
    assert robot._assign_charger_from_fifo() is False
    c = warehouse.charging_counters
    # energy gate fired; legacy + new name both incremented, route path did not
    assert c["charger_energy_infeasible_candidates"] == 1
    assert c["charger_candidate_insufficient_energy"] == 1
    assert c["charger_candidate_unroutable"] == 0
    # a whole pass found no feasible candidate; no claim was created
    assert c["charger_assignment_no_feasible_candidate"] == 1
    assert warehouse.occupied_chargers == {}


def test_unroutable_is_distinct_from_insufficient_energy(monkeypatch):
    warehouse = _dispatch_warehouse()
    warehouse.charger_cells = {(7, 5)}
    warehouse.active_charger_cells = {(7, 5)}
    warehouse.charger_route_graph_by_cell = {(7, 5): "standard"}
    warehouse.graph = _NoRouteGraph()
    warehouse.graph_pod = _NoRouteGraph()
    robot = _robot_on(warehouse, monkeypatch)

    robot.battery_level_j = robot.BATTERY_CAPACITY_J  # plenty of energy
    assert robot._assign_charger_from_fifo() is False
    c = warehouse.charging_counters
    assert c["charger_candidate_unroutable"] == 1
    assert c["charger_route_failures"] == 1  # legacy name preserved
    assert c["charger_candidate_insufficient_energy"] == 0
    assert c["charger_assignment_no_feasible_candidate"] == 1
    assert warehouse.occupied_chargers == {}


def test_occupied_candidate_counted(monkeypatch):
    warehouse = _dispatch_warehouse()
    warehouse.charger_cells = {(7, 5)}
    warehouse.active_charger_cells = {(7, 5)}
    warehouse.charger_route_graph_by_cell = {(7, 5): "standard"}
    warehouse.graph = _RouteGraph()
    warehouse.graph_pod = _RouteGraph()
    warehouse.occupied_chargers = {(7, 5): 999}  # claimed by another robot
    robot = _robot_on(warehouse, monkeypatch)

    robot.battery_level_j = robot.BATTERY_CAPACITY_J
    assert robot._assign_charger_from_fifo() is False
    c = warehouse.charging_counters
    assert c["charger_candidate_occupied"] == 1
    assert c["charger_assignment_no_feasible_candidate"] == 1
    # still someone else's claim — untouched
    assert warehouse.occupied_chargers == {(7, 5): 999}


def test_feasible_claim_records_no_rejection(monkeypatch):
    warehouse = _dispatch_warehouse()
    warehouse.charger_cells = {(7, 5)}
    warehouse.active_charger_cells = {(7, 5)}
    warehouse.charger_route_graph_by_cell = {(7, 5): "standard"}
    warehouse.graph = _RouteGraph()
    warehouse.graph_pod = _RouteGraph()
    robot = _robot_on(warehouse, monkeypatch)

    robot.battery_level_j = robot.BATTERY_CAPACITY_J
    assert robot._assign_charger_from_fifo() is True
    c = warehouse.charging_counters
    assert warehouse.occupied_chargers == {(7, 5): 1}
    assert c["charger_claims_created"] == 1
    assert c["charger_candidate_unroutable"] == 0
    assert c["charger_candidate_occupied"] == 0
    assert c["charger_candidate_insufficient_energy"] == 0
    assert c["charger_assignment_no_feasible_candidate"] == 0
