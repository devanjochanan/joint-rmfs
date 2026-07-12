import json
from types import SimpleNamespace

import pandas as pd
import pytest

from model.inventory import Inventory
from model.robot import Robot
from scripts.data.build_adaptive_hybrid import build as build_adaptive
from scripts.data.build_baseline_random import build as build_reference
from src.rmfs.app import netlogo_api
from src.rmfs.decisions.charging.dispatch import ChargingDispatcher
from src.rmfs.decisions.task_allocation.committed_next import (
    CommittedNextInvariantError,
    CommittedNextRegistry,
    CommittedNextReservation,
    STATUS_CANCELLED,
    STATUS_COMMITTED,
)
from src.rmfs.runtime_io import RunContext
from src.rmfs.runtime_io.detail_db import configure_detail_db


class _QueueRobot:
    object_type = "robot"

    def __init__(self, robot_id, assignments):
        self._id = robot_id
        self.id = robot_id
        self.pos_x = robot_id
        self.pos_y = 0
        self._charging_episode_id = None
        self._charging_queue_entered_at = None
        self._charging_lifecycle_state = None
        self.current_state = "idle"
        self.assignments = assignments
        self.calls = 0

    def _enter_waiting_for_charger(self, reason):
        self.current_state = "waiting_for_charger"

    def _assign_charger_from_fifo(self):
        self.calls += 1
        result = self.assignments.pop(0) if self.assignments else False
        if result:
            self.current_state = "going_to_charge"
        return result


def _dispatch_warehouse():
    warehouse = SimpleNamespace(
        _tick=10.0,
        charging_counters={},
        charging_enabled=True,
        disable_active_charging=False,
        charger_cells=set(),
        active_charger_cells=set(),
        occupied_chargers={},
        charger_station_by_cell={},
    )
    robots = []
    warehouse.get_movable_objects = lambda: robots
    warehouse.robots = robots
    return warehouse


def test_fifo_is_oldest_then_stable_robot_id_and_does_not_poll_per_tick():
    warehouse = _dispatch_warehouse()
    dispatcher = ChargingDispatcher(warehouse)
    older_high_id = _QueueRobot(9, [False, False, True])
    newer_low_id = _QueueRobot(1, [True])
    warehouse.robots.extend([older_high_id, newer_low_id])

    dispatcher.enqueue(older_high_id)
    warehouse._tick = 11.0
    dispatcher.enqueue(newer_low_id)
    assert dispatcher.ordered_entries()[0].robot is older_high_id
    evaluations = warehouse.charging_counters["charging_assignment_evaluations"]
    for _ in range(20):
        dispatcher.reconsider_if_occupancy_changed()
    assert warehouse.charging_counters["charging_assignment_evaluations"] == evaluations

    dispatcher.notify_availability_changed("charger_release")
    assert older_high_id.current_state == "going_to_charge"
    assert newer_low_id.current_state == "going_to_charge"

    warehouse = _dispatch_warehouse()
    dispatcher = ChargingDispatcher(warehouse)
    high = _QueueRobot(9, [False, False, True])
    low = _QueueRobot(1, [False, False, True])
    warehouse.robots.extend([high, low])
    dispatcher.enqueue(high)
    dispatcher.enqueue(low)
    entries = dispatcher.ordered_entries()
    assert [entry.robot._id for entry in entries] == [1, 9]


def test_one_episode_and_no_duplicate_queue_entry():
    warehouse = _dispatch_warehouse()
    dispatcher = ChargingDispatcher(warehouse)
    robot = _QueueRobot(1, [False])
    warehouse.robots.append(robot)
    assert dispatcher.enqueue(robot) is True
    assert dispatcher.enqueue(robot) is False
    assert warehouse.charging_counters["charging_episodes_created"] == 1
    assert warehouse.charging_counters["charging_requests"] == 1


def _install_reservation(registry, inventory, robot):
    pod = SimpleNamespace(pod_id="pod-1")
    job = SimpleNamespace(pod=pod, job_id="job-1")
    reservation = CommittedNextReservation(
        reservation_id="cnr-1",
        owner_robot_id=str(robot._id),
        job=job,
        job_id="job-1",
        pod_id="pod-1",
        picking_station_id="picker-1",
        original_queue_index=0,
        created_time_seconds=1.0,
    )
    registry.reservations_by_id[reservation.reservation_id] = reservation
    registry.robot_id_to_reservation[str(robot._id)] = reservation.reservation_id
    registry.pod_id_to_reservation[reservation.pod_id] = reservation.reservation_id
    registry.job_id_to_reservation[reservation.job_id] = reservation.reservation_id
    registry._set_markers(reservation)
    return reservation, job, pod


def test_charging_cancellation_restores_once_and_blocks_other_causes():
    registry = CommittedNextRegistry()
    inventory = SimpleNamespace(job_queue=[], _tick=10.0)
    robot = SimpleNamespace(_id=7, universe=inventory)
    reservation, job, pod = _install_reservation(registry, inventory, robot)

    registry.cancel_for_robot(robot, "reservation_cancelled_charging")
    assert reservation.status == STATUS_CANCELLED
    assert inventory.job_queue == [job]
    assert registry.lifecycle_counters["committed_next_jobs_restored"] == 1
    assert job.committed_next_reservation_id is None
    assert pod.committed_next_reservation_id is None
    assert registry.get_for_robot(robot) is None
    registry.cancel_for_robot(robot, "reservation_cancelled_charging")
    assert inventory.job_queue == [job]

    for reason in (
        "scheduler_pass", "station_queue_changed", "replenishment_activity",
        "path_delay", "rts_return_rollback",
    ):
        guarded = CommittedNextRegistry()
        guarded_robot = SimpleNamespace(_id=7, universe=SimpleNamespace(job_queue=[]))
        guarded_reservation, guarded_job, _ = _install_reservation(
            guarded, guarded_robot.universe, guarded_robot
        )
        with pytest.raises(CommittedNextInvariantError):
            guarded.cancel_for_robot(guarded_robot, reason)
        assert guarded_reservation.status == STATUS_COMMITTED
        assert guarded.get_for_robot(guarded_robot) is guarded_reservation
        assert guarded_job not in guarded_robot.universe.job_queue


def test_fifo_entry_is_the_charging_cancellation_seam():
    warehouse = _dispatch_warehouse()
    warehouse.job_queue = []
    warehouse.charging_dispatcher = ChargingDispatcher(warehouse)
    registry = warehouse.committed_next_registry = CommittedNextRegistry()
    censored = []
    warehouse.rts_rollout_runtime = SimpleNamespace(
        censor_pending_for_robot=lambda **kwargs: censored.append(kwargs)
    )
    robot = Robot(warehouse)
    robot.setUniverse(warehouse)
    robot.id = 7
    robot._id = 7
    robot.pos_x = 5
    robot.pos_y = 5
    robot.current_state = "idle"
    warehouse.robots.append(robot)
    reservation, job, _ = _install_reservation(registry, warehouse, robot)

    robot._queue_for_charging()
    assert reservation.status == STATUS_CANCELLED
    assert warehouse.job_queue == [job]
    assert censored[0]["reason"] == "reservation_cancelled_charging"
    assert warehouse.charging_counters["committed_next_cancelled_for_charging"] == 1
    robot._queue_for_charging()
    assert warehouse.job_queue == [job]
    assert len(censored) == 1


def test_activation_identity_drift_is_an_invariant_not_a_cancellation():
    registry = CommittedNextRegistry()
    inventory = SimpleNamespace(job_queue=[], _tick=10.0)
    robot = SimpleNamespace(_id=7, universe=inventory)
    reservation, _, _ = _install_reservation(registry, inventory, robot)
    with pytest.raises(CommittedNextInvariantError):
        registry.activate_for_robot(inventory, robot)
    assert reservation.status == STATUS_COMMITTED
    assert registry.get_for_robot(robot) is reservation


def _draw_generated_treatment(tmp_path, monkeypatch, config, source):
    tmp_path.mkdir(parents=True, exist_ok=True)
    config_path = tmp_path / f"{source}.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    context = RunContext.isolated(tmp_path / "runtime", repo_root=netlogo_api.REPO_ROOT if hasattr(netlogo_api, "REPO_ROOT") else None)
    context.ensure_runtime_dirs()
    configure_detail_db(enabled=False, db_path=context.sqlite_db)
    netlogo_api.configure_run_context(context)
    monkeypatch.setenv("RMFS_CHARGING_PLACEMENT_SOURCE", source)
    monkeypatch.setenv("RMFS_CHARGING_CONFIG", str(config_path))
    monkeypatch.setenv("RMFS_CHARGING_ENABLED", "1")
    warehouse = Inventory(runtime_paths={}, sqlite_db_path=str(context.sqlite_db))
    netlogo_api.draw_storage_from_generated_file(warehouse)
    return warehouse


def test_declared_reference_and_adaptive_p2_p3_chargers_validate(tmp_path, monkeypatch):
    grid_path = RunContext.default().generated_pod_csv
    grid = pd.read_csv(grid_path, header=None).to_numpy(dtype=int)

    adaptive, p3_count, p2_count = build_adaptive(grid, 20, 0.6)
    adaptive_warehouse = _draw_generated_treatment(
        tmp_path / "adaptive", monkeypatch, adaptive, "generated_salsa_adaptive"
    )
    assert (p3_count, p2_count) == (6, 6)
    assert adaptive_warehouse.charging_dispatch_validation == {
        "active_count": 12, "p2_count": 6, "p3_count": 6,
    }
    assert adaptive_warehouse.active_charger_cells == adaptive_warehouse.charger_cells
    for cell, graph_name in adaptive_warehouse.charger_route_graph_by_cell.items():
        assert adaptive_warehouse.charger_exit_cell_by_cell[cell] != cell
        if graph_name == "pod":
            assert cell in adaptive_warehouse.charger_station_by_cell

    reference = build_reference(grid.tolist(), num_chargers=10, seed=1)
    reference_warehouse = _draw_generated_treatment(
        tmp_path / "reference", monkeypatch, reference, "generated_reference"
    )
    assert reference_warehouse.charging_dispatch_validation["active_count"] == 10
    assert reference_warehouse.charging_dispatch_validation["p3_count"] == 0
    assert all(
        exit_cell != cell
        for cell, exit_cell in reference_warehouse.charger_exit_cell_by_cell.items()
    )


class _RouteGraph:
    def dijkstra(self, start, end):
        return ["5,5", "6,5", "7,5"]


def test_charger_claim_is_exclusive_and_requires_route_energy(monkeypatch):
    warehouse = _dispatch_warehouse()
    warehouse.active_charger_cells = {(7, 5)}
    warehouse.charger_cells = {(7, 5)}
    warehouse.charger_route_graph_by_cell = {(7, 5): "standard"}
    warehouse.graph = _RouteGraph()
    warehouse.graph_pod = _RouteGraph()
    warehouse.zoning = False
    warehouse.tick_to_second = 0.15
    warehouse.landscape = SimpleNamespace(get_neighbor_object=lambda *args: None)
    robot = Robot(warehouse)
    robot.setUniverse(warehouse)
    robot.id = 1
    robot._id = 1
    robot.pos_x = 5
    robot.pos_y = 5
    robot.heading = 0
    warehouse.robots.append(robot)
    monkeypatch.setattr(robot, "set_move", lambda dest, graph: setattr(robot, "route_stop_points", [dest]))

    robot.battery_level_j = 1.0
    assert robot._assign_charger_from_fifo() is False
    assert warehouse.occupied_chargers == {}
    assert warehouse.charging_counters["charger_energy_infeasible_candidates"] == 1

    robot.battery_level_j = robot.BATTERY_CAPACITY_J * 0.18
    assert robot._assign_charger_from_fifo() is True
    assert warehouse.occupied_chargers == {(7, 5): 1}
    second = Robot(warehouse)
    second.setUniverse(warehouse)
    second.id = 2
    second._id = 2
    second.pos_x = 5
    second.pos_y = 5
    second.heading = 0
    warehouse.robots.append(second)
    assert second._assign_charger_from_fifo() is False


def _charging_robot_at_claim(battery_pct):
    warehouse = _dispatch_warehouse()
    warehouse.tick_to_second = 0.15
    warehouse.charger_cells = {(7, 5)}
    warehouse.active_charger_cells = {(7, 5)}
    warehouse.charger_exit_cell_by_cell = {(7, 5): (8, 5)}
    warehouse.charger_route_graph_by_cell = {(7, 5): "standard"}
    warehouse.graph = object()
    warehouse.graph_pod = object()
    warehouse.job_queue = []
    warehouse.zoning = False
    robot = Robot(warehouse)
    robot.setUniverse(warehouse)
    robot.id = 1
    robot._id = 1
    robot.pos_x = 7
    robot.pos_y = 5
    robot.heading = 0
    robot.route_stop_points = []
    robot._claimed_charger = (7, 5)
    robot._charging_episode_id = "charge:1:0"
    robot._charging_queue_entered_at = 0.0
    robot.BATTERY_LOW_PCT = 18.0
    robot.BATTERY_CHARGED_PCT = 60.0
    robot.BATTERY_INTERRUPT_PCT = 50.0
    robot.battery_level_j = robot.BATTERY_CAPACITY_J * battery_pct / 100.0
    warehouse.occupied_chargers[(7, 5)] = 1
    warehouse.robots.append(robot)
    warehouse.charging_dispatcher = ChargingDispatcher(warehouse)
    robot.set_move = lambda destination, graph: setattr(robot, "route_stop_points", [destination])
    return warehouse, robot


def test_physical_session_start_completion_interruption_and_release():
    warehouse, robot = _charging_robot_at_claim(30.0)
    robot.current_state = "going_to_charge"
    robot.movementPlan()
    assert robot.current_state == "physically_charging"
    assert warehouse.charging_counters["charging_sessions_started"] == 1
    assert robot.is_unavailable_for_work_due_to_charging() is True

    robot.battery_level_j = robot.BATTERY_CAPACITY_J * robot.BATTERY_CHARGED_PCT / 100.0
    robot.movementPlan()
    assert robot.current_state == "leaving_charger"
    assert robot._claimed_charger == (7, 5)
    assert warehouse.occupied_chargers == {(7, 5): 1}
    robot.pos_x, robot.pos_y = 8, 5
    robot.route_stop_points = []
    robot.movementPlan()
    assert robot.current_state == "idle"
    assert robot._claimed_charger is None
    assert warehouse.occupied_chargers == {}
    assert warehouse.charging_counters["charging_sessions_completed"] == 1

    warehouse, robot = _charging_robot_at_claim(robot.BATTERY_INTERRUPT_PCT)
    robot.current_state = "physically_charging"
    robot._charging_lifecycle_state = "physically_charging"
    warehouse.job_queue.append(object())
    robot.movementPlan()
    assert robot.current_state == "leaving_charger"
    assert warehouse.charging_counters["charging_sessions_interrupted"] == 1


def test_leaving_charger_timeout_retries_instead_of_failing_run():
    warehouse, robot = _charging_robot_at_claim(60.0)
    robot.current_state = "leaving_charger"
    robot._charging_lifecycle_state = "leaving_charger"
    robot._leaving_started_at = 0.0
    robot._leaving_progress_at = 0.0
    robot._leaving_progress_pos = (robot.pos_x, robot.pos_y)
    warehouse._tick = robot.LEAVING_CHARGER_MAX_SECONDS

    robot._charging_pre_move()
    assert robot.current_state == "leaving_charger"
    assert robot._claimed_charger == (7, 5)
    assert robot.route_stop_points
    assert warehouse.charging_counters["charging_departure_no_progress_retries"] == 1


def test_busy_low_soc_defers_without_entering_fifo():
    warehouse = _dispatch_warehouse()
    warehouse.tick_to_second = 0.15
    warehouse.disable_active_charging = False
    warehouse.charging_enabled = True
    warehouse.charging_dispatcher = ChargingDispatcher(warehouse)
    robot = Robot(warehouse)
    robot.setUniverse(warehouse)
    robot.id = 1
    robot._id = 1
    robot.current_state = "returning_pod"
    robot.job = SimpleNamespace(is_finished=False)
    robot.battery_level_j = robot.BATTERY_CAPACITY_J * robot.BATTERY_LOW_PCT / 100.0
    warehouse.robots.append(robot)
    robot._charging_pre_move()
    assert robot.charge_after_current_task is True
    assert warehouse.charging_dispatcher.ordered_entries() == ()
    assert robot._charging_queue_entered_at is None


def test_idle_interrupt_soc_queues_charging_before_low_soc():
    warehouse = _dispatch_warehouse()
    warehouse.tick_to_second = 0.15
    warehouse.disable_active_charging = False
    warehouse.charging_enabled = True
    warehouse.charging_dispatcher = ChargingDispatcher(warehouse)
    robot = Robot(warehouse)
    robot.setUniverse(warehouse)
    robot.id = 1
    robot._id = 1
    robot.current_state = "idle"
    robot.job = None
    robot.BATTERY_LOW_PCT = 18.0
    robot.BATTERY_INTERRUPT_PCT = 50.0
    robot.battery_level_j = robot.BATTERY_CAPACITY_J * 0.49
    warehouse.robots.append(robot)

    assert robot.is_unavailable_for_work_due_to_charging() is True
    parked = robot._charging_pre_move()

    assert parked is True
    assert robot.current_state == "waiting_for_charger"
    assert warehouse.charging_counters["charging_requests"] == 1
    assert warehouse.charging_counters["forced_charging_requests"] == 1
    assert warehouse.charging_counters["robots_sent_to_charge_before_job"] == 1


def test_busy_interrupt_soc_defers_charging_before_low_soc():
    warehouse = _dispatch_warehouse()
    warehouse.tick_to_second = 0.15
    warehouse.disable_active_charging = False
    warehouse.charging_enabled = True
    warehouse.charging_dispatcher = ChargingDispatcher(warehouse)
    robot = Robot(warehouse)
    robot.setUniverse(warehouse)
    robot.id = 1
    robot._id = 1
    robot.current_state = "returning_pod"
    robot.job = SimpleNamespace(is_finished=False)
    robot.BATTERY_LOW_PCT = 18.0
    robot.BATTERY_INTERRUPT_PCT = 50.0
    robot.battery_level_j = robot.BATTERY_CAPACITY_J * 0.49
    warehouse.robots.append(robot)

    parked = robot._charging_pre_move()

    assert parked is False
    assert robot.charge_after_current_task is True
    assert warehouse.charging_counters["forced_charging_requests"] == 1
    assert warehouse.charging_dispatcher.ordered_entries() == ()


def test_battery_death_without_charge_request_is_counted():
    warehouse = _dispatch_warehouse()
    warehouse.tick_to_second = 0.15
    warehouse.disable_active_charging = False
    warehouse.charging_enabled = True
    warehouse.charging_dispatcher = ChargingDispatcher(warehouse)
    warehouse.landscape = SimpleNamespace(setObject=lambda *args, **kwargs: None)
    warehouse.committed_next_registry = None
    warehouse.rts_rollout_runtime = None
    robot = Robot(warehouse)
    robot.setUniverse(warehouse)
    robot.id = 1
    robot._id = 1
    robot.current_state = "idle"
    robot.job = None
    robot.battery_level_j = 0.0
    robot.velocity = 0
    robot.acceleration = 0
    robot.heading = 0
    robot.load_mass = 0
    warehouse.robots.append(robot)

    parked = robot._charging_pre_move()

    assert parked is True
    assert robot.current_state == "dead"
    assert warehouse.charging_counters["robots_died_from_battery"] == 1
    assert warehouse.charging_counters["robots_died_without_charge_request"] == 1
