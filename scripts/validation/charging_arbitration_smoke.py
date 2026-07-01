#!/usr/bin/env python3
"""Targeted smoke test for charging arbitration under different battery/reservation scenarios."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

os.environ["RMFS_DETAIL_DB"] = "0"
os.environ["RMFS_FAST_TRAIN"] = "1"

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from engine.netlogo_coordinate import NetLogoCoordinate
from model.inventory import Inventory
from model.pod import Pod
from model.robot import Robot
from model.robot_job import RobotJob
from model.station import Station
from src.rmfs.decisions.rts.types import RTSDestinationContext, RTSDecision
from src.rmfs.rl.rts.action_space import STORE
from src.rmfs.rl.rts.outcome_tracker import RTSRolloutRuntime
from src.rmfs.rl.rts.runtime_config import RTSRuntimeConfig
from src.rmfs.decisions.task_allocation.committed_next import (
    get_committed_next_reservation,
    get_committed_next_job,
)

# Simple Graph Mock
class MockGraph:
    def dijkstra(self, start, end, avoid=None):
        return [start, end]
    def dijkstra_modified(self, start, end, penalties, zone_boundary, avoid=None):
        return self.dijkstra(start, end, avoid)

# Simple RTS Policy Mock
class MockBranchPolicy:
    def select_destination(self, context):
        storage = context.warehouse.storage_manager.storages[0]
        return RTSDecision(
            storage=storage,
            destination=NetLogoCoordinate(storage.pos_x, storage.pos_y),
            policy_name="mock_rts",
            mode="rl",
            action_index=0,
            branch="store",
            zone_id="zone-a",
            storage_id="10:5",
            replenishment_station=None,
            replenishment_station_id=None,
            action_context_id="ctx-1",
            action_context_version=1,
            next_job_proposal_id="proposal-1",
            metadata={
                "selected_action_index": 0,
            }
        )

def setup_mock_universe(committed_enabled=True):
    tmp = tempfile.TemporaryDirectory()
    root = Path(tmp.name)
    inv = Inventory(
        runtime_paths={
            "assign_order_csv": str(root / "assign_order.csv"),
            "pod_info_csv": str(root / "pod_info.csv"),
            "generated_order_csv": str(root / "generated_order.csv"),
        },
        sqlite_db_path=str(root / "warehouse.db"),
    )
    inv._smoke_tmp = tmp
    inv.fast_train = True
    inv.tick_to_second = 1.0
    inv.committed_next_reservations_enabled = committed_enabled
    inv.charging_enabled = True
    inv.disable_active_charging = False

    # Ensure generated_order_csv exists
    Path(inv.generated_order_csv).write_text("order_id,item_id,item_quantity,order_arrival,status,assigned_station,assigned_pod\n")

    # Register 1 charger position at (12, 12)
    inv.charger_cells.add((12, 12))

    # Initialize DB tables
    from model.tools.pod_location import initialize_pod_location_table
    from model.tools.pod_travel import initialize_pod_travel_table
    from model.tools.job_task import initialize_job_task_table
    from model.tools.pre_assign import initialize_pre_assign_table
    from model.tools.order_history import initialize_order_history_table

    initialize_pod_location_table(None, db_path=inv.sqlite_db_path)
    initialize_pod_travel_table(None, db_path=inv.sqlite_db_path)
    initialize_job_task_table(None, db_path=inv.sqlite_db_path)
    initialize_pre_assign_table(None, db_path=inv.sqlite_db_path)
    initialize_order_history_table(None, db_path=inv.sqlite_db_path)

    graph = MockGraph()
    inv.graph = graph
    inv.graph_pod = graph
    inv.rts_rollout_runtime = RTSRolloutRuntime(
        config=RTSRuntimeConfig(
            policy_mode="current_probe",
            rollout_enabled=True,
            zone_ids=("zone-a",),
            committed_next_reservations_enabled=committed_enabled,
        ),
        runtime_root=root,
    )

    picker = Station(1, "picker")
    picker.pos_x = 5
    picker.pos_y = 5
    picker.coordinate = NetLogoCoordinate(5, 5)
    picker.short_path = [NetLogoCoordinate(5, 5)]
    inv.station_manager.add_station(picker)

    # Store pod
    pod1 = Pod(1)
    pod1.pos_x = 5
    pod1.pos_y = 5
    inv.pod_manager.add_pod(pod1)

    storage1 = inv.storage_manager.createStorage(10, 5)
    storage1.rts_zone_id = "zone-a"
    inv.storage_manager.addPodToStorage(pod1, storage1)

    # Queue a job
    pod2 = Pod(2)
    pod2.pos_x = 1
    pod2.pos_y = 1
    inv.pod_manager.add_pod(pod2)
    storage2 = inv.storage_manager.createStorage(1, 1)
    storage2.rts_zone_id = "zone-a"
    inv.storage_manager.addPodToStorage(pod2, storage2)
    inv.pod_manager.mark_pod_not_available(pod2)

    job2 = RobotJob(pod2.coordinate, station_id=picker.station_id, pod=pod2)
    job2.add_picking_task(901, 101, 1)
    inv.job_queue.append(job2)

    inv.storage_manager.initStorageManager()

    robot = Robot(inv)
    robot.pos_x = picker.pos_x
    robot.pos_y = picker.pos_y
    robot.coordinate = NetLogoCoordinate(robot.pos_x, robot.pos_y)
    inv.addObject(robot)

    inv.rts_policy = MockBranchPolicy()

    return inv, picker, robot, pod1, pod2, job2

def assert_ownership_invariants(inv):
    # 1. no duplicate object identity in job_queue
    job_ids = [id(j) for j in inv.job_queue]
    assert len(job_ids) == len(set(job_ids)), "Duplicate job identity in job_queue"

    # 2. no job assigned to two robots
    assigned_jobs = []
    for o in inv.get_movable_objects():
        if o.object_type == "robot" and o.job is not None and not getattr(o.job, "is_finished", False):
            assert o.job not in assigned_jobs, f"Job {o.job} assigned to multiple robots"
            assigned_jobs.append(o.job)

    # 3. no robot has both active customer job and charger claim
    for o in inv.get_movable_objects():
        if o.object_type == "robot":
            has_active_customer_job = o.job is not None and not getattr(o.job, "is_finished", False)
            has_charger_claim = getattr(o, "_claimed_charger", None) is not None or getattr(o, "is_charging", False) or o.current_state in {"going_to_charge", "charging"}
            assert not (has_active_customer_job and has_charger_claim), f"Robot {o} has both active job and charger claim"

    # 4. no charger has two robot owners
    charger_claims = {}
    for o in inv.get_movable_objects():
        if o.object_type == "robot" and getattr(o, "_claimed_charger", None) is not None:
            charger = o._claimed_charger
            assert charger not in charger_claims, f"Charger {charger} claimed by multiple robots"
            charger_claims[charger] = o

    # 5. no pod is both carried and available
    carried_pods = set()
    for o in inv.get_movable_objects():
        if o.object_type == "robot" and o.current_state in {"delivering_pod", "station_processing", "returning_pod"} and o.job is not None:
            if o.job.pod is not None:
                carried_pods.add(o.job.pod.pod_id)
    for pod_id in carried_pods:
        assert not inv.pod_manager.is_idle(pod_id), f"Carried pod {pod_id} is marked as idle/available"

def test_scenario_a_low_battery_while_idle():
    inv, picker, robot, pod1, pod2, job2 = setup_mock_universe()
    robot.current_state = "idle"
    robot.job = None
    robot.battery_level_j = robot.BATTERY_CAPACITY_J * 0.1
    inv.tick()
    assert robot.current_state == "going_to_charge"
    assert robot._claimed_charger == (12, 12)
    assert_ownership_invariants(inv)
    print("Scenario A passed")

def test_scenario_b_low_battery_during_taking_pod():
    inv, picker, robot, pod1, pod2, job2 = setup_mock_universe()
    robot.job = job2
    robot.current_state = "taking_pod"
    robot.battery_level_j = robot.BATTERY_CAPACITY_J * 0.1

    initial_queue_len = len(inv.job_queue)
    inv.tick()
    assert robot.job is job2
    assert robot.current_state == "taking_pod"
    assert robot.charge_after_current_task is True
    assert len(inv.job_queue) == initial_queue_len
    assert_ownership_invariants(inv)
    print("Scenario B passed")

def test_scenario_c_low_battery_during_delivering_pod():
    inv, picker, robot, pod1, pod2, job2 = setup_mock_universe()
    robot.job = job2
    robot.current_state = "delivering_pod"
    robot.battery_level_j = robot.BATTERY_CAPACITY_J * 0.1

    inv.tick()
    assert robot.job is job2
    assert robot.current_state == "delivering_pod"
    assert robot.charge_after_current_task is True
    assert_ownership_invariants(inv)
    print("Scenario C passed")

def test_scenario_d_low_battery_during_returning_pod():
    inv, picker, robot, pod1, pod2, job2 = setup_mock_universe()
    job1 = RobotJob(pod1.coordinate, station_id=picker.station_id, pod=pod1)
    job1.set_job_finish()
    robot.job = job1
    robot.current_state = "returning_pod"
    robot.destination = NetLogoCoordinate(10, 5)
    robot.battery_level_j = robot.BATTERY_CAPACITY_J * 0.1
    robot.charge_after_current_task = True

    robot.advance_state()
    assert robot.current_state == "going_to_charge"
    assert robot._claimed_charger == (12, 12)
    assert robot.charge_after_current_task is False
    assert_ownership_invariants(inv)
    print("Scenario D passed")

def test_scenario_e_rts_committed_next_plus_queued_charging():
    inv, picker, robot, pod1, pod2, job2 = setup_mock_universe()

    job1 = RobotJob(pod1.coordinate, station_id=picker.station_id, pod=pod1)
    job1.set_job_finish()
    robot.job = job1
    robot.current_state = "returning_pod"
    robot.destination = NetLogoCoordinate(10, 5)
    robot.battery_level_j = robot.BATTERY_CAPACITY_J * 0.1
    robot.charge_after_current_task = True

    reservation = inv.ensure_committed_next_reservation(robot)
    assert reservation is not None
    assert get_committed_next_reservation(inv, robot) is not None
    assert get_committed_next_job(inv, robot) is job2

    initial_queue_len = len(inv.job_queue)

    # Assert quantity doesn't change during reservation cancellation
    initial_qty = {sku: val["current_qty"] for sku, val in pod2.skus.items()}

    robot.advance_state()

    assert get_committed_next_reservation(inv, robot) is None
    assert len(inv.job_queue) == initial_queue_len
    assert job2 in inv.job_queue
    assert robot.current_state == "going_to_charge"
    assert robot._claimed_charger == (12, 12)
    assert robot.charge_after_current_task is False

    # Assert no inventory quantity changes during reservation cancellation
    for sku, val in pod2.skus.items():
        assert val["current_qty"] == initial_qty[sku]

    assert_ownership_invariants(inv)
    print("Scenario E passed")

def test_scenario_f_charger_unavailable():
    inv, picker, robot, pod1, pod2, job2 = setup_mock_universe()
    inv.occupied_chargers[(12, 12)] = 999

    robot.current_state = "idle"
    robot.job = None
    robot.battery_level_j = robot.BATTERY_CAPACITY_J * 0.1
    robot.charge_after_current_task = True

    inv.tick()

    assert robot.current_state == "idle"
    assert robot.charge_after_current_task is True
    assert robot._claimed_charger is None
    assert job2 in inv.job_queue
    assert robot.job is None
    assert_ownership_invariants(inv)

    del inv.occupied_chargers[(12, 12)]
    inv.tick()
    assert robot.current_state == "going_to_charge"
    assert robot._claimed_charger == (12, 12)
    assert robot.charge_after_current_task is False
    assert_ownership_invariants(inv)
    print("Scenario F passed")

def test_scenario_g_sufficient_battery():
    inv, picker, robot, pod1, pod2, job2 = setup_mock_universe()

    job1 = RobotJob(pod1.coordinate, station_id=picker.station_id, pod=pod1)
    job1.set_job_finish()
    robot.job = job1
    robot.current_state = "returning_pod"
    robot.destination = NetLogoCoordinate(10, 5)
    robot.battery_level_j = robot.BATTERY_CAPACITY_J * 0.9
    robot.charge_after_current_task = False

    reservation = inv.ensure_committed_next_reservation(robot)
    assert reservation is not None

    robot.advance_state()
    assert robot.current_state == "taking_pod"
    assert robot.job is job2
    assert get_committed_next_reservation(inv, robot) is None
    assert job2 not in inv.job_queue
    assert_ownership_invariants(inv)
    print("Scenario G passed")

def main():
    test_scenario_a_low_battery_while_idle()
    test_scenario_b_low_battery_during_taking_pod()
    test_scenario_c_low_battery_during_delivering_pod()
    test_scenario_d_low_battery_during_returning_pod()
    test_scenario_e_rts_committed_next_plus_queued_charging()
    test_scenario_f_charger_unavailable()
    test_scenario_g_sufficient_battery()
    print("ALL CHARGING ARBITRATION VALIDATION TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    main()
