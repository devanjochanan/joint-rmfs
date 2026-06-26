#!/usr/bin/env python3
"""Bounded smoke for physical RTS-RL branch execution."""

from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile

os.environ.setdefault("RMFS_FAST_TRAIN", "1")
os.environ.setdefault("RMFS_DETAIL_DB", "0")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/rmfs-mpl")

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from engine.netlogo_coordinate import NetLogoCoordinate
from model.inventory import Inventory
from model.pod import Pod
from model.robot import Robot
from model.robot_job import RobotJob
from model.station import Station
from src.rmfs.decisions.rts import CurrentRTSPolicy
from src.rmfs.decisions.rts.types import RTSDecision
from src.rmfs.rl.rts.action_space import REPLENISH_STORE, STORE


class SimpleGraph:
    key = "smoke"

    def __init__(self):
        self.fail = False

    def dijkstra(self, start, end, avoid=None):
        if self.fail:
            return None
        return [start, end]

    def dijkstra_modified(self, start, end, penalties, zone_boundary, avoid=None):
        return self.dijkstra(start, end, avoid)


class CountingRuntime:
    def __init__(self):
        self.decisions = []
        self.return_completed = 0
        self.station_arrivals = []

    def on_decision(self, *, robot, context, decision):
        self.decisions.append(decision)

    def on_return_completed(self, *, robot):
        self.return_completed += 1

    def on_station_arrival(self, *, robot, station):
        self.station_arrivals.append(station.station_id)

    def censor_pending_for_robot(self, *args, **kwargs):
        return None

    def on_committed_next_activated(self, *args, **kwargs):
        return None


class BranchPolicy:
    def __init__(self, branch, storage):
        self.branch = branch
        self.storage = storage

    def select_destination(self, context):
        return RTSDecision(
            storage=self.storage,
            destination=NetLogoCoordinate(self.storage.pos_x, self.storage.pos_y),
            policy_name="smoke_rts_rl",
            mode="rl",
            action_index=0 if self.branch == STORE else 1,
            branch=self.branch,
            zone_id="zone-a",
            storage_id=f"{self.storage.pos_x}:{self.storage.pos_y}",
            replenishment_station=(
                context.warehouse.station_manager.find_available_replenish_station()
                if self.branch == REPLENISH_STORE
                else None
            ),
            replenishment_station_id=(
                context.warehouse.station_manager.find_available_replenish_station().station_id
                if self.branch == REPLENISH_STORE
                and context.warehouse.station_manager.find_available_replenish_station() is not None
                else None
            ),
            metadata={
                "selected_action_branch": self.branch,
                "selected_zone_id": "zone-a",
                "selected_action_index": 0 if self.branch == STORE else 1,
            },
        )


def build_inventory():
    tmp = tempfile.TemporaryDirectory()
    inv = Inventory(
        runtime_paths={
            "assign_order_csv": str(Path(tmp.name) / "assign_order.csv"),
            "pod_info_csv": str(Path(tmp.name) / "pod_info.csv"),
            "generated_order_csv": str(Path(tmp.name) / "generated_order.csv"),
        },
        sqlite_db_path=str(Path(tmp.name) / "warehouse.db"),
    )
    inv._smoke_tmp = tmp
    inv.fast_train = True
    inv.rts_rollout_runtime = CountingRuntime()
    graph = SimpleGraph()
    inv.graph = graph
    inv.graph_pod = graph
    inv.tick_to_second = 1.0
    return inv


def station(station_id, station_type, x, y):
    st = Station(station_id, station_type)
    st.pos_x = x
    st.pos_y = y
    st.coordinate = NetLogoCoordinate(x, y)
    st.short_path = [NetLogoCoordinate(x, y)]
    st.long_path = [NetLogoCoordinate(x, y)]
    return st


def pod_with_stock(pod_id=1, current_qty=2):
    pod = Pod(pod_id)
    pod.pos_x = 1
    pod.pos_y = 1
    pod.add_sku(101, limit_qty=10, current_qty=current_qty, threshold=0.5, weight=2.0)
    return pod


def add_common_objects(inv, pod):
    picker = station(1, "picker", 5, 5)
    repl = station(1, "replenishment", 8, 5)
    inv.station_manager.add_station(picker)
    inv.station_manager.add_station(repl)
    inv.pod_manager.add_pod(pod)
    inv.pod_manager.add_sku_data(101, current_qty=pod.skus[101]["current_qty"], max_qty=10, global_threshold_inv_level=0.5)
    storage = inv.storage_manager.createStorage(10, 5)
    inv.storage_manager.initStorageManager()
    robot = Robot(inv)
    robot.pos_x = picker.pos_x
    robot.pos_y = picker.pos_y
    robot.coordinate = NetLogoCoordinate(robot.pos_x, robot.pos_y)
    inv.addObject(robot)
    job = RobotJob(pod.coordinate, station_id=picker.station_id, pod=pod)
    job.set_job_finish()
    robot.job = job
    robot.current_state = "station_processing"
    robot.route_stop_points = []
    picker.add_pod(pod.pod_id)
    picker.add_robot(robot.robotName())
    inv.pod_manager.mark_pod_not_available(pod)
    return picker, repl, storage, robot, job


def start_return(robot):
    robot.route_stop_points = []
    robot.advance_state_if_needed()


def finish_current_return(robot):
    robot.pos_x = robot.destination.x
    robot.pos_y = robot.destination.y
    robot.coordinate = NetLogoCoordinate(robot.pos_x, robot.pos_y)
    robot.route_stop_points = []
    robot.advance_state_if_needed()


def test_direct_store_branch():
    inv = build_inventory()
    pod = pod_with_stock()
    _picker, repl, storage, robot, job = add_common_objects(inv, pod)
    inv.rts_policy = BranchPolicy(STORE, storage)

    start_return(robot)

    assert inv.rts_rollout_runtime.decisions[0].branch == STORE
    assert len(inv.rts_rollout_runtime.decisions) == 1
    assert storage not in inv.storage_manager.empty_storages
    assert inv.storage_manager.getStorageByPod(pod) is storage
    assert robot.current_state == "returning_pod"
    assert robot.destination.x == storage.pos_x and robot.destination.y == storage.pos_y
    assert job.replenishment_delay == 0
    assert job.replenishment_skus == []
    assert pod.pod_id not in repl.incoming_pod
    assert inv.replenishment_count == 0
    assert inv.replenishment_trips == 0

    finish_current_return(robot)

    assert inv.storage_manager.getStorageByPod(pod) is storage
    assert not pod.rts_return_in_progress
    assert inv.rts_rollout_runtime.return_completed == 1


def test_replenish_store_branch():
    inv = build_inventory()
    pod = pod_with_stock()
    _picker, repl, storage, robot, job = add_common_objects(inv, pod)
    inv.rts_policy = BranchPolicy(REPLENISH_STORE, storage)
    before_pod_qty = pod.skus[101]["current_qty"]
    before_global_qty = inv.pod_manager.skus_data[101]["current_global_qty"]
    before_mass = pod.mass

    start_return(robot)

    assert inv.rts_rollout_runtime.decisions[0].branch == REPLENISH_STORE
    assert len(inv.rts_rollout_runtime.decisions) == 1
    assert robot.job.pod is pod
    assert robot.current_state == "delivering_pod"
    assert job.station_id == repl.station_id
    assert job.replenishment_delay > 0
    assert job.replenishment_skus == [101]
    assert pod.pod_id in repl.incoming_pod
    assert robot.taking_pod_delay == 0

    initial_delay = job.replenishment_delay
    robot.pos_x = repl.pos_x
    robot.pos_y = repl.pos_y
    robot.coordinate = NetLogoCoordinate(robot.pos_x, robot.pos_y)
    robot.route_stop_points = []
    robot.advance_state_if_needed()
    assert robot.current_state == "station_processing"
    job.decrement_delay()
    assert job.replenishment_delay == initial_delay - 1
    job.replenishment_delay = 0
    inv.finish_task_in_job(job)

    restored = 10 - before_pod_qty
    assert pod.skus[101]["current_qty"] == 10
    assert inv.pod_manager.skus_data[101]["current_global_qty"] == before_global_qty + restored
    assert pod.mass == before_mass + restored * pod.skus[101]["weight"]
    assert inv.replenishment_count == 1
    assert inv.replenishment_trips == 1
    assert pod.pod_id not in repl.incoming_pod

    robot.route_stop_points = []
    robot.advance_state_if_needed()
    assert robot.current_state == "returning_pod"
    assert robot.destination.x == storage.pos_x and robot.destination.y == storage.pos_y
    assert len(inv.rts_rollout_runtime.decisions) == 1

    finish_current_return(robot)

    assert inv.rts_rollout_runtime.return_completed == 1
    assert not pod.rts_return_in_progress
    assert inv.storage_manager.getStorageByPod(pod) is storage


def test_post_pick_ownership_gate():
    for controls, expected_calls in ((True, 0), (False, 1)):
        inv = build_inventory()
        pod = pod_with_stock()
        picker, _repl, _storage, _robot, job = add_common_objects(inv, pod)
        calls = []

        def record_queue(queued_pod, critical_skus=None):
            calls.append((queued_pod, list(critical_skus or [])))
            return True

        inv.queue_post_pick_replenishment = record_queue
        inv.rts_controls_post_pick_replenishment = controls
        picker.add_pod(pod.pod_id)
        inv.finish_picking_task(job)
        assert len(calls) == expected_calls
        if calls:
            assert calls[0] == (pod, [])


def test_proactive_conflict_guard():
    inv = build_inventory()
    pod = pod_with_stock()
    _picker, repl, storage, robot, _job = add_common_objects(inv, pod)
    pod.rts_return_in_progress = True
    pod.rts_return_branch = REPLENISH_STORE
    pod.rts_return_zone_id = "zone-a"
    inv.pod_manager.mark_pod_available(pod)

    assert inv.get_most_depleted_eligible_pod_for_sku(101) is None
    inv.pending_replenishment_dispatches.append(
        {"pod_id": pod.pod_id, "skus_to_replenish": [101], "created_tick": 0, "guaranteed_on_release": False}
    )
    pod.has_pending_replenishment_dispatch = True
    assert inv.dispatch_pending_replenishment_requests() == 0
    assert inv.get_pending_replenishment_dispatch(pod.pod_id) is not None

    pod.rts_return_in_progress = False
    assert inv.pod_manager.is_idle(pod.pod_id)

    inv.rts_policy = BranchPolicy(REPLENISH_STORE, storage)
    inv.pod_manager.mark_pod_not_available(pod)
    robot.current_state = "station_processing"
    robot.route_stop_points = []
    start_return(robot)
    assert inv.get_pending_replenishment_dispatch(pod.pod_id) is None
    assert pod.pod_id in repl.incoming_pod


def test_rollback_after_reservation_failure():
    inv = build_inventory()
    pod = pod_with_stock()
    _picker, _repl, storage, robot, _job = add_common_objects(inv, pod)
    inv.rts_policy = BranchPolicy(STORE, storage)
    inv.graph_pod.fail = True

    try:
        start_return(robot)
    except RuntimeError as exc:
        assert "failed to start RTS store return" in str(exc)
    else:
        raise AssertionError("expected RTS store route failure")

    assert storage in inv.storage_manager.empty_storages
    assert inv.storage_manager.getStorageByPod(pod) is None
    assert storage.assigned_pod is None
    assert not pod.rts_return_in_progress
    assert not inv.pod_manager.is_idle(pod.pod_id)


def test_non_rts_nearest_regression():
    inv = build_inventory()
    pod = pod_with_stock(current_qty=9)
    _picker, repl, storage, robot, job = add_common_objects(inv, pod)
    inv.rts_policy = CurrentRTSPolicy()

    start_return(robot)

    assert len(inv.rts_rollout_runtime.decisions) == 1
    decision = inv.rts_rollout_runtime.decisions[0]
    assert decision.mode == "nearest"
    assert decision.branch is None
    assert decision.zone_id is None
    assert inv.storage_manager.getStorageByPod(pod) is storage
    assert robot.current_state == "returning_pod"
    assert pod.pod_id not in repl.incoming_pod
    assert not job.rts_continuation_active

    finish_current_return(robot)
    assert inv.rts_rollout_runtime.return_completed == 1


def main():
    test_direct_store_branch()
    test_replenish_store_branch()
    test_post_pick_ownership_gate()
    test_proactive_conflict_guard()
    test_rollback_after_reservation_failure()
    test_non_rts_nearest_regression()
    print("rts branch execution smoke ok")


if __name__ == "__main__":
    main()
