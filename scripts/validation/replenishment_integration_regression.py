#!/usr/bin/env python3
"""Focused integration checks for proactive replenishment caps/discovery."""

from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from model.robot_job import RobotJob
from scripts.validation.replenishment_policy_regression import (
    StubRobot,
    add_stub_robot,
    make_inventory,
    pod_with,
    register_pod,
    station,
)
from scripts.validation.rts_branch_execution_smoke import (
    BranchPolicy,
    REPLENISH_STORE,
    add_common_objects,
    build_inventory,
    pod_with_stock,
    start_return,
)
from src.rmfs.decisions.rts.types import RTSDestinationContext
from src.rmfs.rl.rts.action_context import selected_context_by_index
from src.rmfs.rl.rts.state import build_state


def _make_queued_proactive(inv, pod_id: int) -> None:
    pod = pod_with(pod_id, 20 + pod_id, 5, {1000 + pod_id: (10, 2, 0.5, 1.0)})
    register_pod(inv, pod)
    job = RobotJob(pod.coordinate, station_id="replenishment-1", pod=pod)
    job.add_replenishment_task(pod, [1000 + pod_id], source="proactive")
    inv.job_queue.append(job)


def test_direct_proactive_discovery_local_and_global() -> None:
    inv, _root = make_inventory()
    local_only = pod_with(1, 2, 2, {101: (10, 3, 0.5, 1.0), 102: (10, 3, 0.5, 1.0)})
    register_pod(inv, local_only, global_current=100, global_max=100, global_threshold=0.2)

    global_only = pod_with(2, 3, 2, {201: (10, 9, 0.5, 1.0), 202: (10, 9, 0.5, 1.0)})
    register_pod(inv, global_only, global_current=100, global_max=100, global_threshold=0.2)
    inv.pod_manager.skus_data[201]["current_global_qty"] = 1
    inv.pod_manager.skus_data[201]["max_global_qty"] = 100
    inv.pod_manager.skus_data[201]["global_inv_level"] = 0.01
    inv.pod_manager.skus_data[201]["global_threshold_inv_level"] = 0.5

    assert not inv.global_critical_skus
    admitted = inv.run_proactive_replenishment_pass()
    assert admitted == 2
    assert inv.get_pending_replenishment_dispatch(1) is not None
    assert inv.get_pending_replenishment_dispatch(2) is not None
    assert inv.total_replenishment_load() == 2
    assert inv.proactive_replenishment_load() == 0
    print("PASS test_direct_proactive_discovery_local_and_global")


def test_unmatched_idle_robot_bypass_after_picking_allocation() -> None:
    inv, _root = make_inventory()
    inv.station_manager.add_station(station(1, "replenishment", 8, 5))
    for pod_id in range(1, 4):
        _make_queued_proactive(inv, pod_id)
    assert inv.proactive_replenishment_load() == 3

    replenishment_pod = pod_with(20, 5, 5, {301: (10, 2, 0.5, 1.0)})
    register_pod(inv, replenishment_pod)
    inv.pod_manager.skus_data[301]["current_global_qty"] = 1
    inv.pod_manager.skus_data[301]["global_inv_level"] = 0.01
    inv.pod_manager.skus_data[301]["global_threshold_inv_level"] = 0.5
    assert inv.run_proactive_replenishment_pass() == 1

    picking_pod = pod_with(30, 6, 5, {401: (10, 9, 0.5, 1.0)})
    register_pod(inv, picking_pod, global_current=100, global_max=100, global_threshold=0.2)
    pick_job = RobotJob(picking_pod.coordinate, station_id="picker-1", pod=picking_pod)
    pick_job.add_picking_task(900, 401, 1)
    inv.job_queue.append(pick_job)
    assert inv._picking_jobs_assignable_now()

    idle_robot_with_pick = StubRobot(1, None, state="idle")
    idle_robot_with_pick.pos_x = 0
    idle_robot_with_pick.pos_y = 0
    unmatched_robot = StubRobot(2, None, state="idle")
    unmatched_robot.pos_x = 10
    unmatched_robot.pos_y = 10
    add_stub_robot(inv, idle_robot_with_pick)
    add_stub_robot(inv, unmatched_robot)

    unmatched = inv._unmatched_idle_robots_after_picking_allocation([idle_robot_with_pick, unmatched_robot])
    assert unmatched == [unmatched_robot]

    # Simulate the existing one-job picking allocation seam: after the single
    # assignable pick is consumed by the first idle robot, only the remaining
    # unmatched robot may bypass the soft cap.
    inv.job_queue.remove(pick_job)
    idle_robot_with_pick.job = pick_job
    idle_robot_with_pick.current_state = "taking_pod"
    dispatched = inv.dispatch_proactive_replenishment_to_unmatched_idle_robots(unmatched)
    assert dispatched == 1
    assert unmatched_robot.job is not None and unmatched_robot.job.is_replenishment_job
    assert idle_robot_with_pick.job is pick_job
    assert inv.proactive_replenishment_load() == 4
    print("PASS test_unmatched_idle_robot_bypass_after_picking_allocation")


def test_returning_replenishment_and_rts_continuation_count_until_storage_return() -> None:
    inv, _root = make_inventory()
    pod = pod_with(50, 5, 5, {501: (10, 2, 0.5, 1.0)})
    register_pod(inv, pod)
    job = RobotJob(pod.coordinate, station_id="replenishment-1", pod=pod)
    job.add_replenishment_task(pod, [501], source="rts")
    job.set_job_finish()
    job.rts_continuation_active = True
    job.rts_branch = "replenish_store"
    job.rts_stage = "to_storage"
    add_stub_robot(inv, StubRobot(7, job, state="returning_pod"))
    commitments = inv.replenishment_commitments_by_pod()
    assert commitments == {50: "rts"}, commitments
    assert inv.total_replenishment_load() == 1
    print("PASS test_returning_replenishment_and_rts_continuation_count_until_storage_return")


def test_rts_replenish_store_invalid_and_execution_guard_at_hard_cap() -> None:
    inv = build_inventory()
    pod = pod_with_stock()
    picker, repl, storage, robot, _job = add_common_objects(inv, pod)
    storage.rts_zone_id = "zone-a"
    for pod_id in range(100, 111):
        inv.pending_replenishment_dispatches.append(
            {"pod_id": pod_id, "skus_to_replenish": [999], "created_tick": 0, "source": "post_pick"}
        )
    assert inv.total_replenishment_load() == 11
    context = RTSDestinationContext(inv, robot, pod, picker)
    state = build_state(context, ("zone-a",))
    store_context = selected_context_by_index(state.action_contexts, 0)
    replenish_context = selected_context_by_index(state.action_contexts, 1)
    assert store_context.action_valid
    assert not replenish_context.action_valid
    assert "replenishment_hard_cap_reached" in replenish_context.invalid_reason_codes

    inv.rts_policy = BranchPolicy(REPLENISH_STORE, storage)
    try:
        start_return(robot)
    except RuntimeError as exc:
        assert "replenishment_hard_cap_reached" in str(exc)
    else:
        raise AssertionError("RTS replenish_store execution guard did not fail at hard cap")
    assert storage in inv.storage_manager.empty_storages
    assert pod.pod_id not in repl.incoming_pod
    print("PASS test_rts_replenish_store_invalid_and_execution_guard_at_hard_cap")


def main() -> None:
    test_direct_proactive_discovery_local_and_global()
    test_unmatched_idle_robot_bypass_after_picking_allocation()
    test_returning_replenishment_and_rts_continuation_count_until_storage_return()
    test_rts_replenish_store_invalid_and_execution_guard_at_hard_cap()
    print("\nALL REPLENISHMENT INTEGRATION REGRESSION TESTS PASSED")


if __name__ == "__main__":
    main()
