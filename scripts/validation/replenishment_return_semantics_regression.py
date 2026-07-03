#!/usr/bin/env python3
"""Focused regression for proactive / post-pick / RTS replenishment return
semantics.

Covers:
  * proactive pod retains exclusive origin ownership while away;
  * another pod cannot claim the origin;
  * proactive return calls RTS zero times and returns to the exact origin;
  * separate post-pick replenishment calls RTS exactly once;
  * RTS REPLENISH_STORE uses one picker-time decision and no second decision;
  * ownership conflict on proactive return fails clearly (no nearest fallback);
  * proactive dispatch failure leaves no metadata, no duplicate/stranded owner.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys

os.environ.setdefault("RMFS_FAST_TRAIN", "1")
os.environ.setdefault("RMFS_DETAIL_DB", "0")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/rmfs-mpl")

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from engine.netlogo_coordinate import NetLogoCoordinate
from model.pod import Pod
from model.robot_job import RobotJob
from src.rmfs.decisions.rts.types import RTSDecision
from src.rmfs.rl.rts.action_space import STORE
from src.rmfs.rl.rts.outcome_tracker import NoopRTSRolloutRuntime
from scripts.validation.rts_committed_next_smoke import build_inventory, pod_with_stock


class CountingPolicy:
    """RTS policy stub that counts select_destination calls and returns a fixed
    return decision (so handle_pod_return executes without deep RL plumbing)."""

    def __init__(self):
        self.calls = 0

    def select_destination(self, context):
        self.calls += 1
        dest = NetLogoCoordinate(context.robot.pos_x, context.robot.pos_y)
        return RTSDecision(
            storage=None,
            destination=dest,
            policy_name="counting_stub",
            mode="fixed",
        )


def _register_pod_at_storage(inv, pod_id, x, y, current_qty=2):
    pod = pod_with_stock(pod_id, x, y, current_qty=current_qty)
    inv.pod_manager.add_pod(pod)
    inv.pod_manager.add_sku_data(101, current_qty=current_qty, max_qty=10, global_threshold_inv_level=0.5)
    storage = inv.storage_manager.createStorage(x, y)
    inv.storage_manager.addPodToStorage(pod, storage)
    inv.storage_manager.initStorageManager()
    return pod, storage


def test_proactive_retains_origin_and_returns_without_rts():
    inv, _picker, repl, _final_storage, robot, _cur, _nj, _policy = build_inventory(branch=STORE, next_jobs=0)
    pod, origin = _register_pod_at_storage(inv, 50, 3, 3)

    ok = inv.send_pod_for_replenishment(pod, repl, [101], source="proactive")
    assert ok is True
    job = inv.job_queue[-1]
    assert job.is_proactive_replenishment()
    assert job.proactive_origin_reserved is True
    assert job.proactive_origin_storage is origin
    assert job.proactive_origin_coordinate == (float(origin.pos_x), float(origin.pos_y))

    # Origin remains owned by the pod and unavailable to others while away.
    assert inv.storage_manager.storage_owned_by_pod(origin, pod) is True
    assert origin.assigned_pod is pod
    assert origin.is_empty is False
    assert origin not in inv.storage_manager.empty_storages

    # On return, RTS must not be consulted and the pod routes to the exact origin.
    counting = CountingPolicy()
    inv.rts_policy = counting
    robot.job = job
    robot.pos_x, robot.pos_y = repl.pos_x, repl.pos_y
    robot.coordinate = NetLogoCoordinate(robot.pos_x, robot.pos_y)
    robot.route_stop_points = []
    robot.handle_pod_return(repl)
    assert counting.calls == 0, "proactive return must call RTS zero times"
    assert (robot.destination.x, robot.destination.y) == (origin.pos_x, origin.pos_y)
    print("PASS test_proactive_retains_origin_and_returns_without_rts")


def test_post_pick_replenishment_calls_rts_once():
    inv, _picker, repl, _final_storage, robot, _cur, _nj, _policy = build_inventory(branch=STORE, next_jobs=0)
    pod, _origin = _register_pod_at_storage(inv, 51, 4, 4)

    job = RobotJob(pod.coordinate, station_id=repl.station_id, pod=pod)
    job.add_replenishment_task(pod, [101], source="post_pick")
    assert job.is_proactive_replenishment() is False
    assert job.proactive_origin_reserved is False

    counting = CountingPolicy()
    inv.rts_policy = counting
    # Neutralize rollout capture (build_inventory's capture requires a picker
    # source); we only assert that RTS destination selection is invoked once.
    inv.rts_rollout_runtime = NoopRTSRolloutRuntime()
    robot.job = job
    robot.pos_x, robot.pos_y = repl.pos_x, repl.pos_y
    robot.coordinate = NetLogoCoordinate(robot.pos_x, robot.pos_y)
    robot.route_stop_points = []
    robot.handle_pod_return(repl)
    assert counting.calls == 1, "post-pick replenishment return must call RTS exactly once"
    print("PASS test_post_pick_replenishment_calls_rts_once")


def test_rts_replenish_store_no_second_decision():
    inv, _picker, repl, final_storage, robot, _cur, _nj, _policy = build_inventory(branch=STORE, next_jobs=0)
    pod, _origin = _register_pod_at_storage(inv, 52, 5, 6)

    # Simulate a job already mid-RTS-REPLENISH_STORE continuation (final storage
    # was selected+reserved at the picker); the return must use it directly.
    job = RobotJob(pod.coordinate, station_id=repl.station_id, pod=pod)
    job.rts_continuation_active = True
    job.rts_stage = "post_replenishment_to_storage"
    job.rts_final_storage = final_storage
    job.rts_final_destination = NetLogoCoordinate(final_storage.pos_x, final_storage.pos_y)

    counting = CountingPolicy()
    inv.rts_policy = counting
    robot.job = job
    robot.pos_x, robot.pos_y = repl.pos_x, repl.pos_y
    robot.coordinate = NetLogoCoordinate(robot.pos_x, robot.pos_y)
    robot.route_stop_points = []
    robot.handle_pod_return(repl)
    assert counting.calls == 0, "RTS REPLENISH_STORE must not invoke a second RTS decision"
    assert (robot.destination.x, robot.destination.y) == (final_storage.pos_x, final_storage.pos_y)
    print("PASS test_rts_replenish_store_no_second_decision")


def test_origin_ownership_conflict_fails_clearly():
    inv, _picker, repl, _final_storage, robot, _cur, _nj, _policy = build_inventory(branch=STORE, next_jobs=0)
    pod, origin = _register_pod_at_storage(inv, 53, 7, 7)
    ok = inv.send_pod_for_replenishment(pod, repl, [101], source="proactive")
    assert ok is True
    job = inv.job_queue[-1]

    # Another pod steals the origin (ownership conflict).
    intruder = Pod(999)
    origin.assigned_pod = intruder

    robot.job = job
    robot.pos_x, robot.pos_y = repl.pos_x, repl.pos_y
    try:
        robot._return_proactive_pod_to_origin()
    except RuntimeError as exc:
        assert "ownership conflict" in str(exc)
    else:
        raise AssertionError("expected an ownership-conflict RuntimeError")
    print("PASS test_origin_ownership_conflict_fails_clearly")


def test_proactive_dispatch_failure_leaves_no_metadata():
    inv, _picker, repl, _final_storage, _robot, _cur, _nj, _policy = build_inventory(branch=STORE, next_jobs=0)
    # A pod that owns no storage cannot be pinned to an origin.
    lonely = pod_with_stock(54, 9, 9, current_qty=2)
    inv.pod_manager.add_pod(lonely)
    inv.pod_manager.add_sku_data(101, current_qty=2, max_qty=10, global_threshold_inv_level=0.5)

    before_queue = len(inv.job_queue)
    ok = inv.send_pod_for_replenishment(lonely, repl, [101], source="proactive")
    assert ok is False, "proactive dispatch must fail cleanly when origin cannot be pinned"
    assert len(inv.job_queue) == before_queue, "no job may be queued on failure"
    assert lonely.is_awaiting_replenishment is False, "no lifecycle mutation on failure"
    assert lonely.pod_id not in repl.incoming_pod, "no station membership on failure"
    assert inv.storage_manager.get_owned_storage_for_pod(lonely) is None
    print("PASS test_proactive_dispatch_failure_leaves_no_metadata")


def main():
    test_proactive_retains_origin_and_returns_without_rts()
    test_post_pick_replenishment_calls_rts_once()
    test_rts_replenish_store_no_second_decision()
    test_origin_ownership_conflict_fails_clearly()
    test_proactive_dispatch_failure_leaves_no_metadata()
    print("\nALL REPLENISHMENT RETURN SEMANTICS REGRESSION TESTS PASSED")


if __name__ == "__main__":
    raise SystemExit(main())
