#!/usr/bin/env python3
"""Focused regression for the approved replenishment semantics.

Covers:
  * OR eligibility (local-only, global-only, neither, globally-low-but-full);
  * full-pod restoration (all SKUs to limit; exact per-SKU global deltas; mass);
  * full-pod service time uses all distinct pod SKUs;
  * soft cap (3) / hard cap (11) accounting by unique committed pods;
  * idle-no-picking-job soft-cap bypass;
  * pending -> active counted once; duplicate request does not increase load;
  * pending stability (survives eligibility recheck; cancellation clears locks;
    aged request cannot stay pick-blocked without ownership).
"""

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
from model.robot_job import RobotJob
from model.station import Station


def make_inventory():
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
    inv._tmp = tmp
    inv.fast_train = True
    inv.tick_to_second = 1.0
    return inv, root


def station(station_id, station_type, x, y):
    st = Station(station_id, station_type)
    st.pos_x = x
    st.pos_y = y
    st.coordinate = NetLogoCoordinate(x, y)
    st.short_path = [NetLogoCoordinate(x, y)]
    st.long_path = [NetLogoCoordinate(x, y)]
    return st


def pod_with(pod_id, x, y, sku_specs):
    pod = Pod(pod_id)
    pod.pos_x = x
    pod.pos_y = y
    for sku, (limit_qty, current_qty, threshold, weight) in sku_specs.items():
        pod.add_sku(sku, limit_qty=limit_qty, current_qty=current_qty, threshold=threshold, weight=weight)
    return pod


def register_pod(inv, pod, *, global_current=None, global_max=None, global_threshold=0.5):
    inv.pod_manager.add_pod(pod)
    for sku, details in pod.skus.items():
        inv.pod_manager.add_sku_data(
            sku,
            current_qty=global_current if global_current is not None else details["current_qty"],
            max_qty=global_max if global_max is not None else details["limit_qty"],
            global_threshold_inv_level=global_threshold,
        )


class StubRobot:
    object_type = "robot"

    def __init__(self, _id, job, state="returning_pod"):
        self._id = _id
        self.id = _id
        self.job = job
        self.current_state = state
        self.is_charging_pending = False
        self.is_charging = False
        self._claimed_charger = None

    def assign_job_and_set_move_to_station(self, job):
        self.job = job
        self.current_state = "delivering_pod"


def add_stub_robot(inv, robot):
    inv._objects.append(robot)


# ----------------------------------------------------------------------------
def test_eligibility_local_only():
    inv, _root = make_inventory()
    # Aggregate local fill below 0.4 (mean of 0.3, 0.3 = 0.3); no globally-low
    # SKU (global inventory healthy).
    pod = pod_with(1, 2, 2, {101: (10, 3, 0.5, 1.0), 102: (10, 3, 0.5, 1.0)})
    register_pod(inv, pod, global_current=100, global_max=100, global_threshold=0.2)
    plan = inv.evaluate_pod_replenishment_eligibility(pod)
    assert plan["local_trigger"] is True, plan
    assert plan["global_trigger"] is False, plan
    assert plan["eligible"] is True, plan
    print("PASS test_eligibility_local_only")


def test_eligibility_global_only():
    inv, _root = make_inventory()
    # Aggregate local fill high (0.9) so local trigger false; one SKU globally
    # low and locally refillable (current 9 < limit 10).
    pod = pod_with(1, 2, 2, {101: (10, 9, 0.5, 1.0), 102: (10, 9, 0.5, 1.0)})
    register_pod(inv, pod)
    # Drive SKU 101 globally low.
    inv.pod_manager.skus_data[101]["current_global_qty"] = 1
    inv.pod_manager.skus_data[101]["max_global_qty"] = 100
    inv.pod_manager.skus_data[101]["global_inv_level"] = 0.01
    inv.pod_manager.skus_data[101]["global_threshold_inv_level"] = 0.5
    plan = inv.evaluate_pod_replenishment_eligibility(pod)
    assert plan["local_trigger"] is False, plan
    assert plan["global_trigger"] is True, plan
    assert 101 in plan["global_low_refillable_skus"], plan
    assert plan["eligible"] is True, plan
    print("PASS test_eligibility_global_only")


def test_eligibility_neither():
    inv, _root = make_inventory()
    pod = pod_with(1, 2, 2, {101: (10, 9, 0.5, 1.0), 102: (10, 9, 0.5, 1.0)})
    register_pod(inv, pod, global_current=100, global_max=100, global_threshold=0.2)
    plan = inv.evaluate_pod_replenishment_eligibility(pod)
    assert plan["eligible"] is False, plan
    print("PASS test_eligibility_neither")


def test_globally_low_but_locally_full_not_refillable():
    inv, _root = make_inventory()
    # SKU 101 is at its limit on this pod (full) -> not refillable even though
    # globally low.
    pod = pod_with(1, 2, 2, {101: (10, 10, 0.5, 1.0), 102: (10, 9, 0.5, 1.0)})
    register_pod(inv, pod, global_current=100, global_max=100, global_threshold=0.2)
    inv.pod_manager.skus_data[101]["current_global_qty"] = 1
    inv.pod_manager.skus_data[101]["global_inv_level"] = 0.01
    inv.pod_manager.skus_data[101]["global_threshold_inv_level"] = 0.5
    refillable = inv.get_globally_low_refillable_skus_for_pod(pod)
    assert 101 not in refillable, f"full compartment must not be refillable: {refillable}"
    print("PASS test_globally_low_but_locally_full_not_refillable")


def test_full_pod_restoration_and_deltas():
    inv, root = make_inventory()
    repl = station(1, "replenishment", 8, 5)
    inv.station_manager.add_station(repl)
    pod = pod_with(1, 5, 5, {101: (10, 2, 0.5, 2.0), 102: (8, 5, 0.5, 3.0), 103: (8, 8, 0.5, 1.0)})
    register_pod(inv, pod)

    before_global = {sku: inv.pod_manager.skus_data[sku]["current_global_qty"] for sku in pod.skus}
    before_mass = pod.mass

    # Trigger subset intentionally lists only SKU 101; restoration must still be
    # full-pod.
    job = RobotJob(pod.coordinate, station_id=repl.station_id, pod=pod)
    job.add_replenishment_task(pod, [101], source="proactive")
    repl.add_pod(pod.pod_id)
    pod.station = repl
    pod.is_awaiting_replenishment = True
    inv.pod_manager.mark_pod_not_available(pod)

    # Service time uses all distinct SKUs (3) x 20.
    assert job.replenishment_delay == 3 * 20, f"delay must be full-pod: {job.replenishment_delay}"

    trips_before = inv.replenishment_trips
    count_before = inv.replenishment_count
    inv.finish_replenishment_task(job)

    assert pod.skus[101]["current_qty"] == 10
    assert pod.skus[102]["current_qty"] == 8
    assert pod.skus[103]["current_qty"] == 8
    # exact per-SKU global deltas match local deltas (101:+8, 102:+3, 103:+0).
    assert inv.pod_manager.skus_data[101]["current_global_qty"] == before_global[101] + 8
    assert inv.pod_manager.skus_data[102]["current_global_qty"] == before_global[102] + 3
    assert inv.pod_manager.skus_data[103]["current_global_qty"] == before_global[103]
    # mass increases by exact restored weight (8*2 + 3*3 + 0).
    assert pod.mass == before_mass + (8 * 2.0) + (3 * 3.0), f"mass mismatch {pod.mass}"
    # count only SKUs actually restored (101, 102) -> 2.
    assert inv.replenishment_count == count_before + 2, inv.replenishment_count
    assert inv.replenishment_trips == trips_before + 1
    print("PASS test_full_pod_restoration_and_deltas")


def _make_pending(inv, pod_id, source):
    inv.pending_replenishment_dispatches.append(
        {"pod_id": pod_id, "skus_to_replenish": [101], "created_tick": 0, "source": source}
    )


def _make_queued_replenishment_job(inv, pod_id, source):
    pod = pod_with(pod_id, 5 + pod_id, 5, {1000 + pod_id: (10, 2, 0.5, 1.0)})
    register_pod(inv, pod)
    job = RobotJob(pod.coordinate, station_id="replenishment-1", pod=pod)
    job.add_replenishment_task(pod, [1000 + pod_id], source=source)
    inv.job_queue.append(job)
    return job


def test_caps_accounting_and_admission():
    inv, _root = make_inventory()
    # Passive pending proactive requests count toward the hard cap but do not
    # consume proactive robot slots.
    _make_pending(inv, 1, "proactive")
    _make_pending(inv, 2, "proactive")
    assert inv.total_replenishment_load() == 2
    assert inv.proactive_replenishment_load() == 0
    assert inv.can_admit_replenishment("proactive") is True
    assert inv.can_admit_replenishment("proactive", consume_robot_slot=True) is True

    # Queued/active proactive jobs consume soft-cap robot slots.
    for pid in range(3, 6):
        _make_queued_replenishment_job(inv, pid, "proactive")
    assert inv.proactive_replenishment_load() == 3
    assert inv.can_admit_replenishment("proactive") is True
    assert inv.can_admit_replenishment("proactive", consume_robot_slot=True) is False
    assert inv.can_admit_replenishment("proactive", consume_robot_slot=True, idle_bypass=True) is True
    # post_pick and rts are not subject to the soft cap (only hard cap).
    assert inv.can_admit_replenishment("post_pick") is True
    assert inv.can_admit_replenishment("rts") is True

    # Fill up to total 10 with mixed sources -> one more admission allowed.
    for pid in range(6, 11):
        _make_pending(inv, pid, "post_pick")
    assert inv.total_replenishment_load() == 10
    assert inv.can_admit_replenishment("post_pick") is True
    assert inv.can_admit_replenishment("rts") is True
    assert inv.can_admit_replenishment("proactive", consume_robot_slot=True, idle_bypass=True) is True

    # total 11 -> every new admission blocked.
    _make_pending(inv, 11, "post_pick")
    assert inv.total_replenishment_load() == 11
    assert inv.can_admit_replenishment("post_pick") is False
    assert inv.can_admit_replenishment("rts") is False
    assert inv.can_admit_replenishment("proactive", consume_robot_slot=True, idle_bypass=True) is False
    print("PASS test_caps_accounting_and_admission")


def test_pending_to_active_counted_once():
    inv, _root = make_inventory()
    repl = station(1, "replenishment", 8, 5)
    pod = pod_with(5, 5, 5, {101: (10, 3, 0.5, 1.0)})
    register_pod(inv, pod)

    # Pending commitment.
    _make_pending(inv, 5, "proactive")
    assert inv.total_replenishment_load() == 1

    # Transition to an active robot replenishment job for the SAME pod: the
    # pending entry is removed and an active job created -> still one commitment.
    inv.remove_pending_replenishment_dispatch(5)
    active_job = RobotJob(pod.coordinate, station_id=repl.station_id, pod=pod)
    active_job.add_replenishment_task(pod, [101], source="proactive")
    add_stub_robot(inv, StubRobot(1, active_job, state="returning_pod"))
    assert inv.total_replenishment_load() == 1, "pending->active must remain a single commitment"
    assert inv.proactive_replenishment_load() == 1
    print("PASS test_pending_to_active_counted_once")


def test_duplicate_request_does_not_increase_load():
    inv, _root = make_inventory()
    pod = pod_with(7, 5, 5, {101: (10, 2, 0.5, 1.0)})
    register_pod(inv, pod)
    inv.pod_manager.mark_pod_available(pod)

    admitted_first = inv.enqueue_pending_replenishment_dispatch(pod, [101], source="proactive")
    admitted_dup = inv.enqueue_pending_replenishment_dispatch(pod, [101], source="proactive")
    assert admitted_first is True
    assert admitted_dup is False, "duplicate request for same pod must not create a new commitment"
    assert inv.total_replenishment_load() == 1
    print("PASS test_duplicate_request_does_not_increase_load")


def test_idle_bypass_permits_beyond_soft_cap():
    inv, _root = make_inventory()
    repl = station(1, "replenishment", 8, 5)
    inv.station_manager.add_station(repl)

    # Saturate proactive robot usage at the soft cap (3).
    for pid in range(1, 4):
        _make_queued_replenishment_job(inv, pid, "proactive")
    assert inv.proactive_replenishment_load() == 3

    # A globally-low SKU with an eligible pod is discovered directly and may
    # wait as pending without consuming a fourth proactive robot slot.
    pod = pod_with(20, 5, 5, {101: (10, 2, 0.5, 1.0)})
    register_pod(inv, pod)
    inv.pod_manager.mark_pod_available(pod)
    # Proactive replenishment returns the pod to its exact origin, so the pod
    # must own a storage before it can be dispatched proactively.
    _origin20 = inv.storage_manager.createStorage(5, 5)
    inv.storage_manager.addPodToStorage(pod, _origin20)
    inv.pod_manager.skus_data[101]["current_global_qty"] = 1
    inv.pod_manager.skus_data[101]["global_inv_level"] = 0.01
    inv.pod_manager.skus_data[101]["global_threshold_inv_level"] = 0.5
    admitted = inv.run_proactive_replenishment_pass()
    assert admitted >= 1, "direct proactive discovery should enqueue an eligible pending request"
    assert inv.get_pending_replenishment_dispatch(20) is not None
    assert inv.proactive_replenishment_load() == 3, inv.proactive_replenishment_load()

    # An unmatched idle robot may dispatch that pending request beyond the soft
    # cap, still subject to the hard cap.
    idle_robot = StubRobot(1, None, state="idle")
    add_stub_robot(inv, idle_robot)
    dispatched = inv.dispatch_proactive_replenishment_to_unmatched_idle_robots([idle_robot])
    assert dispatched == 1
    assert inv.proactive_replenishment_load() == 4, inv.proactive_replenishment_load()
    print("PASS test_idle_bypass_permits_beyond_soft_cap")


def test_pending_stability_and_cancellation():
    inv, _root = make_inventory()
    pod = pod_with(30, 5, 5, {101: (10, 2, 0.5, 1.0)})
    register_pod(inv, pod)
    inv.pod_manager.mark_pod_available(pod)

    # Admit a proactive pending request while eligible.
    assert inv.enqueue_pending_replenishment_dispatch(pod, [101], source="proactive") is True
    assert inv.get_pending_replenishment_dispatch(30) is not None

    # Later eligibility recheck flips to ineligible (pod refilled): the admitted
    # request must survive.
    pod.skus[101]["current_qty"] = 10
    assert inv.is_pod_replenishment_eligible(pod) is False
    assert inv.get_pending_replenishment_dispatch(30) is not None, "admitted request must not be revoked"

    # Cancellation clears all lock flags.
    pod.must_replenish_before_pick = True
    inv.remove_pending_replenishment_dispatch(30)
    assert inv.get_pending_replenishment_dispatch(30) is None
    assert pod.has_pending_replenishment_dispatch is False
    assert pod.must_replenish_before_pick is False
    print("PASS test_pending_stability_and_cancellation")


def test_aged_request_not_pick_blocked_without_ownership():
    inv, _root = make_inventory()
    pod = pod_with(40, 5, 5, {101: (10, 2, 0.5, 1.0)})
    register_pod(inv, pod)
    inv.pod_manager.mark_pod_available(pod)
    # Pod is marked pick-blocked but has NO pending/active/return ownership.
    pod.must_replenish_before_pick = True
    inv._tick = 1000
    inv.refresh_mandatory_replenishment_pods()
    assert pod.must_replenish_before_pick is False, "pick-block must clear without ownership"
    print("PASS test_aged_request_not_pick_blocked_without_ownership")


def main():
    test_eligibility_local_only()
    test_eligibility_global_only()
    test_eligibility_neither()
    test_globally_low_but_locally_full_not_refillable()
    test_full_pod_restoration_and_deltas()
    test_caps_accounting_and_admission()
    test_pending_to_active_counted_once()
    test_duplicate_request_does_not_increase_load()
    test_idle_bypass_permits_beyond_soft_cap()
    test_pending_stability_and_cancellation()
    test_aged_request_not_pick_blocked_without_ownership()
    print("\nALL REPLENISHMENT POLICY REGRESSION TESTS PASSED")


if __name__ == "__main__":
    raise SystemExit(main())
