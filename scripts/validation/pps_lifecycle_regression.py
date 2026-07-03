#!/usr/bin/env python3
"""Focused regression for PPS candidate rejection and pod lifecycle cleanup.

Covers:
  * zero-score PPS candidate returns no pod;
  * positive-score ranking is unchanged;
  * empty PPS construction causes zero mutations;
  * non-empty PPS construction applies each mutation exactly once;
  * add_picking_task_after_pps counts pps_picked_quantity at delivery, not
    creation (including dynamically added tasks);
  * partial delivery does not prematurely mark assign_order.csv status finished;
  * finalize_completed_return releases the old pod (available, no station);
  * completed return releases the old pod before committed-next activation;
  * unavailable unowned pod is flagged by the runtime invariant.
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

import pandas as pd

from engine.netlogo_coordinate import NetLogoCoordinate
from model.inventory import Inventory
from model.order import Order
from model.pod import Pod
from model.robot_job import RobotJob
from model.station import Station
from src.rmfs.rl.rts.runtime_invariants import check_runtime_invariants


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


def register_pod(inv, pod):
    inv.pod_manager.add_pod(pod)
    for sku, details in pod.skus.items():
        inv.pod_manager.add_sku_data(
            sku,
            current_qty=details["current_qty"],
            max_qty=details["limit_qty"],
            global_threshold_inv_level=0.5,
        )


# ----------------------------------------------------------------------------
def test_zero_score_candidate_returns_no_pod():
    inv, _root = make_inventory()
    # Pod holds the SKU key but has zero usable quantity.
    empty_pod = pod_with(1, 2, 2, {101: (10, 0, 0.5, 1.0)})
    register_pod(inv, empty_pod)
    inv.pod_manager.mark_pod_available(empty_pod)

    pod, score = inv.find_best_pod({101: 5}, [101], mode="pile_on")
    assert pod is None, f"expected no pod for zero-value candidate, got {pod}"
    assert score <= 0, f"expected non-positive score, got {score}"
    print("PASS test_zero_score_candidate_returns_no_pod")


def test_positive_score_ranking_unchanged():
    inv, _root = make_inventory()
    low = pod_with(1, 2, 2, {101: (10, 3, 0.5, 1.0)})
    high = pod_with(2, 3, 3, {101: (10, 9, 0.5, 1.0)})
    register_pod(inv, low)
    register_pod(inv, high)
    inv.pod_manager.mark_pod_available(low)
    inv.pod_manager.mark_pod_available(high)

    pod, score = inv.find_best_pod({101: 5}, [101], mode="pile_on")
    # high pod: min(9,5)=5; low pod: min(3,5)=3 -> high wins with score 5.
    assert pod is high, f"expected higher-stock pod to win, got {pod}"
    assert score == 5, f"expected winning score 5, got {score}"
    print("PASS test_positive_score_ranking_unchanged")


def test_empty_pps_construction_zero_mutations():
    inv, _root = make_inventory()
    st = station(1, "picker", 5, 5)
    inv.station_manager.add_station(st)
    pod = pod_with(1, 2, 2, {101: (10, 0, 0.5, 1.0)})  # zero usable stock
    register_pod(inv, pod)
    inv.pod_manager.mark_pod_available(pod)

    order = Order(order_id=900, order_arrival=0)
    order.add_sku(101, 4)
    inv.order_manager.add_order(order)

    before_global = inv.pod_manager.skus_data[101]["current_global_qty"]
    before_visits = inv.pps_pod_visits
    before_picked = inv.pps_picked_quantity

    job = inv.add_picking_task_after_pps(
        st, pod, {101: [(900, 4)]}, {101: 4}
    )

    assert len(job.orders) == 0, "empty plan must yield an empty job"
    assert pod.skus[101]["current_qty"] == 0, "pod stock must be untouched"
    assert inv.pod_manager.skus_data[101]["current_global_qty"] == before_global, "global stock untouched"
    assert order.skus[101]["quantity_committed"] == 0, "no order commitment on empty plan"
    assert pod.pod_id not in st.incoming_pod, "station incoming membership untouched"
    assert pod.station is None, "pod.station untouched"
    assert inv.pod_manager.is_idle(pod.pod_id) is True, "pod must remain available"
    assert inv.pps_pod_visits == before_visits, "pps_pod_visits untouched"
    assert inv.pps_picked_quantity == before_picked, "pps_picked_quantity untouched"
    print("PASS test_empty_pps_construction_zero_mutations")


def test_nonempty_pps_construction_applies_each_mutation_once():
    inv, _root = make_inventory()
    st = station(1, "picker", 5, 5)
    inv.station_manager.add_station(st)
    pod = pod_with(1, 2, 2, {101: (10, 5, 0.5, 2.0)})
    register_pod(inv, pod)
    inv.pod_manager.mark_pod_available(pod)

    order = Order(order_id=900, order_arrival=0)
    order.add_sku(101, 3)
    inv.order_manager.add_order(order)

    before_global = inv.pod_manager.skus_data[101]["current_global_qty"]
    before_visits = inv.pps_pod_visits
    before_picked = inv.pps_picked_quantity

    job = inv.add_picking_task_after_pps(st, pod, {101: [(900, 3)]}, {101: 3})

    assert [o[1:] for o in job.orders] == [(101, 3)], f"unexpected job plan {job.orders}"
    assert pod.skus[101]["current_qty"] == 2, "pod stock reduced exactly once"
    assert inv.pod_manager.skus_data[101]["current_global_qty"] == before_global - 3, "global reduced once"
    assert order.skus[101]["quantity_committed"] == 3, "order committed once"
    assert st.incoming_pod.count(pod.pod_id) == 1, "pod added to station once"
    assert pod.station is st, "pod.station set"
    assert inv.pod_manager.is_idle(pod.pod_id) is False, "pod reserved (unavailable)"
    assert inv.pps_pod_visits == before_visits + 1, "one pod visit counted"
    # pps_picked_quantity is NOT incremented at creation anymore.
    assert inv.pps_picked_quantity == before_picked, "picked qty must not be counted at creation"
    print("PASS test_nonempty_pps_construction_applies_each_mutation_once")


def test_pps_picked_quantity_counted_at_delivery():
    inv, root = make_inventory()
    st = station(1, "picker", 5, 5)
    inv.station_manager.add_station(st)
    pod = pod_with(1, 5, 5, {101: (10, 9, 0.5, 1.0)})
    register_pod(inv, pod)

    pd.DataFrame(
        [{"order_id": 900, "item_id": 101, "item_quantity": 5, "status": -1, "order_finished": 0}]
    ).to_csv(Path(inv.assign_order_csv), index=False)

    order = Order(order_id=900, order_arrival=0)
    order.add_sku(101, 5)
    order.assign_station(st.station_id)
    order.commit_quantity(101, 5)
    inv.order_manager.add_order(order)
    st.add_order(900, order)
    st.add_pod(pod.pod_id)

    job = RobotJob(pod.coordinate, station_id=st.station_id, pod=pod)
    job.add_picking_task(900, 101, 3)
    # Dynamically added task must also be counted at delivery.
    job.add_picking_task(900, 101, 2)

    inv.finish_picking_task(job)
    assert inv.pps_picked_quantity == 5, f"expected 5 delivered counted, got {inv.pps_picked_quantity}"
    print("PASS test_pps_picked_quantity_counted_at_delivery")


def test_partial_delivery_does_not_prematurely_finish_csv():
    inv, root = make_inventory()
    st = station(1, "picker", 5, 5)
    inv.station_manager.add_station(st)
    pod = pod_with(1, 5, 5, {101: (10, 9, 0.5, 1.0)})
    register_pod(inv, pod)

    # assign_order.csv: single order/SKU line requiring 5 units.
    csv_path = Path(inv.assign_order_csv)
    pd.DataFrame(
        [{"order_id": 900, "item_id": 101, "item_quantity": 5, "status": -1, "order_finished": 0}]
    ).to_csv(csv_path, index=False)

    order = Order(order_id=900, order_arrival=0)
    order.add_sku(101, 5)
    order.assign_station(st.station_id)
    order.commit_quantity(101, 5)
    inv.order_manager.add_order(order)
    st.add_order(900, order)
    st.add_pod(pod.pod_id)

    # Partial delivery of 3 of 5.
    partial_job = RobotJob(pod.coordinate, station_id=st.station_id, pod=pod)
    partial_job.add_picking_task(900, 101, 3)
    inv.finish_picking_task(partial_job)

    df = pd.read_csv(csv_path)
    status_after_partial = int(df.loc[df["order_id"] == 900, "status"].iloc[0])
    assert status_after_partial != 1, "partial delivery must not mark CSV finished"

    # Deliver remaining 2 -> aggregate complete.
    st.add_pod(pod.pod_id)
    final_job = RobotJob(pod.coordinate, station_id=st.station_id, pod=pod)
    final_job.add_picking_task(900, 101, 2)
    inv.finish_picking_task(final_job)

    df = pd.read_csv(csv_path)
    status_after_full = int(df.loc[df["order_id"] == 900, "status"].iloc[0])
    assert status_after_full == 1, "aggregate completion must mark CSV finished"
    print("PASS test_partial_delivery_does_not_prematurely_finish_csv")


def test_finalize_completed_return_releases_pod():
    inv, _root = make_inventory()
    st = station(1, "picker", 5, 5)
    inv.station_manager.add_station(st)
    pod = pod_with(1, 5, 5, {101: (10, 5, 0.5, 1.0)})
    register_pod(inv, pod)
    # Pod is currently at the station, unavailable.
    st.add_pod(pod.pod_id)
    pod.station = st
    inv.pod_manager.mark_pod_not_available(pod)

    job = RobotJob(pod.coordinate, station_id=st.station_id, pod=pod)
    inv.finalize_completed_return(job)

    assert inv.pod_manager.is_idle(pod.pod_id) is True, "returned pod must be available"
    assert pod.station is None, "pod.station must be cleared after storage return"
    assert pod.pod_id not in st.incoming_pod, "stale station incoming membership must be removed"
    print("PASS test_finalize_completed_return_releases_pod")


def test_completed_return_releases_old_pod_before_committed_next():
    # Integration: drive the RTS committed-next return flow and confirm the old
    # pod is finalized (available, no station) and committed-next activated.
    from scripts.validation.rts_committed_next_smoke import (
        build_inventory,
        start_return,
        finish_return,
    )
    from src.rmfs.rl.rts.action_space import STORE

    (
        inv,
        picker,
        _repl,
        _final_storage,
        robot,
        current_pod,
        next_jobs_data,
        _policy,
    ) = build_inventory(branch=STORE, next_jobs=1, committed_enabled=True)

    start_return(robot)
    finish_return(robot)

    assert inv.pod_manager.is_idle(current_pod.pod_id) is True, "old pod must be released as available"
    assert current_pod.station is None, "old pod.station must be cleared"
    assert current_pod.pod_id not in picker.incoming_pod, "old pod must leave station incoming set"
    # committed-next activation should have replaced robot.job with the next job.
    next_job, next_pod, _storage = next_jobs_data[0]
    assert robot.job is next_job, "committed-next job must be activated after return"
    assert robot.job.pod is next_pod, "activated job must carry the reserved next pod"
    print("PASS test_completed_return_releases_old_pod_before_committed_next")


def test_unavailable_unowned_pod_flagged():
    inv, _root = make_inventory()
    pod = pod_with(1, 5, 5, {101: (10, 5, 0.5, 1.0)})
    register_pod(inv, pod)
    # Make it unavailable without any legitimate owner.
    inv.pod_manager.mark_pod_not_available(pod)

    result = check_runtime_invariants(inv)
    codes = {v["code"] for v in result["violations"]}
    assert "pod_unavailable_without_owner" in codes, f"expected unowned-pod violation, got {codes}"

    # Give it a legitimate owner (pending replenishment commitment) -> no violation.
    inv.pending_replenishment_dispatches.append(
        {"pod_id": pod.pod_id, "skus_to_replenish": [101], "created_tick": 0, "source": "post_pick"}
    )
    pod.has_pending_replenishment_dispatch = True
    result2 = check_runtime_invariants(inv)
    codes2 = {v["code"] for v in result2["violations"]}
    assert "pod_unavailable_without_owner" not in codes2, f"owned pod must not be flagged, got {codes2}"
    print("PASS test_unavailable_unowned_pod_flagged")


def main():
    test_zero_score_candidate_returns_no_pod()
    test_positive_score_ranking_unchanged()
    test_empty_pps_construction_zero_mutations()
    test_nonempty_pps_construction_applies_each_mutation_once()
    test_pps_picked_quantity_counted_at_delivery()
    test_partial_delivery_does_not_prematurely_finish_csv()
    test_finalize_completed_return_releases_pod()
    test_completed_return_releases_old_pod_before_committed_next()
    test_unavailable_unowned_pod_flagged()
    print("\nALL PPS/LIFECYCLE REGRESSION TESTS PASSED")


if __name__ == "__main__":
    raise SystemExit(main())
