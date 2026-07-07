#!/usr/bin/env python3
"""Deterministic tests for shortest-leg storage resolver.

Layout (Manhattan distances, same line or grid):
  Picker at (5, 5)
  Storage A at (6, 5) — nearest to picker (d=1), far from next pod (d=12)
  Storage B at (5, 15) — far from picker (d=10), near next pod (d=1)
  Next pod at (4, 15)

Shortest-leg costs (loaded + empty):
  Storage A: loaded(1) + empty(12) = 13
  Storage B: loaded(10) + empty(1) = 11  <- wins
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
from model.robot import Robot
from model.robot_job import RobotJob
from model.station import Station
from src.rmfs.decisions.rts.types import RTSDestinationContext
from src.rmfs.rl.rts.action_context import (
    revalidate_selected_context,
    select_candidate_storage,
    select_shortest_leg_storage,
    selected_context_by_index,
)
from src.rmfs.rl.rts.action_space import STORE
from src.rmfs.rl.rts.state import build_state
from src.rmfs.rl.rts.outcome_tracker import RTSRolloutRuntime
from src.rmfs.rl.rts.runtime_config import RTSRuntimeConfig


class SimpleGraph:
    key = "shortest-leg-smoke"

    def __init__(self):
        self.fail = False

    def dijkstra(self, start, end, avoid=None):
        if self.fail:
            return None
        return [start, end]

    def dijkstra_modified(self, start, end, penalties, zone_boundary, avoid=None):
        return self.dijkstra(start, end, avoid)


def make_station(sid, station_type, x, y):
    st = Station(sid, station_type)
    st.pos_x = x
    st.pos_y = y
    st.coordinate = NetLogoCoordinate(x, y)
    st.short_path = [NetLogoCoordinate(x, y)]
    st.long_path = [NetLogoCoordinate(x, y)]
    return st


def make_pod(pod_id, x, y, current_qty=8):
    pod = Pod(pod_id)
    pod.pos_x = x
    pod.pos_y = y
    pod.add_sku(101, limit_qty=10, current_qty=current_qty, threshold=0.5, weight=2.0)
    return pod


def build_fixture(*, next_jobs=1):
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
    inv.committed_next_reservations_enabled = True
    graph = SimpleGraph()
    inv.graph = graph
    inv.graph_pod = graph
    inv.rts_rollout_runtime = RTSRolloutRuntime(
        config=RTSRuntimeConfig(
            policy_mode="current_probe",
            rollout_enabled=True,
            zone_ids=("zone-a",),
            committed_next_reservations_enabled=True,
        ),
        runtime_root=root,
    )

    picker = make_station(1, "picker", 5, 5)
    repl = make_station(1, "replenishment", 8, 5)
    inv.station_manager.add_station(picker)
    inv.station_manager.add_station(repl)

    current_pod = make_pod(1, 5, 5, current_qty=2)
    inv.pod_manager.add_pod(current_pod)
    inv.pod_manager.add_sku_data(
        101, current_qty=2, max_qty=10, global_threshold_inv_level=0.5,
    )

    # Storage A: nearest to picker (d=1)
    storage_a = inv.storage_manager.createStorage(6, 5)
    storage_a.rts_zone_id = "zone-a"

    # Storage B: far from picker (d=10) but near next pod (d=1)
    storage_b = inv.storage_manager.createStorage(5, 15)
    storage_b.rts_zone_id = "zone-a"

    next_jobs_data = []
    if next_jobs > 0:
        next_pod = make_pod(2, 4, 15, current_qty=8)
        inv.pod_manager.add_pod(next_pod)
        next_stg = inv.storage_manager.createStorage(4, 15)
        next_stg.rts_zone_id = "zone-a"
        inv.storage_manager.addPodToStorage(next_pod, next_stg)
        inv.pod_manager.mark_pod_not_available(next_pod)
        job = RobotJob(
            next_pod.coordinate, station_id=picker.station_id, pod=next_pod,
        )
        job.add_picking_task(900, 101, 1)
        inv.job_queue.append(job)
        next_jobs_data.append((job, next_pod, next_stg))

    inv.storage_manager.initStorageManager()

    robot = Robot(inv)
    robot.pos_x = picker.pos_x
    robot.pos_y = picker.pos_y
    robot.coordinate = NetLogoCoordinate(robot.pos_x, robot.pos_y)
    inv.addObject(robot)
    current_job = RobotJob(
        current_pod.coordinate, station_id=picker.station_id, pod=current_pod,
    )
    current_job.set_job_finish()
    robot.job = current_job
    robot.current_state = "station_processing"
    robot.route_stop_points = []
    picker.add_pod(current_pod.pod_id)
    picker.add_robot(robot.robotName())
    inv.pod_manager.mark_pod_not_available(current_pod)

    return inv, picker, repl, storage_a, storage_b, robot, current_pod, next_jobs_data


def test_shortest_leg_wins_over_nearest():
    """Storage B (farther from picker) is selected because its two-leg cost is lower."""
    inv, picker, _repl, storage_a, storage_b, robot, pod, _next = build_fixture(
        next_jobs=1,
    )
    context = RTSDestinationContext(inv, robot, pod, picker)

    nearest, _ = select_candidate_storage(context, "zone-a")
    assert nearest is storage_a, f"expected storage_a as nearest, got {nearest}"

    state = build_state(context, ("zone-a",))
    store_ctx = selected_context_by_index(state.action_contexts, 0)
    replenish_ctx = selected_context_by_index(state.action_contexts, 1)

    assert store_ctx.candidate_storage is storage_b, (
        f"shortest-leg should pick storage_b at (5,15), "
        f"got {store_ctx.candidate_storage_coordinate}"
    )
    assert store_ctx.next_job_proposal is not None
    assert store_ctx.next_job_proposal.has_next_job
    assert store_ctx.next_job_proposal.candidate_storage is storage_b
    assert replenish_ctx.candidate_storage is storage_a, (
        f"replenish_store should keep nearest resolver storage_a, "
        f"got {replenish_ctx.candidate_storage_coordinate}"
    )
    assert replenish_ctx.next_job_proposal is not None
    assert replenish_ctx.next_job_proposal.candidate_storage is storage_a


def test_nearest_fallback_with_no_next_job():
    """Without a next-job proposal, falls back to nearest-in-zone."""
    inv, picker, _repl, storage_a, _storage_b, robot, pod, _ = build_fixture(
        next_jobs=0,
    )
    context = RTSDestinationContext(inv, robot, pod, picker)

    state = build_state(context, ("zone-a",))
    store_ctx = selected_context_by_index(state.action_contexts, 0)

    assert store_ctx.candidate_storage is storage_a, (
        f"no-next-job should pick nearest storage_a, "
        f"got {store_ctx.candidate_storage_coordinate}"
    )


def test_revalidation_retains_shortest_leg():
    """Revalidation keeps or recomputes store-only shortest-leg storage."""
    inv, picker, _repl, storage_a, storage_b, robot, pod, _ = build_fixture(
        next_jobs=1,
    )
    context = RTSDestinationContext(inv, robot, pod, picker)

    state = build_state(context, ("zone-a",))
    store_ctx = selected_context_by_index(state.action_contexts, 0)
    assert store_ctx.candidate_storage is storage_b

    refreshed = revalidate_selected_context(context, store_ctx)
    assert refreshed.candidate_storage is storage_b, (
        f"revalidation should retain storage_b, "
        f"got {refreshed.candidate_storage_coordinate}"
    )
    assert refreshed.next_job_proposal.candidate_storage is storage_b
    assert refreshed.cycle_estimate.next_job_proposal_id == refreshed.next_job_proposal.proposal_id

    storage_c = inv.storage_manager.createStorage(5, 14)
    storage_c.rts_zone_id = "zone-a"
    storage_b.assigned_pod = pod
    storage_b.is_empty = False
    refreshed2 = revalidate_selected_context(context, store_ctx)
    assert refreshed2.candidate_storage is storage_c, (
        f"after storage_b infeasible, should recompute shortest-leg storage_c, "
        f"got {refreshed2.candidate_storage_coordinate}"
    )
    assert refreshed2.next_job_proposal.candidate_storage is storage_c
    assert refreshed2.next_job_proposal.candidate_storage_id == refreshed2.candidate_storage_id
    assert refreshed2.cycle_estimate.next_job_proposal_id == refreshed2.next_job_proposal.proposal_id


def main():
    test_shortest_leg_wins_over_nearest()
    test_nearest_fallback_with_no_next_job()
    test_revalidation_retains_shortest_leg()
    print("rts shortest-leg storage smoke ok")


if __name__ == "__main__":
    main()
