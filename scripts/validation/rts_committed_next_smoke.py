#!/usr/bin/env python3
"""Bounded smoke for RTS committed-next retrieval lifecycle."""

from __future__ import annotations

import json
import os
import pickle
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
from src.rmfs.decisions.rts.types import RTSDestinationContext, RTSDecision
from src.rmfs.decisions.task_allocation.committed_next import (
    get_committed_next_action_proposal,
    get_committed_next_action_proposals,
    get_committed_next_job,
    get_committed_next_pod,
    get_committed_next_reservation,
    get_committed_next_station,
    get_committed_next_zone,
    has_committed_next_retrieval,
)
from src.rmfs.rl.rts.action_space import REPLENISH_STORE, STORE
from src.rmfs.rl.rts.ablation import resolve_ablation
from src.rmfs.rl.rts.action_context import selected_context_by_index
from src.rmfs.rl.rts.features import build_action_feature_names, build_stock_feature_names
from src.rmfs.rl.rts.storage_resolver import find_free_storage_in_zone
from src.rmfs.rl.rts.state import build_state
from src.rmfs.rl.rts.outcome_tracker import RTSRolloutRuntime
from src.rmfs.rl.rts.runtime_config import RTSRuntimeConfig
from src.rmfs.rl.rts.training.checkpoint import write_feature_schema
from src.rmfs.rl.rts.training.on_policy_dataset import build_on_policy_training_steps


def smoke_schema_id() -> str:
    with tempfile.TemporaryDirectory() as tmp:
        schema = write_feature_schema(
            Path(tmp) / "feature_schema.json",
            action_feature_names=build_action_feature_names(("zone-a",)),
            stock_feature_names=build_stock_feature_names(),
        )
        return schema["feature_schema_id"]


SMOKE_SCHEMA_ID = smoke_schema_id()
SMOKE_ABLATION = resolve_ablation("full")


class SimpleGraph:
    key = "committed-next-smoke"

    def __init__(self):
        self.fail = False

    def dijkstra(self, start, end, avoid=None):
        if self.fail:
            return None
        return [start, end]

    def dijkstra_modified(self, start, end, penalties, zone_boundary, avoid=None):
        return self.dijkstra(start, end, avoid)


class BranchPolicy:
    def __init__(self, branch, storage):
        self.branch = branch
        self.storage = storage
        self.calls = 0
        self.reservation_seen = None
        self.proposal_seen = None
        self.proposals_seen = {}

    def select_destination(self, context):
        self.calls += 1
        self.reservation_seen = get_committed_next_reservation(context.warehouse, context.robot)
        state = build_state(context, ("zone-a",))
        selected_action_index = 0 if self.branch == STORE else 1
        action_context = selected_context_by_index(state.action_contexts, selected_action_index)
        self.proposals_seen = get_committed_next_action_proposals(context.warehouse, context.robot)
        self.proposal_seen = get_committed_next_action_proposal(context.warehouse, context.robot, self.branch, "zone-a")
        proposal = action_context.next_job_proposal
        return RTSDecision(
            storage=action_context.candidate_storage or self.storage,
            destination=NetLogoCoordinate(
                (action_context.candidate_storage or self.storage).pos_x,
                (action_context.candidate_storage or self.storage).pos_y,
            ),
            policy_name="smoke_rts_rl",
            mode="rl",
            action_index=selected_action_index,
            branch=self.branch,
            zone_id="zone-a",
            storage_id=action_context.candidate_storage_id,
            replenishment_station=action_context.replenishment_station,
            replenishment_station_id=action_context.replenishment_station_id,
            action_context_id=action_context.context_id,
            action_context_version=action_context.context_version,
            next_job_proposal_id=getattr(proposal, "proposal_id", None),
            metadata={
                "actor_kind": "rts_rl_explicit",
                "policy_checkpoint_id": "smoke_checkpoint",
                "policy_mode": "rts_rl_explicit",
                "feature_schema_id": SMOKE_SCHEMA_ID,
                "feature_ablation": SMOKE_ABLATION.name,
                "feature_ablation_hash": SMOKE_ABLATION.hash,
                "old_log_prob": -0.25,
                "old_value": 0.5,
                "policy_entropy": 0.1,
                "selected_action_branch": self.branch,
                "selected_zone_id": "zone-a",
                "selected_action_index": selected_action_index,
                "action_mask": [1 if ctx.action_valid else 0 for ctx in state.action_contexts],
                "action_context_id": action_context.context_id,
                "action_context_version": action_context.context_version,
                "candidate_storage_id": action_context.candidate_storage_id,
                "selected_replenishment_station_id": action_context.replenishment_station_id,
                "next_job_proposal_id": getattr(proposal, "proposal_id", None),
                "state_json": state.state_json,
            },
        )


def read_jsonl(path: Path):
    with path.open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


def station(station_id, station_type, x, y):
    st = Station(station_id, station_type)
    st.pos_x = x
    st.pos_y = y
    st.coordinate = NetLogoCoordinate(x, y)
    st.short_path = [NetLogoCoordinate(x, y)]
    st.long_path = [NetLogoCoordinate(x, y)]
    return st


def pod_with_stock(pod_id, x, y, current_qty=8):
    pod = Pod(pod_id)
    pod.pos_x = x
    pod.pos_y = y
    pod.add_sku(101, limit_qty=10, current_qty=current_qty, threshold=0.5, weight=2.0)
    return pod


def build_inventory(*, branch=STORE, next_jobs=1, committed_enabled=True):
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
    graph = SimpleGraph()
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

    picker = station(1, "picker", 5, 5)
    repl = station(1, "replenishment", 8, 5)
    inv.station_manager.add_station(picker)
    inv.station_manager.add_station(repl)

    current_pod = pod_with_stock(1, 5, 5, current_qty=2)
    inv.pod_manager.add_pod(current_pod)
    inv.pod_manager.add_sku_data(101, current_qty=current_pod.skus[101]["current_qty"], max_qty=10, global_threshold_inv_level=0.5)

    final_storage = inv.storage_manager.createStorage(10, 5)
    final_storage.rts_zone_id = "zone-a"
    next_jobs_data = []
    for index in range(next_jobs):
        pod = pod_with_stock(2 + index, 1 + index, 1, current_qty=8)
        inv.pod_manager.add_pod(pod)
        storage = inv.storage_manager.createStorage(1 + index, 1)
        storage.rts_zone_id = "zone-a"
        inv.storage_manager.addPodToStorage(pod, storage)
        inv.pod_manager.mark_pod_not_available(pod)
        job = RobotJob(pod.coordinate, station_id=picker.station_id, pod=pod)
        job.add_picking_task(900 + index, 101, 1)
        inv.job_queue.append(job)
        next_jobs_data.append((job, pod, storage))
    inv.storage_manager.initStorageManager()

    robot = Robot(inv)
    robot.pos_x = picker.pos_x
    robot.pos_y = picker.pos_y
    robot.coordinate = NetLogoCoordinate(robot.pos_x, robot.pos_y)
    inv.addObject(robot)
    job = RobotJob(current_pod.coordinate, station_id=picker.station_id, pod=current_pod)
    job.set_job_finish()
    robot.job = job
    robot.current_state = "station_processing"
    robot.route_stop_points = []
    picker.add_pod(current_pod.pod_id)
    picker.add_robot(robot.robotName())
    inv.pod_manager.mark_pod_not_available(current_pod)
    policy = BranchPolicy(branch, final_storage)
    inv.rts_policy = policy
    return inv, picker, repl, final_storage, robot, current_pod, next_jobs_data, policy


def start_return(robot):
    robot.route_stop_points = []
    robot.advance_state_if_needed()


def finish_return(robot):
    robot.warehouse._tick += 5
    robot.pos_x = robot.destination.x
    robot.pos_y = robot.destination.y
    robot.coordinate = NetLogoCoordinate(robot.pos_x, robot.pos_y)
    robot.route_stop_points = []
    robot.advance_state_if_needed()


def arrive_at_next_pod(robot):
    robot.warehouse._tick += 3
    robot.pos_x = robot.destination.x
    robot.pos_y = robot.destination.y
    robot.coordinate = NetLogoCoordinate(robot.pos_x, robot.pos_y)
    robot.route_stop_points = []
    robot.advance_state_if_needed()


def arrive_at_station(robot, station):
    robot.warehouse._tick += 4
    robot.pos_x = station.pos_x
    robot.pos_y = station.pos_y
    robot.coordinate = NetLogoCoordinate(robot.pos_x, robot.pos_y)
    robot.route_stop_points = []
    robot.advance_state_if_needed()


def finish_replenishment_leg(inv, robot, repl):
    inv._tick += 4
    robot.pos_x = repl.pos_x
    robot.pos_y = repl.pos_y
    robot.coordinate = NetLogoCoordinate(robot.pos_x, robot.pos_y)
    robot.route_stop_points = []
    robot.advance_state_if_needed()
    robot.job.replenishment_delay = 0
    inv.finish_task_in_job(robot.job)
    robot.route_stop_points = []
    robot.advance_state_if_needed()


def rows(inv):
    return read_jsonl(Path(inv._smoke_tmp.name) / "rts_rollout.jsonl")


def assert_one_completed_cycle(inv):
    event_rows = rows(inv)
    completed = [row for row in event_rows if row.get("outcome_status") == "paper_cycle_completed"]
    assert len(completed) == 1, event_rows
    assert completed[0]["paper_cycle_status"] == "complete"
    assert completed[0]["committed_next_reservation_id"]
    dataset = build_on_policy_training_steps(
        event_rows,
        required_policy_checkpoint_id="smoke_checkpoint",
        required_feature_schema_id=SMOKE_SCHEMA_ID,
        required_feature_ablation_hash=SMOKE_ABLATION.hash,
        expected_tick_to_second=1.0,
    )
    assert dataset.summary["trainable_step_count"] == 1
    return event_rows, completed[0]


def test_committed_store_lifecycle():
    inv, picker, _repl, final_storage, robot, current_pod, next_jobs, policy = build_inventory(branch=STORE, next_jobs=1)
    next_job, next_pod, next_storage = next_jobs[0]
    before_queue = list(inv.job_queue)
    before_qty = next_pod.skus[101]["current_qty"]

    start_return(robot)

    reservation = get_committed_next_reservation(inv, robot)
    assert policy.calls == 1
    assert policy.reservation_seen is None
    assert policy.proposal_seen is not None
    assert policy.proposal_seen.job is next_job
    assert policy.proposal_seen.candidate_storage is final_storage
    assert reservation is not None
    assert reservation is not None
    assert has_committed_next_retrieval(inv, robot)
    assert get_committed_next_job(inv, robot) is next_job
    assert get_committed_next_pod(inv, robot) is next_pod
    assert get_committed_next_station(inv, robot) is picker
    assert get_committed_next_zone(inv, robot) == "zone-a"
    assert inv.job_queue == before_queue
    assert next_job.orders == [(900, 101, 1)]
    assert next_pod.skus[101]["current_qty"] == before_qty
    assert inv.storage_manager.getStorageByPod(next_pod) is next_storage
    assert final_storage not in inv.storage_manager.empty_storages
    assert next_job.committed_next_reservation_id == reservation.reservation_id
    assert next_pod.committed_next_reservation_id == reservation.reservation_id
    assert not get_committed_next_action_proposals(inv, robot)

    finish_return(robot)

    assert inv.storage_manager.getStorageByPod(current_pod) is final_storage
    assert not current_pod.rts_return_in_progress
    assert robot.job is next_job
    assert robot.current_state == "taking_pod"
    assert next_job not in inv.job_queue
    assert next_job.committed_next_activated_by_robot_id == str(robot._id)
    assert reservation.linked_rts_decision_event_id
    assert reservation.status == "activated"

    arrive_at_next_pod(robot)
    assert robot.current_state == "delivering_pod"
    assert inv.storage_manager.getStorageByPod(next_pod) is None
    assert next_storage in inv.storage_manager.empty_storages
    arrive_at_station(robot, picker)

    event_rows, completed = assert_one_completed_cycle(inv)
    decision_rows = [row for row in event_rows if row.get("event_type") == "decision"]
    return_rows = [row for row in event_rows if row.get("outcome_status") == "return_completed"]
    assert len(decision_rows) == 1
    assert len(return_rows) == 1
    assert completed["committed_next_job_id"] == str(next_job.my_id)
    assert completed["committed_next_pod_id"] == str(next_pod.pod_id)


def test_committed_replenish_store_lifecycle():
    inv, picker, repl, final_storage, robot, current_pod, next_jobs, policy = build_inventory(branch=REPLENISH_STORE, next_jobs=1)
    next_job, next_pod, _next_storage = next_jobs[0]

    start_return(robot)
    assert robot.current_state == "delivering_pod"
    assert robot.job.pod is current_pod
    assert robot.job.station_id == repl.station_id
    finish_replenishment_leg(inv, robot, repl)
    assert robot.current_state == "returning_pod"
    assert robot.destination.x == final_storage.pos_x
    finish_return(robot)

    assert policy.calls == 1
    assert robot.job is next_job
    assert robot.current_state == "taking_pod"
    arrive_at_next_pod(robot)
    arrive_at_station(robot, picker)
    event_rows, _completed = assert_one_completed_cycle(inv)
    assert len([row for row in event_rows if row.get("event_type") == "decision"]) == 1
    assert next_pod.pod_id == int(robot.job.pod.pod_id)


def test_no_next_task_censored_and_dataset_rejected():
    inv, _picker, _repl, _storage, robot, _pod, _next_jobs, _policy = build_inventory(branch=STORE, next_jobs=0)
    start_return(robot)
    finish_return(robot)
    event_rows = rows(inv)
    decisions = [row for row in event_rows if row.get("event_type") == "decision"]
    assert len(decisions) == 1
    assert decisions[0]["eligible_job_pool_size"] == 0
    assert decisions[0]["eligible_job_queue_length"] == 0
    assert decisions[0]["eligible_job_rejection_counts_by_reason"] == {}
    censored = [row for row in event_rows if row.get("paper_cycle_status") == "censored_no_next_task"]
    assert len(censored) == 1
    dataset = build_on_policy_training_steps(event_rows, required_policy_checkpoint_id="smoke_checkpoint")
    assert dataset.summary["trainable_step_count"] == 0
    assert dataset.summary["rejected_missing_completed_paper_cycle_count"] == 1
    assert robot.current_state == "idle"


def test_cancelled_and_route_failure_cleanup():
    inv, _picker, _repl, _storage, robot, _pod, next_jobs, _policy = build_inventory(branch=STORE, next_jobs=1)
    next_job, next_pod, _next_storage = next_jobs[0]
    start_return(robot)
    reservation = get_committed_next_reservation(inv, robot)
    inv.committed_next_registry.cancel_for_robot(robot, "smoke_cancel")
    finish_return(robot)
    assert robot.current_state == "idle"
    assert next_job in inv.job_queue
    assert next_job.committed_next_reservation_id is None
    assert next_pod.committed_next_reservation_id is None
    assert reservation.status == "cancelled"
    assert any(row.get("paper_cycle_status") == "censored_committed_next_cancelled" for row in rows(inv))

    inv2, _picker2, _repl2, _storage2, robot2, _pod2, next_jobs2, _policy2 = build_inventory(branch=STORE, next_jobs=1)
    next_job2, next_pod2, _next_storage2 = next_jobs2[0]
    start_return(robot2)
    inv2.graph.fail = True
    finish_return(robot2)
    assert robot2.current_state == "idle"
    assert next_job2 in inv2.job_queue
    assert next_job2.committed_next_reservation_id is None
    assert next_pod2.committed_next_reservation_id is None
    assert any(row.get("paper_cycle_status") == "censored_committed_next_cancelled" for row in rows(inv2))


def test_conflict_guards_and_multiple_reservations():
    inv, _picker, repl, _storage, robot, _pod, next_jobs, _policy = build_inventory(branch=STORE, next_jobs=2)
    other_robot = Robot(inv)
    other_robot.pos_x = 20
    other_robot.pos_y = 20
    inv.addObject(other_robot)
    start_return(robot)
    primary_reservation = get_committed_next_reservation(inv, robot)
    first_job = primary_reservation.job
    first_pod = first_job.pod

    assert not inv.pod_manager.is_idle(first_pod.pod_id)
    assert inv.get_most_depleted_eligible_pod_for_sku(101) is None
    inv.pending_replenishment_dispatches.append(
        {"pod_id": first_pod.pod_id, "skus_to_replenish": [101], "created_tick": 0, "guaranteed_on_release": False}
    )
    first_pod.has_pending_replenishment_dispatch = True
    assert inv.dispatch_pending_replenishment_requests() == 0
    assert first_pod.pod_id not in repl.incoming_pod

    inv.ensure_committed_next_reservation(other_robot)
    other_reservation = get_committed_next_reservation(inv, other_robot)
    assert other_reservation is not None
    assert other_reservation.job is not first_job


def test_zone_conditioned_proposals_and_storage_alignment():
    inv, picker, _repl, final_storage, robot, _pod, _next_jobs, _policy = build_inventory(branch=STORE, next_jobs=1)
    zone_b_storage = inv.storage_manager.createStorage(30, 5)
    zone_b_storage.rts_zone_id = "zone-b"
    zone_ids = ("zone-a", "zone-b")
    context = RTSDestinationContext(
        warehouse=inv,
        robot=robot,
        pod=robot.job.pod,
        station=picker,
    )

    build_state(context, zone_ids)
    proposals = get_committed_next_action_proposals(inv, robot)
    assert get_committed_next_reservation(inv, robot) is None
    assert set(proposals) == {"zone-a", "zone-b"}

    store_a = proposals["zone-a"]
    store_b = proposals["zone-b"]
    assert store_a is get_committed_next_action_proposal(inv, robot, STORE, "zone-a")
    assert store_a is get_committed_next_action_proposal(inv, robot, REPLENISH_STORE, "zone-a")
    assert store_a.candidate_storage is final_storage
    assert store_b.candidate_storage is zone_b_storage
    assert store_a.proposal_cost != store_b.proposal_cost
    assert store_a.to_state_json()["proposed_next_job_known"] == 1
    assert "regret_score" not in store_a.to_state_json()
    assert find_free_storage_in_zone(context, "zone-a", STORE) is final_storage
    assert find_free_storage_in_zone(context, "zone-b", STORE) is zone_b_storage


def test_pickle_and_run_end_finalization():
    inv, _picker, _repl, _storage, robot, _pod, next_jobs, _policy = build_inventory(branch=STORE, next_jobs=1)
    next_job, next_pod, _next_storage = next_jobs[0]
    start_return(robot)
    loaded = pickle.loads(pickle.dumps(inv))
    loaded_robot = next(o for o in loaded.get_movable_objects() if isinstance(o, Robot))
    loaded_reservation = get_committed_next_reservation(loaded, loaded_robot)
    assert loaded_reservation is not None
    assert loaded_reservation.job.committed_next_reservation_id == loaded_reservation.reservation_id
    assert loaded_reservation.job.pod.committed_next_reservation_id == loaded_reservation.reservation_id

    inv.finalize_committed_next_run_end()
    inv.finalize_committed_next_run_end()
    event_rows = rows(inv)
    censored = [row for row in event_rows if row.get("paper_cycle_status") == "censored_run_end"]
    assert len(censored) == 1
    assert get_committed_next_reservation(inv, robot) is None
    assert next_job.committed_next_reservation_id is None
    assert next_pod.committed_next_reservation_id is None


def main():
    test_committed_store_lifecycle()
    test_committed_replenish_store_lifecycle()
    test_no_next_task_censored_and_dataset_rejected()
    test_cancelled_and_route_failure_cleanup()
    test_conflict_guards_and_multiple_reservations()
    test_zone_conditioned_proposals_and_storage_alignment()
    test_pickle_and_run_end_finalization()
    print("rts committed-next smoke ok")


if __name__ == "__main__":
    main()
