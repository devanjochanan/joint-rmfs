#!/usr/bin/env python3
"""Bounded smoke for Phase 4 RTS action-context alignment."""

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
from src.rmfs.decisions.rts.types import RTSDestinationContext, RTSDecision
from src.rmfs.rl.rts.action_context import (
    revalidate_selected_context,
    selected_context_by_index,
)
from src.rmfs.rl.rts.action_space import REPLENISH_STORE, STORE, build_action_mask_from_contexts
from src.rmfs.rl.rts.features import build_feature_bundle
from src.rmfs.rl.rts.state import build_state

from scripts.validation.rts_branch_execution_smoke import (
    add_common_objects,
    build_inventory,
    finish_current_return,
    pod_with_stock,
    station,
    start_return,
)


class ContextPolicy:
    def __init__(self, branch, zone_id="zone-a", *, before_revalidate=None, after_decision=None):
        self.branch = branch
        self.zone_id = zone_id
        self.before_revalidate = before_revalidate
        self.after_decision = after_decision
        self.calls = 0
        self.selected_context = None
        self.pre_revalidation_context = None
        self.mask = None

    def select_destination(self, context):
        self.calls += 1
        state = build_state(context, (self.zone_id,))
        action_index = 0 if self.branch == STORE else 1
        self.mask = build_action_mask_from_contexts((self.zone_id,), state.action_contexts)
        pre_context = selected_context_by_index(state.action_contexts, action_index)
        self.pre_revalidation_context = pre_context
        if self.before_revalidate is not None:
            self.before_revalidate(context, pre_context)
        action_context = revalidate_selected_context(context, pre_context)
        self.selected_context = action_context
        proposal = action_context.next_job_proposal
        storage = action_context.candidate_storage
        decision = RTSDecision(
            storage=storage,
            destination=NetLogoCoordinate(storage.pos_x, storage.pos_y),
            policy_name="alignment_rts_rl",
            mode="rl",
            action_index=action_index,
            branch=self.branch,
            zone_id=self.zone_id,
            storage_id=action_context.candidate_storage_id,
            replenishment_station=action_context.replenishment_station,
            replenishment_station_id=action_context.replenishment_station_id,
            action_context_id=action_context.context_id,
            action_context_version=action_context.context_version,
            next_job_proposal_id=getattr(proposal, "proposal_id", None),
            metadata={
                "actor_kind": "rts_rl_explicit",
                "policy_checkpoint_id": "alignment_smoke",
                "policy_mode": "rts_rl_explicit",
                "feature_schema_id": "alignment_smoke",
                "selected_action_index": action_index,
                "selected_action_branch": self.branch,
                "selected_zone_id": self.zone_id,
                "action_mask": list(self.mask),
                "action_context_id": action_context.context_id,
                "action_context_version": action_context.context_version,
                "candidate_storage_id": action_context.candidate_storage_id,
                "selected_replenishment_station_id": action_context.replenishment_station_id,
                "next_job_proposal_id": getattr(proposal, "proposal_id", None),
                "state_json": state.state_json,
            },
        )
        if self.after_decision is not None:
            self.after_decision(context, decision)
        return decision


def mark_storage_zone(storage, zone_id="zone-a"):
    storage.rts_zone_id = zone_id
    return storage


def build_context_fixture(*, current_qty=2, committed=True):
    inv = build_inventory()
    inv.committed_next_reservations_enabled = committed
    pod = pod_with_stock(current_qty=current_qty)
    picker, repl, storage, robot, job = add_common_objects(inv, pod)
    mark_storage_zone(storage)
    return inv, picker, repl, storage, robot, job, pod


def snapshot(inv, pod):
    return {
        "empty": tuple(sorted((s.pos_x, s.pos_y) for s in inv.storage_manager.empty_storages)),
        "assigned": tuple(sorted((p.pod_id, s.pos_x, s.pos_y) for p, s in inv.storage_manager.pods_to_storage.items())),
        "station_robots": tuple(sorted((s.station_id, tuple(sorted(s.robot_ids))) for s in inv.station_manager.stations)),
        "station_queues": tuple((s.station_id, tuple(s.robot_queue)) for s in inv.station_manager.stations),
        "job_queue_len": len(inv.job_queue),
        "pod_flags": (
            pod.rts_return_in_progress,
            pod.is_awaiting_replenishment,
            pod.has_pending_replenishment_dispatch,
        ),
        "stock": tuple(sorted((sku, details["current_qty"]) for sku, details in pod.skus.items())),
    }


def test_context_non_mutating_and_mask_equivalence():
    inv, picker, _repl, storage, robot, _job, pod = build_context_fixture()
    before = snapshot(inv, pod)
    context = RTSDestinationContext(inv, robot, pod, picker)
    state = build_state(context, ("zone-a",))
    after = snapshot(inv, pod)
    assert before == after
    mask = build_action_mask_from_contexts(("zone-a",), state.action_contexts)
    for action_context in state.action_contexts:
        assert mask[action_context.action_index] == (1 if action_context.action_valid else 0)
        if action_context.action_valid:
            assert action_context.candidate_storage is storage
            assert action_context.candidate_storage_id == "10:5"
            assert action_context.storage_feasibility.available
            assert action_context.next_job_proposal is not None
            if action_context.branch == REPLENISH_STORE:
                assert action_context.replenishment_station is not None
                assert action_context.station_feasibility.admission_possible


def test_invalid_reason_codes():
    inv, picker, repl, storage, robot, _job, pod = build_context_fixture(current_qty=9)
    mark_storage_zone(storage)
    context = RTSDestinationContext(inv, robot, pod, picker)
    state = build_state(context, ("zone-a",))
    repl_context = selected_context_by_index(state.action_contexts, 1)
    assert not repl_context.action_valid
    assert any(reason.startswith("replenishment_not_eligible") for reason in repl_context.invalid_reason_codes)

    storage.assigned_pod = pod
    storage.is_empty = False
    state = build_state(context, ("zone-a",))
    store_context = selected_context_by_index(state.action_contexts, 0)
    assert not store_context.action_valid
    assert "no_available_storage" in store_context.invalid_reason_codes

    storage.assigned_pod = None
    storage.is_empty = True
    inv.station_manager.replenishment_stations.remove(repl)
    state = build_state(context, ("zone-a",))
    repl_context = selected_context_by_index(state.action_contexts, 1)
    assert not repl_context.action_valid
    assert "no_replenishment_station" in repl_context.invalid_reason_codes


def test_exact_store_alignment():
    inv, _picker, _repl, storage, robot, _job, pod = build_context_fixture()
    policy = ContextPolicy(STORE)
    inv.rts_policy = policy
    start_return(robot)
    decision = inv.rts_rollout_runtime.decisions[0]
    proposal = policy.selected_context.next_job_proposal
    assert policy.selected_context.candidate_storage is storage
    assert proposal.candidate_storage is storage
    assert decision.storage is storage
    assert inv.storage_manager.getStorageByPod(pod) is storage
    assert robot.destination.x == storage.pos_x and robot.destination.y == storage.pos_y
    finish_current_return(robot)
    assert inv.storage_manager.getStorageByPod(pod) is storage
    assert not pod.rts_return_in_progress


def test_exact_replenish_alignment():
    inv, _picker, repl, storage, robot, job, pod = build_context_fixture()
    policy = ContextPolicy(REPLENISH_STORE)
    inv.rts_policy = policy
    start_return(robot)
    decision = inv.rts_rollout_runtime.decisions[0]
    assert policy.selected_context.replenishment_station is repl
    assert decision.replenishment_station is repl
    assert job.rts_replenishment_station is repl
    assert job.rts_final_storage is storage
    assert robot.destination.x == repl.pos_x and robot.destination.y == repl.pos_y
    robot.pos_x = repl.pos_x
    robot.pos_y = repl.pos_y
    robot.coordinate = NetLogoCoordinate(robot.pos_x, robot.pos_y)
    robot.route_stop_points = []
    robot.advance_state_if_needed()
    assert robot.current_state == "station_processing"
    job.replenishment_delay = 0
    inv.finish_task_in_job(job)
    robot.route_stop_points = []
    robot.advance_state_if_needed()
    assert robot.destination.x == storage.pos_x and robot.destination.y == storage.pos_y


def test_no_next_job_does_not_mask_action():
    inv, picker, _repl, _storage, robot, _job, pod = build_context_fixture(committed=True)
    context = RTSDestinationContext(inv, robot, pod, picker)
    state = build_state(context, ("zone-a",))
    mask = build_action_mask_from_contexts(("zone-a",), state.action_contexts)
    assert mask == [1, 1]
    features = build_feature_bundle(("zone-a",), mask, state.state_json)
    next_known_index = features.action_feature_names.index("next_job_known")
    assert features.X_actions[0, next_known_index] == 0.0
    assert features.X_actions[1, next_known_index] == 0.0
    inv.rts_policy = ContextPolicy(STORE)
    start_return(robot)
    assert inv.committed_next_registry.get_for_robot(robot) is None


def test_selected_storage_replacement():
    inv, _picker, _repl, storage, robot, _job, _pod = build_context_fixture()
    replacement = inv.storage_manager.createStorage(11, 5)
    mark_storage_zone(replacement)

    blocker = Pod(99)

    def stale_storage(_context, action_context):
        action_context.candidate_storage.assigned_pod = blocker
        action_context.candidate_storage.is_empty = False
        if action_context.candidate_storage in inv.storage_manager.empty_storages:
            inv.storage_manager.empty_storages.remove(action_context.candidate_storage)

    policy = ContextPolicy(STORE, before_revalidate=stale_storage)
    inv.rts_policy = policy
    start_return(robot)
    assert policy.pre_revalidation_context.candidate_storage is storage
    assert policy.selected_context.candidate_storage is replacement
    assert robot.destination.x == replacement.pos_x


def test_selected_station_replacement():
    inv, _picker, repl, _storage, robot, job, _pod = build_context_fixture()
    replacement = station(2, "replenishment", 9, 5)
    inv.station_manager.add_station(replacement)

    def stale_station(_context, action_context):
        action_context.replenishment_station.max_robots = 0

    policy = ContextPolicy(REPLENISH_STORE, before_revalidate=stale_station)
    inv.rts_policy = policy
    start_return(robot)
    assert policy.pre_revalidation_context.replenishment_station is repl
    assert policy.selected_context.replenishment_station is replacement
    assert job.rts_replenishment_station is replacement


def test_rollback_after_selected_reservation():
    inv, _picker, _repl, storage, robot, _job, pod = build_context_fixture()

    def fail_after_decision(context, _decision):
        context.warehouse.graph_pod.fail = True

    inv.rts_policy = ContextPolicy(STORE, after_decision=fail_after_decision)
    try:
        start_return(robot)
    except RuntimeError as exc:
        assert "failed to start RTS store return" in str(exc)
    else:
        raise AssertionError("expected RTS route failure")
    assert storage in inv.storage_manager.empty_storages
    assert inv.storage_manager.getStorageByPod(pod) is None
    assert storage.assigned_pod is None
    assert not pod.rts_return_in_progress
    assert inv.committed_next_registry.get_for_robot(robot) is None


def main():
    test_context_non_mutating_and_mask_equivalence()
    test_invalid_reason_codes()
    test_exact_store_alignment()
    test_exact_replenish_alignment()
    test_no_next_job_does_not_mask_action()
    test_selected_storage_replacement()
    test_selected_station_replacement()
    test_rollback_after_selected_reservation()
    print("rts action alignment smoke ok")


if __name__ == "__main__":
    main()
