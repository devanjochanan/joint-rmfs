#!/usr/bin/env python3
"""Bounded smoke for Phase 5 RTS structural cycle estimates."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

os.environ.setdefault("RMFS_FAST_TRAIN", "1")
os.environ.setdefault("RMFS_DETAIL_DB", "0")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/rmfs-mpl")

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from engine.netlogo_coordinate import NetLogoCoordinate
from src.rmfs.decisions.rts.types import RTSDestinationContext
from src.rmfs.rl.rts.action_context import selected_context_by_index
from src.rmfs.rl.rts.action_space import REPLENISH_STORE, STORE, build_action_mask_from_contexts
from src.rmfs.rl.rts.features import build_feature_bundle
from src.rmfs.rl.rts.graph_distance import graph_distance_or_fallback
from src.rmfs.rl.rts.state import build_state
from src.rmfs.rl.rts.travel_time import EMPTY_ROBOT, LOADED_ROBOT

from scripts.validation.rts_committed_next_smoke import (
    arrive_at_next_pod,
    arrive_at_station,
    build_inventory,
    finish_return,
    finish_replenishment_leg,
    read_jsonl,
    start_return,
)


def test_topology_specific_distance_cache():
    import networkx as nx

    empty = nx.DiGraph()
    loaded = nx.DiGraph()
    empty.add_edge("0,0", "1,0", weight=2.0)
    loaded.add_edge("0,0", "1,0", weight=5.0)
    warehouse = SimpleNamespace(
        graph=SimpleNamespace(graph=empty),
        graph_pod=SimpleNamespace(graph=loaded),
    )
    src = NetLogoCoordinate(0, 0)
    dst = NetLogoCoordinate(1, 0)
    empty_result = graph_distance_or_fallback(warehouse, src, dst, topology=EMPTY_ROBOT, allow_metric_fallback=False)
    loaded_result = graph_distance_or_fallback(warehouse, src, dst, topology=LOADED_ROBOT, allow_metric_fallback=False)
    assert empty_result.distance == 2.0
    assert loaded_result.distance == 5.0
    assert empty_result.topology == EMPTY_ROBOT
    assert loaded_result.topology == LOADED_ROBOT


def test_store_cycle_components_and_features():
    inv, picker, _repl, _final_storage, robot, _pod, _next_jobs, _policy = build_inventory(branch=STORE, next_jobs=1)
    context = RTSDestinationContext(inv, robot, robot.job.pod, picker)
    state = build_state(context, ("zone-a",))
    action_context = selected_context_by_index(state.action_contexts, 0)
    estimate = action_context.cycle_estimate.to_json_dict()
    assert estimate["known"] is True
    assert estimate["status"] == "available"
    assert estimate["branch"] == STORE
    assert [component["topology"] for component in estimate["route_components"]] == [
        LOADED_ROBOT,
        EMPTY_ROBOT,
        LOADED_ROBOT,
    ]
    assert estimate["estimated_queue_seconds"] == 0.0
    assert estimate["estimated_replenishment_service_seconds"] == 0.0
    assert estimate["estimated_handling_seconds"] == 20.0
    expected = (5.0 + 13.0 + 8.0) / robot.maximum_speed + 20.0
    assert abs(estimate["estimated_cycle_seconds"] - expected) < 1e-9
    mask = build_action_mask_from_contexts(("zone-a",), state.action_contexts)
    features = build_feature_bundle(("zone-a",), mask, state.state_json)
    names = features.action_feature_names
    # v6: the cycle_estimate_known feature was removed; estimated_cycle_time
    # remains and, per the schema-v6 invariant, must be finite for a valid
    # action with a known proposed next job.
    assert "cycle_estimate_known" not in names
    assert "estimated_cycle_time" in names
    assert "estimated_queue_time" not in names
    assert abs(features.X_actions[0, names.index("estimated_cycle_time")] - expected) < 1e-5


def test_replenish_store_cycle_components():
    inv, picker, _repl, _final_storage, robot, _pod, _next_jobs, _policy = build_inventory(branch=REPLENISH_STORE, next_jobs=1)
    context = RTSDestinationContext(inv, robot, robot.job.pod, picker)
    state = build_state(context, ("zone-a",))
    action_context = selected_context_by_index(state.action_contexts, 1)
    estimate = action_context.cycle_estimate.to_json_dict()
    assert estimate["known"] is True
    assert estimate["branch"] == REPLENISH_STORE
    assert [component["name"] for component in estimate["route_components"]] == [
        "loaded_picker_to_replenishment_station",
        "loaded_replenishment_station_to_final_storage",
        "empty_final_storage_to_committed_next_pod",
        "loaded_committed_next_pod_to_picker",
    ]
    assert estimate["queue_estimate"]["queue_semantics"] == "host_parallel_processing_no_serial_service_queue"
    assert estimate["estimated_queue_seconds"] == 0.0
    assert estimate["estimated_replenishment_service_seconds"] == 20.0
    expected = (3.0 + 2.0 + 13.0 + 8.0) / robot.maximum_speed + 20.0 + 20.0
    assert abs(estimate["estimated_cycle_seconds"] - expected) < 1e-9


def test_no_next_job_cycle_unknown_without_masking():
    inv, picker, _repl, _final_storage, robot, _pod, _next_jobs, _policy = build_inventory(branch=STORE, next_jobs=0)
    context = RTSDestinationContext(inv, robot, robot.job.pod, picker)
    state = build_state(context, ("zone-a",))
    mask = build_action_mask_from_contexts(("zone-a",), state.action_contexts)
    assert mask == [1, 1]
    for action_context in state.action_contexts:
        estimate = action_context.cycle_estimate.to_json_dict()
        assert estimate["known"] is False
        assert estimate["status"] == "unavailable_no_next_job"
    features = build_feature_bundle(("zone-a",), mask, state.state_json)
    known_index = features.action_feature_names.index("proposed_next_job_known")
    cycle_index = features.action_feature_names.index("estimated_cycle_time")
    dist_index = features.action_feature_names.index("candidate_to_proposed_next_pod_distance")
    # No proposed next job -> proposed_next_job_known=0, estimated_cycle_time=0,
    # raw candidate-to-next-pod distance=0.
    assert features.X_actions[0, known_index] == 0.0
    assert features.X_actions[1, known_index] == 0.0
    assert features.X_actions[0, cycle_index] == 0.0
    assert features.X_actions[1, cycle_index] == 0.0
    assert features.X_actions[0, dist_index] == 0.0
    assert features.X_actions[1, dist_index] == 0.0


def test_rollout_estimate_and_error_fields():
    inv, picker, _repl, _final_storage, robot, _current_pod, _next_jobs, _policy = build_inventory(branch=STORE, next_jobs=1)
    start_return(robot)
    finish_return(robot)
    arrive_at_next_pod(robot)
    arrive_at_station(robot, picker)
    rows = read_jsonl(Path(inv._smoke_tmp.name) / "rts_rollout.jsonl")
    decisions = [row for row in rows if row.get("event_type") == "decision"]
    completed = [row for row in rows if row.get("outcome_status") == "paper_cycle_completed"]
    assert len(decisions) == 1
    assert len(completed) == 1
    assert decisions[0]["cycle_estimate_known"] == 1
    assert decisions[0]["estimated_cycle_time_at_decision"] is not None
    assert completed[0]["estimated_cycle_time_at_decision"] == decisions[0]["estimated_cycle_time_at_decision"]
    assert completed[0]["cycle_estimate_error"] is not None
    assert completed[0]["cycle_estimate_absolute_error"] >= 0.0
    assert completed[0]["cycle_estimate_relative_error"] >= 0.0


def main():
    test_topology_specific_distance_cache()
    test_store_cycle_components_and_features()
    test_replenish_store_cycle_components()
    test_no_next_job_cycle_unknown_without_masking()
    test_rollout_estimate_and_error_fields()
    print("rts cycle estimator smoke ok")


if __name__ == "__main__":
    main()
