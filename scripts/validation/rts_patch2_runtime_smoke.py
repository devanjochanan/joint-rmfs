#!/usr/bin/env python3
"""Focused Patch 2 RTS runtime checks."""

from __future__ import annotations

import pickle
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

import networkx as nx

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from engine.netlogo_coordinate import NetLogoCoordinate
from src.rmfs.decisions.rts.types import RTSDecision
from src.rmfs.rl.rts.evaluation_summary import summarize_rollout_events
from src.rmfs.rl.rts.outcome_tracker import RTSRolloutRuntime
from src.rmfs.rl.rts.rollout_schema import build_decision_event, build_outcome_event
from src.rmfs.rl.rts.runtime_config import RTSRuntimeConfig
from src.rmfs.rl.rts.static_runtime_index import (
    get_or_build_static_runtime_index,
    get_static_runtime_index,
    install_static_runtime_index,
    invalidate_static_runtime_index,
    rebuild_static_runtime_index,
    reset_static_runtime_index_diagnostics,
    static_runtime_index_diagnostics,
    validate_or_rebuild_static_runtime_index,
    validate_static_runtime_index_identity,
)
from src.rmfs.rl.rts.training.on_policy_dataset import build_on_policy_training_steps


class Obj:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class GraphWrapper:
    def __init__(self, graph):
        self.graph = graph


def main() -> int:
    test_static_index_lifecycle()
    test_minimal_capture_skips_full_state_builders()
    test_minimal_dataset_rejection_and_funnel_summary()
    print("rts patch2 runtime smoke ok")
    return 0


def test_static_index_lifecycle() -> None:
    warehouse = _warehouse()
    reset_static_runtime_index_diagnostics()
    index = install_static_runtime_index(warehouse)
    assert index is get_static_runtime_index(warehouse)
    setup_diag = static_runtime_index_diagnostics()
    assert setup_diag["build_count"] == 1, setup_diag
    assert setup_diag["graph_matrix_build_count"] == 2, setup_diag
    assert setup_diag["empty_matrix_build_count"] == 1, setup_diag
    assert setup_diag["loaded_matrix_build_count"] == 1, setup_diag

    before = static_runtime_index_diagnostics()
    for _ in range(5):
        assert get_or_build_static_runtime_index(warehouse) is index
        assert get_static_runtime_index(warehouse) is index
    after = static_runtime_index_diagnostics()
    assert after["layout_hash_count"] == before["layout_hash_count"], (before, after)
    assert after["storage_hash_count"] == before["storage_hash_count"], (before, after)
    assert after["graph_hash_count"] == before["graph_hash_count"], (before, after)
    assert after["cached_retrieval_count"] > before["cached_retrieval_count"], (before, after)

    assert validate_static_runtime_index_identity(warehouse, index)
    audited = static_runtime_index_diagnostics()
    assert audited["identity_validation_count"] == 1, audited
    assert audited["graph_hash_count"] > after["graph_hash_count"], (after, audited)

    payload = pickle.dumps(warehouse)
    assert b"RTSStaticRuntimeIndex" not in payload
    assert b"distance_matrix" not in payload

    invalidate_static_runtime_index(warehouse)
    assert get_static_runtime_index(warehouse) is None
    rebuilt = rebuild_static_runtime_index(warehouse)
    assert rebuilt is get_static_runtime_index(warehouse)
    assert rebuilt is not index
    final = static_runtime_index_diagnostics()
    assert final["rebuild_count"] == 1, final
    assert final["build_count"] == 2, final
    warehouse.graph.graph.add_edge("0,0", "1,1", weight=3.0)
    changed = validate_or_rebuild_static_runtime_index(warehouse)
    assert changed is not rebuilt
    assert validate_static_runtime_index_identity(warehouse, changed)


def test_minimal_capture_skips_full_state_builders() -> None:
    import src.rmfs.rl.rts.outcome_tracker as tracker_module

    with tempfile.TemporaryDirectory() as tmp:
        runtime = RTSRolloutRuntime(
            config=RTSRuntimeConfig(policy_mode="current", rollout_enabled=True, zone_ids=("A",)),
            runtime_root=Path(tmp),
        )
        context, robot, storage = _minimal_context()
        decision = RTSDecision(
            storage=storage,
            destination=NetLogoCoordinate(storage.pos_x, storage.pos_y),
            policy_name="current",
            mode="nearest",
            action_index=0,
            branch="store",
            zone_id="A",
        )
        original_build_state = tracker_module.build_state
        original_build_feature_bundle = tracker_module.build_feature_bundle

        def forbidden(*args, **kwargs):
            raise AssertionError("minimal capture called a full-state builder")

        tracker_module.build_state = forbidden
        tracker_module.build_feature_bundle = forbidden
        try:
            runtime.on_decision(robot=robot, context=context, decision=decision)
        finally:
            tracker_module.build_state = original_build_state
            tracker_module.build_feature_bundle = original_build_feature_bundle
            runtime.close()
        rows = list(runtime.writer.events)
        assert rows, rows
        decision_row = rows[0]
        assert decision_row["state_capture_mode"] == "minimal"
        assert decision_row["state_available"] is False
        assert decision_row["trainable"] is False
        assert decision_row["state_json"] is None
        assert decision_row["action_mask"] == []
        summary = summarize_rollout_events(rows, policy_mode="current")
        assert summary["invalid_action_selected_count"] == 0, summary


def test_minimal_dataset_rejection_and_funnel_summary() -> None:
    decision = build_decision_event(
        decision_event_id="d-min",
        tick=1,
        robot_id="r1",
        job_id="j1",
        pod_id="p1",
        source_station_id="picker-1",
        source_station_type="picker",
        policy_name="rts_rl_explicit",
        zone_ids=("A",),
        action_mask=None,
        selected_action_index=0,
        selected_action_branch="store",
        selected_zone_id="A",
        selected_storage=Obj(pos_x=5, pos_y=5),
        state_json=None,
        feature_shapes=None,
        actor_kind="rts_rl_explicit",
        policy_checkpoint_id="ckpt",
        policy_mode="greedy",
        old_log_prob=-0.1,
        old_value=0.2,
        netlogo_step=1,
        warehouse_time=0.15,
        tick_to_second=0.15,
        state_capture_mode="minimal",
        state_available=False,
        trainable=False,
        nontrainable_reason="minimal_state_capture",
    )
    decision.update(
        {
            "eligible_job_pool_size": 3,
            "committed_next_candidate_count": 2,
            "next_job_proposal_id": "cnp-1",
            "committed_next_reservation_id": "cnr-1",
            "committed_next_job_id": "job-next",
            "committed_next_pod_id": "pod-next",
            "committed_next_activation_time_seconds": 2.0,
            "proposal_build_ms": 4.0,
            "eligible_pool_build_ms": 1.5,
        }
    )
    outcome = build_outcome_event(
        decision_event_id="d-min",
        tick=4,
        robot_id="r1",
        job_id="job-next",
        pod_id="pod-next",
        outcome_status="paper_cycle_completed",
        return_start_tick=1,
        return_finish_tick=2,
        realized_cycle_time=3,
        destination_x=5,
        destination_y=5,
        reward_json={"reward_computed": True, "reward_value": 1.0},
        paper_cycle_status="complete",
        paper_cycle_complete=1,
        paper_cycle_duration=3,
        paper_cycle_completion_rule="next_order_retrieval_arrival",
    )
    outcome["committed_next_reservation_id"] = "cnr-1"
    outcome["committed_next_activation_time_seconds"] = 2.0
    dataset = build_on_policy_training_steps([decision, outcome], required_policy_checkpoint_id="ckpt")
    assert dataset.summary["trainable_step_count"] == 0
    assert dataset.summary["rejected_minimal_capture_count"] == 1, dataset.summary
    summary = summarize_rollout_events([decision, outcome], policy_mode="rts_rl_explicit")
    assert summary["decisions_with_nonempty_eligible_job_pool"] == 1, summary
    assert summary["decisions_with_known_proposed_next_job"] == 1, summary
    assert summary["reservations_committed"] == 1, summary
    assert summary["reservations_activated"] == 1, summary
    assert summary["completed_paper_cycle_count"] == 1, summary
    assert summary["trainable_transition_count"] == 0, summary


def _warehouse():
    storages = []
    for number, (x, y) in enumerate(((0, 0), (1, 0), (0, 1), (1, 1)), start=1):
        storages.append(
            Obj(
                storage_number=number,
                storage_id=str(number),
                pos_x=x,
                pos_y=y,
                coordinate=NetLogoCoordinate(x, y),
                is_empty=True,
                assigned_pod=None,
            )
        )
    graph = _graph()
    return Obj(
        layout=Obj(layout_id="patch2-smoke"),
        storage_manager=Obj(storages=storages),
        graph=GraphWrapper(graph),
        graph_pod=GraphWrapper(graph.copy()),
        _objects=tuple(storages),
    )


def _graph():
    graph = nx.DiGraph()
    nodes = ("0,0", "1,0", "0,1", "1,1")
    graph.add_nodes_from(nodes)
    for src, dst in (
        ("0,0", "1,0"),
        ("1,0", "0,0"),
        ("0,0", "0,1"),
        ("0,1", "0,0"),
        ("1,0", "1,1"),
        ("1,1", "1,0"),
        ("0,1", "1,1"),
        ("1,1", "0,1"),
    ):
        graph.add_edge(src, dst, weight=1.0)
    return graph


def _minimal_context():
    storage = Obj(pos_x=5, pos_y=5)
    job = Obj(my_id="job-1", pod=Obj(pod_id="pod-1"))
    robot = Obj(_id="r1", id="r1", job=job, destination=NetLogoCoordinate(5, 5))
    station = Obj(station_id="picker-1", station_type="picker")
    warehouse = Obj(_tick=0.15, tick_to_second=0.15, committed_next_registry=None)
    robot.warehouse = warehouse
    robot.universe = warehouse
    return Obj(warehouse=warehouse, robot=robot, pod=job.pod, station=station), robot, storage


if __name__ == "__main__":
    raise SystemExit(main())
