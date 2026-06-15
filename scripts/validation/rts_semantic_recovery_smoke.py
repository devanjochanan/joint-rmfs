#!/usr/bin/env python3
"""Focused smoke for the RTS-RL semantic recovery patch."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.rmfs.decisions.rts.types import RTSDecision
from src.rmfs.rl.rts.action_space import STORE
from src.rmfs.rl.rts.graph_distance import DISTANCE_SEMANTICS_VERSION, graph_distance_or_fallback
from src.rmfs.rl.rts.outcome_tracker import RTSRolloutRuntime
from src.rmfs.rl.rts.runtime_config import RTSRuntimeConfig
from src.rmfs.rl.rts.state import build_state
from src.rmfs.rl.rts.storage_resolver import find_free_storage_in_zone
from src.rmfs.rl.rts.training.checkpoint import write_feature_schema
from src.rmfs.rl.rts.training.on_policy_dataset import build_on_policy_training_steps
from src.rmfs.rl.rts.zone_registry import (
    ZONE_GEOMETRY_VERSION,
    build_zone_registry,
    validate_no_col_zone_ids,
)
from src.rmfs.rl.rts.features import build_action_feature_names, build_stock_feature_names
from src.rmfs.rl.rts.rollout_schema import build_decision_event, build_outcome_event


class Obj:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def build_context():
    import networkx as nx

    pod = Obj(pod_id=7, skus={1: {"current_qty": 1, "limit_qty": 5, "threshold": 0.5}})
    station = Obj(station_id="picker-1", station_type="picker", pos_x=0, pos_y=0)
    job = Obj(my_id="job-1", pod=pod, station_id="picker-1")
    robot = Obj(_id=1, id=1, object_type="robot", pos_x=0, pos_y=0, job=job, destination=Obj(x=5, y=0))
    storages = [
        Obj(storage_number=1, pos_x=0, pos_y=0, is_empty=False, assigned_pod=pod),
        Obj(storage_number=2, pos_x=1, pos_y=0, is_empty=True, assigned_pod=None),
        Obj(storage_number=3, pos_x=4, pos_y=0, is_empty=True, assigned_pod=None),
        Obj(storage_number=4, pos_x=5, pos_y=0, is_empty=True, assigned_pod=None),
        Obj(storage_number=5, pos_x=0, pos_y=4, is_empty=True, assigned_pod=None),
        Obj(storage_number=6, pos_x=1, pos_y=4, is_empty=True, assigned_pod=None),
        Obj(storage_number=7, pos_x=4, pos_y=4, is_empty=True, assigned_pod=None),
        Obj(storage_number=8, pos_x=5, pos_y=4, is_empty=True, assigned_pod=None),
    ]
    graph = nx.DiGraph()
    for coord in ("0,0", "1,0", "4,0", "5,0", "0,4", "1,4", "4,4", "5,4"):
        graph.add_node(coord)
    graph.add_edge("0,0", "4,0", weight=10.0)
    graph.add_edge("4,0", "0,0", weight=10.0)
    graph.add_edge("0,0", "5,0", weight=1.0)
    graph.add_edge("5,0", "0,0", weight=1.0)
    graph.add_edge("0,0", "1,0", weight=2.0)
    graph.add_edge("1,0", "0,0", weight=2.0)
    warehouse = Obj(
        _tick=3,
        tick_to_second=1.0,
        storage_manager=Obj(storages=storages),
        station_manager=Obj(stations=[station, Obj(station_id="replenishment-1", station_type="replenishment", pos_x=0, pos_y=4)]),
        pod_manager=Obj(pods=[pod]),
        graph_pod=Obj(graph=graph),
        _objects=[robot],
    )
    robot.universe = warehouse
    robot.warehouse = warehouse
    return Obj(warehouse=warehouse, robot=robot, pod=pod, station=station), storages


def read_jsonl(path: Path):
    with path.open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


def main():
    context, storages = build_context()
    registry = build_zone_registry(context)
    assert registry.zone_ids == (
        "rts_z_r00_c00",
        "rts_z_r00_c01",
        "rts_z_r01_c00",
        "rts_z_r01_c01",
    )
    assert registry.geometry_version == ZONE_GEOMETRY_VERSION
    assert "rts_z_r00_c00" in registry.zones_by_id["rts_z_r00_c01"].neighbor_zone_ids
    try:
        validate_no_col_zone_ids(("col_0",), context="semantic smoke")
        raise AssertionError("col_* zone id should be rejected")
    except ValueError:
        pass

    state = build_state(context, registry.zone_ids).state_json
    assert state["zone_registry"]["zone_geometry_version"] == ZONE_GEOMETRY_VERSION
    assert state["zone_registry"]["distance_semantics_version"] == DISTANCE_SEMANTICS_VERSION
    assert not any(str(zone_id).startswith("col_") for zone_id in state["zone_registry"]["zone_ids"])

    # Graph distance must beat Manhattan for the tie-break in the selected zone.
    result = graph_distance_or_fallback(context.warehouse, context.station, storages[3])
    assert result.distance == 1.0
    before = [(s.storage_number, s.is_empty, s.assigned_pod) for s in storages]
    selected = find_free_storage_in_zone(context, "rts_z_r00_c01", STORE)
    after = [(s.storage_number, s.is_empty, s.assigned_pod) for s in storages]
    assert selected is storages[3]
    assert before == after

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        runtime = RTSRolloutRuntime(
            config=RTSRuntimeConfig(policy_mode="current_probe", rollout_enabled=True, zone_ids=registry.zone_ids),
            runtime_root=tmp_path,
        )
        runtime.on_decision(
            robot=context.robot,
            context=context,
            decision=RTSDecision(storages[3], Obj(x=5, y=0), "current_rts", "nearest"),
        )
        context.warehouse._tick = 8
        runtime.on_return_completed(robot=context.robot)
        context.warehouse._tick = 13
        runtime.on_station_arrival(robot=context.robot, station=context.station)
        runtime.close()
        rows = read_jsonl(tmp_path / "rts_rollout.jsonl")
        assert [row["event_type"] for row in rows] == ["decision", "outcome", "outcome"]
        assert rows[1]["paper_cycle_status"] == "pending"
        assert rows[2]["paper_cycle_status"] == "complete"
        assert rows[2]["paper_cycle_duration"] == 10

    with tempfile.TemporaryDirectory() as tmp:
        schema = write_feature_schema(
            Path(tmp) / "feature_schema.json",
            action_feature_names=build_action_feature_names(registry.zone_ids),
            stock_feature_names=build_stock_feature_names(),
        )
        assert schema["zone_geometry_version"] == ZONE_GEOMETRY_VERSION
        assert schema["reward_horizon"] == "paper_cycle_duration"
        assert schema["distance_semantics_version"] == DISTANCE_SEMANTICS_VERSION

    decision = build_decision_event(
        decision_event_id="pending-only",
        tick=1,
        robot_id=1,
        job_id="job",
        pod_id="pod",
        source_station_id="picker",
        source_station_type="picker",
        policy_name="rts_rl_explicit",
        zone_ids=registry.zone_ids,
        action_mask=(1, 1, 1, 1, 0, 0, 0, 0),
        selected_action_index=0,
        selected_action_branch=STORE,
        selected_zone_id=registry.zone_ids[0],
        selected_storage=None,
        state_json=state,
        feature_shapes={},
        actor_kind="rts_rl_explicit",
        policy_checkpoint_id="batch_000001",
        old_log_prob=-0.5,
        old_value=0.0,
    )
    pending = build_outcome_event(
        decision_event_id="pending-only",
        tick=2,
        robot_id=1,
        job_id="job",
        pod_id="pod",
        outcome_status="return_completed",
        return_start_tick=1,
        return_finish_tick=2,
        realized_cycle_time=None,
        destination_x=1,
        destination_y=0,
        reward_json={"reward_computed": False},
        paper_cycle_status="pending",
        paper_cycle_complete=0,
        paper_cycle_start_tick=1,
        paper_cycle_storage_arrival_tick=2,
    )
    dataset = build_on_policy_training_steps([decision, pending], required_policy_checkpoint_id="batch_000001")
    assert dataset.summary["trainable_step_count"] == 0
    assert dataset.summary["rejected_missing_completed_paper_cycle_count"] == 1

    print("rts semantic recovery smoke ok")


if __name__ == "__main__":
    main()
