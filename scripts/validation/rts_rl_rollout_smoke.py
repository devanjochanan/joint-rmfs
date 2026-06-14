#!/usr/bin/env python3
"""Pure synthetic smoke for Phase 7 RTS rollout integration."""

from __future__ import annotations

import json
import random
from pathlib import Path
import sys
import tempfile

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.rmfs.decisions.rts import CurrentRTSPolicy
from src.rmfs.decisions.rts.types import RTSDecision
from src.rmfs.rl.rts.action_space import STORE
from src.rmfs.rl.rts.cycle_reference import RTSCycleReference, write_cycle_reference
from src.rmfs.rl.rts.evaluation_policy import RTSRandomValidPolicy
from src.rmfs.rl.rts.evaluation_summary import summarize_rollout_events
from src.rmfs.rl.rts.outcome_tracker import PendingRTSDecision, RTSOutcomeTracker, RTSRolloutRuntime
from src.rmfs.rl.rts.rollout_schema import build_decision_event, build_outcome_event
from src.rmfs.rl.rts.rollout_writer import RTSRolloutWriter
from src.rmfs.rl.rts.runtime_config import RTSRuntimeConfig
from src.rmfs.rl.rts.runtime_registry import configure_rts_runtime, get_rts_runtime_config, reset_rts_runtime
from src.rmfs.rl.rts.state import build_state
from src.rmfs.rl.rts.storage_resolver import find_free_storage_in_zone


class Obj:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def build_context():
    pod = Obj(
        pod_id=7,
        pos_x=1,
        pos_y=1,
        skus={
            1: {"current_qty": 2, "limit_qty": 10, "threshold": 0.3, "weight": 1.0},
            2: {"current_qty": 0, "limit_qty": 8, "threshold": 0.2, "weight": 1.0},
        },
    )
    station = Obj(station_id="picker-1", station_type="picker", pos_x=2, pos_y=3)
    job = Obj(my_id="job-1", pod=pod, station_id="picker-1")
    robot = Obj(_id=1, id=1, object_type="robot", pos_x=5, pos_y=6, job=job, destination=Obj(x=10, y=1))
    storages = [
        Obj(pos_x=10, pos_y=1, is_empty=True, assigned_pod=None, zone_id="A"),
        Obj(pos_x=12, pos_y=5, is_empty=True, assigned_pod=None, zone_id="A"),
        Obj(pos_x=20, pos_y=1, is_empty=False, assigned_pod=pod, zone_id="B"),
        Obj(pos_x=21, pos_y=1, is_empty=True, assigned_pod=None, zone_id="B"),
    ]
    warehouse = Obj(
        _tick=3,
        storage_manager=Obj(storages=storages),
        station_manager=Obj(stations=[station, Obj(station_id="replenishment-1", station_type="replenishment")]),
        pod_manager=Obj(pods=[pod]),
        _objects=[robot],
    )
    robot.universe = warehouse
    robot.warehouse = warehouse
    return Obj(warehouse=warehouse, robot=robot, pod=pod, station=station), storages


def read_jsonl(path: Path):
    with path.open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


def main():
    default = RTSRuntimeConfig()
    assert default.policy_mode == "current"
    assert default.rollout_enabled is False

    configure_rts_runtime({"policy_mode": "current", "rollout_enabled": False})
    assert get_rts_runtime_config().policy_mode == "current"
    reset_rts_runtime()
    assert get_rts_runtime_config() == RTSRuntimeConfig()

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        decision = build_decision_event(
            decision_event_id="d1",
            tick=1,
            robot_id=1,
            job_id="j1",
            pod_id="p1",
            source_station_id="s1",
            source_station_type="picker",
            policy_name="current_rts",
            zone_ids=("A", "B"),
            action_mask=(1, 0, 0, 1),
            selected_action_index=0,
            selected_action_branch=STORE,
            selected_zone_id="A",
            selected_storage=Obj(pos_x=10, pos_y=1),
            state_json={"zone_rows": []},
            feature_shapes={"X_actions": [4, 48]},
        )
        outcome = build_outcome_event(
            decision_event_id="d1",
            tick=2,
            robot_id=1,
            job_id="j1",
            pod_id="p1",
            outcome_status="completed",
            return_start_tick=1,
            return_finish_tick=2,
            realized_cycle_time=1,
            destination_x=10,
            destination_y=1,
            reward_json={"reward_computed": False},
        )
        writer = RTSRolloutWriter(tmp_path / "rollout.jsonl")
        writer.write_decision(decision)
        writer.write_outcome(outcome)
        writer.close()
        rows = read_jsonl(tmp_path / "rollout.jsonl")
        assert len(rows) == 2
        summary = summarize_rollout_events(rows, policy_mode="current_probe")
        assert summary["decision_count"] == 1
        assert summary["outcome_count"] == 1
        assert summary["orphan_count"] == 0

    tracker = RTSOutcomeTracker()
    pending = PendingRTSDecision("d2", "r1", "j2", "p2", 5.0, STORE, {})
    tracker.record_decision(pending)
    assert tracker.complete_return(robot_id="r1", job_id="j2", pod_id="p2") == pending
    assert tracker.orphan_pending() == []

    context, storages = build_context()
    state_bundle = build_state(context, ("A", "B"))
    assert state_bundle.state_json["feature_fidelity"]["destination_robot_pressure"] == "approx_repo_grounded"
    zone_rows = {row["zone_id"]: row for row in state_bundle.state_json["zone_rows"]}
    assert zone_rows["A"]["zone_destination_robot_count"] == 1.0
    assert zone_rows["B"]["neighbor_zone_destination_robot_count"] == 1.0
    before = [(storage.pos_x, storage.pos_y, storage.is_empty, storage.assigned_pod) for storage in storages]
    resolved = find_free_storage_in_zone(context, "A", STORE)
    after = [(storage.pos_x, storage.pos_y, storage.is_empty, storage.assigned_pod) for storage in storages]
    assert resolved is storages[0]
    assert before == after

    policy = RTSRandomValidPolicy()
    rng = random.Random(123)
    for _ in range(50):
        action = policy.select_action(("A", "B"), (0, 1, 0, 0), rng)
        assert action.action_index == 1

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        runtime = RTSRolloutRuntime(
            config=RTSRuntimeConfig(policy_mode="current_probe", rollout_enabled=True, zone_ids=("A", "B")),
            runtime_root=tmp_path,
        )
        current_policy = CurrentRTSPolicy()
        decision = RTSDecision(
            storage=storages[0],
            destination=Obj(x=10, y=1),
            policy_name="current_rts",
            mode="nearest",
        )
        runtime.on_decision(robot=context.robot, context=context, decision=decision)
        context.warehouse._tick = 8
        runtime.on_return_completed(robot=context.robot)
        runtime.close()
        rows = read_jsonl(tmp_path / "rts_rollout.jsonl")
        assert len(rows) == 2
        assert rows[0]["event_type"] == "decision"
        assert rows[1]["event_type"] == "outcome"
        assert rows[1]["reward_json"]["reward_computed"] is False
        assert isinstance(current_policy, CurrentRTSPolicy)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        reference_path = tmp_path / "cycle_reference.json"
        write_cycle_reference(reference_path, RTSCycleReference(10.0, 8.0, 12.0, store_action_count=1))
        context, storages = build_context()
        runtime = RTSRolloutRuntime(
            config=RTSRuntimeConfig(
                policy_mode="current_probe",
                rollout_enabled=True,
                zone_ids=("A", "B"),
                reward_reference_path=str(reference_path),
            ),
            runtime_root=tmp_path,
        )
        runtime.on_decision(
            robot=context.robot,
            context=context,
            decision=RTSDecision(storages[0], Obj(x=10, y=1), "current_rts", "nearest"),
        )
        context.warehouse._tick = 9
        runtime.on_return_completed(robot=context.robot)
        runtime.close()
        rows = read_jsonl(tmp_path / "rts_rollout.jsonl")
        assert rows[-1]["reward_json"]["reward_computed"] is True
    # Check that rollout enabled + zero events still writes rts_rollout_summary.json on close
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        runtime = RTSRolloutRuntime(
            config=RTSRuntimeConfig(policy_mode="current_probe", rollout_enabled=True, zone_ids=("A", "B")),
            runtime_root=tmp_path,
        )
        runtime.close()
        summary_file = tmp_path / "rts_rollout_summary.json"
        assert summary_file.exists()
        with summary_file.open() as fh:
            summary_data = json.load(fh)
        assert summary_data["decision_count"] == 0
        assert summary_data["outcome_count"] == 0
        assert summary_data["policy_mode"] == "current_probe"

    # Check that random_valid storage policy only executes store branch (forces replenish invalid)
    from src.rmfs.rl.rts.evaluation_policy import RTSRandomValidStoragePolicy
    context, storages = build_context()
    context.warehouse.storage_manager.storages[0].is_empty = True
    context.warehouse.storage_manager.storages[0].assigned_pod = None
    random_valid_policy = RTSRandomValidStoragePolicy(zone_ids=("A", "B"), random_seed=123)
    for _ in range(50):
        decision_val = random_valid_policy.select_destination(context)
        assert decision_val.metadata["selected_action_branch"] == STORE
        mask_in_meta = decision_val.metadata["action_mask"]
        assert mask_in_meta[2] == 0
        assert mask_in_meta[3] == 0

    import sys
    from scripts.run.local_executor_smoke import main as cli_main
    old_argv = sys.argv
    try:
        sys.argv = ["local_executor_smoke.py", "--output-root", "tmp_cli", "--rts-policy-mode", "current_probe"]
        cli_main()
        raise AssertionError("cli_main should fail without --rts-rollout")
    except SystemExit:
        pass
    finally:
        sys.argv = old_argv

    print("rts rl rollout smoke ok")


if __name__ == "__main__":
    main()
