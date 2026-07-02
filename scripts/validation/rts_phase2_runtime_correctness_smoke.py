"""Focused Phase 2 RTS runtime correctness checks.

This script is intentionally unit-sized: it validates lifecycle and I/O
hardening without running a long NetLogo simulation.
"""

from __future__ import annotations

import pickle
import tempfile
from pathlib import Path
from types import SimpleNamespace

from engine.netlogo_coordinate import NetLogoCoordinate
from model.robot_job import RobotJob
from src.rmfs.rl.rts.evaluation_summary import RolloutSummaryAccumulator, summarize_rollout_events
from src.rmfs.rl.rts.outcome_tracker import (
    PAPER_CYCLE_STATUS_CENSORED_NO_NEXT_TASK,
    PAPER_CYCLE_STATUS_COMPLETE,
    PendingRTSDecision,
    RTSOutcomeTracker,
    RTSRolloutRuntime,
)
from src.rmfs.rl.rts.rollout_schema import build_decision_event, build_outcome_event
from src.rmfs.rl.rts.rollout_writer import RTSRolloutWriter
from src.rmfs.rl.rts.runtime_config import RTSRuntimeConfig


def main() -> int:
    test_job_identity()
    test_rollout_writer_pickle_excludes_events()
    test_incremental_summary_matches_reference()
    test_negative_return_duration_fails()
    test_no_next_task_censors_after_return()
    test_exact_committed_next_lineage()
    print("rts phase2 runtime correctness smoke ok")
    return 0


def test_job_identity() -> None:
    first = RobotJob(NetLogoCoordinate(1, 2), "picker-1", SimpleNamespace(pod_id=1))
    second = RobotJob(NetLogoCoordinate(2, 3), "picker-1", SimpleNamespace(pod_id=2))
    assert isinstance(first.job_id, int), first.job_id
    assert first.job_id == first.my_id
    assert second.job_id != first.job_id
    restored = pickle.loads(pickle.dumps(first))
    assert restored.job_id == first.job_id


def test_rollout_writer_pickle_excludes_events() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        writer = RTSRolloutWriter(Path(tmp) / "rollout.jsonl", enabled=True)
        writer.write_decision({"event_type": "decision", "decision_event_id": "d1"})
        assert len(writer.events) == 1
        state = writer.__getstate__()
        assert state["events"] == []
        writer.close()


def test_incremental_summary_matches_reference() -> None:
    events = [
        build_decision_event(
            decision_event_id="d1",
            tick=1,
            robot_id="r1",
            job_id="j1",
            pod_id="p1",
            source_station_id="s1",
            source_station_type="picker",
            policy_name="current_probe",
            zone_ids=["z1"],
            action_mask=[1, 1],
            selected_action_index=0,
            selected_action_branch="store",
            selected_zone_id="z1",
            selected_storage=None,
            state_json={},
            feature_shapes={},
        ),
        build_outcome_event(
            decision_event_id="d1",
            tick=3,
            robot_id="r1",
            job_id="j1",
            pod_id="p1",
            outcome_status="paper_cycle_completed",
            return_start_tick=1,
            return_finish_tick=2,
            realized_cycle_time=2,
            destination_x=1,
            destination_y=1,
            reward_json={"reward_computed": True},
            paper_cycle_status=PAPER_CYCLE_STATUS_COMPLETE,
            paper_cycle_complete=1,
            paper_cycle_duration=2,
        ),
    ]
    accumulator = RolloutSummaryAccumulator(policy_mode="current_probe")
    for event in events:
        accumulator.add_event(event)
    assert accumulator.to_summary() == summarize_rollout_events(events, policy_mode="current_probe")


def test_negative_return_duration_fails() -> None:
    tracker = RTSOutcomeTracker()
    tracker.record_decision(_pending(return_start_tick=10.0))
    try:
        tracker.mark_return_completed(robot_id="1", job_id="10", pod_id="20", return_finish_tick=9.0)
    except ValueError:
        return
    raise AssertionError("negative return duration did not fail")


def test_no_next_task_censors_after_return() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        runtime = _runtime(tmp)
        runtime.tracker.record_decision(_pending(committed_next_reservation_id=None))
        robot = _robot(job_id="10", pod_id="20", station_id="picker-1", tick=5.0)
        runtime.on_return_completed(robot=robot)
        runtime.close()
        statuses = [row.get("paper_cycle_status") for row in runtime.writer.events]
        assert PAPER_CYCLE_STATUS_CENSORED_NO_NEXT_TASK in statuses, statuses


def test_exact_committed_next_lineage() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        runtime = _runtime(tmp)
        pending = _pending(
            committed_next_reservation_id="cnr-1",
            committed_next_job_id="10",
            committed_next_pod_id="20",
            committed_next_station_id="picker-1",
            return_finish_tick=2.0,
        )
        runtime.tracker.record_decision(pending)
        wrong_robot = _robot(job_id="999", pod_id="20", station_id="picker-1", tick=5.0)
        assert not runtime._matches_committed_next_arrival(
            robot=wrong_robot,
            station=SimpleNamespace(station_id="picker-1"),
            pending=pending,
        )
        right_robot = _robot(job_id="10", pod_id="20", station_id="picker-1", tick=5.0)
        assert runtime._matches_committed_next_arrival(
            robot=right_robot,
            station=SimpleNamespace(station_id="picker-1"),
            pending=pending,
        )


def _runtime(tmp: str) -> RTSRolloutRuntime:
    return RTSRolloutRuntime(
        config=RTSRuntimeConfig(policy_mode="current_probe", rollout_enabled=True),
        runtime_root=Path(tmp),
    )


def _pending(**overrides) -> PendingRTSDecision:
    payload = dict(
        decision_event_id="d1",
        robot_id="1",
        job_id="10",
        pod_id="20",
        return_start_tick=1.0,
        selected_action_branch="store",
        metadata={},
        tick_to_second=0.15,
    )
    payload.update(overrides)
    return PendingRTSDecision(**payload)


def _robot(*, job_id: str, pod_id: str, station_id: str, tick: float):
    job = SimpleNamespace(
        my_id=job_id,
        pod=SimpleNamespace(pod_id=pod_id),
        committed_next_activated_by_robot_id="1",
        committed_next_reservation_id=None,
    )
    return SimpleNamespace(
        _id="1",
        job=job,
        destination=SimpleNamespace(x=1, y=1),
        universe=SimpleNamespace(_tick=tick),
        warehouse=SimpleNamespace(_tick=tick, tick_to_second=0.15),
    )


if __name__ == "__main__":
    raise SystemExit(main())
