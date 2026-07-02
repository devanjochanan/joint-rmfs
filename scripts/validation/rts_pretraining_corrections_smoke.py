#!/usr/bin/env python3
"""Focused pre-training RTS correction checks."""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.validation.rts_ppo_update_smoke import synthetic_state
from src.rmfs.decisions.task_allocation.committed_next import CommittedNextProposal, CommittedNextRegistry
from src.rmfs.rl.rts.evaluation_summary import summarize_rollout_events
from src.rmfs.rl.rts.outcome_tracker import (
    PAPER_CYCLE_STATUS_CENSORED_MAXIMUM_HORIZON,
    PendingRTSDecision,
    RTSRolloutRuntime,
)
from src.rmfs.rl.rts.rollout_schema import build_decision_event, build_outcome_event
from src.rmfs.rl.rts.runtime_config import RTSRuntimeConfig
import src.rmfs.rl.rts.state as state_module
from src.rmfs.rl.rts.training.on_policy_dataset import build_on_policy_training_steps


class Obj:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def main() -> int:
    test_reward_uses_realized_paper_cycle_duration()
    test_proposal_availability_and_candidate_summary()
    test_reservation_cancellation_separate_from_censoring()
    test_same_decision_no_job_proposal_revalidation()
    test_explicit_proposal_failure_is_fatal()
    test_training_metric_names_are_explicit()
    print("rts pretraining corrections smoke ok")
    return 0


def test_reward_uses_realized_paper_cycle_duration() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        runtime = RTSRolloutRuntime(
            config=RTSRuntimeConfig(policy_mode="current", rollout_enabled=True, zone_ids=("A",)),
            runtime_root=Path(tmp),
        )
        captured = {}

        def reward_json(branch: str, paper_cycle_duration: float):
            captured["branch"] = branch
            captured["paper_cycle_duration"] = paper_cycle_duration
            return {
                "reward_computed": True,
                "reward_value": paper_cycle_duration,
                "components": {"cycle_time": paper_cycle_duration, "cycle_time_source": "paper_cycle_duration"},
            }

        runtime._reward_json = reward_json  # type: ignore[method-assign]
        pending = PendingRTSDecision(
            decision_event_id="d-realized",
            robot_id="r1",
            job_id="current-job",
            pod_id="current-pod",
            return_start_tick=10.0,
            selected_action_branch="store",
            metadata={},
            tick_to_second=0.15,
            return_finish_tick=18.0,
            return_duration=8.0,
            committed_next_reservation_id="cnr-1",
            committed_next_job_id="next-job",
            committed_next_pod_id="next-pod",
            committed_next_station_id="picker-1",
            estimated_cycle_time_at_decision=20.0,
            cycle_estimate_known=True,
        )
        runtime.tracker.pending_by_robot_id["r1"] = pending
        warehouse = Obj(_tick=47.5, committed_next_registry=None)
        station = Obj(station_id="picker-1", station_type="picker")
        exact_robot = _arrival_robot("r1", warehouse, "next-job", "next-pod", "picker-1")

        assert not runtime._matches_committed_next_arrival(
            robot=_arrival_robot("r2", warehouse, "next-job", "next-pod", "picker-1"),
            station=station,
            pending=pending,
        )
        assert not runtime._matches_committed_next_arrival(
            robot=_arrival_robot("r1", warehouse, "next-job", "wrong-pod", "picker-1"),
            station=station,
            pending=pending,
        )
        assert not runtime._matches_committed_next_arrival(
            robot=exact_robot,
            station=Obj(station_id="picker-2", station_type="picker"),
            pending=pending,
        )
        assert runtime._matches_committed_next_arrival(robot=exact_robot, station=station, pending=pending)
        runtime._complete_paper_cycle(robot=exact_robot, station=station, pending=pending)
        outcome = runtime.writer.events[-1]
        assert outcome["paper_cycle_duration"] == 37.5, outcome
        assert outcome["realized_cycle_time"] == 37.5, outcome
        assert captured["paper_cycle_duration"] == 37.5, captured
        assert outcome["return_duration"] == 8.0, outcome
        assert outcome["estimated_cycle_time_at_decision"] == 20.0, outcome
        assert outcome["reward_json"]["reward_value"] == 37.5, outcome

        pending_censor = PendingRTSDecision(
            decision_event_id="d-censored",
            robot_id="r3",
            job_id="job",
            pod_id="pod",
            return_start_tick=50.0,
            selected_action_branch="store",
            metadata={},
        )
        runtime.tracker.pending_by_robot_id["r3"] = pending_censor
        censor_robot = Obj(_id="r3", warehouse=Obj(_tick=55.0, committed_next_registry=None), universe=Obj(_tick=55.0), job=None, destination=None)
        runtime._censor_paper_cycle(
            robot=censor_robot,
            station=None,
            pending=pending_censor,
            status=PAPER_CYCLE_STATUS_CENSORED_MAXIMUM_HORIZON,
            reason="maximum_horizon",
        )
        censored = runtime.writer.events[-1]
        assert censored["realized_cycle_time"] is None, censored
        assert censored["paper_cycle_duration"] is None, censored
        assert censored["reward_json"]["reward_computed"] is False, censored

        decision = _full_decision("d-realized", reward_checkpoint="ckpt", reward_value=37.5)
        outcome["decision_event_id"] = "d-realized"
        dataset = build_on_policy_training_steps([decision, outcome], required_policy_checkpoint_id="ckpt")
        assert dataset.summary["trainable_step_count"] == 1, dataset.summary
        assert dataset.steps[0].reward == 37.5, dataset.steps[0]


def test_proposal_availability_and_candidate_summary() -> None:
    empty = _minimal_decision("empty")
    empty.update(
        {
            "selected_proposal_id": "cnp-empty",
            "selected_proposal_has_next_job": False,
            "selected_proposal_job_id": None,
            "selected_proposal_pod_id": None,
            "selected_proposal_picker_id": None,
            "selected_proposal_candidate_count": 0,
            "trainable": True,
        }
    )
    real = _minimal_decision("real")
    real.update(
        {
            "selected_proposal_id": "cnp-real",
            "selected_proposal_has_next_job": True,
            "selected_proposal_job_id": "job-next",
            "selected_proposal_pod_id": "pod-next",
            "selected_proposal_picker_id": "picker-1",
            "selected_proposal_candidate_count": 2,
        }
    )
    missing = _minimal_decision("missing")
    summary = summarize_rollout_events([empty, real, missing], policy_mode="rts_rl_explicit")
    assert summary["decisions_with_known_proposed_next_job"] == 1, summary
    assert summary["proposals_selected_for_commitment"] == 1, summary
    assert summary["on_policy_candidate_decision_count"] == 1, summary
    assert summary["trainable_transition_count"] is None, summary


def test_reservation_cancellation_separate_from_censoring() -> None:
    no_reservation_censor = _outcome("c1", status="censored_maximum_horizon", reason="maximum_horizon")
    activated = _outcome("a1", status="complete", reservation_id="cnr-active", activation_time=4.0)
    cancelled = _outcome(
        "x1",
        status="censored_next_task_charging",
        reason="next_task_charging",
        reservation_id="cnr-cancel",
        reservation_status="cancelled",
        cancelled=True,
        cancellation_reason="charging_override",
    )
    summary = summarize_rollout_events([no_reservation_censor, activated, cancelled], policy_mode="current")
    assert summary["censored_paper_cycle_count"] == 2, summary
    assert summary["paper_cycle_censor_counts_by_reason"]["maximum_horizon"] == 1, summary
    assert summary["paper_cycle_censor_counts_by_reason"]["next_task_charging"] == 1, summary
    assert summary["reservations_activated"] == 1, summary
    assert summary["reservations_cancelled"] == 1, summary
    assert summary["reservation_cancellation_counts_by_reason"]["charging_override"] == 1, summary


def test_same_decision_no_job_proposal_revalidation() -> None:
    registry = CommittedNextRegistry()
    storage = Obj(storage_id="s1", storage_number=1, pos_x=1.0, pos_y=2.0)
    robot = Obj(_id="r1")
    warehouse = Obj(_tick=12.5)
    proposal = CommittedNextProposal(
        proposal_id="cnp-no-job",
        owner_robot_id="r1",
        zone_id="A",
        candidate_storage=storage,
        candidate_storage_id="s1",
        destination_x=1.0,
        destination_y=2.0,
        job=None,
        job_id=None,
        pod_id=None,
        picking_station_id=None,
        original_queue_index=None,
        created_time_seconds=12.5,
        candidate_count=0,
    )
    valid, reason = registry.validate_selected_action_proposal(
        warehouse,
        robot,
        proposal,
        zone_id="A",
        storage=storage,
    )
    assert valid, reason
    warehouse._tick = 12.65
    valid, reason = registry.validate_selected_action_proposal(
        warehouse,
        robot,
        proposal,
        zone_id="A",
        storage=storage,
    )
    assert not valid and reason == "proposal_stale_time", reason


def test_explicit_proposal_failure_is_fatal() -> None:
    context = Obj(
        warehouse=Obj(
            rts_rollout_runtime=Obj(config=RTSRuntimeConfig(policy_mode="rts_rl_explicit", rollout_enabled=True, committed_next_reservations_enabled=True, policy_checkpoint_dir="x", policy_checkpoint_id="ckpt")),
            committed_next_reservations_enabled=True,
            ensure_committed_next_action_proposals=lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("boom")),
        ),
        robot=Obj(job=Obj(rts_continuation_active=True)),
        pod=Obj(),
        station=Obj(station_type="picker", station_id="picker-1"),
    )
    originals = {
        "get_static_runtime_index": state_module.get_static_runtime_index,
        "resolve_runtime_zone_ids": state_module.resolve_runtime_zone_ids,
        "get_or_build_static_state_context": state_module.get_or_build_static_state_context,
        "stock_rows_from_pod": state_module.stock_rows_from_pod,
        "build_replenishment_snapshot": state_module.build_replenishment_snapshot,
        "build_zone_rows": state_module.build_zone_rows,
        "build_zone_registry_metadata": state_module.build_zone_registry_metadata,
        "build_physical_zone_contexts": state_module.build_physical_zone_contexts,
        "build_action_contexts": state_module.build_action_contexts,
    }
    state_module.get_static_runtime_index = lambda warehouse: Obj(zone_registry=Obj(zone_ids=("A",), zones_by_id={"A": object()}))
    state_module.resolve_runtime_zone_ids = lambda warehouse, zone_ids: tuple(zone_ids)
    state_module.get_or_build_static_state_context = lambda warehouse: Obj(
        norm_x=lambda value: float(value or 0.0),
        norm_y=lambda value: float(value or 0.0),
        pod_rank=lambda pod: 0.0,
        pod_count=lambda pod: 0.0,
        distance_metadata={"distance_normalization_denominator": 1.0},
        historical_metadata={},
        layout_metadata={},
    )
    state_module.stock_rows_from_pod = lambda *args, **kwargs: []
    state_module.build_replenishment_snapshot = lambda *args, **kwargs: Obj(eligible=False, to_json_dict=lambda: {"eligible": False})
    state_module.build_zone_rows = lambda *args, **kwargs: ([{"zone_id": "A", "zone_row_index": 0, "zone_col_index": 0}], [])
    state_module.build_zone_registry_metadata = lambda *args, **kwargs: {"distance_semantics_version": "test", "distance_fallback_count": 0}
    state_module.build_physical_zone_contexts = lambda *args, **kwargs: {"A": Obj(storage=Obj(storage_id="s1"))}
    state_module.build_action_contexts = lambda *args, **kwargs: ()
    try:
        try:
            state_module.build_state(context, ("A",))
        except RuntimeError as exc:
            assert "ValueError: boom" in str(exc), exc
        else:
            raise AssertionError("explicit RTS-RL proposal failure did not fail")
    finally:
        for name, value in originals.items():
            setattr(state_module, name, value)


def test_training_metric_names_are_explicit() -> None:
    controller_text = (REPO_ROOT / "src/rmfs/rl/rts/training/controller.py").read_text()
    assert "rts/completed_paper_cycles" in controller_text
    assert "rts/average_paper_cycle_duration" in controller_text
    assert "rts/trainable_transitions" in controller_text
    assert "warehouse/orders_completed_total" not in controller_text


def _arrival_robot(robot_id: str, warehouse: Obj, job_id: str, pod_id: str, picker_id: str) -> Obj:
    pod = Obj(pod_id=pod_id)
    job = Obj(
        my_id=job_id,
        pod=pod,
        committed_next_activated_by_robot_id="r1",
        committed_next_reservation_id=None,
        station_id=picker_id,
    )
    return Obj(_id=robot_id, warehouse=warehouse, universe=warehouse, job=job, destination=None)


def _full_decision(event_id: str, *, reward_checkpoint: str, reward_value: float) -> dict:
    state = synthetic_state()
    return build_decision_event(
        decision_event_id=event_id,
        tick=10,
        robot_id="r1",
        job_id="current-job",
        pod_id="current-pod",
        source_station_id="picker-0",
        source_station_type="picker",
        policy_name="rts_rl_explicit",
        zone_ids=("A", "B"),
        action_mask=(1, 1, 0, 0),
        selected_action_index=0,
        selected_action_branch="store",
        selected_zone_id="A",
        selected_storage=None,
        state_json=state,
        feature_shapes={},
        actor_kind="rts_rl_explicit",
        policy_checkpoint_id=reward_checkpoint,
        policy_mode="greedy",
        old_log_prob=-0.1,
        old_value=0.2,
        feature_schema_id=state["feature_schema_id"] if "feature_schema_id" in state else None,
        netlogo_step=66,
        warehouse_time=10.0,
        tick_to_second=0.15,
    )


def _minimal_decision(event_id: str) -> dict:
    return build_decision_event(
        decision_event_id=event_id,
        tick=1,
        robot_id="r1",
        job_id="job",
        pod_id="pod",
        source_station_id="picker",
        source_station_type="picker",
        policy_name="current",
        zone_ids=("A",),
        action_mask=None,
        selected_action_index=0,
        selected_action_branch="store",
        selected_zone_id="A",
        selected_storage=Obj(pos_x=1, pos_y=2),
        state_json=None,
        feature_shapes=None,
        state_capture_mode="minimal",
        state_available=False,
        trainable=False,
        nontrainable_reason="minimal_state_capture",
    )


def _outcome(
    event_id: str,
    *,
    status: str,
    reason: str = "",
    reservation_id: str | None = None,
    activation_time: float | None = None,
    reservation_status: str | None = None,
    cancelled: bool = False,
    cancellation_reason: str | None = None,
) -> dict:
    row = build_outcome_event(
        decision_event_id=event_id,
        tick=5,
        robot_id="r1",
        job_id="job",
        pod_id="pod",
        outcome_status="paper_cycle_completed" if status == "complete" else "paper_cycle_censored",
        return_start_tick=1,
        return_finish_tick=2,
        realized_cycle_time=4 if status == "complete" else None,
        destination_x=1,
        destination_y=2,
        reward_json={"reward_computed": status == "complete", "reward_value": 1.0 if status == "complete" else None},
        paper_cycle_status=status,
        paper_cycle_complete=1 if status == "complete" else 0,
        paper_cycle_duration=4 if status == "complete" else None,
        paper_cycle_censor_reason=reason,
        paper_cycle_completion_rule="next_order_retrieval_arrival" if status == "complete" else status,
    )
    row.update(
        {
            "committed_next_reservation_id": reservation_id,
            "committed_next_activation_time_seconds": activation_time,
            "committed_next_reservation_status": reservation_status,
            "committed_next_cancelled": cancelled,
            "committed_next_cancellation_reason": cancellation_reason,
        }
    )
    return row


if __name__ == "__main__":
    raise SystemExit(main())
