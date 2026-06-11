#!/usr/bin/env python3
"""Pure smoke for on-policy RTS dataset filtering."""

from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.validation.rts_ppo_update_smoke import synthetic_state
from src.rmfs.rl.rts.rollout_schema import build_decision_event, build_outcome_event
from src.rmfs.rl.rts.training.on_policy_dataset import build_on_policy_ppo_batch, build_on_policy_training_steps


def decision(event_id: str, actor_kind: str, checkpoint: str = "batch_000001", **overrides):
    payload = build_decision_event(
        decision_event_id=event_id,
        tick=1,
        robot_id=1,
        job_id="job",
        pod_id="pod",
        source_station_id="picker",
        source_station_type="picker",
        policy_name=actor_kind,
        zone_ids=("A", "B"),
        action_mask=(1, 1, 0, 0),
        selected_action_index=0,
        selected_action_branch="store",
        selected_zone_id="A",
        selected_storage=None,
        state_json=synthetic_state(),
        feature_shapes={},
        actor_kind=actor_kind,
        policy_checkpoint_id=checkpoint,
        policy_mode=actor_kind,
        old_log_prob=-0.7,
        old_value=1.0,
        netlogo_step=1,
        warehouse_time=0.5,
        tick_to_second=0.5,
    )
    payload.update(overrides)
    return payload


def outcome(event_id: str):
    return build_outcome_event(
        decision_event_id=event_id,
        tick=2,
        robot_id=1,
        job_id="job",
        pod_id="pod",
        outcome_status="completed",
        return_start_tick=1,
        return_finish_tick=2,
        realized_cycle_time=1,
        destination_x=10,
        destination_y=1,
        reward_json={"reward_computed": True, "reward_value": 1.0},
    )


def main():
    events = [decision("good", "rts_rl_explicit"), outcome("good")]
    for actor in ("current_probe", "random_valid", "current", "heuristic", "synthetic"):
        events.extend([decision(actor, actor), outcome(actor)])
    events.extend([decision("mismatch", "rts_rl_explicit", checkpoint="old"), outcome("mismatch")])
    events.extend([decision("missing_logprob", "rts_rl_explicit", old_log_prob=None), outcome("missing_logprob")])
    events.extend([decision("missing_value", "rts_rl_explicit", old_value=None), outcome("missing_value")])
    dataset = build_on_policy_training_steps(events, required_policy_checkpoint_id="batch_000001")
    assert dataset.summary["trainable_step_count"] == 1
    assert dataset.summary["rejected_non_on_policy_count"] == 5
    assert dataset.summary["rejected_checkpoint_mismatch_count"] == 1
    assert dataset.summary["rejected_missing_old_log_prob_count"] == 1
    assert dataset.summary["rejected_missing_old_value_count"] == 1
    batch = build_on_policy_ppo_batch(dataset, gamma=0.99, gae_lambda=0.95)
    assert batch.old_log_probs.shape[0] == 1
    assert batch.old_values.shape[0] == 1
    print("rts on policy dataset smoke ok")


if __name__ == "__main__":
    main()

