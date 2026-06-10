#!/usr/bin/env python3
"""Pure synthetic smoke for the RTS PPO update core."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.rmfs.rl.rts.model import RTSMaskedActorCritic
from src.rmfs.rl.rts.rollout_schema import build_decision_event, build_outcome_event
from src.rmfs.rl.rts.training.config import RTSTrainingConfig
from src.rmfs.rl.rts.training.ppo import (
    build_offline_ppo_batch,
    compute_log_probs_values,
    masked_categorical_from_logits,
    run_ppo_update,
)
from src.rmfs.rl.rts.training.rollout_dataset import build_feature_tensors_from_steps, build_training_steps


def synthetic_state(offset: float = 0.0) -> dict:
    zones = ("A", "B")
    return {
        "turnover_rank": 0.1 + offset,
        "turnover_value": 0.2 + offset,
        "estimated_queue_time": 0.0,
        "next_retrieval_zone_known": 1,
        "next_retrieval_zone_one_hot": {"A": 1, "B": 0},
        "zone_rows": [
            {
                "zone_id": "A",
                "zone_row_index": 0.0,
                "zone_col_index": 0.0,
                "occupation_level": 0.2,
                "free_slot_count": 2.0,
                "zone_destination_robot_count": 0.0,
                "neighbor_zone_destination_robot_count": 0.0,
                "superzone_destination_robot_count": 0.0,
                "zone_present_robot_count": 1.0,
                "neighbor_zone_present_robot_count": 0.0,
                "superzone_present_robot_count": 0.0,
                "storage_cycle_time_estimate": 4.0,
                "replenish_cycle_time_estimate": 6.0,
                "sku_similarity": 0.5,
                "candidate_zone_to_selected_replenishment_station_distance": 0.2,
                "candidate_zone_to_nearest_replenishment_station_distance": 0.2,
                "store_action_valid": 1.0,
                "replenish_store_action_valid": 0.0,
            },
            {
                "zone_id": "B",
                "zone_row_index": 1.0,
                "zone_col_index": 1.0,
                "occupation_level": 0.4,
                "free_slot_count": 1.0,
                "zone_destination_robot_count": 0.0,
                "neighbor_zone_destination_robot_count": 0.0,
                "superzone_destination_robot_count": 0.0,
                "zone_present_robot_count": 0.0,
                "neighbor_zone_present_robot_count": 0.0,
                "superzone_present_robot_count": 1.0,
                "storage_cycle_time_estimate": 5.0,
                "replenish_cycle_time_estimate": 7.0,
                "sku_similarity": 0.3,
                "candidate_zone_to_selected_replenishment_station_distance": 0.3,
                "candidate_zone_to_nearest_replenishment_station_distance": 0.3,
                "store_action_valid": 1.0,
                "replenish_store_action_valid": 0.0,
            },
        ],
        "stock_rows": [
            {
                "sku_id": "sku-1",
                "current_qty": 2.0,
                "limit_qty": 10.0,
                "fill_ratio": 0.2,
                "pod_below_threshold": 1.0,
                "below_threshold": 1.0,
                "is_zero_qty": 0.0,
                "is_zero_and_global_low": 0.0,
                "shortage_depth": 0.1,
                "global_low_depth": 0.0,
            }
        ],
        "spatial_context": {
            "source_station_is_picking": 1.0,
            "source_station_is_replenishment": 0.0,
            "source_station_x_norm": 0.1,
            "source_station_y_norm": 0.1,
            "picking_station_count": 1.0,
            "replenishment_station_count": 1.0,
            "selected_replenishment_station_x_norm": 0.0,
            "selected_replenishment_station_y_norm": 0.0,
            "selected_replenishment_station_logical_load": 0.0,
            "total_robot_count": 2.0,
            "active_pod_total": 2.0,
            "arrival_rate_order_cycle_time": 0.0,
            "zone_row_min": 0.0,
            "zone_row_max": 1.0,
            "zone_col_min": 0.0,
            "zone_col_max": 1.0,
        },
    }


def synthetic_events():
    events = []
    for index, selected in enumerate((0, 1), start=1):
        event_id = f"d{index}"
        events.append(
            build_decision_event(
                decision_event_id=event_id,
                tick=index,
                robot_id=index,
                job_id=f"job-{index}",
                pod_id=f"pod-{index}",
                source_station_id="picker-1",
                source_station_type="picker",
                policy_name="synthetic",
                zone_ids=("A", "B"),
                action_mask=(1, 1, 0, 0),
                selected_action_index=selected,
                selected_action_branch="store",
                selected_zone_id="A" if selected == 0 else "B",
                selected_storage=None,
                state_json=synthetic_state(float(index) * 0.01),
                feature_shapes={},
            )
        )
        events.append(
            build_outcome_event(
                decision_event_id=event_id,
                tick=index + 1,
                robot_id=index,
                job_id=f"job-{index}",
                pod_id=f"pod-{index}",
                outcome_status="completed",
                return_start_tick=index,
                return_finish_tick=index + 1,
                realized_cycle_time=1.0 + index,
                destination_x=10,
                destination_y=1,
                reward_json={"reward_computed": True, "reward_value": float(index)},
            )
        )
    return events


def assert_raises(fn, expected):
    try:
        fn()
    except expected:
        return
    raise AssertionError(f"expected {expected.__name__}")


def main():
    torch.manual_seed(42)
    dataset = build_training_steps(synthetic_events())
    assert dataset.summary["eligible_step_count"] == 2
    padded = build_feature_tensors_from_steps(dataset.steps)
    model = RTSMaskedActorCritic(
        action_feature_dim=padded.X_actions.shape[-1],
        stock_feature_dim=padded.X_stock.shape[-1],
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    config = RTSTrainingConfig(
        artifact_label="ppo_update_smoke",
        output_root=Path("data/runtime/rts_training_smoke/ppo_update_smoke"),
        learning_rate=1e-3,
        ppo_epochs=2,
        minibatch_size=1,
        tensorboard_enabled=False,
    )
    before = [param.detach().clone() for param in model.parameters()]
    batch = build_offline_ppo_batch(model, padded, "cpu", config.gamma, config.gae_lambda)
    result = run_ppo_update(model, optimizer, batch, config, "cpu")
    assert result.optimizer_steps > 0
    for value in (
        result.total_loss_mean,
        result.policy_loss_mean,
        result.value_loss_mean,
        result.entropy_mean,
    ):
        assert np.isfinite(value)
    changed = any(not torch.allclose(prev, param.detach()) for prev, param in zip(before, model.parameters()))
    assert changed
    logits = torch.zeros((1, 4), dtype=torch.float32)
    assert_raises(lambda: masked_categorical_from_logits(logits, torch.zeros((1, 4), dtype=torch.int64)), ValueError)
    invalid = padded.__class__(
        X_actions=padded.X_actions,
        M_actions=padded.M_actions,
        X_stock=padded.X_stock,
        M_stock=padded.M_stock,
        selected_action_indices=np.asarray([3, 1], dtype=np.int64),
        rewards=padded.rewards,
        terminated=padded.terminated,
        truncated=padded.truncated,
        action_feature_names=padded.action_feature_names,
        stock_feature_names=padded.stock_feature_names,
        decision_event_ids=padded.decision_event_ids,
    )
    assert_raises(lambda: compute_log_probs_values(model, invalid, "cpu"), ValueError)
    if torch.cuda.is_available():
        with torch.no_grad():
            cuda_model = RTSMaskedActorCritic(
                action_feature_dim=padded.X_actions.shape[-1],
                stock_feature_dim=padded.X_stock.shape[-1],
            ).to("cuda")
            compute_log_probs_values(cuda_model, padded, "cuda")
    print("rts ppo update smoke ok")


if __name__ == "__main__":
    main()

