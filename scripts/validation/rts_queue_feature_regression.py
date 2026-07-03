#!/usr/bin/env python3
"""Focused regression for RTS v5 replenishment queue action features."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.rmfs.rl.rts.features import (
    ACTION_FEATURE_SCHEMA_VERSION,
    build_action_feature_names,
    build_feature_bundle,
)
from src.rmfs.rl.rts.stock_features import STOCK_FEATURE_NAMES
from src.rmfs.rl.rts.model import RTSMaskedActorCritic
from src.rmfs.rl.rts.training.checkpoint import atomic_torch_save, write_feature_schema
from src.rmfs.rl.rts.training.policy_loader import load_policy_from_checkpoint


def _state_with_queue_estimate() -> dict:
    queue_estimate = {
        "known": True,
        "estimated_wait_seconds": 7.5,
        "active_robot_count": 2,
        "queued_robot_count": 1,
        "server_count": 2,
        "active_replenishment_work_seconds": 12.0,
    }
    return {
        "state_contract_version": "rts_rl_state.v4",
        "historical_pod_request_rank": 0.5,
        "committed_next_action_proposals": {
            "A": {
                "proposed_next_job_known": 0,
                "candidate_to_proposed_next_pod_distance": 0.0,
            }
        },
        "zone_rows": [
            {
                "zone_id": "A",
                "candidate_storage_x": 1.0,
                "candidate_storage_y": 1.0,
                "free_slot_ratio": 1.0,
                "sku_similarity_fraction": 0.5,
                "zone_present_robot_pressure": 0.0,
                "zone_destination_robot_pressure": 0.0,
                "macro_region_present_robot_pressure": 0.0,
                "macro_region_destination_robot_pressure": 0.0,
                "selected_replenishment_station_destination_pressure": 0.25,
            }
        ],
        "stock_rows": [
            {
                "sku_id": "sku-1",
                "local_fill_ratio": 0.2,
                "local_below_threshold": 1.0,
                "global_fill_ratio": 0.3,
                "global_below_threshold": 1.0,
            }
        ],
        "replenishment_snapshot": {"eligible_skus": ["sku-1"]},
        "spatial_context": {
            "source_picker_x_norm": 0.1,
            "source_picker_y_norm": 0.2,
            "distance_normalization_denominator": 100.0,
        },
        "layout_normalization": {
            "x_min": 0.0,
            "x_max": 10.0,
            "y_min": 0.0,
            "y_max": 10.0,
        },
        "rts_action_contexts": [
            {
                "action_index": 0,
                "branch": "store",
                "zone_id": "A",
                "cycle_estimate": {
                    "known": True,
                    "estimated_cycle_seconds": 20.0,
                    "queue_estimate": queue_estimate,
                },
            },
            {
                "action_index": 1,
                "branch": "replenish_store",
                "zone_id": "A",
                "selected_replenishment_station_destination_pressure": 0.25,
                "cycle_estimate": {
                    "known": False,
                    "estimated_cycle_seconds": None,
                    "queue_estimate": queue_estimate,
                },
            },
        ],
    }


def test_queue_features_are_appended_and_branch_scoped() -> None:
    zones = ("A",)
    names = build_action_feature_names(zones)
    assert ACTION_FEATURE_SCHEMA_VERSION == "rts_action_features.v5"
    assert names[-3:] == (
        "replenishment_queue_estimate_known",
        "estimated_replenishment_queue_seconds",
        "replenishment_station_load_pressure",
    )
    bundle = build_feature_bundle(zones, [1, 1], _state_with_queue_estimate())
    assert bundle.X_actions.shape == (2, 21)
    queue_known_idx = names.index("replenishment_queue_estimate_known")
    queue_seconds_idx = names.index("estimated_replenishment_queue_seconds")
    pressure_idx = names.index("replenishment_station_load_pressure")
    cycle_known_idx = names.index("cycle_estimate_known")
    cycle_seconds_idx = names.index("estimated_cycle_time")

    # STORE rows always zero the three new replenishment queue features.
    assert bundle.X_actions[0, queue_known_idx] == 0.0
    assert bundle.X_actions[0, queue_seconds_idx] == 0.0
    assert bundle.X_actions[0, pressure_idx] == 0.0

    # Queue features come from the nested queue estimate even when the overall
    # cycle estimate is unknown because no committed-next job is available.
    assert bundle.X_actions[1, cycle_known_idx] == 0.0
    assert bundle.X_actions[1, cycle_seconds_idx] == 0.0
    assert bundle.X_actions[1, queue_known_idx] == 1.0
    assert bundle.X_actions[1, queue_seconds_idx] == 7.5
    assert bundle.X_actions[1, pressure_idx] == 1.0
    print("PASS test_queue_features_are_appended_and_branch_scoped")


def test_legacy_v4_checkpoint_fails_clearly() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        checkpoint = Path(tmp)
        model = RTSMaskedActorCritic(action_feature_dim=18, stock_feature_dim=4)
        atomic_torch_save(model.state_dict(), checkpoint / "model.pt")
        (checkpoint / "metadata.json").write_text(json.dumps({"training_config": {}}), encoding="utf-8")
        (checkpoint / "feature_schema.json").write_text(
            json.dumps(
                {
                    "action_feature_schema_version": "rts_action_features.v4",
                    "action_feature_dim": 18,
                    "stock_feature_dim": 4,
                    "action_feature_names": [f"legacy_{idx}" for idx in range(18)],
                    "stock_feature_names": [
                        "local_fill_ratio",
                        "local_below_threshold",
                        "global_fill_ratio",
                        "global_below_threshold",
                    ],
                }
            ),
            encoding="utf-8",
        )
        try:
            load_policy_from_checkpoint(checkpoint, device="cpu")
        except ValueError as exc:
            message = str(exc)
            assert "rts_action_features.v5" in message
            assert "width 21" in message
            assert "fresh training" in message
        else:
            raise AssertionError("legacy v4 width-18 checkpoint was accepted")
    print("PASS test_legacy_v4_checkpoint_fails_clearly")


def test_v5_checkpoint_reloads_successfully() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        checkpoint = Path(tmp)
        action_names = list(build_action_feature_names(("A",)))
        model = RTSMaskedActorCritic(action_feature_dim=len(action_names), stock_feature_dim=4)
        atomic_torch_save(model.state_dict(), checkpoint / "model.pt")
        (checkpoint / "metadata.json").write_text(
            json.dumps({"training_config": {}, "policy_checkpoint_id": "v5_smoke"}),
            encoding="utf-8",
        )
        write_feature_schema(
            checkpoint / "feature_schema.json",
            action_feature_names=tuple(action_names),
            stock_feature_names=tuple(STOCK_FEATURE_NAMES),
        )
        loaded = load_policy_from_checkpoint(checkpoint, device="cpu")
        assert loaded.feature_schema["action_feature_schema_version"] == ACTION_FEATURE_SCHEMA_VERSION
        assert loaded.feature_schema["action_feature_dim"] == 21
    print("PASS test_v5_checkpoint_reloads_successfully")


def main() -> None:
    test_queue_features_are_appended_and_branch_scoped()
    test_legacy_v4_checkpoint_fails_clearly()
    test_v5_checkpoint_reloads_successfully()
    print("\nALL RTS QUEUE FEATURE REGRESSION TESTS PASSED")


if __name__ == "__main__":
    main()
