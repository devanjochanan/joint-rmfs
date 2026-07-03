#!/usr/bin/env python3
"""Regression: RTS v6 removed the four replenishment-queue/cycle-known action
features from the policy tensor (retaining the queue estimator internally)."""

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


REMOVED_FEATURES = (
    "cycle_estimate_known",
    "replenishment_queue_estimate_known",
    "estimated_replenishment_queue_seconds",
    "replenishment_station_load_pressure",
)


def test_queue_features_removed_from_tensor() -> None:
    zones = ("A",)
    names = build_action_feature_names(zones)
    assert ACTION_FEATURE_SCHEMA_VERSION == "rts_action_features.v6"
    assert len(names) == 17
    for removed in REMOVED_FEATURES:
        assert removed not in names, f"{removed} must be absent from v6 tensor"
    # The renamed raw distance is present; the old normalized name is gone.
    assert "candidate_to_proposed_next_pod_distance" in names
    assert "candidate_to_proposed_next_pod_distance_norm" not in names
    print("PASS test_queue_features_removed_from_tensor")


def test_v5_width21_checkpoint_fails_clearly() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        checkpoint = Path(tmp)
        model = RTSMaskedActorCritic(action_feature_dim=21, stock_feature_dim=4)
        atomic_torch_save(model.state_dict(), checkpoint / "model.pt")
        (checkpoint / "metadata.json").write_text(json.dumps({"training_config": {}}), encoding="utf-8")
        (checkpoint / "feature_schema.json").write_text(
            json.dumps(
                {
                    "action_feature_schema_version": "rts_action_features.v5",
                    "action_feature_dim": 21,
                    "stock_feature_dim": 4,
                    "action_feature_names": [f"legacy_{idx}" for idx in range(21)],
                    "stock_feature_names": list(STOCK_FEATURE_NAMES),
                }
            ),
            encoding="utf-8",
        )
        try:
            load_policy_from_checkpoint(checkpoint, device="cpu")
        except ValueError as exc:
            message = str(exc)
            assert "rts_action_features.v6" in message, message
            assert "width 17" in message, message
            assert "fresh training" in message, message
        else:
            raise AssertionError("legacy v5 width-21 checkpoint was accepted")
    print("PASS test_v5_width21_checkpoint_fails_clearly")


def test_v6_checkpoint_reloads_successfully() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        checkpoint = Path(tmp)
        action_names = list(build_action_feature_names(("A",)))
        model = RTSMaskedActorCritic(action_feature_dim=len(action_names), stock_feature_dim=4)
        atomic_torch_save(model.state_dict(), checkpoint / "model.pt")
        (checkpoint / "metadata.json").write_text(
            json.dumps({"training_config": {}, "policy_checkpoint_id": "v6_smoke"}),
            encoding="utf-8",
        )
        write_feature_schema(
            checkpoint / "feature_schema.json",
            action_feature_names=tuple(action_names),
            stock_feature_names=tuple(STOCK_FEATURE_NAMES),
        )
        loaded = load_policy_from_checkpoint(checkpoint, device="cpu")
        assert loaded.feature_schema["action_feature_schema_version"] == ACTION_FEATURE_SCHEMA_VERSION
        assert loaded.feature_schema["action_feature_dim"] == 17
    print("PASS test_v6_checkpoint_reloads_successfully")


def main() -> None:
    test_queue_features_removed_from_tensor()
    test_v5_width21_checkpoint_fails_clearly()
    test_v6_checkpoint_reloads_successfully()
    print("\nALL RTS QUEUE FEATURE REMOVAL REGRESSION TESTS PASSED")


if __name__ == "__main__":
    main()
