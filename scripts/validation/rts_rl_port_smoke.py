#!/usr/bin/env python3
"""Pure validation smoke for the Phase 6 RTS-RL port."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.rmfs.decisions.rts import CurrentRTSPolicy
from src.rmfs.decisions.rts.rl_policy import RTSRLPolicy
from src.rmfs.rl.rts.action_space import (
    REPLENISH_STORE,
    STORE,
    action_mask_entry,
    action_space_size,
    build_action_mask,
    decode_action,
    encode_action,
    validate_action_mask,
)
from src.rmfs.rl.rts.cycle_reference import RTSCycleReference, read_cycle_reference, write_cycle_reference
from src.rmfs.rl.rts.features import build_feature_bundle
from src.rmfs.rl.rts.inference import masked_softmax, run_inference, select_greedy_action
from src.rmfs.rl.rts.model import build_rts_actor_critic_model
from src.rmfs.rl.rts.reward import (
    RTSRewardReference,
    build_reward_components_from_estimated_cycle,
    build_reward_components_from_realized_cycle,
    compute_reward,
)
from src.rmfs.rl.rts.state import build_state
from src.rmfs.rl.rts.validation import (
    validate_feature_matrix_shape,
    validate_model_output_shape,
    validate_no_raw_threshold_features,
    validate_reward_result,
    validate_stock_matrix_shape,
)


class Obj:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def build_synthetic_context():
    pod = Obj(
        pod_id=7,
        skus={
            1: {"current_qty": 2, "limit_qty": 10, "threshold": 0.3, "weight": 1.0},
            2: {"current_qty": 0, "limit_qty": 8, "threshold": 0.2, "weight": 1.0},
        },
    )
    station = Obj(station_id="picker-1", station_type="picker", pos_x=2, pos_y=3)
    robot = Obj(_id=1, object_type="robot", pos_x=5, pos_y=6)
    storages = [
        Obj(pos_x=10, pos_y=1, is_empty=True, assigned_pod=None, zone_id="A"),
        Obj(pos_x=20, pos_y=1, is_empty=False, assigned_pod=pod, zone_id="B"),
        Obj(pos_x=21, pos_y=1, is_empty=True, assigned_pod=None, zone_id="B"),
    ]
    warehouse = Obj(
        storage_manager=Obj(storages=storages),
        station_manager=Obj(stations=[station, Obj(station_id="replenishment-1", station_type="replenishment")]),
        pod_manager=Obj(pods=[pod]),
        _objects=[robot],
    )
    return Obj(warehouse=warehouse, robot=robot, pod=pod, station=station)


def assert_raises(fn, expected):
    try:
        fn()
    except expected:
        return
    raise AssertionError(f"expected {expected.__name__}")


def main():
    zone_ids = ("A", "B")
    assert action_space_size(zone_ids) == 4
    for branch in (STORE, REPLENISH_STORE):
        for zone in zone_ids:
            encoded = encode_action(branch, zone, zone_ids)
            decoded = decode_action(encoded, zone_ids)
            assert decoded.branch == branch and decoded.zone_id == zone
    mask = build_action_mask(zone_ids, store_valid_by_zone={"A": True, "B": False}, replenish_valid_by_zone={"A": False, "B": True})
    assert len(mask) == 4
    assert action_mask_entry(0, zone_ids, mask) == 1
    assert_raises(lambda: encode_action(STORE, "Z", zone_ids), ValueError)
    assert_raises(lambda: validate_action_mask(zone_ids, [0, 0, 0, 0]), ValueError)

    context = build_synthetic_context()
    state = build_state(context, zone_ids)
    json.dumps(state.state_json)
    features = build_feature_bundle(zone_ids, mask, state.state_json)
    validate_feature_matrix_shape(features.X_actions, features.action_feature_names)
    validate_stock_matrix_shape(features.X_stock, features.stock_feature_names)
    validate_no_raw_threshold_features(features.action_feature_names)
    validate_no_raw_threshold_features(features.stock_feature_names)

    reference = RTSRewardReference(
        reference_overall_cycle_time=10.0,
        reference_avg_storage_cycle_time=8.0,
        reference_avg_replenish_cycle_time=12.0,
        alpha=0.5,
    )
    reward = compute_reward(
        build_reward_components_from_realized_cycle(selected_action_branch=STORE, realized_cycle_time=7.0),
        reference,
    )
    validate_reward_result(reward)
    missing = compute_reward(
        build_reward_components_from_estimated_cycle(selected_action_branch=STORE, estimated_cycle_time=7.0),
        None,
    )
    assert not missing.reward_computed

    cycle_reference = RTSCycleReference(10.0, 8.0, 12.0, store_action_count=1, replenish_action_count=1, alpha=0.5)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "cycle_reference.json"
        write_cycle_reference(path, cycle_reference)
        assert read_cycle_reference(path) == cycle_reference

    model = build_rts_actor_critic_model(
        action_feature_dim=features.X_actions.shape[1],
        stock_feature_dim=features.X_stock.shape[1],
    )
    result = run_inference(
        model,
        action_features=features.X_actions,
        action_mask=features.M_actions,
        stock_features=features.X_stock,
        stock_mask=features.M_stock,
        zone_ids=features.zone_ids,
        device="cpu",
    )
    assert mask[result.selected_action_index] == 1
    with torch.no_grad():
        logits, values = model(
            torch.as_tensor(features.X_actions, dtype=torch.float32).unsqueeze(0),
            torch.as_tensor(features.M_actions, dtype=torch.int64).unsqueeze(0),
            torch.as_tensor(features.X_stock, dtype=torch.float32).unsqueeze(0),
            torch.as_tensor(features.M_stock, dtype=torch.int64).unsqueeze(0),
        )
    validate_model_output_shape(logits, values, 1, len(mask))
    assert_raises(lambda: masked_softmax(torch.zeros(4), torch.zeros(4, dtype=torch.int64)), ValueError)
    assert_raises(lambda: select_greedy_action(torch.zeros(4), torch.zeros(4, dtype=torch.int64)), ValueError)

    if torch.cuda.is_available():
        cuda_model = build_rts_actor_critic_model(
            action_feature_dim=features.X_actions.shape[1],
            stock_feature_dim=features.X_stock.shape[1],
        )
        cuda_result = run_inference(
            cuda_model,
            action_features=features.X_actions,
            action_mask=features.M_actions,
            stock_features=features.X_stock,
            stock_mask=features.M_stock,
            zone_ids=features.zone_ids,
            device="cuda",
        )
        assert mask[cuda_result.selected_action_index] == 1

    assert RTSRLPolicy is not None
    assert CurrentRTSPolicy is not None
    print("rts rl port smoke ok")


if __name__ == "__main__":
    main()
