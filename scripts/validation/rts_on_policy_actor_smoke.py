#!/usr/bin/env python3
"""Pure smoke for the explicit RTS on-policy actor."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.rmfs.rl.rts.action_space import action_mask_entry, build_action_mask
from src.rmfs.rl.rts.features import build_feature_bundle
from src.rmfs.rl.rts.model import RTSMaskedActorCritic
from src.rmfs.rl.rts.state import build_state
from src.rmfs.rl.rts.training.policy_actor import RTSOnPolicyActor, RTSOnPolicyActorConfig


class Obj:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def build_context():
    pod = Obj(
        pod_id=1,
        skus={"sku-1": {"current_qty": 2, "limit_qty": 10, "threshold": 0.3}},
    )
    station = Obj(station_id="picker-1", station_type="picker", pos_x=1, pos_y=1)
    robot = Obj(_id=1, object_type="robot", pos_x=2, pos_y=2)
    storages = [
        Obj(pos_x=10, pos_y=1, is_empty=True, assigned_pod=None, zone_id="A"),
        Obj(pos_x=20, pos_y=1, is_empty=True, assigned_pod=None, zone_id="B"),
    ]
    warehouse = Obj(
        _tick=3,
        tick_to_second=0.5,
        storage_manager=Obj(storages=storages),
        station_manager=Obj(stations=[station, Obj(station_id="repl-1", station_type="replenishment")]),
        pod_manager=Obj(pods=[pod]),
        _objects=[robot],
    )
    return Obj(warehouse=warehouse, robot=robot, pod=pod, station=station)


def main():
    torch.manual_seed(7)
    context = build_context()
    zones = ("A", "B")
    state = build_state(context, zones)
    store_valid = {row["zone_id"]: bool(row["store_action_valid"]) for row in state.state_json["zone_rows"]}
    repl_valid = {row["zone_id"]: bool(row["replenish_store_action_valid"]) for row in state.state_json["zone_rows"]}
    mask = build_action_mask(zones, store_valid_by_zone=store_valid, replenish_valid_by_zone=repl_valid)
    features = build_feature_bundle(zones, mask, state.state_json)
    model = RTSMaskedActorCritic(
        action_feature_dim=features.X_actions.shape[-1],
        stock_feature_dim=features.X_stock.shape[-1],
    )
    actor = RTSOnPolicyActor(
        model=model,
        zone_ids=zones,
        config=RTSOnPolicyActorConfig(
            policy_checkpoint_id="batch_000001",
            policy_action_mode="sample",
            policy_device="cpu",
            feature_schema_id="schema-1",
        ),
    )
    decision = actor.select_destination(context)
    metadata = dict(decision.metadata)
    assert metadata["actor_kind"] == "rts_rl_explicit"
    assert metadata["policy_mode"] == "sample"
    assert decision.mode == "rl"
    assert np.isfinite(float(metadata["old_log_prob"]))
    assert np.isfinite(float(metadata["old_value"]))
    assert metadata["policy_checkpoint_id"] == "batch_000001"
    assert action_mask_entry(metadata["selected_action_index"], zones, mask) == 1
    assert metadata["selected_action_branch"]
    assert metadata["selected_zone_id"]
    print("rts on policy actor smoke ok")


if __name__ == "__main__":
    main()
