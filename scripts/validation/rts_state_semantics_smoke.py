#!/usr/bin/env python3
"""Phase 3 RTS-RL state semantics smoke."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np
import pandas as pd
import torch

os.environ.setdefault("RMFS_FAST_TRAIN", "1")
os.environ.setdefault("RMFS_DETAIL_DB", "0")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/rmfs-mpl")

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.rmfs.decisions.rts.types import RTSDestinationContext
from src.rmfs.decisions.task_allocation.committed_next import (
    get_committed_next_action_proposals,
    get_committed_next_reservation,
)
from src.rmfs.rl.rts.action_space import STORE, build_action_mask
from src.rmfs.rl.rts.features import (
    ACTION_FEATURE_SCHEMA_VERSION,
    build_action_feature_names,
    build_feature_bundle,
    build_stock_feature_names,
)
from src.rmfs.rl.rts.model import RTSMaskedActorCritic
from src.rmfs.rl.rts.state import STATE_CONTRACT_VERSION, build_state
from src.rmfs.rl.rts.static_state_context import (
    HISTORICAL_POD_RANK_VERSION,
    LAYOUT_NORMALIZATION_VERSION,
    build_static_state_context,
)
from src.rmfs.rl.rts.stock_features import STOCK_FEATURE_SCHEMA_VERSION, STOCK_SOURCE_VERSION, stock_rows_from_pod, stock_summary
from src.rmfs.rl.rts.training.checkpoint import atomic_torch_save, write_feature_schema
from src.rmfs.rl.rts.training.policy_loader import load_policy_from_checkpoint
from src.rmfs.rl.rts.validation import validate_no_removed_placeholder_features

from scripts.validation.rts_committed_next_smoke import build_inventory, start_return


class Obj:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def make_pod(pod_id, sku_data):
    pod = Obj(pod_id=pod_id, skus={})
    for sku, values in sku_data.items():
        pod.skus[sku] = dict(values)
    return pod


def storage(number, x, y, zone_id, pod=None, empty=True):
    return Obj(storage_number=number, pos_x=x, pos_y=y, zone_id=zone_id, rts_zone_id=zone_id, assigned_pod=pod, is_empty=empty)


def station(station_id, station_type, x, y):
    return Obj(station_id=station_id, station_type=station_type, pos_x=x, pos_y=y)


def write_orders(path: Path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def base_context(order_rows: list[dict]):
    tmp = tempfile.TemporaryDirectory()
    root = Path(tmp.name)
    order_path = root / "generated_order.csv"
    write_orders(order_path, order_rows)
    pod1 = make_pod(
        1,
        {
            101: {"current_qty": 0, "limit_qty": 10, "threshold": 0.3, "weight": 1.0},
            102: {"current_qty": 5, "limit_qty": 30, "threshold": 0.2, "weight": 1.0},
        },
    )
    pod2 = make_pod(2, {102: {"current_qty": 7, "limit_qty": 10, "threshold": 0.2, "weight": 1.0}})
    pod3 = make_pod(3, {103: {"current_qty": 4, "limit_qty": 10, "threshold": 0.2, "weight": 1.0}})
    picker = station("picker", "picker", 10, 100)
    repl = station("repl", "replenishment", 50, 100)
    storages = [
        storage(1, 10, 100, "A", pod=pod2, empty=False),
        storage(2, 30, 100, "A", pod=Obj(pod_id=4, skus={101: {"current_qty": 0, "limit_qty": 10}}), empty=False),
        storage(3, 90, 100, "B", pod=pod3, empty=False),
        storage(4, 110, 100, "B", pod=None, empty=True),
    ]
    robot_a = Obj(_id=1, object_type="robot", pos_x=10, pos_y=100, destination=Obj(x=90, y=100))
    robot_b = Obj(_id=2, object_type="robot", pos_x=30, pos_y=100, destination=Obj(x=10, y=100))
    warehouse = Obj(
        _tick=0,
        pod_replenishment_threshold=0.5,
        generated_order_csv=str(order_path),
        storage_manager=Obj(storages=storages),
        station_manager=Obj(stations=[picker, repl]),
        pod_manager=Obj(
            pods=[pod1, pod2, pod3],
            skus_data={
                101: {"current_global_qty": 10, "max_global_qty": 100, "global_inv_level": 0.1, "global_threshold_inv_level": 0.2},
                102: {"current_global_qty": 80, "max_global_qty": 100, "global_inv_level": 0.8, "global_threshold_inv_level": 0.2},
                103: {"current_global_qty": 10, "max_global_qty": 100, "global_inv_level": 0.1, "global_threshold_inv_level": 0.2},
            },
        ),
        order_manager=Obj(orders=[]),
        _objects=[robot_a, robot_b],
        global_critical_skus=set(),
        pending_replenishment_dispatches=[],
        replenishment_trips=0,
        replenishment_count=0,
    )
    context = Obj(warehouse=warehouse, robot=robot_a, pod=pod1, station=picker)
    context._tmp = tmp
    return context


def test_historical_rank_and_layout():
    rows = [
        {"source_order_id": "o1", "order_id": 1, "item_id": 101, "item_quantity": 1},
        {"source_order_id": "o1", "order_id": 1, "item_id": 102, "item_quantity": 99},
        {"source_order_id": "o2", "order_id": 2, "item_id": 103, "item_quantity": 1},
        {"source_order_id": "o3", "order_id": 3, "item_id": 102, "item_quantity": 1},
    ]
    context = base_context(rows)
    static = build_static_state_context(context.warehouse)
    assert static.pod_request_count["1"] == 2
    assert static.pod_request_count["2"] == 2
    assert static.pod_request_count["3"] == 1
    assert static.pod_request_rank["1"] == 1.0
    assert static.pod_request_rank["2"] == 0.5
    assert static.historical_metadata["historical_pod_rank_version"] == HISTORICAL_POD_RANK_VERSION
    assert static.historical_metadata["valid_unique_source_order_count"] == 3
    reordered = base_context(list(reversed(rows)))
    assert build_static_state_context(reordered.warehouse).pod_request_rank == static.pod_request_rank
    state = build_state(context, ("A", "B")).state_json
    assert state["state_contract_version"] == STATE_CONTRACT_VERSION
    assert state["layout_normalization"]["layout_normalization_version"] == LAYOUT_NORMALIZATION_VERSION
    assert state["spatial_context"]["source_station_x_norm"] == 0.0
    assert state["spatial_context"]["source_station_y_norm"] == 0.0
    assert state["layout_normalization"]["x_max"] == 110.0


def test_stock_and_replenishment_semantics():
    context = base_context([
        {"source_order_id": "o1", "order_id": 1, "item_id": 101, "item_quantity": 1},
    ])
    rows = stock_rows_from_pod(context.pod, context.warehouse)
    by_sku = {row["sku_id"]: row for row in rows}
    assert by_sku["101"]["local_fill_ratio"] == 0.0
    assert by_sku["101"]["global_fill_ratio"] == 0.1
    assert by_sku["101"]["local_zero_and_global_low"] == 1.0
    assert by_sku["102"]["local_zero_and_global_low"] == 0.0
    summary = stock_summary(rows)
    assert np.isclose(summary["pod_fill_ratio"], 5 / 40)
    assert not np.isclose(summary["pod_fill_ratio"], (0.0 + (5 / 30)) / 2)
    before = (
        set(context.warehouse.global_critical_skus),
        list(context.warehouse.pending_replenishment_dispatches),
        context.warehouse.replenishment_trips,
        context.warehouse.replenishment_count,
    )
    state = build_state(context, ("A", "B")).state_json
    snapshot = state["replenishment_snapshot"]
    assert snapshot["eligible"] is True
    assert snapshot["eligible_skus"] == [101]
    after = (
        set(context.warehouse.global_critical_skus),
        list(context.warehouse.pending_replenishment_dispatches),
        context.warehouse.replenishment_trips,
        context.warehouse.replenishment_count,
    )
    assert before == after
    missing = base_context([])
    del missing.warehouse.pod_manager.skus_data[101]
    try:
        stock_rows_from_pod(missing.pod, missing.warehouse)
    except ValueError as exc:
        assert "missing RTS-RL global SKU data" in str(exc)
    else:
        raise AssertionError("missing global SKU data did not fail")


def test_similarity_pressure_features_and_placeholders():
    context = base_context([
        {"source_order_id": "o1", "order_id": 1, "item_id": 101, "item_quantity": 1},
    ])
    state = build_state(context, ("A", "B")).state_json
    zone_a, zone_b = state["zone_rows"]
    assert zone_a["sku_similarity_count"] == 1.0
    assert np.isclose(zone_a["sku_similarity_fraction"], 1.0)
    assert zone_b["sku_similarity_fraction"] == 0.0
    assert 0.0 <= zone_a["zone_present_robot_pressure"] <= 1.0
    assert zone_a["robot_pressure_denominator"] == 2.0
    mask = build_action_mask(
        ("A", "B"),
        store_valid_by_zone={row["zone_id"]: bool(row["store_action_valid"]) for row in state["zone_rows"]},
        replenish_valid_by_zone={row["zone_id"]: bool(row["replenish_store_action_valid"]) for row in state["zone_rows"]},
    )
    features = build_feature_bundle(("A", "B"), mask, state)
    validate_no_removed_placeholder_features(features.action_feature_names)
    assert "turnover_value" not in features.action_feature_names
    assert features.action_feature_names == build_action_feature_names(("A", "B"))
    assert len(features.action_feature_names) == 21
    assert "estimated_queue_time" not in features.action_feature_names
    assert "cycle_estimate_known" in features.action_feature_names
    assert "is_store_action" not in features.action_feature_names
    assert "is_replenish_store_action" not in features.action_feature_names
    assert "allocator_cost_norm" not in features.action_feature_names
    assert "regret_score_norm" not in features.action_feature_names
    assert "one_robot_degenerate" not in features.action_feature_names
    assert all(not name.startswith("next_pod_zone_one_hot__") for name in features.action_feature_names)
    assert all(not name.startswith("next_retrieval_zone_one_hot__") for name in features.action_feature_names)
    assert features.X_actions.shape[1] == len(features.action_feature_names)
    assert features.X_stock.shape[1] == len(features.stock_feature_names)
    assert np.all(np.isfinite(features.X_actions))
    assert np.all(np.isfinite(features.X_stock))
    bounded_indices = [i for i, name in enumerate(features.action_feature_names) if name.endswith("_ratio") or name.endswith("_pressure") or name.endswith("_rank") or name.endswith("_norm")]
    assert np.all(features.X_actions[:, bounded_indices] >= 0.0)
    assert np.all(features.X_actions[:, bounded_indices] <= 1.0)
    assert build_action_feature_names(("A", "B")) == build_action_feature_names(("A", "B"))
    assert build_stock_feature_names() == build_stock_feature_names()


def test_phase2_preservation():
    inv, picker, _repl, final_storage, robot, _pod, next_jobs, policy = build_inventory(branch=STORE, next_jobs=1)
    context = RTSDestinationContext(warehouse=inv, robot=robot, pod=robot.job.pod, station=picker)
    state = build_state(context, ("zone-a",)).state_json
    proposals = get_committed_next_action_proposals(inv, robot)
    assert proposals
    assert get_committed_next_reservation(inv, robot) is None
    proposal = proposals["zone-a"]
    assert proposal.candidate_storage is final_storage
    assert proposal.job is next_jobs[0][0]
    start_return(robot)
    reservation = get_committed_next_reservation(inv, robot)
    assert policy.reservation_seen is None
    assert reservation is not None
    assert reservation.job is next_jobs[0][0]
    assert proposal.candidate_storage_id == reservation.selected_storage_id
    assert state["committed_next_action_proposals"]["zone-a"]["candidate_storage_id"] == reservation.selected_storage_id
    assert "regret_score" not in state["committed_next_action_proposals"]["zone-a"]


def test_checkpoint_schema_rejection():
    with tempfile.TemporaryDirectory() as tmp:
        checkpoint = Path(tmp)
        action_names = ("is_store_action",)
        stock_names = ("local_fill_ratio",)
        model = RTSMaskedActorCritic(action_feature_dim=1, stock_feature_dim=1)
        atomic_torch_save(model.state_dict(), checkpoint / "model.pt")
        write_feature_schema(
            checkpoint / "feature_schema.json",
            action_feature_names=build_action_feature_names(("A",)),
            stock_feature_names=build_stock_feature_names(),
        )
        good_schema = json.loads((checkpoint / "feature_schema.json").read_text())
        assert good_schema["action_feature_schema_version"] == ACTION_FEATURE_SCHEMA_VERSION
        assert good_schema["stock_feature_schema_version"] == STOCK_FEATURE_SCHEMA_VERSION
        assert good_schema["stock_source_version"] == STOCK_SOURCE_VERSION
        old_checkpoint = checkpoint / "old"
        old_checkpoint.mkdir()
        old_model = RTSMaskedActorCritic(action_feature_dim=1, stock_feature_dim=1)
        atomic_torch_save(old_model.state_dict(), old_checkpoint / "model.pt")
        (old_checkpoint / "feature_schema.json").write_text(
            json.dumps({"action_feature_names": list(action_names), "stock_feature_names": list(stock_names), "action_feature_dim": 1, "stock_feature_dim": 1}),
            encoding="utf-8",
        )
        (old_checkpoint / "metadata.json").write_text(json.dumps({"training_config": {}}), encoding="utf-8")
        try:
            load_policy_from_checkpoint(old_checkpoint, device="cpu")
        except ValueError as exc:
            assert "action_feature_schema_version" in str(exc)
        else:
            raise AssertionError("old checkpoint schema was accepted")


def main():
    test_historical_rank_and_layout()
    test_stock_and_replenishment_semantics()
    test_similarity_pressure_features_and_placeholders()
    test_phase2_preservation()
    test_checkpoint_schema_rejection()
    print("rts state semantics smoke ok")


if __name__ == "__main__":
    main()
