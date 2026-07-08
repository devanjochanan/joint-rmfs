#!/usr/bin/env python3
"""Focused deterministic checks for RTS VRSLA teacher pretraining."""

from __future__ import annotations

import random
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.validation.rts_ppo_update_smoke import synthetic_state
from src.rmfs.rl.rts.features import ACTION_FEATURE_SCHEMA_VERSION, build_feature_bundle
from src.rmfs.rl.rts.static_runtime_index import (
    install_static_runtime_index,
    reset_static_runtime_index_diagnostics,
    static_runtime_index_diagnostics,
)
from src.rmfs.rl.rts.static_state_context import (
    HISTORICAL_POD_RANK_VERSION,
    _pod_request_counts,
    _pod_request_ranks,
    get_or_build_static_state_context,
)
from src.rmfs.rl.rts.training.config import RTSTrainingConfig
from src.rmfs.rl.rts.training.checkpoint import load_training_checkpoint
from src.rmfs.rl.rts.training.policy_loader import load_policy_from_checkpoint
from src.rmfs.rl.rts.training.vrsla_pretrain import (
    run_behavior_cloning,
    save_behavior_cloning_checkpoint,
    teacher_batch_from_rows,
)
from src.rmfs.rl.rts.vrsla_teacher import choose_vrsla_teacher_branch, greedy_vrsla_assignments


class Warehouse:
    pass


class GraphWrapper:
    def __init__(self, graph):
        self.graph = graph


def obj(**kwargs):
    return SimpleNamespace(**kwargs)


def storage(sid, x, y, zone="zone-a", empty=True):
    return obj(
        storage_id=str(sid),
        storage_number=int(sid),
        pos_x=float(x),
        pos_y=float(y),
        rts_zone_id=zone,
        is_empty=bool(empty),
        assigned_pod=None,
    )


def picker(station_id, x, y):
    return obj(station_id=str(station_id), station_type="picker", pos_x=float(x), pos_y=float(y))


def build_graph(storages, picker_station):
    import networkx as nx

    graph = nx.DiGraph()
    nodes = [f"{int(s.pos_x)},{int(s.pos_y)}" for s in storages] + [f"{int(picker_station.pos_x)},{int(picker_station.pos_y)}"]
    graph.add_nodes_from(nodes)
    pnode = f"{int(picker_station.pos_x)},{int(picker_station.pos_y)}"
    for stg in storages:
        snode = f"{int(stg.pos_x)},{int(stg.pos_y)}"
        graph.add_edge(snode, pnode, weight=float(getattr(stg, "to_picker", 1.0)))
        graph.add_edge(pnode, snode, weight=float(getattr(stg, "from_picker", 1.0)))
    return graph


def build_warehouse():
    stg1 = storage(1, 1, 0)
    stg1.to_picker = 4.0
    stg1.from_picker = 4.0
    stg2 = storage(2, 2, 0)
    stg2.to_picker = 1.0
    stg2.from_picker = 1.0
    stg3 = storage(3, 3, 0)
    stg3.to_picker = 1.0
    stg3.from_picker = 1.0
    p = picker("picker-1", 0, 0)
    wh = Warehouse()
    wh.storage_manager = obj(storages=[stg1, stg2, stg3])
    wh.station_manager = obj(stations=[p], picking_stations=[p], replenishment_stations=[])
    wh._objects = [stg1, stg2, stg3, p]
    graph = build_graph([stg1, stg2, stg3], p)
    wh.graph = GraphWrapper(graph)
    wh.graph_pod = GraphWrapper(graph)
    wh.pod_manager = obj(pods=[])
    wh.order_manager = obj(orders=[])
    return wh, p, stg1, stg2, stg3


def test_pod_velocity_union_and_rank():
    pod_a = obj(pod_id="1", skus={"A": {}, "B": {}})
    pod_b = obj(pod_id="2", skus={"C": {}})
    orders = {
        "o1": {"A", "B"},
        "o2": {"B", "C"},
        "o3": {"D"},
        "o4": {"A"},
    }
    counts = _pod_request_counts([pod_a, pod_b], orders)
    assert counts["1"] == 3, counts
    assert counts["2"] == 1, counts
    inclusion_exclusion = 2 + 2 - 1
    assert counts["1"] == inclusion_exclusion
    ranks = _pod_request_ranks([pod_b, pod_a], counts)
    assert ranks["1"] == 1.0 and ranks["2"] == 0.0, ranks
    assert HISTORICAL_POD_RANK_VERSION == "rts_historical_pod_request_rank.v1"


def test_static_caching_and_directed_slot_scores():
    wh, p, stg1, stg2, stg3 = build_warehouse()
    reset_static_runtime_index_diagnostics()
    first = install_static_runtime_index(wh)
    second = install_static_runtime_index(wh)
    assert first is second
    diag = static_runtime_index_diagnostics()
    assert diag["build_count"] == 1, diag
    assert diag["vrsla_slot_score_build_count"] == 1, diag
    assert first.vrsla_cycle_distance(p, stg1) == 8.0
    assert first.vrsla_cycle_distance(p, stg2) == 2.0
    assert first.vrsla_cycle_distance(p, stg3) == 2.0
    ordered = first.vrsla_sorted_storage_ids(p)
    assert ordered[:3] == ("2", "3", "1"), ordered
    assert first.vrsla_hotness_rank(p, stg2) == 1.0
    assert first.vrsla_hotness_rank(p, stg1) == 0.0

    # Static pod context also caches on the warehouse object.
    pod = obj(pod_id="10", skus={"A": {}})
    wh.pod_manager.pods = [pod]
    wh.order_manager.orders = [obj(order_id="o1", skus={"A": 1}), obj(order_id="o2", skus={"B": 1})]
    ctx1 = get_or_build_static_state_context(wh)
    ctx2 = get_or_build_static_state_context(wh)
    assert ctx1 is ctx2
    assert ctx1.vrsla_pod_velocity_count(pod) == 1
    assert ctx1.historical_metadata["pod_velocity_semantics"] == "distinct_order_union"


def test_greedy_matching_and_unavailable_exclusion():
    wh, p, stg1, stg2, _stg3 = build_warehouse()
    index = install_static_runtime_index(wh)
    hot = obj(pod_id="1")
    cold = obj(pod_id="2")
    context = obj(warehouse=wh, station=p, robot=obj(_id="r"), pod=hot)
    items = [
        obj(context=context, station=p, pod=cold),
        obj(context=context, station=p, pod=hot),
    ]
    static_context = obj(
        vrsla_pod_velocity_count=lambda pod: 10 if str(pod.pod_id) == "1" else 1,
        vrsla_pod_velocity_rank=lambda pod: 1.0 if str(pod.pod_id) == "1" else 0.0,
    )
    assignments = greedy_vrsla_assignments(items, zone_ids=index.zone_ids, static_index=index, static_context=static_context)
    assert assignments["1"].storage is stg2, assignments
    assert assignments["2"].storage is not stg2, assignments

    stg2.is_empty = False
    assignments = greedy_vrsla_assignments(items, zone_ids=index.zone_ids, static_index=index, static_context=static_context)
    assert assignments["1"].storage is not stg2
    assert assignments["1"].storage_id in {"3", "1"}
    stg2.is_empty = True
    stg1.assigned_pod = hot
    assignments = greedy_vrsla_assignments(items, zone_ids=index.zone_ids, static_index=index, static_context=static_context)
    assert assignments["1"].storage is stg2


def test_seeded_branch_choice():
    rng = random.Random(42)
    first = choose_vrsla_teacher_branch(
        rng=rng,
        store_valid=True,
        replenish_valid=True,
        store_index=0,
        replenish_index=1,
        store_mask=1,
        replenish_mask=1,
    )
    second = choose_vrsla_teacher_branch(
        rng=rng,
        store_valid=True,
        replenish_valid=True,
        store_index=0,
        replenish_index=1,
        store_mask=1,
        replenish_mask=1,
    )
    assert first["branch"] == "replenish_store" and first["rng_draw"] > 0.5, first
    assert second["branch"] == "store" and second["rng_draw"] < 0.5, second
    assert choose_vrsla_teacher_branch(
        rng=rng,
        store_valid=True,
        replenish_valid=False,
        store_index=0,
        replenish_index=1,
        store_mask=1,
        replenish_mask=0,
    )["action_index"] == 0
    assert choose_vrsla_teacher_branch(
        rng=rng,
        store_valid=False,
        replenish_valid=True,
        store_index=0,
        replenish_index=1,
        store_mask=0,
        replenish_mask=1,
    )["action_index"] == 1
    assert choose_vrsla_teacher_branch(
        rng=rng,
        store_valid=False,
        replenish_valid=False,
        store_index=0,
        replenish_index=1,
        store_mask=0,
        replenish_mask=0,
    ) is None


def teacher_row(selected, branch, zone):
    state = synthetic_state(0.0)
    state["rts_action_contexts"] = [
        {"action_index": 0, "branch": "store", "zone_id": "A", "vrsla_slot_hotness_rank": 1.0},
        {"action_index": 1, "branch": "store", "zone_id": "B", "vrsla_slot_hotness_rank": 0.5},
        {"action_index": 2, "branch": "replenish_store", "zone_id": "A", "vrsla_slot_hotness_rank": 1.0},
        {"action_index": 3, "branch": "replenish_store", "zone_id": "B", "vrsla_slot_hotness_rank": 0.5},
    ]
    bundle = build_feature_bundle(("A", "B"), [1, 1, 1, 0], state)
    return {
        "X_actions": bundle.X_actions.tolist(),
        "M_actions": bundle.M_actions.tolist(),
        "X_stock": bundle.X_stock.tolist(),
        "M_stock": bundle.M_stock.tolist(),
        "action_feature_names": list(bundle.action_feature_names),
        "stock_feature_names": list(bundle.stock_feature_names),
        "zone_ids": ["A", "B"],
        "teacher_action_index": selected,
        "teacher_branch": branch,
        "teacher_zone_id": zone,
        "teacher_storage_id": f"s-{selected}",
        "run_id": f"run-{selected}",
        "seed": 100 + selected,
        "pod_velocity_score_cache_identity": "pod-cache",
        "slot_score_cache_identity": "slot-cache",
    }


def test_behavior_cloning_and_checkpoint_reload():
    assert ACTION_FEATURE_SCHEMA_VERSION == "rts_action_features.v6"
    torch.manual_seed(7)
    rows = tuple(teacher_row(selected, branch, zone) for selected, branch, zone in [
        (0, "store", "A"),
        (2, "replenish_store", "A"),
        (0, "store", "A"),
        (2, "replenish_store", "A"),
    ])
    batch = teacher_batch_from_rows(rows)
    with tempfile.TemporaryDirectory() as tmp:
        config = RTSTrainingConfig(
            artifact_label="vrsla_bc_smoke",
            output_root=Path(tmp),
            learning_rate=1e-3,
            minibatch_size=2,
            tensorboard_enabled=False,
        )
        model, result = run_behavior_cloning(batch, config=config, device="cpu", max_epochs=20, patience=5)
        assert result.train_loss_final <= result.train_loss_initial + 1e-6, result
        assert result.invalid_action_rate == 0.0, result
        checkpoint = save_behavior_cloning_checkpoint(
            model=model,
            config=config,
            batch_id=10,
            teacher_batch=batch,
            result=result,
            teacher_batch_count=10,
        )
        loaded = load_policy_from_checkpoint(checkpoint, device="cpu")
        optimizer = torch.optim.Adam(loaded.model.parameters(), lr=1e-3)
        metadata = load_training_checkpoint(checkpoint, model=loaded.model, optimizer=optimizer, device="cpu")
        assert metadata["initialization_method"] == "vrsla_behavior_cloning"
        assert metadata["teacher_batch_count"] == 10
        assert metadata["critic_pretrained"] is False


def main():
    test_pod_velocity_union_and_rank()
    test_static_caching_and_directed_slot_scores()
    test_greedy_matching_and_unavailable_exclusion()
    test_seeded_branch_choice()
    test_behavior_cloning_and_checkpoint_reload()
    print("rts vrsla teacher smoke ok")


if __name__ == "__main__":
    main()
