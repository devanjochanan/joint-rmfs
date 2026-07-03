#!/usr/bin/env python3
"""Focused RTS v5 action-feature contract smoke."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np
import torch

os.environ.setdefault("RMFS_FAST_TRAIN", "1")
os.environ.setdefault("RMFS_DETAIL_DB", "0")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/rmfs-mpl")

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.validation.rts_committed_next_smoke import build_inventory
from src.rmfs.decisions.rts.types import RTSDestinationContext
from src.rmfs.rl.rts.action_context import revalidate_selected_context, selected_context_by_index
from src.rmfs.rl.rts.action_space import ACTION_BRANCHES
from src.rmfs.rl.rts.features import ACTION_FEATURE_SCHEMA_VERSION, build_action_feature_names, build_feature_bundle
from src.rmfs.rl.rts.model import RTSMaskedActorCritic
from src.rmfs.rl.rts.queue_estimator import estimate_replenishment_queue
from src.rmfs.rl.rts.state import build_state
from src.rmfs.rl.rts.stock_features import STOCK_FEATURE_NAMES, STOCK_FEATURE_SCHEMA_VERSION
from src.rmfs.rl.rts.training.checkpoint import atomic_torch_save, write_feature_schema
from src.rmfs.rl.rts.training.policy_actor import RTSOnPolicyActor, RTSOnPolicyActorConfig
from src.rmfs.rl.rts.training.policy_loader import load_policy_from_checkpoint
from src.rmfs.rl.rts.training.rollout_dataset import RTSTrainingStep, build_feature_tensors_from_steps


def zone_ids(count: int) -> tuple[str, ...]:
    return tuple(f"z{index:02d}" for index in range(count))


def synthetic_state(zones: tuple[str, ...], *, picker_x: float = 0.1, picker_y: float = 0.2) -> dict:
    rows = []
    proposals = {}
    for index, zone in enumerate(zones):
        x = float(index % 5)
        y = float(index // 5)
        rows.append(
            {
                "zone_id": zone,
                "candidate_storage_x": x,
                "candidate_storage_y": y,
                "free_slot_ratio": 0.5,
                "sku_similarity_fraction": 0.25,
                "zone_present_robot_pressure": 0.1,
                "zone_destination_robot_pressure": 0.2,
                "macro_region_present_robot_pressure": 0.3,
                "macro_region_destination_robot_pressure": 0.4,
                "store_action_valid": 1.0,
                "replenish_store_action_valid": 1.0,
            }
        )
        proposals[zone] = {
            "proposal_semantics_version": "rts_nearest_next_job_proposal.v1",
            "proposed_next_job_known": 1,
            "candidate_to_proposed_next_pod_distance": 1.0 + index,
            "next_pod_to_picker_distance": 2.0 + index,
            "proposal_cost": 3.0 + index,
        }
    return {
        "state_contract_version": "rts_rl_state.v4",
        "historical_pod_request_rank": 0.75,
        "committed_next_action_proposals": proposals,
        "zone_rows": rows,
        "stock_rows": [
            {
                "sku_id": "sku-1",
                "local_fill_ratio": 0.2,
                "local_below_threshold": 1.0,
                "global_fill_ratio": 0.8,
                "global_below_threshold": 0.0,
            }
        ],
        "replenishment_snapshot": {"eligible_skus": ["sku-1"]},
        "spatial_context": {
            "source_picker_x_norm": picker_x,
            "source_picker_y_norm": picker_y,
            "distance_normalization_denominator": 100.0,
        },
        "layout_normalization": {
            "x_min": 0.0,
            "x_max": 4.0,
            "y_min": 0.0,
            "y_max": 4.0,
        },
    }


def test_feature_contract_variable_zones():
    expected = (
        "is_replenish_store",
        "historical_pod_request_rank",
        "replenishment_eligible_sku_ratio",
        "source_picker_x_norm",
        "source_picker_y_norm",
        "candidate_storage_x_norm",
        "candidate_storage_y_norm",
        "free_slot_ratio",
        "sku_similarity_fraction",
        "zone_present_robot_pressure",
        "zone_destination_robot_pressure",
        "macro_region_present_robot_pressure",
        "macro_region_destination_robot_pressure",
        "selected_replenishment_station_destination_pressure",
        "proposed_next_job_known",
        "candidate_to_proposed_next_pod_distance",
        "estimated_cycle_time",
    )
    assert build_action_feature_names(zone_ids(9)) == expected
    assert build_action_feature_names(zone_ids(16)) == expected
    assert build_action_feature_names(zone_ids(25)) == expected
    assert ACTION_FEATURE_SCHEMA_VERSION == "rts_action_features.v6"
    assert STOCK_FEATURE_SCHEMA_VERSION == "rts_stock_features.v3"
    assert tuple(STOCK_FEATURE_NAMES) == (
        "local_fill_ratio",
        "local_below_threshold",
        "global_fill_ratio",
        "global_below_threshold",
    )
    for count, expected_rows in ((9, 18), (16, 32), (25, 50)):
        zones = zone_ids(count)
        bundle = build_feature_bundle(zones, [1] * (2 * count), synthetic_state(zones))
        assert bundle.X_actions.shape == (expected_rows, 17)
        assert bundle.X_stock.shape[1] == 4
        assert not any("one_hot" in name for name in bundle.action_feature_names)
        removed = {"allocator_cost_norm", "regret_score_norm", "one_robot_degenerate", "estimated_queue_time"}
        assert removed.isdisjoint(bundle.action_feature_names)


def test_picker_count_compatibility():
    names = build_action_feature_names(zone_ids(9))
    widths = []
    source_x_values = []
    source_index = names.index("source_picker_x_norm")
    for picker_count, picker_x in ((2, 0.1), (3, 0.5), (4, 0.9)):
        state = synthetic_state(zone_ids(9), picker_x=picker_x, picker_y=0.25)
        state["spatial_context"]["picking_station_count"] = float(picker_count)
        bundle = build_feature_bundle(zone_ids(9), [1] * 18, state)
        widths.append(bundle.X_actions.shape[1])
        source_x_values.append(float(bundle.X_actions[0, source_index]))
        assert "picking_station_count" not in bundle.action_feature_names
        assert all("picker_one_hot" not in name for name in bundle.action_feature_names)
    assert widths == [17, 17, 17]
    assert np.allclose(source_x_values, [0.1, 0.5, 0.9])


def test_mixed_zone_batch_and_model_forward():
    steps = []
    for count, selected in ((9, 0), (16, 15), (25, 49)):
        zones = zone_ids(count)
        steps.append(
            RTSTrainingStep(
                decision_event_id=f"d{count}",
                zone_ids=zones,
                action_mask=np.ones((2 * count,), dtype=np.int64),
                selected_action_index=selected,
                reward=1.0,
                terminated=True,
                truncated=False,
                state_json=synthetic_state(zones),
                selected_action_branch=None,
                selected_zone_id=None,
                realized_cycle_time=1.0,
                policy_name="synthetic",
            )
        )
    padded = build_feature_tensors_from_steps(steps)
    assert padded.X_actions.shape == (3, 50, 17)
    assert padded.M_actions[0, 18:].sum() == 0
    assert padded.M_actions[1, 32:].sum() == 0
    assert padded.M_actions[2].sum() == 50
    model = RTSMaskedActorCritic(action_feature_dim=17, stock_feature_dim=4)
    with torch.no_grad():
        logits, values = model(
            torch.as_tensor(padded.X_actions, dtype=torch.float32),
            torch.as_tensor(padded.M_actions, dtype=torch.int64),
            torch.as_tensor(padded.X_stock, dtype=torch.float32),
            torch.as_tensor(padded.M_stock, dtype=torch.int64),
        )
        masked_choice = logits.masked_fill(torch.as_tensor(padded.M_actions) <= 0, torch.finfo(logits.dtype).min).argmax(dim=-1)
        assert all(int(masked_choice[i]) < int(padded.M_actions[i].sum()) for i in range(3))
        altered = torch.as_tensor(padded.X_actions, dtype=torch.float32)
        altered[0, 18:, :] = 999.0
        altered[1, 32:, :] = -999.0
        _logits2, values2 = model(
            altered,
            torch.as_tensor(padded.M_actions, dtype=torch.int64),
            torch.as_tensor(padded.X_stock, dtype=torch.float32),
            torch.as_tensor(padded.M_stock, dtype=torch.int64),
        )
        assert torch.allclose(values, values2, atol=1e-5)


def test_checkpoint_cross_zone_compatibility():
    with tempfile.TemporaryDirectory() as tmp:
        checkpoint = Path(tmp)
        action_names = build_action_feature_names(zone_ids(9))
        stock_names = tuple(STOCK_FEATURE_NAMES)
        model = RTSMaskedActorCritic(action_feature_dim=len(action_names), stock_feature_dim=len(stock_names))
        atomic_torch_save(model.state_dict(), checkpoint / "model.pt")
        write_feature_schema(
            checkpoint / "feature_schema.json",
            action_feature_names=action_names,
            stock_feature_names=stock_names,
            schema_metadata={
                "training_zone_count": 9,
                "training_zone_ids": list(zone_ids(9)),
                "zone_order": list(reversed(zone_ids(9))),
            },
        )
        (checkpoint / "metadata.json").write_text(json.dumps({"training_config": {}}), encoding="utf-8")
        loaded = load_policy_from_checkpoint(checkpoint, device="cpu")
        for count in (16, 25):
            actor = RTSOnPolicyActor(
                model=loaded.model,
                zone_ids=zone_ids(count),
                config=RTSOnPolicyActorConfig(policy_checkpoint_id=loaded.policy_checkpoint_id),
            )
            assert len(actor.zone_ids) == count

        bad = checkpoint / "bad_schema"
        bad.mkdir()
        atomic_torch_save(model.state_dict(), bad / "model.pt")
        schema = json.loads((checkpoint / "feature_schema.json").read_text())
        schema["action_feature_names"][0] = "wrong_first_feature"
        (bad / "feature_schema.json").write_text(json.dumps(schema), encoding="utf-8")
        (bad / "metadata.json").write_text(json.dumps({"training_config": {}}), encoding="utf-8")
        try:
            load_policy_from_checkpoint(bad, device="cpu")
        except ValueError as exc:
            assert "action_feature_names" in str(exc)
        else:
            raise AssertionError("schema name drift was accepted")

        v3 = checkpoint / "v3"
        v3.mkdir()
        atomic_torch_save(model.state_dict(), v3 / "model.pt")
        schema = json.loads((checkpoint / "feature_schema.json").read_text())
        schema["action_feature_schema_version"] = "rts_action_features.v3"
        (v3 / "feature_schema.json").write_text(json.dumps(schema), encoding="utf-8")
        (v3 / "metadata.json").write_text(json.dumps({"training_config": {}}), encoding="utf-8")
        try:
            load_policy_from_checkpoint(v3, device="cpu")
        except ValueError as exc:
            assert "action_feature_schema_version" in str(exc)
            assert "incompatible" in str(exc)
        else:
            raise AssertionError("legacy v3 checkpoint was accepted")


def test_selected_zone_revalidation_refreshes_only_selected():
    inv, picker, _repl, final_storage, robot, _pod, _next_jobs, _policy = build_inventory(branch="store", next_jobs=1)
    zone_b_storage = inv.storage_manager.createStorage(30, 5)
    zone_b_storage.rts_zone_id = "zone-b"
    context = RTSDestinationContext(inv, robot, robot.job.pod, picker)
    state = build_state(context, ("zone-a", "zone-b"))
    proposals = inv.committed_next_registry.get_action_proposals_for_robot(robot)
    proposals["zone-a"] = proposals["zone-b"]
    inv.committed_next_registry.robot_id_to_action_proposals[str(robot._id)] = proposals
    calls = {"count": 0}
    original = inv.committed_next_registry.refresh_action_proposal_for_zone

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    inv.committed_next_registry.refresh_action_proposal_for_zone = wrapped
    refreshed = revalidate_selected_context(context, selected_context_by_index(state.action_contexts, 0))
    assert calls["count"] == 1
    assert refreshed.next_job_proposal.candidate_storage is final_storage
    assert inv.committed_next_registry.get_action_proposal(robot, "replenish_store", "zone-b").candidate_storage is zone_b_storage


def test_queue_time_gate_structurally_dead():
    class Station:
        station_id = "repl"
        max_robots = 2
        robot_ids = {}
        robot_queue = []

    class Warehouse:
        _objects = []
        tick_to_second = 1.0

    class Context:
        warehouse = Warehouse()

    known = estimate_replenishment_queue(Context(), Station())
    assert known.known is True
    assert known.estimated_wait_seconds == 0.0
    full = Station()
    full.robot_ids = {"r1": object(), "r2": object()}
    unknown = estimate_replenishment_queue(Context(), full)
    assert unknown.known is False
    assert unknown.estimated_wait_seconds is None
    assert "estimated_queue_time" not in build_action_feature_names(zone_ids(9))


def main():
    test_feature_contract_variable_zones()
    test_picker_count_compatibility()
    test_mixed_zone_batch_and_model_forward()
    test_checkpoint_cross_zone_compatibility()
    test_selected_zone_revalidation_refreshes_only_selected()
    test_queue_time_gate_structurally_dead()
    print("rts phase1 contract smoke ok")


if __name__ == "__main__":
    main()
