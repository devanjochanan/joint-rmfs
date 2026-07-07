#!/usr/bin/env python3
"""Focused regression for the RTS action-feature schema v6 (width 17).

Covers:
  * exact width 17 and stable name ordering;
  * the four removed fields are absent;
  * the normalized next-pod distance is renamed to a raw, unbounded distance;
  * a raw distance above 1 is preserved (not clipped);
  * a no-next-job row uses proposed_next_job_known=0, distance=0, cycle=0;
  * the cycle-estimate invariant (valid + known next job -> finite cycle);
  * a legacy width-21 checkpoint is rejected clearly;
  * a fresh v6 schema reloads successfully.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.rmfs.rl.rts.features import (
    ACTION_FEATURE_SCHEMA_VERSION,
    build_action_feature_names,
    build_feature_bundle,
)
from src.rmfs.rl.rts.action_context import (
    RTSActionContext,
    RTSStorageFeasibility,
    cycle_estimate_invariant_holds,
    _mask_for_cycle_invariant,
)
from src.rmfs.rl.rts.stock_features import STOCK_FEATURE_NAMES
from src.rmfs.rl.rts.model import RTSMaskedActorCritic
from src.rmfs.rl.rts.training.checkpoint import atomic_torch_save, write_feature_schema
from src.rmfs.rl.rts.training.policy_loader import load_policy_from_checkpoint


EXPECTED_NAMES = (
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

REMOVED = (
    "cycle_estimate_known",
    "replenishment_queue_estimate_known",
    "estimated_replenishment_queue_seconds",
    "replenishment_station_load_pressure",
    "candidate_to_proposed_next_pod_distance_norm",
)


def _state(*, proposed_known: int, raw_distance: float, cycle_known: bool, cycle_seconds):
    return {
        "state_contract_version": "rts_rl_state.v4",
        "historical_pod_request_rank": 0.5,
        "committed_next_action_proposals": {
            "A": {
                "proposed_next_job_known": proposed_known,
                "candidate_to_proposed_next_pod_distance": raw_distance,
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
        "spatial_context": {"source_picker_x_norm": 0.1, "source_picker_y_norm": 0.2},
        "layout_normalization": {"x_min": 0.0, "x_max": 10.0, "y_min": 0.0, "y_max": 10.0},
        "rts_action_contexts": [
            {
                "action_index": 0,
                "branch": "store",
                "zone_id": "A",
                "next_job_proposal": {
                    "proposed_next_job_known": proposed_known,
                    "candidate_to_proposed_next_pod_distance": raw_distance,
                },
                "cycle_estimate": {"known": cycle_known, "estimated_cycle_seconds": cycle_seconds},
            },
            {
                "action_index": 1,
                "branch": "replenish_store",
                "zone_id": "A",
                "next_job_proposal": {
                    "proposed_next_job_known": proposed_known,
                    "candidate_to_proposed_next_pod_distance": raw_distance,
                },
                "cycle_estimate": {"known": cycle_known, "estimated_cycle_seconds": cycle_seconds},
            },
        ],
    }


def test_width_and_names_stable():
    for count in (1, 9, 16, 25):
        names = build_action_feature_names(tuple(f"z{i}" for i in range(count)))
        assert names == EXPECTED_NAMES, names
        assert len(names) == 17
    assert ACTION_FEATURE_SCHEMA_VERSION == "rts_action_features.v6"
    for removed in REMOVED:
        assert removed not in EXPECTED_NAMES
    print("PASS test_width_and_names_stable")


def test_raw_distance_unclipped():
    names = build_action_feature_names(("A",))
    dist_idx = names.index("candidate_to_proposed_next_pod_distance")
    # Raw directed empty-robot graph distance far above 1.0 is preserved.
    state = _state(proposed_known=1, raw_distance=42.75, cycle_known=True, cycle_seconds=20.0)
    bundle = build_feature_bundle(("A",), [1, 1], state)
    assert bundle.X_actions.shape == (2, 17)
    assert bundle.X_actions[0, dist_idx] == 42.75, bundle.X_actions[0, dist_idx]
    assert bundle.X_actions[0, dist_idx] > 1.0
    print("PASS test_raw_distance_unclipped")


def test_no_next_job_zeros():
    names = build_action_feature_names(("A",))
    known_idx = names.index("proposed_next_job_known")
    dist_idx = names.index("candidate_to_proposed_next_pod_distance")
    cycle_idx = names.index("estimated_cycle_time")
    state = _state(proposed_known=0, raw_distance=0.0, cycle_known=False, cycle_seconds=None)
    bundle = build_feature_bundle(("A",), [1, 1], state)
    for row in (0, 1):
        assert bundle.X_actions[row, known_idx] == 0.0
        assert bundle.X_actions[row, dist_idx] == 0.0
        assert bundle.X_actions[row, cycle_idx] == 0.0
    print("PASS test_no_next_job_zeros")


def test_cycle_invariant_helper():
    def ctx(action_valid, has_next, known, seconds):
        return SimpleNamespace(
            action_valid=action_valid,
            branch="store",
            store_valid=action_valid,
            replenish_store_valid=False,
            invalid_reason_codes=(),
            next_job_proposal=SimpleNamespace(has_next_job=has_next),
            cycle_estimate=SimpleNamespace(to_json_dict=lambda: {"known": known, "estimated_cycle_seconds": seconds}),
        )

    # valid + known next job + finite cycle -> holds
    assert cycle_estimate_invariant_holds(ctx(True, True, True, 20.0)) is True
    # valid + known next job + unknown cycle -> violated
    bad = ctx(True, True, False, None)
    assert cycle_estimate_invariant_holds(bad) is False
    # Masking must operate on a real RTSActionContext dataclass instance.
    feasible = RTSStorageFeasibility(True, True, True, True, True, ())
    real_bad = RTSActionContext(
        action_index=0,
        branch="store",
        zone_id="A",
        candidate_storage=None,
        candidate_storage_id=None,
        candidate_storage_coordinate=None,
        replenishment_station=None,
        replenishment_station_id=None,
        replenishment_station_coordinate=None,
        replenishment_eligibility=False,
        replenishment_eligible_skus=(),
        storage_feasibility=feasible,
        station_feasibility=None,
        store_valid=True,
        replenish_store_valid=False,
        action_valid=True,
        invalid_reason_codes=(),
        next_job_proposal=SimpleNamespace(has_next_job=True),
        cycle_estimate=SimpleNamespace(to_json_dict=lambda: {"known": False, "estimated_cycle_seconds": None}),
    )
    assert cycle_estimate_invariant_holds(real_bad) is False
    masked = _mask_for_cycle_invariant(real_bad)
    assert masked.action_valid is False
    assert "cycle_estimate_invariant_violation" in masked.invalid_reason_codes
    # no next job -> trivially holds (may emit zero cycle)
    assert cycle_estimate_invariant_holds(ctx(True, False, False, None)) is True
    # invalid action -> trivially holds
    assert cycle_estimate_invariant_holds(ctx(False, True, False, None)) is True
    print("PASS test_cycle_invariant_helper")


def test_legacy_width21_checkpoint_rejected_and_v6_reloads():
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
            msg = str(exc)
            assert "rts_action_features.v6" in msg and "width 17" in msg and "fresh training" in msg, msg
        else:
            raise AssertionError("v5 width-21 checkpoint was accepted")

    with tempfile.TemporaryDirectory() as tmp:
        checkpoint = Path(tmp)
        names = list(build_action_feature_names(("A",)))
        model = RTSMaskedActorCritic(action_feature_dim=len(names), stock_feature_dim=4)
        atomic_torch_save(model.state_dict(), checkpoint / "model.pt")
        (checkpoint / "metadata.json").write_text(
            json.dumps({"training_config": {}, "policy_checkpoint_id": "v6_smoke"}), encoding="utf-8"
        )
        write_feature_schema(
            checkpoint / "feature_schema.json",
            action_feature_names=tuple(names),
            stock_feature_names=tuple(STOCK_FEATURE_NAMES),
        )
        loaded = load_policy_from_checkpoint(checkpoint, device="cpu")
        assert loaded.feature_schema["action_feature_dim"] == 17
        assert loaded.feature_schema["action_feature_schema_version"] == "rts_action_features.v6"
    print("PASS test_legacy_width21_checkpoint_rejected_and_v6_reloads")


def main():
    test_width_and_names_stable()
    test_raw_distance_unclipped()
    test_no_next_job_zeros()
    test_cycle_invariant_helper()
    test_legacy_width21_checkpoint_rejected_and_v6_reloads()
    print("\nALL RTS FEATURE SCHEMA V6 REGRESSION TESTS PASSED")


if __name__ == "__main__":
    raise SystemExit(main())
