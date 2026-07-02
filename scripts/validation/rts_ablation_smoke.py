#!/usr/bin/env python3
"""Validate RTS-RL name-based feature and branch ablations."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.validation.rts_ppo_update_smoke import synthetic_state
from src.rmfs.rl.rts.ablation import ACTION_FEATURE_ZERO_MAP, SUPPORTED_ABLATIONS, apply_ablation_to_arrays, resolve_ablation
from src.rmfs.rl.rts.features import build_feature_bundle


def main() -> None:
    zones = ("A", "B")
    mask = np.asarray([1, 1, 1, 1], dtype=np.int64)
    bundle = build_feature_bundle(zones, mask, synthetic_state())
    assert bundle.X_actions.shape[-1] == 18
    assert bundle.X_stock.shape[-1] == 4
    full = resolve_ablation("full")
    full_arrays = apply_ablation_to_arrays(
        X_actions=bundle.X_actions,
        M_actions=bundle.M_actions,
        X_stock=bundle.X_stock,
        M_stock=bundle.M_stock,
        action_feature_names=bundle.action_feature_names,
        zone_ids=zones,
        ablation=full,
    )
    assert np.array_equal(full_arrays[0], bundle.X_actions)
    assert np.array_equal(full_arrays[1], bundle.M_actions)

    name_to_idx = {name: idx for idx, name in enumerate(bundle.action_feature_names)}
    for name in SUPPORTED_ABLATIONS:
        ablation = resolve_ablation(name)
        X_actions, M_actions, X_stock, M_stock = apply_ablation_to_arrays(
            X_actions=bundle.X_actions,
            M_actions=bundle.M_actions,
            X_stock=bundle.X_stock,
            M_stock=bundle.M_stock,
            action_feature_names=bundle.action_feature_names,
            zone_ids=zones,
            ablation=ablation,
        )
        assert X_actions.shape == bundle.X_actions.shape
        assert M_actions.shape == bundle.M_actions.shape
        assert X_stock.shape == bundle.X_stock.shape
        assert M_stock.shape == bundle.M_stock.shape
        for feature_name in ACTION_FEATURE_ZERO_MAP.get(name, ()):
            assert np.all(X_actions[:, name_to_idx[feature_name]] == 0.0), name
        if name == "no_stock_encoder":
            assert np.all(X_stock == 0.0)
            assert np.all(M_stock == 0)
        if name == "store_only":
            assert M_actions.tolist() == [1, 1, 0, 0]

    worker_masked = apply_ablation_to_arrays(
        X_actions=bundle.X_actions,
        M_actions=bundle.M_actions,
        X_stock=bundle.X_stock,
        M_stock=bundle.M_stock,
        action_feature_names=bundle.action_feature_names,
        zone_ids=zones,
        ablation=resolve_ablation("no_zone_pressure"),
    )
    controller_masked = apply_ablation_to_arrays(
        X_actions=bundle.X_actions.copy(),
        M_actions=bundle.M_actions.copy(),
        X_stock=bundle.X_stock.copy(),
        M_stock=bundle.M_stock.copy(),
        action_feature_names=bundle.action_feature_names,
        zone_ids=zones,
        ablation=resolve_ablation("no_zone_pressure"),
    )
    assert all(np.array_equal(a, b) for a, b in zip(worker_masked, controller_masked))
    print("rts ablation smoke ok")


if __name__ == "__main__":
    main()
