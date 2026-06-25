#!/usr/bin/env python3
"""Bounded smoke for Lukman order generation and pod/scenario inputs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.rmfs.order_generation import generate_orders_from_raw_bootstrap
from src.rmfs.runtime_io.layout_randomization import randomize_pod_locations, read_pod_storage_slots
from src.rmfs.runtime_io.run_profiles import resolve_run_profile
from src.rmfs.runtime_io.scenario_bundle import activate_scenario_inputs, list_available_scenarios


ARTIFACT_ROOT = REPO_ROOT / "data" / "runtime" / "lukman_order_pod_scenario_smoke"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def generate_order_run(seed: int, target: Path) -> dict:
    profile = resolve_run_profile("smoke", run_horizon_ticks=100, bootstrap_n_orders=25, seed=seed)
    generate_orders_from_raw_bootstrap(
        seed=seed,
        n_orders=profile.bootstrap_n_orders,
        source_path=REPO_ROOT / "data" / "input" / "base" / "raw_order.csv",
        target_dir=target,
        items_csv_path=REPO_ROOT / "data" / "input" / "base" / "items.csv",
        run_horizon_ticks=profile.run_horizon_ticks,
        demand_horizon_ticks=profile.demand_horizon_ticks,
        demand_buffer_ticks=profile.demand_buffer_ticks,
        order_generation_mode=profile.order_generation_mode,
        full_raw_order_replay=profile.full_raw_order_replay,
        profile=profile.profile,
    )
    with (target / "generated_order_meta.json").open(encoding="utf-8") as fh:
        return json.load(fh)


def main() -> int:
    shutil.rmtree(ARTIFACT_ROOT, ignore_errors=True)
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    try:
        same_a = ARTIFACT_ROOT / "orders_seed_123_a"
        same_b = ARTIFACT_ROOT / "orders_seed_123_b"
        diff = ARTIFACT_ROOT / "orders_seed_456"
        meta_a = generate_order_run(123, same_a)
        meta_b = generate_order_run(123, same_b)
        meta_diff = generate_order_run(456, diff)

        order_a = same_a / "generated_order.csv"
        order_b = same_b / "generated_order.csv"
        order_diff = diff / "generated_order.csv"
        assert digest(order_a) == digest(order_b), "same order seed produced different generated orders"
        assert digest(order_a) != digest(order_diff), "different order seed produced identical generated orders"
        assert Path(meta_a["target_dir"]).resolve().is_relative_to(ARTIFACT_ROOT.resolve())
        assert meta_a["seed"] == meta_b["seed"] == 123
        assert meta_diff["seed"] == 456

        items = pd.read_csv(REPO_ROOT / "data" / "input" / "base" / "items.csv")
        generated_orders = pd.read_csv(order_a)
        assert set(generated_orders["item_id"].astype(int)).issubset(set(items["item_id"].astype(int)))

        generated_pod = REPO_ROOT / "data" / "input" / "base" / "generated_pod.csv"
        pods = pd.read_csv(REPO_ROOT / "data" / "input" / "base" / "pods.csv")
        slots = read_pod_storage_slots(generated_pod)
        same_pods_a = randomize_pod_locations(generated_pod, ARTIFACT_ROOT / "pods_seed_77_a.csv", seed=77)
        same_pods_b = randomize_pod_locations(generated_pod, ARTIFACT_ROOT / "pods_seed_77_b.csv", seed=77)
        diff_pods = randomize_pod_locations(generated_pod, ARTIFACT_ROOT / "pods_seed_88.csv", seed=88)
        assert digest(same_pods_a) == digest(same_pods_b), "same pod seed produced different pod locations"
        if len(slots) > 1:
            assert digest(same_pods_a) != digest(diff_pods), "different pod seed did not change pod locations"
            pod_seed_status = "different pod seed changed pod-location allocation"
        else:
            pod_seed_status = "different pod seed check skipped: only one storage slot"

        pod_location_ids = set(pd.read_csv(same_pods_a)["pod_id"].astype(int))
        assert set(pods["pod_id"].astype(int)).issubset(pod_location_ids)

        scenarios = list_available_scenarios()
        selected_scenario = "cindy_s1" if "cindy_s1" in scenarios else scenarios[0]
        scenario_metadata = activate_scenario_inputs(
            selected_scenario,
            target_root=ARTIFACT_ROOT / "scenario_dry_run_target",
            dry_run=True,
        )
        assert scenario_metadata is not None and scenario_metadata["dry_run"] is True
        assert scenario_metadata["scenario_name"] == selected_scenario
        assert scenario_metadata["items_rows"] > 0
        assert scenario_metadata["pods_rows"] > 0

        print("lukman order pod scenario smoke ok")
        print("same order seed deterministic; different order seed changed generated orders")
        print(pod_seed_status)
        print(f"scenario dry run: {selected_scenario}")
        return 0
    finally:
        shutil.rmtree(ARTIFACT_ROOT, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
