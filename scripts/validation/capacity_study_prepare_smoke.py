#!/usr/bin/env python3
"""Bounded validation for the capacity study (robot count x order rate)."""

from __future__ import annotations

import json
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.experiments import capacity_study_order_rate as study


def args(root: Path, *, treatment: str = "all_off", execute: bool = False, seconds: float = 87000.0, **filters):
    return SimpleNamespace(
        prepare_only=not execute,
        execute=execute,
        treatment=treatment,
        max_workers=filters.pop("max_workers", 1),
        replications=20,
        simulated_seconds=seconds,
        output_root=str(root),
        resume=filters.pop("resume", False),
        limit_runs=filters.pop("limit_runs", None),
        robots=filters.pop("robots", None),
        order_rate=filters.pop("order_rate", None),
        replication=filters.pop("replication", None),
        seed=42,
        pps_model_path=str(study.DEFAULT_PPS_MODEL_PATH),
        rts_checkpoint_dir=str(study.DEFAULT_RTS_CHECKPOINT_DIR),
        charging_config_path=str(study.DEFAULT_CHARGING_CONFIG_PATH),
        progress=False,
    )


def assert_prepare_manifest(manifest: dict, *, treatment: str) -> None:
    conditions = manifest["conditions"]
    assert len(conditions) == 300, f"expected 300, got {len(conditions)}"
    assert len({c["run_id"] for c in conditions}) == 300, "duplicate run_ids"
    assert manifest["ticks"] == 580000, f"expected 580000 ticks, got {manifest['ticks']}"
    assert manifest["simulated_seconds"] == 87000.0
    assert manifest["tick_to_second"] == 0.15
    assert manifest["treatment"] == treatment

    assert Counter(c["robot_count"] for c in conditions) == {10: 60, 15: 60, 20: 60, 25: 60, 30: 60}
    assert Counter(c["order_rate"] for c in conditions) == {400: 100, 500: 100, 600: 100}
    assert Counter(c["replication"] for c in conditions) == {i: 15 for i in range(1, 21)}
    assert all(c["picker_count"] == 3 for c in conditions)
    assert all(c["replenishment_count"] == 1 for c in conditions)

    input_meta = manifest["input_meta"]
    assert input_meta["layout"]["pod_capacity"] == 121
    assert input_meta["layout"]["picking_station_count"] == 3
    assert input_meta["layout"]["replenishment_station_count"] == 1
    assert input_meta["layout"]["graph_connected"]
    assert input_meta["layout"]["legal_robot_spawn_cell_count"] >= 30

    layout_hashes = {c["layout_hash"] for c in conditions}
    assert len(layout_hashes) == 1, f"expected one shared layout hash, got {len(layout_hashes)}"

    by_rep_seed = defaultdict(set)
    for condition in conditions:
        by_rep_seed[condition["replication"]].add(condition["run_seed"])
    assert all(len(seeds) == 1 for seeds in by_rep_seed.values()), "unified seed per replication violated"

    for rep in range(1, 21):
        rep_conditions = [c for c in conditions if c["replication"] == rep]
        seeds = {c["run_seed"] for c in rep_conditions}
        assert len(seeds) == 1, f"replication {rep} has {len(seeds)} distinct seeds"
        assert seeds.pop() == 42 + rep - 1, f"replication {rep} seed mismatch"


def first_worker_summary(root: Path) -> dict:
    summaries = sorted((root / "runs").glob("*/worker_summary.json"))
    assert summaries, f"no worker summary under {root}"
    return json.loads(summaries[0].read_text(encoding="utf-8"))


def main() -> int:
    base = Path(tempfile.mkdtemp(prefix="rmfs_capacity_study_validation_"))

    prepare_root = base / "prepare"
    prepare_args = args(prepare_root)
    prepare_manifest, prepare_conditions, prepare_failures = study.prepare_conditions(prepare_args)
    assert prepare_failures == []
    study.write_outputs(prepare_root, prepare_manifest, prepare_conditions, prepare_failures)
    assert_prepare_manifest(prepare_manifest, treatment="all_off")
    print("  [ok] prepare manifest 300-run structure")

    prepare_root_2 = base / "prepare_again"
    prepare_again, prepare_again_conditions, prepare_again_failures = study.prepare_conditions(args(prepare_root_2))
    assert prepare_again_failures == []
    assert [c["run_id"] for c in prepare_conditions] == [c["run_id"] for c in prepare_again_conditions]
    print("  [ok] idempotent preparation")

    exec_root = base / "exec_bounded"
    exec_args = args(exec_root, execute=True, seconds=0.45)
    exec_manifest, exec_conditions, exec_failures = study.prepare_conditions(exec_args)
    assert exec_failures == []

    run_10_400 = args(
        exec_root,
        execute=True,
        seconds=0.45,
        robots=[10],
        order_rate=[400],
        replication=[1],
        limit_runs=1,
    )
    study.execute_selected(run_10_400, exec_manifest, exec_conditions)
    study.write_outputs(exec_root, exec_manifest, exec_conditions, exec_failures)
    s1 = first_worker_summary(exec_root)
    assert s1["status"] == "success"
    assert s1["realized_robot_count"] == 10
    assert s1["realized_picking_station_count"] == 3
    assert s1["realized_replenishment_station_count"] == 1
    assert s1["pps_mode"] == "heuristic"
    assert s1.get("rts_policy_mode", "current") == "current"
    print("  [ok] bounded run: 10 robots, 400/hr")

    all_on_root = base / "all_on_prepare"
    all_on_manifest, all_on_conditions, all_on_failures = study.prepare_conditions(args(all_on_root, treatment="all_on_rl"))
    study.write_outputs(all_on_root, all_on_manifest, all_on_conditions, all_on_failures)
    assert all_on_manifest["feature_flags"]["pps_rl_enabled"] is True
    assert all_on_manifest["feature_flags"]["rts_rl_enabled"] is True
    assert all_on_manifest["feature_flags"]["charging_enabled"] is True
    assert all_on_manifest["rl_assets"]["pps_station_compatibility"]["status"] == "compatible"
    assert all_on_failures == []
    assert len(all_on_conditions) == 300
    assert_prepare_manifest(all_on_manifest, treatment="all_on_rl")
    print("  [ok] all_on_rl prepare 300-run structure")

    print("capacity study prepare smoke ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
