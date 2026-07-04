#!/usr/bin/env python3
"""Bounded validation for the capacity study (robot count × order rate)."""

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


def args(root: Path, *, execute: bool = False, seconds: float = 87000.0, **filters):
    return SimpleNamespace(
        prepare_only=not execute,
        execute=execute,
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
        progress=False,
    )


def assert_prepare_manifest(manifest: dict) -> None:
    conditions = manifest["conditions"]
    assert len(conditions) == 300, f"expected 300, got {len(conditions)}"
    assert len({c["run_id"] for c in conditions}) == 300, "duplicate run_ids"
    assert manifest["ticks"] == 580000, f"expected 580000 ticks, got {manifest['ticks']}"
    assert manifest["simulated_seconds"] == 87000.0
    assert manifest["tick_to_second"] == 0.15

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

    # 1. Prepare-only: verify 300-run matrix structure
    prepare_root = base / "prepare"
    prepare_args = args(prepare_root)
    prepare_manifest, prepare_conditions = study.prepare_conditions(prepare_args)
    study.write_outputs(prepare_root, prepare_manifest, prepare_conditions)
    assert_prepare_manifest(prepare_manifest)
    print("  [ok] prepare manifest 300-run structure")

    # 2. Idempotent preparation
    prepare_root_2 = base / "prepare_again"
    prepare_again, prepare_again_conditions = study.prepare_conditions(args(prepare_root_2))
    assert [c["run_id"] for c in prepare_conditions] == [c["run_id"] for c in prepare_again_conditions]
    print("  [ok] idempotent preparation")

    # 3. Bounded execution: 10 robots, 400/hr, 1 replication (3 backend steps)
    exec_root = base / "exec_bounded"
    exec_args = args(exec_root, execute=True, seconds=0.45)
    exec_manifest, exec_conditions = study.prepare_conditions(exec_args)

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
    study.write_outputs(exec_root, exec_manifest, exec_conditions)
    s1 = first_worker_summary(exec_root)
    assert s1["status"] == "success"
    assert s1["realized_robot_count"] == 10
    assert s1["realized_picking_station_count"] == 3
    assert s1["realized_replenishment_station_count"] == 1
    assert s1["pps_mode"] == "heuristic"
    assert s1.get("rts_policy_mode", "current") == "current"
    print("  [ok] bounded run: 10 robots, 400/hr")

    # 4. Bounded execution: 20 robots, 500/hr
    run_20_500 = args(
        exec_root,
        execute=True,
        seconds=0.45,
        robots=[20],
        order_rate=[500],
        replication=[1],
        limit_runs=1,
    )
    study.execute_selected(run_20_500, exec_manifest, exec_conditions)
    study.write_outputs(exec_root, exec_manifest, exec_conditions)
    summaries_20 = sorted((exec_root / "runs").glob("*r20*arr500*rep001*/worker_summary.json"))
    assert summaries_20, "no 20-robot 500/hr summary"
    s2 = json.loads(summaries_20[0].read_text(encoding="utf-8"))
    assert s2["status"] == "success"
    assert s2["realized_robot_count"] == 20
    print("  [ok] bounded run: 20 robots, 500/hr")

    # 5. Bounded execution: 30 robots, 600/hr
    run_30_600 = args(
        exec_root,
        execute=True,
        seconds=0.45,
        robots=[30],
        order_rate=[600],
        replication=[1],
        limit_runs=1,
    )
    study.execute_selected(run_30_600, exec_manifest, exec_conditions)
    study.write_outputs(exec_root, exec_manifest, exec_conditions)
    summaries_30 = sorted((exec_root / "runs").glob("*r30*arr600*rep001*/worker_summary.json"))
    assert summaries_30, "no 30-robot 600/hr summary"
    s3 = json.loads(summaries_30[0].read_text(encoding="utf-8"))
    assert s3["status"] == "success"
    assert s3["realized_robot_count"] == 30
    print("  [ok] bounded run: 30 robots, 600/hr")

    # 6. Two concurrent bounded workers have isolated runtime roots
    concurrent_args = args(
        exec_root,
        execute=True,
        seconds=0.45,
        robots=[10],
        order_rate=[400],
        limit_runs=2,
        max_workers=2,
    )
    study.execute_selected(concurrent_args, exec_manifest, exec_conditions)
    study.write_outputs(exec_root, exec_manifest, exec_conditions)
    runtime_roots = [
        json.loads(path.read_text(encoding="utf-8"))["runtime_root"]
        for path in sorted((exec_root / "runs").glob("*/worker_summary.json"))
    ]
    assert len(set(runtime_roots)) >= 2
    print("  [ok] concurrent workers have isolated roots")

    # 7. Resume skips a matching successful bounded run
    resume_args = args(
        exec_root,
        execute=True,
        seconds=0.45,
        robots=[10],
        order_rate=[400],
        replication=[1],
        limit_runs=1,
        resume=True,
    )
    study.execute_selected(resume_args, exec_manifest, exec_conditions)
    assert len(exec_manifest["execution_selection"]["skipped_successful_resume"]) == 1
    print("  [ok] resume skips matching successful run")

    # 8. Verify input hashes unchanged between preparations
    hash_a = prepare_manifest["input_meta"]["layout"]["layout_sha256"]
    hash_b = prepare_again["input_meta"]["layout"]["layout_sha256"]
    assert hash_a == hash_b, "input hashes changed between preparations"
    print("  [ok] input hashes stable across preparations")

    # 9. Verify order metadata records requested rate
    order_meta_files = sorted((exec_root / "runs").glob("*/generated_order_metadata.json"))
    for meta_path in order_meta_files:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        rate = meta.get("order_rate_per_hour") or meta.get("requested_order_rate_per_hour")
        assert rate is not None and int(rate) in (400, 500, 600), f"unexpected order rate in {meta_path}: {rate}"
    if order_meta_files:
        print(f"  [ok] order metadata validated ({len(order_meta_files)} files)")
    else:
        print("  [skip] no generated_order_metadata.json files (expected for 3-step runs)")

    print("capacity study prepare smoke ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
