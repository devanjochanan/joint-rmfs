#!/usr/bin/env python3
"""Bounded validation for the two-treatment capacity-study preparation."""

from __future__ import annotations

import json
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

from scripts.experiments import four_researcher_compatibility_matrix as study


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
        treatment=filters.pop("treatment", None),
        robots=filters.pop("robots", None),
        pickers=filters.pop("pickers", None),
        replication=filters.pop("replication", None),
        seed=42,
    )


def assert_prepare_manifest(manifest: dict) -> None:
    conditions = manifest["conditions"]
    assert len(conditions) == 600
    assert len({c["run_id"] for c in conditions}) == 600
    assert manifest["ticks"] == 580000
    assert manifest["simulated_seconds"] == 87000.0
    assert manifest["tick_to_second"] == 0.15
    assert Counter(c["treatment"] for c in conditions) == {"all_off": 300, "all_on": 300}
    assert Counter(c["robot_count"] for c in conditions) == {10: 120, 15: 120, 20: 120, 25: 120, 30: 120}
    assert Counter(c["picker_count"] for c in conditions) == {2: 200, 3: 200, 4: 200}
    assert Counter(c["replication"] for c in conditions) == {i: 30 for i in range(1, 21)}
    assert {c["replenishment_count"] for c in conditions} == {1}

    by_rep_seed = defaultdict(set)
    for condition in conditions:
        by_rep_seed[condition["replication"]].add(condition["root_seed"])
    assert all(len(seeds) == 1 for seeds in by_rep_seed.values())

    input_roots = manifest["input_roots"]
    assert len(input_roots) == 6
    assert len({entry["layout"]["layout_sha256"] for entry in input_roots}) == 3
    assert all(entry["layout"]["graph_connected"] for entry in input_roots)
    assert all(entry["layout"]["legal_robot_spawn_cell_count"] >= 30 for entry in input_roots)
    assert all("data/input/base" not in entry["input_root"] for entry in input_roots)

    charging = manifest["charging_configs"]
    assert len(charging) == 30
    expected_budgets = {10: 6, 15: 9, 20: 12, 25: 15, 30: 18}
    for cfg in charging:
        assert cfg["charger_budget"] == expected_budgets[cfg["robot_count"]]
        assert Path(cfg["config_path"]).exists()

    failures = [c for c in conditions if c["compatibility_failures"]]
    assert len(failures) == 200
    assert all(c["treatment"] == "all_on" and c["picker_count"] in {2, 4} for c in failures)
    assert manifest["pps_checkpoint"] is not None
    assert manifest["pps_checkpoint"]["trained_station_count"] == 3
    assert manifest["rts_checkpoint"] is not None


def first_worker_summary(root: Path) -> dict:
    summaries = sorted((root / "runs").glob("*/worker_summary.json"))
    assert summaries, f"no worker summary under {root}"
    return json.loads(summaries[0].read_text(encoding="utf-8"))


def write_outputs(root: Path, manifest: dict, conditions: list[dict]) -> None:
    study.write_outputs(root, manifest, conditions)


def main() -> int:
    base = Path(tempfile.mkdtemp(prefix="rmfs_capacity_study_validation_"))

    prepare_root = base / "prepare"
    prepare_args = args(prepare_root)
    prepare_manifest, prepare_conditions = study.prepare_conditions(prepare_args)
    write_outputs(prepare_root, prepare_manifest, prepare_conditions)
    assert_prepare_manifest(prepare_manifest)

    prepare_root_2 = base / "prepare_again"
    prepare_again, prepare_again_conditions = study.prepare_conditions(args(prepare_root_2))
    assert [c["run_id"] for c in prepare_conditions] == [c["run_id"] for c in prepare_again_conditions]

    exec_root = base / "exec_3_steps"
    exec_args = args(exec_root, execute=True, seconds=0.45)
    exec_manifest, exec_conditions = study.prepare_conditions(exec_args)

    off_args = args(
        exec_root,
        execute=True,
        seconds=0.45,
        treatment=["all_off"],
        robots=[10],
        pickers=[2],
        replication=[1],
        limit_runs=1,
    )
    study.execute_selected(off_args, exec_manifest, exec_conditions)
    write_outputs(exec_root, exec_manifest, exec_conditions)
    off_summary = first_worker_summary(exec_root)
    assert off_summary["status"] == "success"
    assert off_summary["realized_robot_count"] == 10
    assert off_summary["realized_picking_station_count"] == 2

    on_args = args(
        exec_root,
        execute=True,
        seconds=0.45,
        treatment=["all_on"],
        robots=[10],
        pickers=[3],
        replication=[1],
        limit_runs=1,
    )
    study.execute_selected(on_args, exec_manifest, exec_conditions)
    write_outputs(exec_root, exec_manifest, exec_conditions)
    summaries = sorted((exec_root / "runs").glob("*/worker_summary.json"))
    on_summary = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in summaries
        if "all_on" in path.as_posix()
    ][0]
    assert on_summary["status"] == "success"
    assert on_summary["pps_mode"] == "ppo"
    assert on_summary["pps_rl_enabled"] is True
    assert on_summary["realized_picking_station_count"] == 3

    concurrent_args = args(
        exec_root,
        execute=True,
        seconds=0.45,
        treatment=["all_off"],
        robots=[10],
        pickers=[2],
        limit_runs=2,
        max_workers=2,
    )
    study.execute_selected(concurrent_args, exec_manifest, exec_conditions)
    write_outputs(exec_root, exec_manifest, exec_conditions)
    runtime_roots = [
        json.loads(path.read_text(encoding="utf-8"))["runtime_root"]
        for path in sorted((exec_root / "runs").glob("*/worker_summary.json"))
    ]
    assert len(set(runtime_roots)) >= 2

    resume_args = args(
        exec_root,
        execute=True,
        seconds=0.45,
        treatment=["all_off"],
        robots=[10],
        pickers=[2],
        replication=[1],
        limit_runs=1,
        resume=True,
    )
    study.execute_selected(resume_args, exec_manifest, exec_conditions)
    assert len(exec_manifest["execution_selection"]["skipped_successful_resume"]) == 1

    blocked_args = args(
        exec_root,
        execute=True,
        seconds=0.45,
        treatment=["all_on"],
        pickers=[2],
        limit_runs=1,
    )
    try:
        study.execute_selected(blocked_args, exec_manifest, exec_conditions)
    except SystemExit as exc:
        assert "incompatible" in str(exc)
    else:
        raise AssertionError("incompatible all_on p2 execution unexpectedly succeeded")

    print("capacity study prepare smoke ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
