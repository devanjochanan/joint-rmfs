#!/usr/bin/env python3
"""RMFS capacity-study: one non-RL treatment, varied robot count and order rate.

Uses the canonical training NetLogo environment (data/input/base) as the fixed
warehouse configuration.  Varies robot count and order arrival rate across
replications.  Reuses existing RunSpec, run_specs scheduler, local_executor,
and worker_summary infrastructure.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.rmfs.experiments.identity import short_hash
from src.rmfs.orchestration.local_executor import git_value, load_worker_summary, run_specs
from src.rmfs.orchestration.run_spec import RunSpec
from src.rmfs.runtime_io.run_profiles import TICK_TO_SECOND

MATRIX_SCHEMA_VERSION = "capacity_study_order_rate.v1"
ROBOT_COUNTS = (10, 15, 20, 25, 30)
ORDER_RATES = (400, 500, 600)
PICKER_COUNT = 3
REPLENISHMENT_COUNT = 1
REPLICATIONS = 20
OPERATIONAL_METRIC_FIELDS = (
    "orders_completed",
    "average_order_cycle_time",
    "total_energy",
    "stop_and_go_count",
    "turning_count",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def file_digest_map(root: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            rel = path.relative_to(root).as_posix()
            out[rel] = {"size": path.stat().st_size, "sha256": sha256_file(path)}
    return out


def ticks_from_seconds(seconds: float) -> int:
    ticks = int(round(float(seconds) / TICK_TO_SECOND))
    if not math.isclose(ticks * TICK_TO_SECOND, float(seconds), rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"simulated seconds {seconds} is not exactly representable at tick_to_second={TICK_TO_SECOND}")
    return ticks


def station_coordinates(grid: np.ndarray) -> dict[str, list[list[int]]]:
    picking = []
    replenishment = []
    for row in range(grid.shape[0]):
        for col in range(grid.shape[1]):
            value = int(grid[row, col])
            if value == 11:
                picking.append([row, col])
            elif value == 21:
                replenishment.append([row, col])
    return {"picking": picking, "replenishment": replenishment}


def legal_spawn_cells(grid: np.ndarray) -> list[list[int]]:
    row_count, col_count = grid.shape
    x_min = min(5, max(0, col_count - 1))
    x_max = max(x_min, col_count - 6)
    cells = []
    blocked_values = {1, 2, 11, 21}
    graph_values = {0, 1, 2, 3, 4, 5, 6, 7, 12, 13, 14, 16, 17, 18, 19, 22, 23, 24, 26, 27, 28, 29}
    for y in range(row_count):
        for x in range(x_min, x_max + 1):
            value = int(grid[y, x])
            if value in graph_values and value not in blocked_values:
                cells.append([y, x])
    return cells


def graph_connected(grid: np.ndarray) -> bool:
    traversable = {0, 1, 2, 3, 4, 5, 6, 7, 12, 13, 14, 16, 17, 18, 19, 22, 23, 24, 26, 27, 28, 29, 99}
    nodes = [(r, c) for r in range(grid.shape[0]) for c in range(grid.shape[1]) if int(grid[r, c]) in traversable]
    if not nodes:
        return False
    seen = {nodes[0]}
    stack = [nodes[0]]
    while stack:
        r, c = stack.pop()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1] and int(grid[nr, nc]) in traversable:
                if (nr, nc) not in seen:
                    seen.add((nr, nc))
                    stack.append((nr, nc))
    return len(seen) == len(nodes)


def layout_metadata(layout_path: Path) -> dict[str, Any]:
    frame = pd.read_csv(layout_path, header=None)
    grid = frame.to_numpy(dtype=int)
    coords = station_coordinates(grid)
    spawn = legal_spawn_cells(grid)
    pod_count = int((grid == 1).sum())
    storage_count = int(((grid == 0) | (grid == 1)).sum())
    if len(coords["picking"]) != PICKER_COUNT:
        raise RuntimeError(f"layout picker count mismatch: expected {PICKER_COUNT}, got {len(coords['picking'])}")
    if len(coords["replenishment"]) != REPLENISHMENT_COUNT:
        raise RuntimeError(f"layout replenishment count mismatch: expected {REPLENISHMENT_COUNT}, got {len(coords['replenishment'])}")
    if pod_count != 121:
        raise RuntimeError(f"layout pod count mismatch: expected 121, got {pod_count}")
    if len(spawn) < max(ROBOT_COUNTS):
        raise RuntimeError(f"layout has only {len(spawn)} legal spawn cells; need {max(ROBOT_COUNTS)}")
    return {
        "layout_path": str(layout_path),
        "layout_sha256": sha256_file(layout_path),
        "dimensions": [int(grid.shape[0]), int(grid.shape[1])],
        "picking_station_count": len(coords["picking"]),
        "replenishment_station_count": len(coords["replenishment"]),
        "pod_capacity": pod_count,
        "storage_cell_count": storage_count,
        "station_coordinates": coords,
        "graph_connected": graph_connected(grid),
        "legal_robot_spawn_cell_count": len(spawn),
    }


def validate_input_root(input_root: Path) -> dict[str, Any]:
    required = ["generated_pod.csv", "items.csv", "pods.csv", "raw_order.csv"]
    for name in required:
        if not (input_root / name).exists():
            raise RuntimeError(f"input root missing required file: {input_root / name}")
    layout_meta = layout_metadata(input_root / "generated_pod.csv")
    return {
        "input_root": str(input_root),
        "layout": layout_meta,
        "file_digests": file_digest_map(input_root),
    }


def run_prefix(robot_count: int, order_rate: int, replication: int) -> str:
    return f"all_off__r{robot_count}__arr{order_rate}__rep{replication:03d}"


def build_run_id(prefix: str, identity: dict[str, Any]) -> str:
    return f"{prefix}__{short_hash(identity)}"


def build_spec(
    condition: dict[str, Any],
    repo_root: Path,
    python_executable: str,
    branch: str,
    commit: str,
) -> RunSpec:
    return RunSpec(
        run_id=condition["run_id"],
        ticks=int(condition["ticks"]),
        runtime_root=Path(condition["runtime_root"]),
        repo_root=repo_root,
        input_root=Path(condition["input_root"]),
        branch=branch,
        commit=commit,
        python_executable=python_executable,
        timestamp=condition["spec_timestamp"],
        rts_policy_mode="current",
        rts_rollout_enabled=False,
        committed_next_reservations_enabled=False,
        rts_seed_base=int(condition["run_seed"]),
        rts_random_seed=int(condition["run_seed"]),
        rts_state_capture_mode="auto",
        robot_count=int(condition["robot_count"]),
        expected_picking_station_count=PICKER_COUNT,
        expected_replenishment_station_count=REPLENISHMENT_COUNT,
        pps_mode="heuristic",
        pps_model_path=None,
        keep_runtime_artifacts=False,
        detail_db=False,
        timing=False,
        worker_status_cadence=1000,
        run_profile="training",
        run_horizon_ticks=int(condition["ticks"]),
        demand_horizon_ticks=int(condition["ticks"]) + 1000,
        demand_buffer_ticks=1000,
        order_generation_mode="shuffled_historical_cycle",
        full_raw_order_replay=False,
        order_rate_per_hour=int(condition["order_rate"]),
        pod_location_mode="randomize_slots",
        pod_location_seed=int(condition["run_seed"]),
        experiment_id=condition["experiment_id"],
        scenario_id=condition["scenario_id"],
        artifact_label=condition["run_id"],
        batch_id=1,
        worker_id=int(condition["replication"]),
        robot_task_allocator="legacy_nearest",
        regret_k=None,
        task_allocator_scope="active_job_queue",
        rts_torch_threads=1,
        rts_torch_interop_threads=1,
    )


def existing_success_matches(spec: RunSpec) -> bool:
    spec_path = spec.runtime_root / "run_spec.json"
    summary_path = spec.runtime_root / "worker_summary.json"
    if not spec_path.exists() or not summary_path.exists():
        return False
    try:
        current = spec.to_json_dict()
        previous = json.loads(spec_path.read_text(encoding="utf-8"))
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    finalized = summary.get("finalization", {})
    return previous == current and summary.get("status") == "success" and bool(finalized.get("finalized"))


def condition_summary_row(condition: dict[str, Any], worker_summary: dict[str, Any] | None = None) -> dict[str, Any]:
    metrics = worker_summary.get("final_metrics", {}) if worker_summary else {}
    finalization = worker_summary.get("finalization", {}) if worker_summary else {}
    row = {
        "run_id": condition["run_id"],
        "treatment": "all_off",
        "robot_count": condition["robot_count"],
        "order_rate": condition["order_rate"],
        "picker_count": PICKER_COUNT,
        "replenishment_count": REPLENISHMENT_COUNT,
        "replication": condition["replication"],
        "run_seed": condition["run_seed"],
        "runtime_status": None if worker_summary is None else worker_summary.get("status"),
        "requested_robot_count": condition["robot_count"],
        "realized_robot_count": None if worker_summary is None else worker_summary.get("realized_robot_count"),
        "expected_picking_station_count": PICKER_COUNT,
        "realized_picking_station_count": None if worker_summary is None else worker_summary.get("realized_picking_station_count"),
        "expected_replenishment_station_count": REPLENISHMENT_COUNT,
        "realized_replenishment_station_count": None if worker_summary is None else worker_summary.get("realized_replenishment_station_count"),
        "scenario_id": condition["scenario_id"],
        "scenario_hash": condition["scenario_hash"],
        "layout_hash": condition["layout_hash"],
        "ticks_requested": condition["ticks"],
        "ticks_completed": None if worker_summary is None else worker_summary.get("ticks_completed"),
        "completed_simulated_seconds": None if worker_summary is None else worker_summary.get("warehouse_time_elapsed"),
        "termination_reason": finalization.get("reason"),
        "worker_wall_time_elapsed": None if worker_summary is None else worker_summary.get("worker_wall_time_elapsed"),
    }
    for name in OPERATIONAL_METRIC_FIELDS:
        row[name] = None
        row[f"{name}_available"] = False
    if metrics:
        mapping = {
            "orders_completed": "warehouse_orders_completed",
            "average_order_cycle_time": "warehouse_average_order_cycle_time",
            "total_energy": "total_energy",
            "stop_and_go_count": "stop_and_go",
            "turning_count": "total_turning",
        }
        for out_name, source_name in mapping.items():
            row[out_name] = metrics.get(source_name)
            row[f"{out_name}_available"] = metrics.get(f"{source_name}_available", metrics.get(source_name) is not None)
    return row


def write_outputs(output_root: Path, manifest: dict[str, Any], conditions: list[dict[str, Any]]) -> None:
    write_json(output_root / "matrix_manifest.json", manifest)
    rows = []
    for condition in conditions:
        summary = load_worker_summary(Path(condition["runtime_root"])) if (Path(condition["runtime_root"]) / "worker_summary.json").exists() else None
        rows.append(condition_summary_row(condition, summary))
    write_json(output_root / "condition_summary.json", rows)
    if rows:
        with (output_root / "condition_summary.csv").open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


def apply_filters(conditions: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    selected = conditions
    if args.robots:
        selected = [c for c in selected if c["robot_count"] in set(args.robots)]
    if args.order_rate:
        selected = [c for c in selected if c["order_rate"] in set(args.order_rate)]
    if args.replication:
        selected = [c for c in selected if c["replication"] in set(args.replication)]
    if args.limit_runs is not None:
        selected = selected[: int(args.limit_runs)]
    return selected


def validate_counts(conditions: list[dict[str, Any]]) -> None:
    assert len(conditions) == 300, f"expected 300 conditions, got {len(conditions)}"
    assert len({c["run_id"] for c in conditions}) == 300, "duplicate run_ids"
    assert Counter(c["robot_count"] for c in conditions) == {10: 60, 15: 60, 20: 60, 25: 60, 30: 60}
    assert Counter(c["order_rate"] for c in conditions) == {400: 100, 500: 100, 600: 100}
    assert Counter(c["replication"] for c in conditions) == {i: 15 for i in range(1, 21)}
    assert all(c["picker_count"] == PICKER_COUNT for c in conditions)
    assert all(c["replenishment_count"] == REPLENISHMENT_COUNT for c in conditions)


def prepare_conditions(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    output_root = Path(args.output_root).resolve() if Path(args.output_root).is_absolute() else (REPO_ROOT / args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    branch = git_value(REPO_ROOT, "rev-parse", "--abbrev-ref", "HEAD")
    commit = git_value(REPO_ROOT, "rev-parse", "HEAD")
    ticks = ticks_from_seconds(args.simulated_seconds)

    input_root = REPO_ROOT / "data" / "input" / "base"
    input_meta = validate_input_root(input_root)
    layout_hash = input_meta["layout"]["layout_sha256"]
    scenario_identity = {
        "input_root": "data/input/base",
        "file_digests": {
            key: value["sha256"]
            for key, value in input_meta["file_digests"].items()
        },
    }
    scenario_hash = short_hash(scenario_identity)
    scenario_id = f"scenario_{scenario_hash}"

    conditions = []
    for robot_count in ROBOT_COUNTS:
        for order_rate in ORDER_RATES:
            for replication in range(1, int(args.replications) + 1):
                run_seed = int(args.seed) + replication - 1
                identity = {
                    "treatment": "all_off",
                    "robot_count": robot_count,
                    "order_rate": order_rate,
                    "picker_count": PICKER_COUNT,
                    "replenishment_count": REPLENISHMENT_COUNT,
                    "replication": replication,
                    "run_seed": run_seed,
                    "input_hash": scenario_hash,
                    "layout_hash": layout_hash,
                    "horizon_ticks": ticks,
                    "simulated_seconds": args.simulated_seconds,
                    "branch": branch,
                    "commit": commit,
                }
                prefix = run_prefix(robot_count, order_rate, replication)
                run_id = build_run_id(prefix, identity)
                runtime_root = output_root / "runs" / run_id
                conditions.append({
                    "run_id": run_id,
                    "run_prefix": prefix,
                    "experiment_id": "capacity_study_order_rate",
                    "treatment": "all_off",
                    "robot_count": robot_count,
                    "order_rate": order_rate,
                    "picker_count": PICKER_COUNT,
                    "replenishment_count": REPLENISHMENT_COUNT,
                    "replication": replication,
                    "run_seed": run_seed,
                    "ticks": ticks,
                    "simulated_seconds": args.simulated_seconds,
                    "input_root": str(input_root),
                    "runtime_root": str(runtime_root),
                    "scenario_id": scenario_id,
                    "scenario_hash": scenario_hash,
                    "layout_hash": layout_hash,
                    "identity": identity,
                    "spec_timestamp": MATRIX_SCHEMA_VERSION,
                })

    validate_counts(conditions)
    manifest = {
        "schema_version": MATRIX_SCHEMA_VERSION,
        "mode": "prepare_only" if not args.execute else "execute",
        "repo_root": str(REPO_ROOT),
        "output_root": str(output_root),
        "branch": branch,
        "commit": commit,
        "tick_to_second": TICK_TO_SECOND,
        "simulated_seconds": args.simulated_seconds,
        "ticks": ticks,
        "order_rates": list(ORDER_RATES),
        "intended_max_workers": int(args.max_workers),
        "robot_counts": list(ROBOT_COUNTS),
        "picker_count": PICKER_COUNT,
        "replenishment_count": REPLENISHMENT_COUNT,
        "replications": int(args.replications),
        "total_runs": len(conditions),
        "input_root": str(input_root),
        "input_meta": input_meta,
        "conditions": conditions,
    }
    return manifest, conditions


def execute_selected(args: argparse.Namespace, manifest: dict[str, Any], conditions: list[dict[str, Any]]) -> None:
    selected = apply_filters(conditions, args)
    specs = []
    skipped = []
    for condition in selected:
        spec = build_spec(condition, REPO_ROOT, sys.executable, manifest["branch"], manifest["commit"])
        if args.resume and existing_success_matches(spec):
            skipped.append(spec.run_id)
            continue
        specs.append(spec)
    manifest["execution_selection"] = {
        "selected_runs": len(selected),
        "skipped_successful_resume": skipped,
        "launched_runs": [spec.run_id for spec in specs],
    }
    if specs:
        run_specs(specs, max_workers=int(args.max_workers), progress=bool(args.progress))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare or run the RMFS capacity study (robot count × order rate).")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare-only", action="store_true", default=False)
    mode.add_argument("--execute", action="store_true", default=False)
    parser.add_argument("--max-workers", type=int, default=16)
    parser.add_argument("--replications", type=int, default=20)
    parser.add_argument("--simulated-seconds", type=float, default=87000.0)
    parser.add_argument("--output-root", default="data/runtime/capacity_study_order_rate")
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--limit-runs", type=int, default=None)
    parser.add_argument("--robots", action="append", type=int, choices=ROBOT_COUNTS)
    parser.add_argument("--order-rate", action="append", type=int, choices=ORDER_RATES)
    parser.add_argument("--replication", action="append", type=int)
    parser.add_argument("--seed", type=int, default=42)
    progress_group = parser.add_mutually_exclusive_group()
    progress_group.add_argument("--progress", action="store_true", default=True)
    progress_group.add_argument("--no-progress", dest="progress", action="store_false")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.prepare_only and not args.execute:
        args.prepare_only = True
    if int(args.replications) != 20:
        raise SystemExit("this study definition requires --replications 20")
    if int(args.max_workers) < 1:
        raise SystemExit("--max-workers must be >= 1")
    manifest, conditions = prepare_conditions(args)
    if args.execute:
        execute_selected(args, manifest, conditions)
    write_outputs(Path(manifest["output_root"]), manifest, conditions)
    print(
        f"[capacity] wrote 300-run prepare manifest under {manifest['output_root']} "
        f"(ticks={manifest['ticks']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
