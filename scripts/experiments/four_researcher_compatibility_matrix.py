#!/usr/bin/env python3
"""Two-treatment RMFS capacity-study preparation and bounded execution.

This keeps the historical four-researcher compatibility script as the study
entry point while resolving the matrix to two explicit treatments:
``all_off`` and ``all_on``.  Prepare-only is the default.  Execution uses the
Stage 1 ``RunSpec`` contract and ``run_specs`` scheduler.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
import random
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from model.layout import Layout
from scripts.data.build_adaptive_hybrid import build as build_adaptive_hybrid
from scripts.data.build_baseline_random import build as build_baseline_random
from src.rmfs.decisions.pps.model_paths import DEFAULT_PPS_MODEL_PATH
from src.rmfs.decisions.pps.runtime import (
    PPS_RL_NUM_STATIONS,
    load_pps_rl_model_strict,
)
from src.rmfs.experiments.identity import short_hash
from src.rmfs.orchestration.local_executor import git_value, load_worker_summary, run_specs
from src.rmfs.orchestration.run_spec import RunSpec
from src.rmfs.rl.rts.training.policy_loader import load_policy_from_checkpoint
from src.rmfs.runtime_io.run_profiles import DEFAULT_RTS_ORDER_RATE_PER_HOUR, TICK_TO_SECOND
from src.rmfs.runtime_io.scenario_bundle import activate_scenario_inputs, read_scenario_inputs


MATRIX_SCHEMA_VERSION = "capacity_study_two_treatment.v1"
ROBOT_COUNTS = (10, 15, 20, 25, 30)
PICKER_COUNTS = (2, 3, 4)
REPLENISHMENT_COUNTS = (1,)
TREATMENTS = ("all_off", "all_on")
OPERATIONAL_METRIC_FIELDS = (
    "orders_completed",
    "average_order_cycle_time",
    "total_energy",
    "stop_and_go_count",
    "turning_count",
)


@dataclass
class FactorResolution:
    ok: bool
    requested: str
    actual: str
    detail: dict[str, Any] = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)


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


def derive_seed(root_seed: int, *parts: Any) -> int:
    payload = json.dumps([int(root_seed), *parts], sort_keys=True, separators=(",", ":"), default=str)
    return int(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8], 16)


def charger_budget(robot_count: int) -> int:
    return int(round(0.6 * int(robot_count)))


def ticks_from_seconds(seconds: float) -> int:
    ticks = int(round(float(seconds) / TICK_TO_SECOND))
    if not math.isclose(ticks * TICK_TO_SECOND, float(seconds), rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"simulated seconds {seconds} is not exactly representable at tick_to_second={TICK_TO_SECOND}")
    return ticks


def resolve_dewa(treatment: str, rts_checkpoint: dict[str, Any] | None) -> FactorResolution:
    if treatment == "all_off":
        return FactorResolution(True, "all_off.dewa", "current_nearest", {
            "rts_policy_mode": "current",
            "rts_rollout_enabled": False,
            "committed_next_reservations_enabled": False,
        })
    failures = [] if rts_checkpoint else ["rts_checkpoint_missing_or_incompatible"]
    return FactorResolution(not failures, "all_on.dewa", "rts_rl_explicit", {
        "rts_policy_mode": "rts_rl_explicit",
        "rts_rollout_enabled": True,
        "committed_next_reservations_enabled": True,
        "rts_policy_checkpoint_dir": None if rts_checkpoint is None else rts_checkpoint["path"],
        "rts_policy_checkpoint_id": None if rts_checkpoint is None else rts_checkpoint["policy_checkpoint_id"],
        "rts_policy_action_mode": "greedy",
        "rts_policy_device": "cpu",
        "rts_zone_ids": ["auto"],
    }, failures)


def resolve_devan(treatment: str, picker_count: int, pps_checkpoint: dict[str, Any] | None) -> FactorResolution:
    if treatment == "all_off":
        return FactorResolution(True, "all_off.devan", "rika_heuristic", {
            "pps_mode": "heuristic",
            "pps_model_path": None,
            "pps_rl_enabled": False,
        })
    failures: list[str] = []
    if pps_checkpoint is None:
        failures.append("pps_checkpoint_missing_or_incompatible")
    if int(picker_count) != int(PPS_RL_NUM_STATIONS):
        failures.append(
            f"pps_fixed_station_count:{PPS_RL_NUM_STATIONS}:requested_pickers:{int(picker_count)}"
        )
    return FactorResolution(not failures, "all_on.devan", "ppo_strict_legacy_checkpoint", {
        "pps_mode": "ppo",
        "pps_model_path": None if pps_checkpoint is None else pps_checkpoint["path"],
        "pps_rl_enabled": True,
        "strict_loading": True,
        "trained_station_count": PPS_RL_NUM_STATIONS,
    }, failures)


def resolve_lukman(treatment: str, bundle: str) -> FactorResolution:
    try:
        canonical, items, pods = read_scenario_inputs(bundle)
        detail = {
            "bundle_name": canonical,
            "items_rows": int(len(items)),
            "pods_rows": int(len(pods)),
            "unique_items": int(items["item_id"].nunique()),
            "unique_pods": int(pods["pod_id"].nunique()),
        }
        return FactorResolution(True, f"{treatment}.lukman", f"bundle:{canonical}", detail)
    except Exception as exc:
        return FactorResolution(False, f"{treatment}.lukman", "unresolved", {
            "bundle_name": bundle,
        }, [f"bundle_resolution_failed:{bundle}:{type(exc).__name__}:{exc}"])


def resolve_salsa(treatment: str, charging_config: dict[str, Any]) -> FactorResolution:
    if treatment == "all_off":
        return FactorResolution(True, "all_off.salsa", "yang_random_baseline", {
            "charging_enabled": True,
            "charging_method": "yang_random_baseline",
            **charging_config,
        })
    return FactorResolution(True, "all_on.salsa", "adaptive_hybrid", {
        "charging_enabled": True,
        "charging_method": "adaptive_hybrid",
        **charging_config,
    })


def discover_rts_checkpoint() -> dict[str, Any] | None:
    candidates = []
    for checkpoint in sorted((REPO_ROOT / "data" / "runtime").rglob("checkpoint")):
        if (checkpoint / "model.pt").exists():
            candidates.append(checkpoint)
    for checkpoint in reversed(candidates):
        try:
            loaded = load_policy_from_checkpoint(checkpoint, device="cpu")
        except Exception:
            continue
        return {
            "path": str(checkpoint),
            "policy_checkpoint_id": loaded.policy_checkpoint_id,
            "model_sha256": sha256_file(checkpoint / "model.pt"),
            "feature_schema_sha256": sha256_file(checkpoint / "feature_schema.json"),
            "action_feature_dim": int(loaded.feature_schema.get("action_feature_dim")),
            "stock_feature_dim": int(loaded.feature_schema.get("stock_feature_dim")),
        }
    return None


def inspect_pps_checkpoint() -> dict[str, Any] | None:
    if not DEFAULT_PPS_MODEL_PATH.exists():
        return None
    model = load_pps_rl_model_strict(DEFAULT_PPS_MODEL_PATH)
    spaces = model.observation_space.spaces
    return {
        "path": str(DEFAULT_PPS_MODEL_PATH),
        "sha256": sha256_file(DEFAULT_PPS_MODEL_PATH),
        "action_space": str(model.action_space),
        "observation_shapes": {key: list(value.shape) for key, value in spaces.items()},
        "trained_station_count": PPS_RL_NUM_STATIONS,
    }


def generate_layout(path: Path, picker_count: int, replenishment_count: int, seed: int) -> None:
    random.seed(int(seed))
    layout = Layout()
    layout.order_picker_total = int(picker_count)
    layout.order_replenishment_total = int(replenishment_count)
    layout.total_charging_stations = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    layout.generate(output_path=str(path))


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


def layout_metadata(layout_path: Path, picker_count: int, replenishment_count: int) -> dict[str, Any]:
    frame = pd.read_csv(layout_path, header=None)
    grid = frame.to_numpy(dtype=int)
    coords = station_coordinates(grid)
    spawn = legal_spawn_cells(grid)
    pod_count = int((grid == 1).sum())
    storage_count = int(((grid == 0) | (grid == 1)).sum())
    if len(coords["picking"]) != int(picker_count):
        raise RuntimeError(f"layout picker count mismatch for {layout_path}")
    if len(coords["replenishment"]) != int(replenishment_count):
        raise RuntimeError(f"layout replenishment count mismatch for {layout_path}")
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


def materialize_input_root(
    output_root: Path,
    treatment: str,
    picker_count: int,
    replenishment_count: int,
    bundle: str,
    seed: int,
) -> dict[str, Any]:
    root = output_root / "inputs" / f"{treatment}__p{picker_count}__q{replenishment_count}"
    root.mkdir(parents=True, exist_ok=True)
    scenario_meta = activate_scenario_inputs(bundle, target_root=root, metadata_path=root / "active_scenario.json", dry_run=False)
    layout_path = root / "generated_pod.csv"
    generate_layout(layout_path, picker_count, replenishment_count, seed)
    meta = {
        "treatment": treatment,
        "input_root": str(root),
        "scenario": scenario_meta,
        "layout": layout_metadata(layout_path, picker_count, replenishment_count),
        "file_digests": file_digest_map(root),
    }
    write_json(root / "source_metadata.json", meta)
    return meta


def build_charging_config(
    output_root: Path,
    treatment: str,
    picker_count: int,
    robot_count: int,
    layout_meta: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    grid = pd.read_csv(layout_meta["layout_path"], header=None).to_numpy(dtype=int)
    budget = charger_budget(robot_count)
    if treatment == "all_off":
        cfg = build_baseline_random(grid.tolist(), num_chargers=budget, seed=seed)
        method = "yang_random_baseline"
        extra = {}
    else:
        cfg, n_picker, n_depot = build_adaptive_hybrid(grid, n_robots=robot_count, rho=0.6)
        method = "adaptive_hybrid"
        extra = {"adaptive_picker_cells": n_picker, "adaptive_depot_cells": n_depot}
    cfg_path = output_root / "charging" / treatment / f"p{picker_count}" / f"robots_{robot_count}.json"
    write_json(cfg_path, cfg)
    return {
        "method": method,
        "robot_count": int(robot_count),
        "charger_budget": budget,
        "positions": cfg.get("charger_positions", []),
        "source_layout_hash": layout_meta["layout_sha256"],
        "config_path": str(cfg_path),
        "config_sha256": sha256_file(cfg_path),
        "generation_seed": int(seed),
        **extra,
    }


def run_prefix(treatment: str, robot_count: int, picker_count: int, replenishment_count: int, replication: int) -> str:
    return f"{treatment}__r{robot_count}__p{picker_count}__q{replenishment_count}__rep{replication:03d}"


def build_run_id(prefix: str, identity: dict[str, Any]) -> str:
    return f"{prefix}__{short_hash(identity)}"


def build_spec(
    condition: dict[str, Any],
    repo_root: Path,
    python_executable: str,
    branch: str,
    commit: str,
) -> RunSpec:
    treatment = condition["treatment"]
    dewa = condition["resolutions"]["dewa"]["detail"]
    devan = condition["resolutions"]["devan"]["detail"]
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
        rts_policy_mode=dewa["rts_policy_mode"],
        rts_rollout_enabled=bool(dewa["rts_rollout_enabled"]),
        rts_zone_ids=dewa.get("rts_zone_ids"),
        rts_policy_checkpoint_dir=dewa.get("rts_policy_checkpoint_dir"),
        rts_policy_checkpoint_id=dewa.get("rts_policy_checkpoint_id"),
        rts_policy_action_mode=dewa.get("rts_policy_action_mode", "sample"),
        rts_policy_device=dewa.get("rts_policy_device", "cpu"),
        rts_state_capture_mode="full" if treatment == "all_on" else "auto",
        committed_next_reservations_enabled=bool(dewa["committed_next_reservations_enabled"]),
        robot_count=int(condition["robot_count"]),
        expected_picking_station_count=int(condition["picker_count"]),
        expected_replenishment_station_count=int(condition["replenishment_count"]),
        pps_mode=devan["pps_mode"],
        pps_model_path=devan.get("pps_model_path"),
        charging_enabled=True,
        charging_config_path=condition["charging"]["config_path"],
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
        order_rate_per_hour=DEFAULT_RTS_ORDER_RATE_PER_HOUR,
        pod_location_mode="randomize_slots",
        pod_location_seed=int(condition["seeds"]["pod_location_seed"]),
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
        "treatment": condition["treatment"],
        "robot_count": condition["robot_count"],
        "picker_count": condition["picker_count"],
        "replenishment_count": condition["replenishment_count"],
        "replication": condition["replication"],
        "root_seed": condition["root_seed"],
        "runtime_status": None if worker_summary is None else worker_summary.get("status"),
        "compatibility_failure_count": len(condition["compatibility_failures"]),
        "compatibility_failures": ";".join(condition["compatibility_failures"]),
        "requested_robot_count": condition["robot_count"],
        "realized_robot_count": None if worker_summary is None else worker_summary.get("realized_robot_count"),
        "expected_picking_station_count": condition["picker_count"],
        "realized_picking_station_count": None if worker_summary is None else worker_summary.get("realized_picking_station_count"),
        "expected_replenishment_station_count": condition["replenishment_count"],
        "realized_replenishment_station_count": None if worker_summary is None else worker_summary.get("realized_replenishment_station_count"),
        "scenario_id": condition["scenario_id"],
        "scenario_hash": condition["scenario_hash"],
        "layout_hash": condition["layout"]["layout_sha256"],
        "charging_method": condition["charging"]["method"],
        "charging_config_hash": condition["charging"]["config_sha256"],
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
    failures = {c["run_id"]: c["compatibility_failures"] for c in conditions if c["compatibility_failures"]}
    write_json(output_root / "compatibility_failures.json", {
        "schema_version": MATRIX_SCHEMA_VERSION,
        "total_failed_conditions": len(failures),
        "condition_failures": failures,
    })
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
    if args.treatment:
        selected = [c for c in selected if c["treatment"] in set(args.treatment)]
    if args.robots:
        selected = [c for c in selected if c["robot_count"] in set(args.robots)]
    if args.pickers:
        selected = [c for c in selected if c["picker_count"] in set(args.pickers)]
    if args.replication:
        selected = [c for c in selected if c["replication"] in set(args.replication)]
    if args.limit_runs is not None:
        selected = selected[: int(args.limit_runs)]
    return selected


def validate_counts(conditions: list[dict[str, Any]]) -> None:
    assert len(conditions) == 600
    assert len({c["run_id"] for c in conditions}) == 600
    assert Counter(c["treatment"] for c in conditions) == {"all_off": 300, "all_on": 300}
    assert Counter(c["robot_count"] for c in conditions) == {10: 120, 15: 120, 20: 120, 25: 120, 30: 120}
    assert Counter(c["picker_count"] for c in conditions) == {2: 200, 3: 200, 4: 200}
    assert Counter(c["replication"] for c in conditions) == {i: 30 for i in range(1, 21)}
    assert {c["replenishment_count"] for c in conditions} == {1}


def prepare_conditions(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    output_root = Path(args.output_root).resolve() if Path(args.output_root).is_absolute() else (REPO_ROOT / args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    branch = git_value(REPO_ROOT, "rev-parse", "--abbrev-ref", "HEAD")
    commit = git_value(REPO_ROOT, "rev-parse", "HEAD")
    ticks = ticks_from_seconds(args.simulated_seconds)
    rts_checkpoint = discover_rts_checkpoint()
    pps_checkpoint = inspect_pps_checkpoint()

    input_roots = {}
    for treatment in TREATMENTS:
        bundle = "cindy_s3" if treatment == "all_off" else "scenario4_sij"
        for picker_count in PICKER_COUNTS:
            seed = derive_seed(args.seed, "layout", picker_count, 1)
            input_roots[(treatment, picker_count, 1)] = materialize_input_root(
                output_root, treatment, picker_count, 1, bundle, seed
            )

    charging_configs = {}
    for treatment in TREATMENTS:
        for picker_count in PICKER_COUNTS:
            layout_meta = input_roots[(treatment, picker_count, 1)]["layout"]
            for robot_count in ROBOT_COUNTS:
                seed = derive_seed(args.seed, "charging", treatment, picker_count, robot_count)
                charging_configs[(treatment, picker_count, robot_count)] = build_charging_config(
                    output_root, treatment, picker_count, robot_count, layout_meta, seed
                )

    conditions = []
    for treatment in TREATMENTS:
        for robot_count in ROBOT_COUNTS:
            for picker_count in PICKER_COUNTS:
                for replenishment_count in REPLENISHMENT_COUNTS:
                    for replication in range(1, int(args.replications) + 1):
                        root_seed = derive_seed(args.seed, "replication", replication)
                        input_meta = input_roots[(treatment, picker_count, replenishment_count)]
                        charging = charging_configs[(treatment, picker_count, robot_count)]
                        dewa = resolve_dewa(treatment, rts_checkpoint)
                        devan = resolve_devan(treatment, picker_count, pps_checkpoint)
                        lukman = resolve_lukman(treatment, input_meta["scenario"]["scenario_name"])
                        salsa = resolve_salsa(treatment, charging)
                        failures = dewa.failures + devan.failures + lukman.failures + salsa.failures
                        scenario_identity = {
                            "treatment": treatment,
                            "scenario": input_meta["scenario"]["scenario_name"],
                            "scenario_files": {
                                key: value["sha256"]
                                for key, value in input_meta["file_digests"].items()
                                if key in {"items.csv", "pods.csv", "raw_order.csv"}
                            },
                        }
                        scenario_id = f"scenario_{short_hash(scenario_identity)}"
                        identity = {
                            "treatment": treatment,
                            "robot_count": robot_count,
                            "picker_count": picker_count,
                            "replenishment_count": replenishment_count,
                            "replication": replication,
                            "root_seed": root_seed,
                            "scenario_identity": scenario_identity,
                            "layout_hash": input_meta["layout"]["layout_sha256"],
                            "charging_config_hash": charging["config_sha256"],
                            "horizon_ticks": ticks,
                            "simulated_seconds": args.simulated_seconds,
                            "rts_checkpoint": None if rts_checkpoint is None else rts_checkpoint["policy_checkpoint_id"],
                            "pps_checkpoint": None if pps_checkpoint is None else pps_checkpoint["sha256"],
                            "branch": branch,
                            "commit": commit,
                        }
                        prefix = run_prefix(treatment, robot_count, picker_count, replenishment_count, replication)
                        run_id = build_run_id(prefix, identity)
                        runtime_root = output_root / "runs" / run_id
                        conditions.append({
                            "run_id": run_id,
                            "run_prefix": prefix,
                            "experiment_id": "capacity_study_two_treatment",
                            "treatment": treatment,
                            "robot_count": robot_count,
                            "picker_count": picker_count,
                            "replenishment_count": replenishment_count,
                            "replication": replication,
                            "root_seed": root_seed,
                            "seeds": {
                                "replication_seed": root_seed,
                                "pod_location_seed": derive_seed(root_seed, "pod_location", robot_count, picker_count),
                                "rts_random_seed": derive_seed(root_seed, "rts", treatment),
                            },
                            "ticks": ticks,
                            "simulated_seconds": args.simulated_seconds,
                            "input_root": input_meta["input_root"],
                            "runtime_root": str(runtime_root),
                            "scenario_id": scenario_id,
                            "scenario_hash": short_hash(scenario_identity),
                            "layout": input_meta["layout"],
                            "charging": charging,
                            "resolutions": {
                                "dewa": dewa.__dict__,
                                "devan": devan.__dict__,
                                "lukman": lukman.__dict__,
                                "salsa": salsa.__dict__,
                            },
                            "compatibility_failures": failures,
                            "identity": identity,
                            "spec_timestamp": "capacity_study_two_treatment.v1",
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
        "order_rate_per_hour": DEFAULT_RTS_ORDER_RATE_PER_HOUR,
        "intended_max_workers": int(args.max_workers),
        "treatments": list(TREATMENTS),
        "robot_counts": list(ROBOT_COUNTS),
        "picker_counts": list(PICKER_COUNTS),
        "replenishment_counts": list(REPLENISHMENT_COUNTS),
        "replications": int(args.replications),
        "total_runs": len(conditions),
        "input_roots": [input_roots[key] for key in sorted(input_roots)],
        "charging_configs": [charging_configs[key] for key in sorted(charging_configs)],
        "rts_checkpoint": rts_checkpoint,
        "pps_checkpoint": pps_checkpoint,
        "pps_compatibility_note": (
            f"Legacy PPO checkpoint is fixed to {PPS_RL_NUM_STATIONS} picking stations; "
            "all_on 2-picker and 4-picker conditions are incompatible unless PPS is explicitly adapted."
        ),
        "conditions": conditions,
    }
    return manifest, conditions


def execute_selected(args: argparse.Namespace, manifest: dict[str, Any], conditions: list[dict[str, Any]]) -> None:
    selected = apply_filters(conditions, args)
    incompatible = [c for c in selected if c["compatibility_failures"]]
    if incompatible:
        raise SystemExit(
            "selected runs include incompatible conditions; see compatibility_failures.json. "
            "Use filters to select compatible runs only."
        )
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
        run_specs(specs, max_workers=int(args.max_workers), progress=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare or run the two-treatment RMFS capacity study.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare-only", action="store_true", default=False)
    mode.add_argument("--execute", action="store_true", default=False)
    parser.add_argument("--max-workers", type=int, default=16)
    parser.add_argument("--replications", type=int, default=20)
    parser.add_argument("--simulated-seconds", type=float, default=87000.0)
    parser.add_argument("--output-root", default="data/runtime/capacity_study_two_treatment")
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--limit-runs", type=int, default=None)
    parser.add_argument("--treatment", action="append", choices=TREATMENTS)
    parser.add_argument("--robots", action="append", type=int, choices=ROBOT_COUNTS)
    parser.add_argument("--pickers", action="append", type=int, choices=PICKER_COUNTS)
    parser.add_argument("--replication", action="append", type=int)
    parser.add_argument("--seed", type=int, default=42)
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
        f"[capacity] wrote 600-run prepare manifest under {manifest['output_root']} "
        f"(ticks={manifest['ticks']}, incompatible={sum(1 for c in conditions if c['compatibility_failures'])})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
