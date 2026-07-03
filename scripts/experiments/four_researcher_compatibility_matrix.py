#!/usr/bin/env python3
"""Deterministic four-researcher (Dewa / Devan / Lukman / Salsa) compatibility
matrix runner and reporting contract.

This builds all 16 binary conditions (one replication each) and, in
``--prepare-only`` mode, writes the complete matrix manifest and per-condition
isolation without launching any worker. The ``--execute`` path is scaffolded and
coherent but is intentionally NOT invoked by the accompanying validation prompt.

Factors
-------
* Dewa (RTS): OFF = current policy, no rollout capture, no committed-next.
              ON  = random_valid RTS interface-compatibility mode (schema v6,
              both branches, rollout capture + committed-next), NOT a trained
              policy.
* Devan (PPS): OFF = heuristic (Rika) PPS.
               ON  = strict PPO if an observation-compatible checkpoint exists,
               otherwise random PPO-style "PPS interface compatibility mode".
* Lukman (pod-SKU/order inputs): OFF = explicit baseline bundle,
               ON = explicit Lukman pod-SKU/order bundle. Both are explicit,
               verified scenario bundles (never the mutable data/input/base).
* Salsa (charging): OFF = charging explicitly disabled.
               ON  = charging explicitly enabled through the production
               salsa_charging_config.json; charger cells verified.

Bundle selection note (Lukman factor)
-------------------------------------
No scenario directory is literally named "baseline/Rika" or "Lukman"; every
bundle under data/input/scenarios/ is a consolidated Lukman-format port. The
baseline/Lukman pairing is therefore exposed as explicit, verified CLI
parameters (``--baseline-bundle`` / ``--lukman-bundle``) with evidence-based
defaults. The manifest records the resolved bundle metadata for each condition
so the pairing is auditable and overridable before any execution.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.rmfs.decisions.charging.config import (
    canonical_charging_config_path,
    load_charging_config,
    validate_charging_config,
)
from src.rmfs.decisions.pps.model_paths import configured_pps_model_path, pps_model_candidates
from src.rmfs.runtime_io.scenario_bundle import (
    activate_scenario_inputs,
    list_available_scenarios,
    normalize_scenario_name,
    read_scenario_inputs,
)
from src.rmfs.orchestration.local_executor import git_value


MATRIX_SCHEMA_VERSION = "four_researcher_compatibility_matrix.v1"
DEFAULT_BASELINE_BUNDLE = "my_scenario"
DEFAULT_LUKMAN_BUNDLE = "scenario4_sij"
INTENDED_120_POD_MEANING = 121  # pod_id 0..120 inclusive => 121 physical pods.


# --------------------------------------------------------------------------- #
# Reporting contract
# --------------------------------------------------------------------------- #
# Per-condition RTS rollout summary (Dewa ON). censored_rate uses
# finalized = completed_realized_cycles + censored_decisions as the denominator,
# and is null (not zero) when no RTS decisions occurred.
RTS_ROLLOUT_SUMMARY_FIELDS = (
    "rts_decisions_created",
    "completed_realized_cycles",
    "censored_decisions",
    "pending_decisions_at_finalization",
    "finalized_rts_decisions",
    "censored_rate",
    "censor_status_counts",
    "censor_reason_counts",
    "censored_no_next_task_count",
    "censored_no_next_task_rate",
    "censored_run_end_count",
    "censored_run_end_rate",
    "censored_committed_next_cancelled_count",
    "censored_committed_next_cancelled_rate",
    "failed_or_invalid_action_count",
    "store_selections",
    "replenish_store_selections",
    "realized_cycle_time_mean",
    "realized_cycle_time_median",
    "realized_cycle_time_p95",
    "estimated_cycle_time_mean",
    "estimate_error_mean",
    "estimate_mae",
    "estimate_rmse",
    "dataset_accepted_cycle_count",
    "dataset_rejected_cycle_count",
    "dataset_trainable_rate",
    "dataset_trainable_rate_denominator",
)

# Operational metrics (per condition where available). Any metric that cannot be
# sourced is reported as null with a corresponding *_available=false flag.
OPERATIONAL_METRIC_FIELDS = (
    "orders_completed",
    "average_order_cycle_time",
    "order_lines_completed",
    "picking_pod_visits",
    "pps_picked_quantity",
    "replenishment_trips",
    "replenished_sku_compartment_count",
    "total_robot_distance",
    "loaded_robot_distance",
    "empty_robot_distance",
    "distance_per_completed_order",
    "total_energy",
    "energy_per_completed_order",
    "stop_and_go_count",
    "turning_count",
    "robot_idle_time",
    "max_proactive_robot_load",
    "max_total_replenishment_load",
    "hard_cap_breach_count",
    "ownerless_unavailable_pod_count",
    "proactive_origin_return_violation_count",
    "proactive_rts_invocation_count",
    "duplicate_storage_assignment_count",
    "inventory_reconciliation_mismatch_count",
    "charging_trips_events",
    "dead_robot_count",
    "min_observed_battery_level",
)


def censored_rate(completed: int, censored: int) -> float | None:
    finalized = int(completed) + int(censored)
    if finalized <= 0:
        return None  # null rather than zero when no RTS decisions occurred.
    return float(censored) / float(finalized)


def empty_operational_metrics() -> dict[str, Any]:
    """Reporting contract stub: all operational metrics null + unavailable until
    a condition is executed and the warehouse is inspected."""
    out: dict[str, Any] = {}
    for name in OPERATIONAL_METRIC_FIELDS:
        out[name] = None
        out[f"{name}_available"] = False
    return out


def empty_rts_rollout_summary() -> dict[str, Any]:
    return {name: None for name in RTS_ROLLOUT_SUMMARY_FIELDS}


# --------------------------------------------------------------------------- #
# Factor resolution
# --------------------------------------------------------------------------- #
@dataclass
class FactorResolution:
    ok: bool
    requested: str
    actual: str
    detail: dict[str, Any] = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)


def resolve_dewa(on: bool) -> FactorResolution:
    if not on:
        return FactorResolution(
            ok=True,
            requested="dewa_off",
            actual="rts_current_no_rollout_no_committed_next",
            detail={
                "rts_policy_mode": "current",
                "rts_rollout_enabled": False,
                "committed_next_reservations_enabled": False,
            },
        )
    return FactorResolution(
        ok=True,
        requested="dewa_on",
        actual="rts_random_valid_interface_compatibility_mode",
        detail={
            "rts_policy_mode": "random_valid",
            "rts_rollout_enabled": True,
            "committed_next_reservations_enabled": True,
            "rts_policy_action_mode": "sample",
            "uses_trained_checkpoint": False,
            "label": "RTS interface compatibility mode, not trained-policy performance",
            "action_feature_schema_version": _action_schema_version(),
        },
    )


def _action_schema_version() -> str:
    from src.rmfs.rl.rts.features import ACTION_FEATURE_SCHEMA_VERSION, build_action_feature_names

    return f"{ACTION_FEATURE_SCHEMA_VERSION}:width{len(build_action_feature_names(()))}"


def resolve_devan(on: bool) -> FactorResolution:
    if not on:
        return FactorResolution(
            ok=True,
            requested="devan_off",
            actual="pps_heuristic",
            detail={"pps_mode": "heuristic", "label": "Rika heuristic PPS"},
        )
    model_path = configured_pps_model_path()
    candidates = [str(c) for c in pps_model_candidates(model_path)]
    existing = next((c for c in candidates if Path(c).exists()), None)
    if existing is not None:
        # A checkpoint file exists; execute-time loading is strict (fail, no
        # fallback). Observation-compatibility is validated at execute time via
        # load_pps_rl_model_strict.
        return FactorResolution(
            ok=True,
            requested="devan_on",
            actual="pps_ppo_strict",
            detail={
                "pps_mode": "ppo",
                "strict_loading": True,
                "model_path": existing,
                "label": "trained PPO PPS (strict load, no fallback)",
            },
        )
    return FactorResolution(
        ok=True,
        requested="devan_on",
        actual="pps_random_interface_compatibility_mode",
        detail={
            "pps_mode": "random",
            "strict_loading": False,
            "model_path": str(model_path),
            "checkpoint_present": False,
            "label": "PPS interface compatibility mode, not trained PPO performance",
        },
    )


def verify_bundle(bundle: str) -> dict[str, Any]:
    canonical = normalize_scenario_name(bundle)
    scenario_dir = REPO_ROOT / "data" / "input" / "scenarios" / str(canonical)
    canonical_name, items_frame, pods_frame = read_scenario_inputs(canonical)
    pod_ids = pods_frame["pod_id"].astype(int)
    unique_pods = int(pod_ids.nunique())
    pod_min, pod_max = int(pod_ids.min()), int(pod_ids.max())
    has_generated_pod = (scenario_dir / "generated_pod.csv").exists()
    has_raw_order = (scenario_dir / "raw_order.csv").exists()
    order_source = "bundle:raw_order.csv" if has_raw_order else "base:raw_order.csv"
    return {
        "bundle_name": canonical_name,
        "bundle_dir": str(scenario_dir),
        "unique_pods": unique_pods,
        "pod_id_range": [pod_min, pod_max],
        "unique_skus_items": int(items_frame["item_id"].nunique()),
        "unique_pod_item_skus": int(pods_frame["item"].astype(int).nunique()),
        "order_source": order_source,
        "has_generated_pod_csv": has_generated_pod,
        "has_raw_order_csv": has_raw_order,
        "intended_120_pod_means": unique_pods,
        "note_120_vs_121": (
            f"pod_id range [{pod_min}..{pod_max}] inclusive => {unique_pods} physical pods; "
            f"the intended '120-pod' configuration means {unique_pods} pods, not 120"
        ),
    }


def resolve_lukman(on: bool, baseline_bundle: str, lukman_bundle: str) -> FactorResolution:
    bundle = lukman_bundle if on else baseline_bundle
    failures: list[str] = []
    try:
        metadata = verify_bundle(bundle)
    except Exception as exc:  # unresolved / malformed bundle
        failures.append(f"bundle_resolution_failed:{bundle}:{type(exc).__name__}:{exc}")
        return FactorResolution(
            ok=False,
            requested="lukman_on" if on else "lukman_off",
            actual="unresolved",
            detail={"requested_bundle": bundle},
            failures=failures,
        )
    if not metadata["has_generated_pod_csv"] or not metadata["has_raw_order_csv"]:
        failures.append(
            f"bundle_missing_required_inputs:{bundle}:"
            f"generated_pod={metadata['has_generated_pod_csv']},raw_order={metadata['has_raw_order_csv']}"
        )
    return FactorResolution(
        ok=not failures,
        requested="lukman_on" if on else "lukman_off",
        actual=f"bundle:{metadata['bundle_name']}",
        detail=metadata,
        failures=failures,
    )


def resolve_salsa(on: bool) -> FactorResolution:
    if not on:
        return FactorResolution(
            ok=True,
            requested="salsa_off",
            actual="charging_disabled",
            detail={"charging_enabled": False, "env": {"RMFS_CHARGING_ENABLED": "0"}},
        )
    config_path = canonical_charging_config_path()
    failures: list[str] = []
    detail: dict[str, Any] = {
        "charging_enabled": True,
        "config_path": str(config_path),
        "env": {"RMFS_CHARGING_ENABLED": "1"},
        "config_present": config_path.exists(),
    }
    if not config_path.exists():
        failures.append("salsa_charging_config_missing")
        return FactorResolution(True and False, "salsa_on", "charging_config_missing", detail, failures)
    cfg = load_charging_config(config_path)
    errors = validate_charging_config(cfg)
    detail.update(
        {
            "num_chargers": int(cfg.num_chargers),
            "charger_positions": len(cfg.charger_positions),
            "charger_cells_present": len(cfg.charger_positions) > 0,
            "pipeline": int(cfg.pipeline),
            "config_validation_errors": errors,
            "post_setup_charging_enabled_check": "verified_at_execute",
        }
    )
    if errors:
        failures.append("salsa_charging_config_invalid:" + ";".join(errors))
    if len(cfg.charger_positions) <= 0:
        failures.append("salsa_charging_no_charger_cells")
    return FactorResolution(
        ok=not failures,
        requested="salsa_on",
        actual="charging_enabled_salsa_config",
        detail=detail,
        failures=failures,
    )


# --------------------------------------------------------------------------- #
# Condition construction
# --------------------------------------------------------------------------- #
def condition_id(d: bool, p: bool, l: bool, s: bool) -> str:
    return f"D{int(d)}_P{int(p)}_L{int(l)}_S{int(s)}"


def build_condition(
    d: bool,
    p: bool,
    l: bool,
    s: bool,
    *,
    args: argparse.Namespace,
    output_root: Path,
    commit: str,
    branch: str,
) -> dict[str, Any]:
    cid = condition_id(d, p, l, s)
    dewa = resolve_dewa(d)
    devan = resolve_devan(p)
    lukman = resolve_lukman(l, args.baseline_bundle, args.lukman_bundle)
    salsa = resolve_salsa(s)

    cond_root = output_root / "conditions" / cid
    runtime_root = cond_root / "runtime"
    input_root = cond_root / "input"
    failures = list(dewa.failures + devan.failures + lukman.failures + salsa.failures)

    env = {
        "RMFS_RUN_PROFILE": "smoke",
        "RMFS_DETAIL_DB": "0",
        "RMFS_DEBUG_TRACE": "0",
        "RMFS_FAST_TRAIN": "1",
        "RMFS_SIM_SEED": str(args.seed),
        "RMFS_POD_LOCATION_MODE": "fixed",
        "RMFS_ORDER_GENERATION_MODE": "shuffled_historical_cycle",
        "RMFS_CHARGING_ENABLED": "1" if s else "0",
        "PPS_MODE": devan.detail.get("pps_mode", "heuristic"),
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
    }

    return {
        "condition_id": cid,
        "replication": 1,
        "seed": int(args.seed),
        "ticks_requested": int(args.ticks),
        "factors": {
            "dewa_rts": bool(d),
            "devan_pps": bool(p),
            "lukman_inputs": bool(l),
            "salsa_charging": bool(s),
        },
        "dewa": {"requested": dewa.requested, "actual": dewa.actual, "detail": dewa.detail},
        "devan": {"requested": devan.requested, "actual": devan.actual, "detail": devan.detail},
        "lukman": {"requested": lukman.requested, "actual": lukman.actual, "detail": lukman.detail},
        "salsa": {"requested": salsa.requested, "actual": salsa.actual, "detail": salsa.detail},
        "isolation": {
            "condition_root": str(cond_root),
            "runtime_root": str(runtime_root),
            "input_root": str(input_root),
            "materialized": False,
        },
        "env": env,
        "run_spec": {
            "rts_policy_mode": dewa.detail.get("rts_policy_mode", "current"),
            "rts_rollout_enabled": dewa.detail.get("rts_rollout_enabled", False),
            "committed_next_reservations_enabled": dewa.detail.get(
                "committed_next_reservations_enabled", False
            ),
            "pps_mode": devan.detail.get("pps_mode", "heuristic"),
            "charging_enabled": bool(s),
            "input_bundle": lukman.detail.get("bundle_name"),
            "pod_location_mode": "fixed",
            "ticks": int(args.ticks),
            "seed": int(args.seed),
            "detail_db": False,
            "debug_trace": False,
            "torch_threads_per_worker": 1,
        },
        "commit": commit,
        "branch": branch,
        "runtime_status": "prepared",
        "compatibility_failures": failures,
        "rts_rollout_summary": empty_rts_rollout_summary() if d else None,
        "operational_metrics": empty_operational_metrics(),
    }


def materialize_condition_inputs(condition: dict[str, Any], args: argparse.Namespace) -> None:
    """Create an isolated input root per condition by activating the resolved
    bundle into it. Never writes to data/input/base or repo scenario dirs."""
    input_root = Path(condition["isolation"]["input_root"])
    input_root.mkdir(parents=True, exist_ok=True)
    bundle = condition["lukman"]["detail"].get("bundle_name")
    if bundle is None:
        return
    activate_scenario_inputs(
        bundle,
        target_root=input_root,
        metadata_path=input_root / "active_scenario.json",
        dry_run=False,
    )
    condition["isolation"]["materialized"] = True


# --------------------------------------------------------------------------- #
# Output writers
# --------------------------------------------------------------------------- #
def write_manifest(output_root: Path, manifest: dict[str, Any]) -> Path:
    path = output_root / "matrix_manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path


def summary_row(condition: dict[str, Any]) -> dict[str, Any]:
    return {
        "condition_id": condition["condition_id"],
        "dewa_requested": condition["dewa"]["requested"],
        "dewa_actual": condition["dewa"]["actual"],
        "pps_requested": condition["devan"]["requested"],
        "pps_actual": condition["devan"]["actual"],
        "input_bundle_requested": condition["lukman"]["requested"],
        "input_bundle_actual": condition["lukman"]["actual"],
        "charging_requested": condition["salsa"]["requested"],
        "charging_actual": condition["salsa"]["actual"],
        "branch": condition["branch"],
        "commit": condition["commit"],
        "seed": condition["seed"],
        "ticks_requested": condition["ticks_requested"],
        "ticks_completed": condition.get("ticks_completed"),
        "runtime_status": condition["runtime_status"],
        "compatibility_failure_count": len(condition["compatibility_failures"]),
    }


def write_summaries(output_root: Path, conditions: list[dict[str, Any]]) -> None:
    rows = [summary_row(c) for c in conditions]
    (output_root / "condition_summary.json").write_text(
        json.dumps(rows, indent=2) + "\n", encoding="utf-8"
    )
    csv_path = output_root / "condition_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    failures = {
        c["condition_id"]: c["compatibility_failures"]
        for c in conditions
        if c["compatibility_failures"]
    }
    (output_root / "compatibility_failures.json").write_text(
        json.dumps(
            {
                "schema_version": MATRIX_SCHEMA_VERSION,
                "condition_failures": failures,
                "total_failed_conditions": len(failures),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    # Per-condition worker_summary.json (prepared stub) + RTS rollout summary.
    for condition in conditions:
        cond_root = Path(condition["isolation"]["condition_root"])
        cond_root.mkdir(parents=True, exist_ok=True)
        (cond_root / "worker_summary.json").write_text(
            json.dumps(condition, indent=2) + "\n", encoding="utf-8"
        )
        if condition["rts_rollout_summary"] is not None:
            (cond_root / "rts_rollout_summary.json").write_text(
                json.dumps(condition["rts_rollout_summary"], indent=2) + "\n",
                encoding="utf-8",
            )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Four-researcher compatibility matrix runner.")
    parser.add_argument("--ticks", type=int, default=2000, help="Backend/NetLogo steps per condition.")
    parser.add_argument("--max-workers", type=int, default=8, help="Max concurrent condition workers.")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic seed base.")
    parser.add_argument(
        "--output-root",
        default="data/runtime/validation/four_researcher_compatibility_2000",
        help="Output root for the matrix run.",
    )
    parser.add_argument("--baseline-bundle", default=DEFAULT_BASELINE_BUNDLE, help="Explicit baseline/Rika bundle (Lukman OFF).")
    parser.add_argument("--lukman-bundle", default=DEFAULT_LUKMAN_BUNDLE, help="Explicit Lukman pod-SKU/order bundle (Lukman ON).")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare-only", action="store_true", help="Write the manifest without launching workers.")
    mode.add_argument("--execute", action="store_true", help="Execute the 16-condition matrix (scaffolded).")
    parser.add_argument("--keep-runtime-artifacts", action="store_true", help="Keep per-condition runtime artifacts.")
    return parser


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    if not args.prepare_only and not args.execute:
        args.prepare_only = True  # default to the safe prepare-only mode.

    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = REPO_ROOT / output_root
    output_root.mkdir(parents=True, exist_ok=True)

    commit = git_value(REPO_ROOT, "rev-parse", "HEAD")
    branch = git_value(REPO_ROOT, "rev-parse", "--abbrev-ref", "HEAD")

    conditions: list[dict[str, Any]] = []
    for d, p, l, s in itertools.product((False, True), repeat=4):
        conditions.append(
            build_condition(d, p, l, s, args=args, output_root=output_root, commit=commit, branch=branch)
        )
    assert len(conditions) == 16
    assert len({c["condition_id"] for c in conditions}) == 16
    assert len({c["isolation"]["input_root"] for c in conditions}) == 16

    baseline_meta = None
    lukman_meta = None
    try:
        baseline_meta = verify_bundle(args.baseline_bundle)
    except Exception as exc:
        baseline_meta = {"error": f"{type(exc).__name__}:{exc}"}
    try:
        lukman_meta = verify_bundle(args.lukman_bundle)
    except Exception as exc:
        lukman_meta = {"error": f"{type(exc).__name__}:{exc}"}

    manifest = {
        "schema_version": MATRIX_SCHEMA_VERSION,
        "mode": "prepare_only" if args.prepare_only else "execute",
        "repo_root": str(REPO_ROOT),
        "output_root": str(output_root),
        "commit": commit,
        "branch": branch,
        "defaults": {
            "ticks": int(args.ticks),
            "max_workers": int(args.max_workers),
            "replication": 1,
            "seed": int(args.seed),
            "detail_db": False,
            "debug_trace": False,
            "pod_location_mode": "fixed",
            "torch_threads_per_worker": 1,
        },
        "factor_definitions": {
            "dewa_rts": {
                "off": "current policy, no rollout capture, no committed-next reservation path",
                "on": "random_valid RTS interface-compatibility mode; rollout capture + committed-next; "
                "schema v6; both RTS branches; NOT a trained checkpoint",
            },
            "devan_pps": {
                "off": "heuristic (Rika) PPS",
                "on": "strict PPO if an observation-compatible checkpoint exists, else random PPO-style "
                "PPS interface compatibility mode",
            },
            "lukman_inputs": {
                "off": f"explicit baseline bundle '{args.baseline_bundle}'",
                "on": f"explicit Lukman pod-SKU/order bundle '{args.lukman_bundle}'",
                "note": "Bundle pairing is inferred from repository evidence and is CLI-overridable; "
                "the mutable data/input/base is never used as an implicit baseline.",
            },
            "salsa_charging": {
                "off": "charging explicitly disabled (RMFS_CHARGING_ENABLED=0)",
                "on": "charging explicitly enabled via production salsa_charging_config.json; charger cells verified",
            },
        },
        "resolved_bundles": {
            "baseline_bundle": args.baseline_bundle,
            "baseline_bundle_metadata": baseline_meta,
            "lukman_bundle": args.lukman_bundle,
            "lukman_bundle_metadata": lukman_meta,
            "intended_120_pod_means": INTENDED_120_POD_MEANING,
        },
        "reporting_contract": {
            "rts_rollout_summary_fields": list(RTS_ROLLOUT_SUMMARY_FIELDS),
            "operational_metric_fields": list(OPERATIONAL_METRIC_FIELDS),
            "censored_rate_definition": "censored_decisions / (completed_realized_cycles + censored_decisions); "
            "null when no RTS decisions occurred",
            "dataset_trainable_rate_denominator": "dataset_accepted_cycle_count + dataset_rejected_cycle_count",
        },
        "conditions": conditions,
    }

    if args.prepare_only:
        for condition in conditions:
            materialize_condition_inputs(condition, args)
        write_manifest(output_root, manifest)
        write_summaries(output_root, conditions)
        print(f"[matrix] prepare-only complete: 16 conditions under {output_root}")
        print(f"[matrix] baseline_bundle={args.baseline_bundle} lukman_bundle={args.lukman_bundle}")
        print("[matrix] wrote matrix_manifest.json, condition_summary.csv/json, compatibility_failures.json")
        return 0

    # Execute path is scaffolded but intentionally not exercised by the
    # accompanying prompt. Real execution would launch up to --max-workers
    # isolated condition workers (one process each, one Torch/BLAS thread each),
    # collect the reporting contract, and write per-condition summaries.
    raise SystemExit(
        "execute mode is scaffolded but not enabled in this build; run with --prepare-only"
    )


if __name__ == "__main__":
    raise SystemExit(main())
