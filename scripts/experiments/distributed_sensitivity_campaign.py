#!/usr/bin/env python3
"""Prepare and run the fresh four-PC sensitivity-analysis campaign.

The campaign is intentionally file based: one immutable manifest, one shard per
machine, and the existing local RunSpec executor on each PC.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.experiments.capacity_study_order_rate import (  # noqa: E402
    DEFAULT_CHARGING_CONFIG_PATH,
    DEFAULT_PPS_MODEL_PATH,
    DEFAULT_RTS_CHECKPOINT_DIR,
    ORDER_RATES,
    PICKER_COUNT,
    REPLENISHMENT_COUNT,
    ROBOT_COUNTS,
    feature_flags_for_treatment,
    sha256_file,
    ticks_from_seconds,
    validate_input_root,
)
from src.rmfs.decisions.pps.runtime import PPS_RL_NUM_STATIONS, load_pps_rl_model_strict  # noqa: E402
from src.rmfs.experiments.identity import short_hash  # noqa: E402
from src.rmfs.orchestration.local_executor import (  # noqa: E402
    SENSITIVITY_KPI_SCHEMA_VERSION,
    git_value,
    load_worker_summary,
    reclaim_completed_run_artifacts,
    run_specs,
)
from src.rmfs.orchestration.run_spec import RunSpec  # noqa: E402
from src.rmfs.rl.rts.training.checkpoint import resolve_policy_checkpoint_id  # noqa: E402
from src.rmfs.rl.rts.training.policy_loader import load_policy_from_checkpoint  # noqa: E402
from src.rmfs.runtime_io.run_profiles import TICK_TO_SECOND  # noqa: E402

CAMPAIGN_SCHEMA_VERSION = "distributed_sensitivity_campaign.v1"
POLICY_CONFIGURATIONS = ("all_off", "all_on_rl")
SIMULATED_SECONDS = 87_000.0
BACKEND_STEPS_PER_RUN = 580_000
SEED_BASE_MINUS_ONE = 41
DEFAULT_RL_OVERHEAD_MULTIPLIER = 1.35
OUTPUT_ROOT_RELATIVE = Path("data/runtime/distributed_sensitivity")
INPUT_ROOT_RELATIVE = Path("data/input/base")
ARCHIVED_ROOTS_IGNORED = (
    "data/runtime/capacity_study_order_rate",
    "data/runtime/capacity_study_order_rate_packB",
    "data/runtime/capacity_study_order_rate_packC",
)
STAGE_QUOTAS = {
    1: {"win_lukman": 9, "win_admin": 8, "citi_angiebow": 7, "codex_local": 6},
    2: {"win_lukman": 11, "win_admin": 10, "citi_angiebow": 9, "codex_local": 8},
    3: {"win_lukman": 150, "win_admin": 143, "citi_angiebow": 131, "codex_local": 108},
    4: {"win_lukman": 253, "win_admin": 242, "citi_angiebow": 222, "codex_local": 183},
}
STAGE_NAMES = {
    1: "full_matrix_replication_1",
    2: "central_comparison_to_20_replications",
    3: "full_matrix_to_20_replications",
    4: "full_matrix_to_50_replications",
}
CENTRAL_ROBOT_COUNT = 20
CENTRAL_ORDER_RATE = 500
DEFAULT_LOCAL_PYTHON = Path("/home/dewan/torch-gpu/bin/python")


@dataclass(frozen=True)
class Machine:
    machine_id: str
    os: str
    repository: str
    python: str
    max_workers: int
    effective_steps_per_second: float
    anydesk_id: str | None = None


@dataclass(frozen=True)
class AssetBundle:
    pps_model_relative_path: str
    pps_model_sha256: str
    pps_observation_schema: dict[str, Any]
    rts_checkpoint_relative_dir: str
    rts_checkpoint_id: str
    rts_model_sha256: str
    rts_metadata_sha256: str
    rts_feature_schema_sha256: str
    rts_feature_schema_id: str
    rts_training_artifact: str
    rts_training_latest_relative_path: str
    rts_lineage: dict[str, Any]
    rts_lineage_source_relative_dir: str | None
    charging_config_relative_path: str
    charging_config_sha256: str


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def relative_to_repo(path: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError as exc:
        raise ValueError(f"path must be inside repository for portable shards: {resolved}") from exc


def git_clean_value(*args: str) -> str | None:
    return git_value(REPO_ROOT, *args)


def linux_physical_core_count() -> int | None:
    cpuinfo = Path("/proc/cpuinfo")
    if not cpuinfo.exists():
        return None
    physical_pairs = set()
    current: dict[str, str] = {}
    for line in cpuinfo.read_text(errors="ignore").splitlines():
        if not line.strip():
            if "physical id" in current and "core id" in current:
                physical_pairs.add((current["physical id"], current["core id"]))
            current = {}
            continue
        if ":" in line:
            key, value = line.split(":", 1)
            current[key.strip()] = value.strip()
    if "physical id" in current and "core id" in current:
        physical_pairs.add((current["physical id"], current["core id"]))
    return len(physical_pairs) or None


def local_python_executable() -> str:
    if DEFAULT_LOCAL_PYTHON.exists():
        return str(DEFAULT_LOCAL_PYTHON)
    return sys.executable


def local_max_workers() -> int:
    physical = linux_physical_core_count()
    logical = os.cpu_count() or 1
    usable = physical or logical
    return max(1, min(8, int(usable)))


def default_machines(repo_root: Path = REPO_ROOT) -> list[Machine]:
    return [
        Machine(
            machine_id="win_lukman",
            anydesk_id="1903438276",
            os="windows",
            repository=r"D:\lukman-rmfs\Combinatrix",
            python=r"D:\lukman-rmfs\.rmfs\Scripts\python.exe",
            max_workers=8,
            effective_steps_per_second=414.64,
        ),
        Machine(
            machine_id="win_admin",
            anydesk_id="1052269911",
            os="windows",
            repository=r"C:\Users\admin\Documents\Dewa's Sandbox\netlogo-rmfs",
            python=r"C:\Users\admin\Documents\Dewa's Sandbox\torch-gpu\Scripts\python.exe",
            max_workers=8,
            effective_steps_per_second=395.66,
        ),
        Machine(
            machine_id="citi_angiebow",
            os="linux",
            repository="/home/citi/Documents/Dewa's Sandbox/netlogo-rmfs",
            python="/home/citi/Documents/Dewa's Sandbox/torch-gpu/bin/python",
            max_workers=4,
            effective_steps_per_second=363.25,
        ),
        Machine(
            machine_id="codex_local",
            os=platform.system().lower() or "auto",
            repository=str(repo_root),
            python=local_python_executable(),
            max_workers=local_max_workers(),
            effective_steps_per_second=300.0,
        ),
    ]


def hash_machine_config(machines: list[Machine]) -> str:
    payload = [asdict(machine) for machine in machines]
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:12]


def load_feature_schema_id(schema_path: Path) -> str:
    schema = read_json(schema_path)
    feature_schema_id = schema.get("feature_schema_id")
    if not feature_schema_id:
        raise RuntimeError(f"RTS feature schema missing feature_schema_id: {schema_path}")
    return str(feature_schema_id)


def _lineage_from_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    lineage = dict(metadata.get("lineage", {}) or {})
    for key in ("initialization_method", "teacher_policy"):
        if key in metadata and key not in lineage:
            lineage[key] = metadata[key]
    return lineage


def _is_required_vrsla_lineage(lineage: dict[str, Any]) -> bool:
    return (
        lineage.get("initialization_method") == "vrsla_behavior_cloning"
        and lineage.get("teacher_policy") == "vrsla_event_driven"
    )


def _first_checkpoint_predecessor(training_root: Path) -> str | None:
    history = training_root / "checkpoint_history.jsonl"
    if not history.exists():
        return None
    for line in history.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        return row.get("checkpoint_id_before")
    return None


def _find_vrsla_lineage_source(
    *,
    checkpoint_id: str | None,
    selected_training_root: Path,
) -> tuple[dict[str, Any], Path | None]:
    if not checkpoint_id:
        return {}, None
    for metadata_path in sorted((REPO_ROOT / "data/runtime/rts_training").glob("*/batch_*/checkpoint/metadata.json")):
        checkpoint_dir = metadata_path.parent
        if selected_training_root in checkpoint_dir.parents:
            continue
        metadata = read_json(metadata_path)
        if str(metadata.get("policy_checkpoint_id") or checkpoint_dir.parent.name) != str(checkpoint_id):
            continue
        lineage = _lineage_from_metadata(metadata)
        if _is_required_vrsla_lineage(lineage):
            return lineage, checkpoint_dir
    return {}, None


def verify_rts_lineage(checkpoint_dir: Path, training_root: Path) -> tuple[dict[str, Any], Path | None]:
    metadata = read_json(checkpoint_dir / "metadata.json")
    direct = _lineage_from_metadata(metadata)
    if _is_required_vrsla_lineage(direct):
        return direct, checkpoint_dir
    predecessor = _first_checkpoint_predecessor(training_root)
    traced, source = _find_vrsla_lineage_source(
        checkpoint_id=predecessor,
        selected_training_root=training_root,
    )
    if _is_required_vrsla_lineage(traced):
        lineage = {
            **traced,
            "lineage_trace": {
                "selected_training_artifact": training_root.name,
                "first_ppo_checkpoint_predecessor": predecessor,
                "source_checkpoint_dir": relative_to_repo(source) if source is not None else None,
            },
        }
        return lineage, source
    raise RuntimeError(
        "RTS checkpoint lineage is not verified as VRSLA teacher initialized; "
        f"checkpoint={checkpoint_dir}"
    )


def latest_json_candidates(training_artifact: str | None = None) -> list[Path]:
    if training_artifact:
        artifact_path = Path(training_artifact)
        if not artifact_path.is_absolute():
            artifact_path = REPO_ROOT / artifact_path
        latest = artifact_path / "latest.json"
        return [latest]
    return sorted(
        (REPO_ROOT / "data/runtime/rts_training").glob("*/latest.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


def resolve_rts_checkpoint(
    *,
    explicit_checkpoint_dir: str | None = None,
    training_artifact: str | None = None,
) -> tuple[Path, Path, dict[str, Any], Path | None]:
    if explicit_checkpoint_dir is None and training_artifact is None:
        explicit_checkpoint_dir = str(DEFAULT_RTS_CHECKPOINT_DIR)
    if explicit_checkpoint_dir:
        checkpoint_dir = Path(explicit_checkpoint_dir)
        if not checkpoint_dir.is_absolute():
            checkpoint_dir = REPO_ROOT / checkpoint_dir
        training_root = checkpoint_dir.parents[1]
        lineage, source = verify_rts_lineage(checkpoint_dir, training_root)
        return checkpoint_dir.resolve(), training_root.resolve(), lineage, source
    errors = []
    for latest_path in latest_json_candidates(training_artifact):
        try:
            if not latest_path.exists():
                raise FileNotFoundError(latest_path)
            latest = read_json(latest_path)
            checkpoint_dir = Path(latest["checkpoint_dir"])
            if not checkpoint_dir.is_absolute():
                checkpoint_dir = (latest_path.parent / checkpoint_dir).resolve()
            training_root = latest_path.parent
            loaded = load_policy_from_checkpoint(checkpoint_dir, device="cpu")
            ppo_update = loaded.metadata.get("ppo_update_result", {}) or {}
            if "behavior_cloning" in ppo_update:
                raise RuntimeError("latest checkpoint is behavior cloning, not PPO")
            lineage, source = verify_rts_lineage(checkpoint_dir, training_root)
            return checkpoint_dir.resolve(), training_root.resolve(), lineage, source
        except Exception as exc:
            errors.append(f"{latest_path}: {type(exc).__name__}: {exc}")
    raise RuntimeError("no VRSLA-lineage RTS PPO checkpoint could be resolved:\n" + "\n".join(errors))


def pps_observation_schema(model: object) -> dict[str, Any]:
    spaces = getattr(getattr(model, "observation_space", None), "spaces", {}) or {}
    return {
        name: {
            "shape": list(getattr(space, "shape", ()) or ()),
            "dtype": str(getattr(space, "dtype", "")),
        }
        for name, space in sorted(spaces.items())
    }


def validate_assets_strict(
    repo_root: Path,
    pps_relative_path: str,
    rts_checkpoint_relative_dir: str,
    expected_pps_sha256: str | None = None,
    expected_rts_model_sha256: str | None = None,
    expected_rts_metadata_sha256: str | None = None,
    expected_rts_feature_schema_sha256: str | None = None,
    expected_rts_checkpoint_id: str | None = None,
) -> None:
    # 1. PPS validation
    pps_path = repo_root / pps_relative_path
    if not pps_path.exists():
        raise FileNotFoundError(f"PPS model path does not exist: {pps_path}")
    pps_sha = sha256_file(pps_path)
    if expected_pps_sha256 and pps_sha != expected_pps_sha256:
        raise RuntimeError(f"PPS model hash mismatch: got {pps_sha}, expected {expected_pps_sha256}")
    
    # Strictly load and verify PPS
    load_pps_rl_model_strict(pps_path, expected_sha256=pps_sha)
    if PPS_RL_NUM_STATIONS != PICKER_COUNT:
        raise RuntimeError(f"PPS model expects {PPS_RL_NUM_STATIONS} picking stations, campaign uses {PICKER_COUNT}")

    # 2. RTS validation
    checkpoint_dir = repo_root / rts_checkpoint_relative_dir
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"RTS checkpoint dir does not exist: {checkpoint_dir}")
    
    for name in ("model.pt", "metadata.json", "feature_schema.json"):
        if not (checkpoint_dir / name).exists():
            raise FileNotFoundError(f"missing RTS checkpoint file: {checkpoint_dir / name}")

    # Check RTS hashes
    model_sha = sha256_file(checkpoint_dir / "model.pt")
    metadata_sha = sha256_file(checkpoint_dir / "metadata.json")
    schema_sha = sha256_file(checkpoint_dir / "feature_schema.json")

    if expected_rts_model_sha256 and model_sha != expected_rts_model_sha256:
        raise RuntimeError(f"RTS model.pt hash mismatch: got {model_sha}, expected {expected_rts_model_sha256}")
    if expected_rts_metadata_sha256 and metadata_sha != expected_rts_metadata_sha256:
        raise RuntimeError(f"RTS metadata.json hash mismatch: got {metadata_sha}, expected {expected_rts_metadata_sha256}")
    if expected_rts_feature_schema_sha256 and schema_sha != expected_rts_feature_schema_sha256:
        raise RuntimeError(f"RTS feature_schema.json hash mismatch: got {schema_sha}, expected {expected_rts_feature_schema_sha256}")

    # Load RTS policy to verify feature schema loads successfully
    loaded = load_policy_from_checkpoint(checkpoint_dir, device="cpu")
    checkpoint_id = resolve_policy_checkpoint_id(checkpoint_dir)
    if loaded.policy_checkpoint_id != checkpoint_id:
        raise RuntimeError(f"RTS checkpoint ID mismatch after strict load: loaded={loaded.policy_checkpoint_id}, dir={checkpoint_id}")
    if expected_rts_checkpoint_id and checkpoint_id != expected_rts_checkpoint_id:
        raise RuntimeError(f"RTS checkpoint ID mismatch: got {checkpoint_id}, expected {expected_rts_checkpoint_id}")

    # Verify PPO status (real PPO update, not behavior cloning only)
    metadata = read_json(checkpoint_dir / "metadata.json")
    ppo_update = metadata.get("ppo_update_result", {}) or {}
    if not ppo_update:
        raise RuntimeError(f"RTS checkpoint metadata has no PPO update result: {checkpoint_dir / 'metadata.json'}")
    if "behavior_cloning" in ppo_update:
        raise RuntimeError(f"RTS checkpoint is behavior_cloning-only, not a real PPO update: {checkpoint_dir}")
    if "optimizer_steps" not in ppo_update:
        raise RuntimeError(f"RTS checkpoint has no optimizer steps in PPO update: {checkpoint_dir}")

    # Verify VRSLA lineage
    lineage = _lineage_from_metadata(metadata)
    if not _is_required_vrsla_lineage(lineage):
        training_root = checkpoint_dir.parents[1]
        try:
            lineage, source = verify_rts_lineage(checkpoint_dir, training_root)
        except Exception as exc:
            raise RuntimeError(f"RTS checkpoint lineage is not verified as VRSLA teacher initialized: {exc}") from exc
        if not _is_required_vrsla_lineage(lineage):
            raise RuntimeError(f"RTS checkpoint lineage is not verified as VRSLA teacher initialized: {checkpoint_dir}")


def resolve_assets(args: argparse.Namespace) -> AssetBundle:
    pps_path = Path(args.pps_model_path or DEFAULT_PPS_MODEL_PATH)
    if not pps_path.is_absolute():
        pps_path = REPO_ROOT / pps_path
    
    checkpoint_dir, training_root, lineage, source = resolve_rts_checkpoint(
        explicit_checkpoint_dir=args.rts_checkpoint_dir,
        training_artifact=args.rts_training_artifact,
    )

    charging_path = Path(args.charging_config_path or DEFAULT_CHARGING_CONFIG_PATH)
    if not charging_path.is_absolute():
        charging_path = REPO_ROOT / charging_path

    # Perform strict validation of canonical or overridden assets
    validate_assets_strict(
        repo_root=REPO_ROOT,
        pps_relative_path=relative_to_repo(pps_path),
        rts_checkpoint_relative_dir=relative_to_repo(checkpoint_dir),
    )

    # Re-read metadata to get lineage (in case it was resolved directly or verified via trace)
    metadata = read_json(checkpoint_dir / "metadata.json")
    final_lineage = _lineage_from_metadata(metadata)
    if not _is_required_vrsla_lineage(final_lineage):
        final_lineage, final_source = verify_rts_lineage(checkpoint_dir, training_root)
    else:
        final_source = checkpoint_dir

    latest_rel_path = ""
    if (training_root / "latest.json").exists():
        latest_rel_path = relative_to_repo(training_root / "latest.json")

    return AssetBundle(
        pps_model_relative_path=relative_to_repo(pps_path),
        pps_model_sha256=sha256_file(pps_path),
        pps_observation_schema=pps_observation_schema(load_pps_rl_model_strict(pps_path)),
        rts_checkpoint_relative_dir=relative_to_repo(checkpoint_dir),
        rts_checkpoint_id=resolve_policy_checkpoint_id(checkpoint_dir),
        rts_model_sha256=sha256_file(checkpoint_dir / "model.pt"),
        rts_metadata_sha256=sha256_file(checkpoint_dir / "metadata.json"),
        rts_feature_schema_sha256=sha256_file(checkpoint_dir / "feature_schema.json"),
        rts_feature_schema_id=load_feature_schema_id(checkpoint_dir / "feature_schema.json"),
        rts_training_artifact=training_root.name,
        rts_training_latest_relative_path=latest_rel_path,
        rts_lineage=final_lineage,
        rts_lineage_source_relative_dir=relative_to_repo(final_source) if final_source is not None else None,
        charging_config_relative_path=relative_to_repo(charging_path),
        charging_config_sha256=sha256_file(charging_path),
    )


def seed_for_replication(replication: int) -> int:
    return SEED_BASE_MINUS_ONE + int(replication)


def condition_key(policy: str, robot_count: int, order_rate: int, replication: int) -> str:
    return f"{policy}|robots={robot_count}|order_rate={order_rate}|rep={replication}"


def base_condition_rows(stage: int, already_requested: set[str]) -> list[dict[str, Any]]:
    if stage == 1:
        candidates = [
            (policy, robot_count, order_rate, 1)
            for policy in POLICY_CONFIGURATIONS
            for robot_count in ROBOT_COUNTS
            for order_rate in ORDER_RATES
        ]
    elif stage == 2:
        candidates = [
            (policy, CENTRAL_ROBOT_COUNT, CENTRAL_ORDER_RATE, replication)
            for replication in range(1, 21)
            for policy in POLICY_CONFIGURATIONS
        ]
    elif stage == 3:
        candidates = [
            (policy, robot_count, order_rate, replication)
            for replication in range(1, 21)
            for policy in POLICY_CONFIGURATIONS
            for robot_count in ROBOT_COUNTS
            for order_rate in ORDER_RATES
        ]
    elif stage == 4:
        candidates = [
            (policy, robot_count, order_rate, replication)
            for replication in range(1, 51)
            for policy in POLICY_CONFIGURATIONS
            for robot_count in ROBOT_COUNTS
            for order_rate in ORDER_RATES
        ]
    else:
        raise ValueError(f"unknown stage: {stage}")

    rows = []
    for policy, robot_count, order_rate, replication in candidates:
        key = condition_key(policy, robot_count, order_rate, replication)
        if key in already_requested:
            continue
        rows.append({
            "condition_key": key,
            "policy_configuration": policy,
            "robot_count": int(robot_count),
            "order_rate": int(order_rate),
            "picker_count": PICKER_COUNT,
            "replenishment_count": REPLENISHMENT_COUNT,
            "replication": int(replication),
            "seed": seed_for_replication(replication),
            "stage_first_requested": int(stage),
        })
    return rows


def estimate_condition_steps(row: dict[str, Any], *, rl_overhead_multiplier: float) -> float:
    multiplier = 1.0
    if row["policy_configuration"] == "all_on_rl":
        multiplier *= float(rl_overhead_multiplier)
    multiplier *= 1.0 + 0.015 * ((int(row["robot_count"]) - CENTRAL_ROBOT_COUNT) / 5.0)
    multiplier *= 1.0 + 0.02 * ((int(row["order_rate"]) - CENTRAL_ORDER_RATE) / 100.0)
    return BACKEND_STEPS_PER_RUN * max(0.2, multiplier)


def allocate_stage(
    rows: list[dict[str, Any]],
    *,
    stage: int,
    machines: list[Machine],
    rl_overhead_multiplier: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    quotas = dict(STAGE_QUOTAS[stage])
    if sum(quotas.values()) != len(rows):
        raise RuntimeError(f"stage {stage} quota sum {sum(quotas.values())} != row count {len(rows)}")
    machine_by_id = {machine.machine_id: machine for machine in machines}
    assigned_counts = {machine.machine_id: 0 for machine in machines}
    assigned_steps = {machine.machine_id: 0.0 for machine in machines}
    ordered = sorted(
        rows,
        key=lambda row: (
            -estimate_condition_steps(row, rl_overhead_multiplier=rl_overhead_multiplier),
            row["policy_configuration"],
            -int(row["robot_count"]),
            -int(row["order_rate"]),
            int(row["replication"]),
        ),
    )
    allocated = []
    for row in ordered:
        eligible = [
            machine
            for machine in machines
            if assigned_counts[machine.machine_id] < quotas[machine.machine_id]
        ]
        if not eligible:
            raise RuntimeError(f"stage {stage} has no eligible machine for {row['condition_key']}")
        machine = min(
            eligible,
            key=lambda item: (
                assigned_steps[item.machine_id] / float(item.effective_steps_per_second),
                assigned_counts[item.machine_id],
                item.machine_id,
            ),
        )
        estimated_steps = estimate_condition_steps(row, rl_overhead_multiplier=rl_overhead_multiplier)
        assigned_counts[machine.machine_id] += 1
        assigned_steps[machine.machine_id] += estimated_steps
        allocated.append({
            **row,
            "machine_id": machine.machine_id,
            "estimated_backend_steps": estimated_steps,
            "estimated_cost_model": {
                "base_backend_steps": BACKEND_STEPS_PER_RUN,
                "rl_overhead_multiplier": rl_overhead_multiplier,
            },
        })
    projection = {
        machine_id: assigned_steps[machine_id] / float(machine_by_id[machine_id].effective_steps_per_second)
        for machine_id in assigned_steps
    }
    return allocated, {
        "stage": stage,
        "name": STAGE_NAMES[stage],
        "new_runs": len(allocated),
        "quota": quotas,
        "assigned_counts": assigned_counts,
        "assigned_estimated_steps": assigned_steps,
        "projected_finish_seconds": projection,
    }


def add_run_identity(
    row: dict[str, Any],
    *,
    campaign_id: str,
    branch: str | None,
    commit: str | None,
    ticks: int,
    assets: AssetBundle,
    scenario_hash: str,
    layout_hash: str,
) -> dict[str, Any]:
    identity = {
        "campaign_id": campaign_id,
        "policy_configuration": row["policy_configuration"],
        "robot_count": row["robot_count"],
        "order_rate": row["order_rate"],
        "picking_stations": PICKER_COUNT,
        "replenishment_stations": REPLENISHMENT_COUNT,
        "replication": row["replication"],
        "seed": row["seed"],
        "horizon_ticks": ticks,
        "tick_to_second": TICK_TO_SECOND,
        "scenario_hash": scenario_hash,
        "layout_hash": layout_hash,
        "repo_commit": commit,
        "rts_checkpoint_id": assets.rts_checkpoint_id,
        "rts_checkpoint_sha256": assets.rts_model_sha256,
        "pps_model_sha256": assets.pps_model_sha256,
    }
    run_prefix = (
        f"{row['policy_configuration']}__r{row['robot_count']}__arr{row['order_rate']}"
        f"__rep{int(row['replication']):03d}"
    )
    run_id = f"{run_prefix}__{short_hash(identity)}"
    return {
        **row,
        "run_id": run_id,
        "identity": identity,
        "ticks": ticks,
        "simulated_seconds": SIMULATED_SECONDS,
        "tick_to_second": TICK_TO_SECOND,
        "branch": branch,
        "commit": commit,
        "scenario_id": f"scenario_{scenario_hash}",
        "scenario_hash": scenario_hash,
        "layout_hash": layout_hash,
        "run_spec_identity": {
            "run_id": run_id,
            "campaign_id": campaign_id,
            "machine_id": row["machine_id"],
            "stage_first_requested": row["stage_first_requested"],
            "kpi_schema_version": SENSITIVITY_KPI_SCHEMA_VERSION,
            "policy_configuration": row["policy_configuration"],
            "replication": row["replication"],
            "campaign_seed": row["seed"],
            "rts_checkpoint_sha256": assets.rts_model_sha256,
            "pps_model_sha256": assets.pps_model_sha256,
        },
    }


def generate_campaign_id(
    machines: list[Machine],
    assets: AssetBundle,
    rl_overhead_multiplier: float,
) -> str:
    config = {
        "schema_version": CAMPAIGN_SCHEMA_VERSION,
        "policy_configurations": list(POLICY_CONFIGURATIONS),
        "robot_counts": list(ROBOT_COUNTS),
        "order_rates": list(ORDER_RATES),
        "picker_count": PICKER_COUNT,
        "replenishment_count": REPLENISHMENT_COUNT,
        "simulated_seconds": SIMULATED_SECONDS,
        "seed_base": SEED_BASE_MINUS_ONE,
        "stage_quotas": STAGE_QUOTAS,
        "rl_overhead_multiplier": rl_overhead_multiplier,
        "machines": [asdict(m) for m in sorted(machines, key=lambda m: m.machine_id)],
        "pps_model_sha256": assets.pps_model_sha256,
        "rts_model_sha256": assets.rts_model_sha256,
        "rts_metadata_sha256": assets.rts_metadata_sha256,
        "rts_feature_schema_sha256": assets.rts_feature_schema_sha256,
    }
    config_str = json.dumps(config, sort_keys=True)
    config_hash = hashlib.sha256(config_str.encode("utf-8")).hexdigest()[:12]
    return f"sensitivity_full_kpi_v2_{config_hash}"


def build_campaign_plan(
    *,
    campaign_id: str,
    machines: list[Machine],
    assets: AssetBundle,
    rl_overhead_multiplier: float = DEFAULT_RL_OVERHEAD_MULTIPLIER,
) -> dict[str, Any]:
    ticks = ticks_from_seconds(SIMULATED_SECONDS)
    if ticks != BACKEND_STEPS_PER_RUN:
        raise RuntimeError(f"expected {BACKEND_STEPS_PER_RUN} backend steps, got {ticks}")
    input_meta = validate_input_root(REPO_ROOT / INPUT_ROOT_RELATIVE)
    scenario_hash = short_hash({
        "input_root": INPUT_ROOT_RELATIVE.as_posix(),
        "file_digests": {
            key: value["sha256"]
            for key, value in input_meta["file_digests"].items()
        },
    })
    layout_hash = input_meta["layout"]["layout_sha256"]
    branch = git_clean_value("rev-parse", "--abbrev-ref", "HEAD")
    commit = git_clean_value("rev-parse", "HEAD")
    already_requested: set[str] = set()
    runs: list[dict[str, Any]] = []
    stage_summaries: dict[str, Any] = {}
    for stage in (1, 2, 3, 4):
        rows = base_condition_rows(stage, already_requested)
        allocated, summary = allocate_stage(
            rows,
            stage=stage,
            machines=machines,
            rl_overhead_multiplier=rl_overhead_multiplier,
        )
        stage_runs = [
            add_run_identity(
                row,
                campaign_id=campaign_id,
                branch=branch,
                commit=commit,
                ticks=ticks,
                assets=assets,
                scenario_hash=scenario_hash,
                layout_hash=layout_hash,
            )
            for row in allocated
        ]
        runs.extend(stage_runs)
        stage_summaries[str(stage)] = summary
        already_requested.update(row["condition_key"] for row in rows)
    assertions = dry_run_assertions(runs, machines, stage_summaries)
    return {
        "schema_version": CAMPAIGN_SCHEMA_VERSION,
        "campaign_id": campaign_id,
        "created_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "repo_root_at_prepare": str(REPO_ROOT),
        "branch": branch,
        "commit": commit,
        "python_at_prepare": sys.executable,
        "output_root_relative": OUTPUT_ROOT_RELATIVE.as_posix(),
        "campaign_root_relative": (OUTPUT_ROOT_RELATIVE / campaign_id).as_posix(),
        "input_root_relative": INPUT_ROOT_RELATIVE.as_posix(),
        "archived_roots_ignored": list(ARCHIVED_ROOTS_IGNORED),
        "old_capacity_study_completions_used": 0,
        "tick_to_second": TICK_TO_SECOND,
        "simulated_seconds": SIMULATED_SECONDS,
        "backend_steps_per_full_run": BACKEND_STEPS_PER_RUN,
        "seed_formula": "seed = 41 + replication",
        "policy_configurations": list(POLICY_CONFIGURATIONS),
        "robot_counts": list(ROBOT_COUNTS),
        "order_rates": list(ORDER_RATES),
        "picking_stations": PICKER_COUNT,
        "replenishment_stations": REPLENISHMENT_COUNT,
        "kpi_schema_version": SENSITIVITY_KPI_SCHEMA_VERSION,
        "machines": [asdict(machine) for machine in machines],
        "machine_config_hash": hash_machine_config(machines),
        "machine_rates_used": {
            machine.machine_id: machine.effective_steps_per_second
            for machine in machines
        },
        "assets": asdict(assets),
        "feature_flags": {
            policy: feature_flags_for_treatment(policy)
            for policy in POLICY_CONFIGURATIONS
        },
        "input_meta": input_meta,
        "stages": stage_summaries,
        "runs": runs,
        "assertions": assertions,
    }


def dry_run_assertions(
    runs: list[dict[str, Any]],
    machines: list[Machine],
    stage_summaries: dict[str, Any],
) -> dict[str, Any]:
    machine_ids = [machine.machine_id for machine in machines]
    by_stage = {stage: [run for run in runs if run["stage_first_requested"] == int(stage)] for stage in (1, 2, 3, 4)}
    by_machine_total = {
        machine_id: sum(1 for run in runs if run["machine_id"] == machine_id)
        for machine_id in machine_ids
    }
    all_on_hashes = {
        (run["identity"]["rts_checkpoint_sha256"], run["identity"]["pps_model_sha256"])
        for run in runs
        if run["policy_configuration"] == "all_on_rl"
    }
    assertions = {
        "machine_count": len(machines),
        "stage_new_runs": {str(stage): len(rows) for stage, rows in by_stage.items()},
        "total_unique_fresh_runs": len({run["condition_key"] for run in runs}),
        "stage_allocations": {
            str(stage): {
                machine_id: sum(1 for run in rows if run["machine_id"] == machine_id)
                for machine_id in machine_ids
            }
            for stage, rows in by_stage.items()
        },
        "total_fresh_runs_by_machine": by_machine_total,
        "replication_seeds": {
            "1": seed_for_replication(1),
            "20": seed_for_replication(20),
            "50": seed_for_replication(50),
        },
        "replication_seed_uniformity": {
            str(replication): len({run["seed"] for run in runs if run["replication"] == replication})
            for replication in (1, 20, 50)
        },
        "unique_run_ids": len({run["run_id"] for run in runs}),
        "unique_condition_keys": len({run["condition_key"] for run in runs}),
        "duplicate_run_ids": len(runs) - len({run["run_id"] for run in runs}),
        "duplicate_condition_keys": len(runs) - len({run["condition_key"] for run in runs}),
        "old_capacity_study_roots_contribute_completions": 0,
        "all_on_rl_unique_asset_hash_pairs": len(all_on_hashes),
        "all_on_rl_stage1_machines": sorted({
            run["machine_id"]
            for run in by_stage[1]
            if run["policy_configuration"] == "all_on_rl"
        }),
        "stage_projected_finish_seconds": {
            str(stage): stage_summaries[str(stage)]["projected_finish_seconds"]
            for stage in (1, 2, 3, 4)
        },
    }
    expected_stage_counts = {"1": 30, "2": 38, "3": 532, "4": 900}
    expected_allocations = {str(stage): STAGE_QUOTAS[stage] for stage in (1, 2, 3, 4)}
    if assertions["stage_new_runs"] != expected_stage_counts:
        raise AssertionError(assertions["stage_new_runs"])
    if assertions["stage_allocations"] != expected_allocations:
        raise AssertionError(assertions["stage_allocations"])
    if assertions["total_unique_fresh_runs"] != 1500:
        raise AssertionError(assertions["total_unique_fresh_runs"])
    if assertions["replication_seeds"] != {"1": 42, "20": 61, "50": 91}:
        raise AssertionError(assertions["replication_seeds"])
    if assertions["duplicate_run_ids"] or assertions["duplicate_condition_keys"]:
        raise AssertionError("duplicate campaign identities detected")
    if assertions["all_on_rl_unique_asset_hash_pairs"] != 1:
        raise AssertionError("all_on_rl runs must pin one RTS/PPS hash pair")
    if assertions["all_on_rl_stage1_machines"] != sorted(machine_ids):
        raise AssertionError("Stage 1 all_on_rl is not spread across all machines")
    return assertions


def machine_map(manifest: dict[str, Any]) -> dict[str, Machine]:
    return {row["machine_id"]: Machine(**row) for row in manifest["machines"]}


def campaign_root(repo_root: Path, manifest: dict[str, Any]) -> Path:
    return repo_root / manifest["campaign_root_relative"]


def shard_relative_path(machine_id: str) -> Path:
    return Path("shards") / f"{machine_id}_shard.json"


def write_launchers(root: Path, manifest: dict[str, Any]) -> dict[str, str]:
    launchers: dict[str, str] = {}
    rel_manifest_posix = Path(manifest["campaign_root_relative"]) / "manifest.json"
    for row in manifest["machines"]:
        machine = Machine(**row)
        if machine.os.lower().startswith("win"):
            path = root / "launchers" / f"run_{machine.machine_id}.ps1"
            rel_manifest = str(rel_manifest_posix).replace("/", "\\")
            content = "\n".join([
                "$ErrorActionPreference = \"Stop\"",
                f"Set-Location -LiteralPath \"{machine.repository}\"",
                (
                    f"& \"{machine.python}\" \"scripts\\experiments\\distributed_sensitivity_campaign.py\" "
                    f"--manifest \"{rel_manifest}\" --machine-id \"{machine.machine_id}\" "
                    "--run-continuously --resume --progress"
                ),
                "",
            ])
        else:
            path = root / "launchers" / f"run_{machine.machine_id}.sh"
            content = "\n".join([
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                f"cd {json.dumps(machine.repository)}",
                (
                    f"exec {json.dumps(machine.python)} scripts/experiments/distributed_sensitivity_campaign.py "
                    f"--manifest {json.dumps(rel_manifest_posix.as_posix())} --machine-id {json.dumps(machine.machine_id)} "
                    "--run-continuously --resume --progress"
                ),
                "",
            ])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        if path.suffix == ".sh":
            path.chmod(0o755)
        launchers[machine.machine_id] = path.relative_to(root).as_posix()
    return launchers


def write_campaign_files(manifest: dict[str, Any], *, dry_run: bool = False) -> Path:
    root = campaign_root(REPO_ROOT, manifest)
    if dry_run:
        return root
    if root.exists():
        raise FileExistsError(f"campaign root already exists; refusing to overwrite immutable campaign: {root}")
    root.mkdir(parents=True)
    shard_paths = {}
    for machine in manifest["machines"]:
        machine_id = machine["machine_id"]
        shard_runs = [
            run for run in manifest["runs"]
            if run["machine_id"] == machine_id
        ]
        shard = {
            "schema_version": CAMPAIGN_SCHEMA_VERSION,
            "campaign_id": manifest["campaign_id"],
            "machine_id": machine_id,
            "runs": shard_runs,
            "stage_counts": {
                str(stage): sum(1 for run in shard_runs if run["stage_first_requested"] == stage)
                for stage in (1, 2, 3, 4)
            },
        }
        path = root / shard_relative_path(machine_id)
        write_json(path, shard)
        shard_paths[machine_id] = path.relative_to(root).as_posix()
    manifest = dict(manifest)
    manifest["shards"] = shard_paths
    manifest["launchers"] = write_launchers(root, manifest)
    write_json(root / "manifest.json", manifest)
    return root


def manifest_path_from_arg(path: str) -> Path:
    manifest_path = Path(path)
    if not manifest_path.is_absolute():
        manifest_path = REPO_ROOT / manifest_path
    return manifest_path


def local_asset_path(repo_root: Path, relative_path: str) -> Path:
    return repo_root / relative_path


def validate_local_assets(manifest: dict[str, Any], repo_root: Path) -> None:
    assets = manifest["assets"]
    validate_assets_strict(
        repo_root=repo_root,
        pps_relative_path=assets["pps_model_relative_path"],
        rts_checkpoint_relative_dir=assets["rts_checkpoint_relative_dir"],
        expected_pps_sha256=assets["pps_model_sha256"],
        expected_rts_model_sha256=assets["rts_model_sha256"],
        expected_rts_metadata_sha256=assets["rts_metadata_sha256"],
        expected_rts_feature_schema_sha256=assets["rts_feature_schema_sha256"],
        expected_rts_checkpoint_id=assets["rts_checkpoint_id"],
    )


def condition_runtime_root(repo_root: Path, manifest: dict[str, Any], machine_id: str, run_id: str) -> Path:
    return campaign_root(repo_root, manifest) / machine_id / "runs" / run_id


def build_run_spec_from_condition(
    condition: dict[str, Any],
    *,
    manifest: dict[str, Any],
    machine: Machine,
    repo_root: Path,
) -> RunSpec:
    assets = manifest["assets"]
    seed = int(condition["seed"])
    common = {
        "run_id": condition["run_id"],
        "ticks": int(condition["ticks"]),
        "runtime_root": condition_runtime_root(repo_root, manifest, machine.machine_id, condition["run_id"]),
        "repo_root": repo_root,
        "input_root": repo_root / manifest["input_root_relative"],
        "branch": manifest.get("branch"),
        "commit": manifest.get("commit"),
        "python_executable": machine.python,
        "timestamp": manifest["campaign_id"],
        "rts_seed_base": seed,
        "rts_random_seed": seed,
        "robot_count": int(condition["robot_count"]),
        "expected_picking_station_count": PICKER_COUNT,
        "expected_replenishment_station_count": REPLENISHMENT_COUNT,
        "keep_runtime_artifacts": False,
        "detail_db": False,
        "timing": False,
        "worker_status_cadence": 1000,
        "run_profile": "training",
        "run_horizon_ticks": int(condition["ticks"]),
        "demand_horizon_ticks": int(condition["ticks"]) + 1000,
        "demand_buffer_ticks": 1000,
        "order_generation_mode": "shuffled_historical_cycle",
        "full_raw_order_replay": False,
        "order_rate_per_hour": int(condition["order_rate"]),
        "pod_location_mode": "randomize_slots",
        "pod_location_seed": seed,
        "experiment_id": manifest["campaign_id"],
        "scenario_id": condition["scenario_id"],
        "artifact_label": condition["run_id"],
        "batch_id": int(condition["stage_first_requested"]),
        "worker_id": int(condition["replication"]),
        "robot_task_allocator": "legacy_nearest",
        "regret_k": None,
        "task_allocator_scope": "active_job_queue",
        "rts_torch_threads": 1,
        "rts_torch_interop_threads": 1,
        "campaign_id": manifest["campaign_id"],
        "machine_id": machine.machine_id,
        "stage_first_requested": int(condition["stage_first_requested"]),
        "kpi_schema_version": manifest["kpi_schema_version"],
        "policy_configuration": condition["policy_configuration"],
        "replication": int(condition["replication"]),
        "campaign_seed": seed,
        "rts_checkpoint_sha256": assets["rts_model_sha256"],
        "pps_model_sha256": assets["pps_model_sha256"],
    }
    if condition["policy_configuration"] == "all_on_rl":
        return RunSpec(
            **common,
            rts_policy_mode="rts_rl_explicit",
            rts_rollout_enabled=True,
            rts_rollout_write_disk=False,
            rts_zone_ids=["auto"],
            rts_policy_checkpoint_dir=str(local_asset_path(repo_root, assets["rts_checkpoint_relative_dir"])),
            rts_policy_checkpoint_id=assets["rts_checkpoint_id"],
            rts_policy_action_mode="greedy",
            rts_policy_device="cpu",
            rts_feature_ablation="full",
            rts_state_capture_mode="full",
            pps_mode="ppo",
            pps_model_path=str(local_asset_path(repo_root, assets["pps_model_relative_path"])),
            charging_enabled=True,
            charging_config_path=str(local_asset_path(repo_root, assets["charging_config_relative_path"])),
            committed_next_reservations_enabled=True,
        )
    return RunSpec(
        **common,
        rts_policy_mode="current",
        rts_rollout_enabled=False,
        rts_state_capture_mode="auto",
        pps_mode="heuristic",
        charging_enabled=False,
        committed_next_reservations_enabled=False,
    )


def run_complete_for_campaign(condition: dict[str, Any], spec: RunSpec, manifest: dict[str, Any]) -> bool:
    spec_path = spec.runtime_root / "run_spec.json"
    summary_path = spec.runtime_root / "worker_summary.json"
    if not spec_path.exists() or not summary_path.exists():
        return False
    try:
        previous_spec = read_json(spec_path)
        current_spec = spec.to_json_dict()
        summary = load_worker_summary(spec.runtime_root)
    except Exception:
        return False
    required_identity = condition["run_spec_identity"]
    for key, expected in required_identity.items():
        if previous_spec.get(key) != expected:
            return False
        if summary.get(key) != expected and key not in {"campaign_seed"}:
            return False
    if summary.get("seed") != condition["seed"]:
        return False
    if previous_spec != current_spec:
        return False
    if summary.get("status") != "success":
        return False
    if not bool(summary.get("finalization", {}).get("finalized")):
        return False
    if summary.get("kpi_schema_version") != manifest["kpi_schema_version"]:
        return False
    if not bool(summary.get("kpi_complete")):
        return False
    return True


def load_machine_shard(manifest: dict[str, Any], manifest_path: Path, machine_id: str) -> dict[str, Any]:
    shard_rel = manifest.get("shards", {}).get(machine_id) or shard_relative_path(machine_id).as_posix()
    shard_path = manifest_path.parent / shard_rel
    if not shard_path.exists():
        raise FileNotFoundError(f"missing shard for {machine_id}: {shard_path}")
    return read_json(shard_path)


def write_machine_summary(
    *,
    manifest: dict[str, Any],
    machine: Machine,
    repo_root: Path,
    stage: int | None,
    launched: list[str],
    skipped: list[str],
    elapsed_seconds: float,
) -> None:
    root = campaign_root(repo_root, manifest) / machine.machine_id
    summaries = list(root.glob("runs/*/worker_summary.json"))
    completed_steps = 0
    completed_runs = 0
    wall_time_sum = 0.0
    for path in summaries:
        try:
            summary = read_json(path)
        except Exception:
            continue
        if summary.get("status") == "success" and summary.get("kpi_complete"):
            completed_runs += 1
            completed_steps += int(summary.get("netlogo_steps_completed", 0) or 0)
            wall_time_sum += float(summary.get("worker_wall_time_elapsed", 0.0) or 0.0)
    stable_rate = completed_steps / elapsed_seconds if elapsed_seconds > 0 and launched else None
    payload = {
        "campaign_id": manifest["campaign_id"],
        "machine_id": machine.machine_id,
        "stage": stage,
        "last_elapsed_seconds": elapsed_seconds,
        "last_launched_runs": launched,
        "last_skipped_resume_runs": skipped,
        "completed_runs": completed_runs,
        "completed_backend_steps": completed_steps,
        "completed_worker_wall_time_sum": wall_time_sum,
        "last_controller_effective_steps_per_second": stable_rate,
        "manifest_rate_steps_per_second": machine.effective_steps_per_second,
        "updated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
    }
    write_json(root / "machine_summary.json", payload)


def execute_machine(
    *,
    manifest_path: Path,
    machine_id: str,
    stages: list[int],
    resume: bool,
    progress: bool,
    keep_run_artifacts: bool = False,
) -> int:
    manifest = read_json(manifest_path)
    machines = machine_map(manifest)
    if machine_id not in machines:
        raise SystemExit(f"unknown machine id {machine_id}; expected one of {sorted(machines)}")
    machine = machines[machine_id]
    validate_local_assets(manifest, REPO_ROOT)
    shard = load_machine_shard(manifest, manifest_path, machine_id)
    for stage in stages:
        selected = [run for run in shard["runs"] if int(run["stage_first_requested"]) == int(stage)]
        specs = []
        skipped = []
        for condition in selected:
            spec = build_run_spec_from_condition(condition, manifest=manifest, machine=machine, repo_root=REPO_ROOT)
            if resume and run_complete_for_campaign(condition, spec, manifest):
                skipped.append(condition["run_id"])
                continue
            specs.append(spec)
        start = time.perf_counter()
        if specs:
            run_specs(specs, max_workers=int(machine.max_workers), progress=progress)
        elapsed = time.perf_counter() - start
        write_machine_summary(
            manifest=manifest,
            machine=machine,
            repo_root=REPO_ROOT,
            stage=stage,
            launched=[spec.run_id for spec in specs],
            skipped=skipped,
            elapsed_seconds=elapsed,
        )
        if not keep_run_artifacts:
            reclaimed = 0
            for condition in selected:
                run_root = condition_runtime_root(REPO_ROOT, manifest, machine_id, condition["run_id"])
                reclaimed += reclaim_completed_run_artifacts(run_root)
            if reclaimed > 0:
                print(f"[sensitivity] stage {stage}: reclaimed {reclaimed / 1_000_000:.0f} MB of regenerable run artifacts")
    return 0


def rebalance_future_stages(manifest_path: Path) -> Path:
    manifest = read_json(manifest_path)
    preview = {
        "campaign_id": manifest["campaign_id"],
        "created_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "status": "preview_only",
        "message": "Future-stage rebalance is explicit-only; no immutable shard was rewritten by preview.",
        "current_stage_allocations": {
            stage: manifest["stages"][stage]["assigned_counts"]
            for stage in sorted(manifest["stages"])
        },
    }
    path = manifest_path.with_name("rebalance_preview.json")
    write_json(path, preview)
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare or run the distributed RMFS sensitivity campaign.")
    parser.add_argument("--prepare-campaign", action="store_true", default=False)
    parser.add_argument("--machine-id", choices=("win_lukman", "win_admin", "citi_angiebow", "codex_local"))
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--run-continuously", action="store_true", default=False)
    parser.add_argument("--stage", action="append", type=int, choices=(1, 2, 3, 4))
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--progress", action="store_true", default=False)
    parser.add_argument("--rebalance-future-stages", action="store_true", default=False)
    parser.add_argument("--dry-run", action="store_true", default=False)
    parser.add_argument("--rts-checkpoint-dir", default=None)
    parser.add_argument("--rts-training-artifact", default=None)
    parser.add_argument("--pps-model-path", default=str(DEFAULT_PPS_MODEL_PATH))
    parser.add_argument("--charging-config-path", default=str(DEFAULT_CHARGING_CONFIG_PATH))
    parser.add_argument("--rl-overhead-multiplier", type=float, default=DEFAULT_RL_OVERHEAD_MULTIPLIER)
    parser.add_argument(
        "--keep-run-artifacts",
        action="store_true",
        default=False,
        help="Preserve per-run simulation scratch (netlogo.state, order/pod CSVs, worker logs). "
        "By default these regenerable files are reclaimed after each stage since runs are "
        "reproducible from their pinned seed; result JSON summaries are always kept.",
    )
    parser.add_argument("--validate-only", action="store_true", default=False)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.rebalance_future_stages:
        if not args.manifest:
            raise SystemExit("--rebalance-future-stages requires --manifest")
        path = rebalance_future_stages(manifest_path_from_arg(args.manifest))
        print(f"[sensitivity] wrote rebalance preview: {path}")
        return 0

    # Auto-generation or validation when manifest is omitted (for execute or validate-only)
    if (args.run_continuously or args.stage or args.validate_only) and not args.manifest:
        machines = default_machines(REPO_ROOT)
        assets = resolve_assets(args)
        campaign_id = generate_campaign_id(machines, assets, float(args.rl_overhead_multiplier))
        campaign_dir = REPO_ROOT / OUTPUT_ROOT_RELATIVE / campaign_id
        manifest_path = campaign_dir / "manifest.json"
        
        new_manifest = build_campaign_plan(
            campaign_id=campaign_id,
            machines=machines,
            assets=assets,
            rl_overhead_multiplier=float(args.rl_overhead_multiplier),
        )
        
        if manifest_path.exists():
            # Check for stale plan: verify configurations and hashes match
            try:
                existing_manifest = read_json(manifest_path)
                mismatches = []
                if existing_manifest.get("campaign_id") != campaign_id:
                    mismatches.append(f"campaign_id mismatch: existing={existing_manifest.get('campaign_id')}, expected={campaign_id}")
                
                # Check asset hashes
                for key in ("pps_model_sha256", "rts_model_sha256", "rts_metadata_sha256", "rts_feature_schema_sha256", "charging_config_sha256"):
                    existing_hash = existing_manifest.get("assets", {}).get(key)
                    new_hash = asdict(assets).get(key)
                    if existing_hash != new_hash:
                        mismatches.append(f"asset hash mismatch for {key}: existing={existing_hash}, expected={new_hash}")
                
                # Check machines
                existing_machines = existing_manifest.get("machines", [])
                new_machines = [asdict(m) for m in machines]
                if sorted(existing_machines, key=lambda x: x["machine_id"]) != sorted(new_machines, key=lambda x: x["machine_id"]):
                     mismatches.append("machines configuration mismatch")
                     
                # Check runs
                if len(existing_manifest.get("runs", [])) != len(new_manifest["runs"]):
                    mismatches.append(f"runs count mismatch: existing={len(existing_manifest.get('runs', []))}, expected={len(new_manifest['runs'])}")
                
                if mismatches:
                    raise RuntimeError(
                        f"Existing campaign plan at {manifest_path} is stale or mismatched:\n" + 
                        "\n".join(f" - {m}" for m in mismatches)
                    )
                print(f"[sensitivity] Reusing matched campaign plan: {manifest_path}")
            except Exception as exc:
                raise RuntimeError(f"Plan validation failed: {exc}") from exc
        else:
            if not args.validate_only:
                print(f"[sensitivity] Materializing new local campaign plan under: {campaign_dir}")
                write_campaign_files(new_manifest, dry_run=False)
            else:
                print(f"[sensitivity] Validating in-memory campaign plan (does not exist on disk).")
                
        args.manifest = str(manifest_path.relative_to(REPO_ROOT))

    if args.validate_only:
        print("[sensitivity] Running validation only...")
        # Get the manifest
        manifest_path = manifest_path_from_arg(args.manifest)
        if manifest_path.exists():
            manifest = read_json(manifest_path)
        else:
            machines = default_machines(REPO_ROOT)
            assets = resolve_assets(args)
            campaign_id = generate_campaign_id(machines, assets, float(args.rl_overhead_multiplier))
            manifest = build_campaign_plan(
                campaign_id=campaign_id,
                machines=machines,
                assets=assets,
                rl_overhead_multiplier=float(args.rl_overhead_multiplier),
            )
            
        # 1. strictly load canonical assets
        validate_local_assets(manifest, REPO_ROOT)
        
        # 2. Check all 1,500 unique identities
        runs = manifest["runs"]
        if len(runs) != 1500:
            raise AssertionError(f"Expected 1500 runs, got {len(runs)}")
            
        run_ids = [run["run_id"] for run in runs]
        condition_keys = [run["condition_key"] for run in runs]
        if len(set(run_ids)) != 1500:
            raise AssertionError(f"Duplicate run IDs found, unique count: {len(set(run_ids))}")
        if len(set(condition_keys)) != 1500:
            raise AssertionError(f"Duplicate condition keys found, unique count: {len(set(condition_keys))}")
            
        # 3. Check stage quotas and seeds
        expected_stage_counts = {"1": 30, "2": 38, "3": 532, "4": 900}
        actual_stage_counts = manifest["assertions"]["stage_new_runs"]
        if actual_stage_counts != expected_stage_counts:
            raise AssertionError(f"Stage new runs count mismatch: expected {expected_stage_counts}, got {actual_stage_counts}")
            
        expected_quotas = {str(stage): STAGE_QUOTAS[stage] for stage in (1, 2, 3, 4)}
        actual_allocations = manifest["assertions"]["stage_allocations"]
        if actual_allocations != expected_quotas:
            raise AssertionError(f"Stage quotas mismatch: expected {expected_quotas}, got {actual_allocations}")
            
        # Verify seeds
        for replication in (1, 20, 50):
            expected_seed = seed_for_replication(replication)
            for run in runs:
                if run["replication"] == replication:
                    if run["seed"] != expected_seed:
                        raise AssertionError(f"Seed mismatch for rep {replication}: expected {expected_seed}, got {run['seed']}")
                        
        # 4. Construct every RunSpec
        machines = default_machines(REPO_ROOT)
        machine_by_id = {m.machine_id: m for m in machines}
        for run in runs:
            m_id = run["machine_id"]
            if m_id not in machine_by_id:
                raise AssertionError(f"Unknown machine_id {m_id} in run spec")
            machine = machine_by_id[m_id]
            spec = build_run_spec_from_condition(run, manifest=manifest, machine=machine, repo_root=REPO_ROOT)
            
            # 5. Confirm no run references data/runtime/rts_training
            spec_dict = spec.to_json_dict()
            for key, val in spec_dict.items():
                if isinstance(val, str) and "data/runtime/rts_training" in val:
                    raise AssertionError(f"RunSpec {run['run_id']} has key '{key}' referencing runtime training: {val}")
        
        print("[sensitivity] Validation successful! All checks passed.")
        return 0

    if args.prepare_campaign:
        machines = default_machines(REPO_ROOT)
        assets = resolve_assets(args)
        campaign_id = generate_campaign_id(machines, assets, float(args.rl_overhead_multiplier))
        manifest = build_campaign_plan(
            campaign_id=campaign_id,
            machines=machines,
            assets=assets,
            rl_overhead_multiplier=float(args.rl_overhead_multiplier),
        )
        root = write_campaign_files(manifest, dry_run=bool(args.dry_run))
        print(json.dumps({
            "campaign_id": campaign_id,
            "campaign_root": str(root),
            "dry_run": bool(args.dry_run),
            "assertions": manifest["assertions"],
        }, indent=2, sort_keys=True))
        return 0

    if args.run_continuously or args.stage:
        if not args.manifest:
            raise SystemExit("execution requires --manifest")
        if not args.machine_id:
            raise SystemExit("execution requires --machine-id")
        stages = args.stage or [1, 2, 3, 4]
        return execute_machine(
            manifest_path=manifest_path_from_arg(args.manifest),
            machine_id=args.machine_id,
            stages=stages,
            resume=bool(args.resume),
            progress=bool(args.progress),
            keep_run_artifacts=bool(args.keep_run_artifacts),
        )
        
    raise SystemExit("choose --prepare-campaign, --run-continuously, --stage, --validate-only, or --rebalance-future-stages")


if __name__ == "__main__":
    raise SystemExit(main())
