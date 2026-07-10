#!/usr/bin/env python3
"""Prepare and run the fresh four-PC sensitivity-analysis campaign.

The campaign is intentionally file based: one immutable manifest, one shard per
machine, and the existing local RunSpec executor on each PC.
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.experiments.capacity_study_order_rate import (  # noqa: E402
    # Retained as a legacy-union compatibility constant only.  Primary campaign
    # preparation never resolves, validates, or hashes this static config.
    DEFAULT_CHARGING_CONFIG_PATH,
    DEFAULT_RTS_CHECKPOINT_DIR,
    PICKER_COUNT,
    REPLENISHMENT_COUNT,
    sha256_file,
    ticks_from_seconds,
    validate_input_root,
)
# Reference (all_off) charging generator — campaign-local, layout-adaptive.
from scripts.data.build_baseline_random import build as build_reference_config  # noqa: E402
from src.rmfs.decisions.pps.runtime import (  # noqa: E402
    PPS_RL_NUM_STATIONS,
    load_pps_rl_inference_policy,
)
from src.rmfs.experiments.identity import short_hash  # noqa: E402
from src.rmfs.orchestration.local_executor import (  # noqa: E402
    FAILED_RECLAIMABLE_RUN_ARTIFACTS,
    SENSITIVITY_KPI_SCHEMA_VERSION,
    SENSITIVITY_KPI_FIELDS,
    SUCCESS_RECLAIMABLE_RUN_ARTIFACTS,
    directory_size_bytes,
    git_value,
    load_worker_summary,
    reclaim_completed_run_artifacts_with_stats,
    reclaim_failed_run_artifacts_with_stats,
    reclaim_interrupted_run_artifacts_with_stats,
    run_specs,
)
from src.rmfs.orchestration.run_spec import RunSpec  # noqa: E402
from src.rmfs.orchestration.source_identity import SourceIdentity  # noqa: E402
from src.rmfs.rl.rts.training.checkpoint import resolve_policy_checkpoint_id  # noqa: E402
from src.rmfs.rl.rts.training.policy_loader import load_policy_from_checkpoint  # noqa: E402
from src.rmfs.runtime_io.run_profiles import TICK_TO_SECOND  # noqa: E402
from src.rmfs.orchestration.kpi_schema import FULL_KPI_V3_FIELDS, FULL_KPI_V3_REQUIRED_FIELDS, FULL_KPI_V3_SCHEMA_VERSION  # noqa: E402
from scripts.data.build_adaptive_hybrid import build as build_salsa_adaptive_config, load_grid as load_salsa_grid  # noqa: E402
from src.rmfs.experiments.sensitivity_allocation import (  # noqa: E402
    MachineAllocationProfile,
    allocation_config_rows as treatment_allocation_config_rows,
    allocate_stage as treatment_allocate_stage,
)

CAMPAIGN_SCHEMA_VERSION = "distributed_sensitivity_campaign.v3"
SIMULATION_SEMANTICS_ID = "sensitivity_simulation_semantics.v1"
ALLOCATION_PATCH_LABEL = "allocation_patch_0001"
CANONICAL_RTS_CHECKPOINT_ID = "batch_000014"
# Campaign PPS asset: compact policy-only inference artifact (NOT the full PPO
# training archive). Exact deterministic parity is verified separately.
DEFAULT_PPS_INFERENCE_ARTIFACT = REPO_ROOT / "data/models/pps/pps_rl_policy_inference.zip"
# Bundled two-treatment comparison (charging enabled in BOTH; different packages).
TREATMENTS = ("all_off", "all_on_rl")
POLICY_CONFIGURATIONS = TREATMENTS          # `policy_configuration` field == treatment name
# Campaign-local factor grid — do NOT import from another experiment.
ROBOT_COUNTS = (15, 20, 25)
ORDER_RATES = (400, 500, 600)
REPLICATIONS = range(1, 41)                 # 40 paired replications
SIMULATED_SECONDS = 87_000.0
BACKEND_STEPS_PER_RUN = 580_000
SEED_BASE_MINUS_ONE = 41
DEFAULT_RL_OVERHEAD_MULTIPLIER = 1.35
DEFAULT_MIN_FREE_DISK_GB = 20.0
OUTPUT_ROOT_RELATIVE = Path("data/runtime/distributed_sensitivity")
INPUT_ROOT_RELATIVE = Path("data/input/base")
ARCHIVED_ROOTS_IGNORED = (
    "data/runtime/capacity_study_order_rate",
    "data/runtime/capacity_study_order_rate_packB",
    "data/runtime/capacity_study_order_rate_packC",
)
STAGE_NAMES = {
    1: "one_rep_sensitivity",
    2: "central_all_off_to_40",
    3: "central_all_on_rl_to_40",
    4: "full_sensitivity_to_40",
}
CENTRAL_ROBOT_COUNT = 20
CENTRAL_ORDER_RATE = 500
DEFAULT_LOCAL_PYTHON = Path("/home/dewan/torch-gpu/bin/python")
MACHINE_IDS = (
    "win_lukman",
    "win_admin",
    "citi_angiebow",
    "codex_local",
    "alisha_pc",
    "dewa_macbook",
    "citi_gojira",
)
SCIENTIFIC_TRACKED_PATHS = (
    "data/input/base/generated_pod.csv",
    "data/input/base/items.csv",
    "data/input/base/pods.csv",
    "data/input/base/raw_order.csv",
    "data/models/pps/pps_rl_policy_inference.zip",
    "data/models/pps/pps_rl_policy_inference.metadata.json",
    "data/models/rts/batch_000014/checkpoint/model.pt",
    "data/models/rts/batch_000014/checkpoint/metadata.json",
    "data/models/rts/batch_000014/checkpoint/feature_schema.json",
)
_SALSA_CONFIG_CACHE: dict[tuple[str, int], dict[str, Any]] = {}


@dataclass(frozen=True)
class Machine:
    machine_id: str
    os: str
    repository: str
    python: str
    max_workers: int
    effective_steps_per_second: float
    eligible_stages: tuple[int, ...]
    anydesk_id: str | None = None
    allowed_treatments: tuple[str, ...] = ("all_off", "all_on_rl")
    reference_effective_steps_per_second: float | None = None
    rl_effective_steps_per_second: float | None = None
    max_rl_workers: int | None = None
    rl_rate_source: str = "fallback_reference_measured"


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


def replace_with_retry(tmp_path: Path, path: Path, *, attempts: int = 7) -> None:
    delay = 0.05
    last_exc: OSError | None = None
    for attempt in range(attempts):
        try:
            tmp_path.replace(path)
            return
        except OSError as exc:
            winerror = getattr(exc, "winerror", None)
            errno = getattr(exc, "errno", None)
            if winerror not in {32, 33} and errno not in {13, 16}:
                raise
            last_exc = exc
            if attempt >= attempts - 1:
                break
            time.sleep(delay)
            delay = min(delay * 2.0, 1.0)
    raise RuntimeError(
        f"failed to replace {path} after {attempts} attempts: "
        f"{type(last_exc).__name__}: {last_exc}"
    ) from last_exc


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    replace_with_retry(tmp_path, path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_json_sha256(path: Path) -> str:
    payload = read_json(path)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def git_blob_identity(repo_relative_path: str) -> str | None:
    value = git_clean_value("rev-parse", f"HEAD:{repo_relative_path}")
    return value or None


def normalized_text_sha256(path: Path) -> str:
    text = Path(path).read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def tracked_text_identity(path: Path) -> str:
    rel = relative_to_repo(path)
    return git_blob_identity(rel) or normalized_text_sha256(path)


def relative_to_repo(path: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError as exc:
        raise ValueError(f"path must be inside repository for portable shards: {resolved}") from exc


def git_clean_value(*args: str) -> str | None:
    return git_value(REPO_ROOT, *args)


def validation_warning(message: str) -> None:
    print(f"[sensitivity] warning: {message}")


def dirty_tracked_files(repo_root: Path = REPO_ROOT) -> list[str]:
    if not (Path(repo_root) / ".git").exists():
        # Autonomous hosts may run an extracted immutable source snapshot.
        # SourceIdentity, not Git, is the executable identity there.
        validation_warning(".git is unavailable; skipping Git dirty-state advisory")
        return []
    try:
        output = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=repo_root,
            text=True,
            stderr=subprocess.STDOUT,
        )
    except (subprocess.CalledProcessError, OSError) as exc:
        validation_warning(f"could not inspect Git dirty state: {exc}")
        return []
    dirty = []
    for line in output.splitlines():
        if not line.strip():
            continue
        dirty.append(line[3:] if len(line) > 3 else line.strip())
    return dirty


def ensure_clean_tracked_scientific_files(repo_root: Path = REPO_ROOT) -> None:
    if not (Path(repo_root) / ".git").exists():
        return
    scientific = set(SCIENTIFIC_TRACKED_PATHS)
    dirty = [
        path
        for path in dirty_tracked_files(repo_root)
        if path in scientific or path.split(" -> ")[-1] in scientific
    ]
    if dirty:
        preview = "\n".join(f" - {path}" for path in dirty[:20])
        suffix = "" if len(dirty) <= 20 else f"\n - ... {len(dirty) - 20} more"
        validation_warning(
            "tracked scientific files differ from HEAD; running with local files and recording "
            "their identities in the local manifest:\n"
            f"{preview}{suffix}"
        )


def validate_source_identity(manifest: dict[str, Any], repo_root: Path = REPO_ROOT) -> SourceIdentity:
    """Validate the executable source tree in both Git and extracted snapshots."""
    source = SourceIdentity.compute(repo_root)
    expected = manifest.get("source_tree_hash")
    if expected and source.source_tree_hash != expected:
        validation_warning(
            "source-tree hash differs from the local manifest; continuing with the current "
            f"host source tree (got {source.source_tree_hash}, manifest {expected})"
        )
    # Git is advisory for portable operation.  The host records local source
    # and immutable-file identities but does not refuse an otherwise runnable
    # campaign for provenance-only differences.
    ensure_clean_tracked_scientific_files(repo_root)
    return source


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
    all_stages = (1, 2, 3, 4)
    return [
        Machine(
            machine_id="win_lukman",
            anydesk_id="1903438276",
            os="windows",
            repository=r"D:\lukman-rmfs\Combinatrix",
            python=r"D:\lukman-rmfs\.rmfs\Scripts\python.exe",
            max_workers=8,
            effective_steps_per_second=414.64,
            eligible_stages=all_stages,
            reference_effective_steps_per_second=414.64,
            rl_effective_steps_per_second=414.64,
            max_rl_workers=8,
        ),
        Machine(
            machine_id="win_admin",
            anydesk_id="1052269911",
            os="windows",
            repository=r"C:\Users\admin\Documents\Dewa's Sandbox\netlogo-rmfs",
            python=r"C:\Users\admin\Documents\Dewa's Sandbox\torch-gpu\Scripts\python.exe",
            max_workers=8,
            effective_steps_per_second=395.66,
            eligible_stages=all_stages,
            reference_effective_steps_per_second=395.66,
            rl_effective_steps_per_second=395.66,
            max_rl_workers=8,
        ),
        Machine(
            machine_id="citi_angiebow",
            os="linux",
            repository="/home/citi/Documents/Dewa's Sandbox/netlogo-rmfs",
            python="/home/citi/Documents/Dewa's Sandbox/torch-gpu/bin/python",
            max_workers=4,
            effective_steps_per_second=363.25,
            eligible_stages=all_stages,
            reference_effective_steps_per_second=363.25,
            rl_effective_steps_per_second=363.25,
            max_rl_workers=4,
        ),
        Machine(
            machine_id="codex_local",
            os=platform.system().lower() or "auto",
            repository=str(repo_root),
            python=local_python_executable(),
            max_workers=8,
            effective_steps_per_second=300.0,
            eligible_stages=all_stages,
            reference_effective_steps_per_second=300.0,
            rl_effective_steps_per_second=300.0,
            max_rl_workers=8,
        ),
        Machine(
            machine_id="alisha_pc",
            os="windows",
            repository=r"C:\Users\ayual\Program TA\netlogo-rmfs",
            python=r"C:\Users\ayual\Program TA\torch-gpu\Scripts\python.exe",
            max_workers=8,
            effective_steps_per_second=330.0,
            eligible_stages=all_stages,
            allowed_treatments=("all_off",),
            reference_effective_steps_per_second=330.0,
            max_rl_workers=0,
            rl_rate_source="not_applicable",
        ),
        Machine(
            machine_id="dewa_macbook",
            os="macos",
            repository="/Users/710502/netlogo-rmfs",
            python="/Users/710502/torch-gpu/bin/python",
            max_workers=8,
            effective_steps_per_second=260.0,
            eligible_stages=all_stages,
            allowed_treatments=("all_off",),
            reference_effective_steps_per_second=260.0,
            max_rl_workers=0,
            rl_rate_source="not_applicable",
        ),
        Machine(
            machine_id="citi_gojira",
            os="linux",
            repository="/home/ma012/Documents/I like Gojira/netlogo-rmfs",
            python="/home/ma012/Documents/I like Gojira/torch-gpu/bin/python",
            max_workers=24,
            effective_steps_per_second=850.0,
            eligible_stages=all_stages,
            reference_effective_steps_per_second=850.0,
            rl_effective_steps_per_second=850.0,
            max_rl_workers=24,
        ),
    ]


def stable_json_hash(payload: Any, length: int = 12) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")).hexdigest()[:length]


def machine_allocation_profiles(machines: list[Machine]) -> list[MachineAllocationProfile]:
    profiles = []
    for machine in machines:
        allowed = tuple(machine.allowed_treatments)
        reference_rate = float(machine.reference_effective_steps_per_second or machine.effective_steps_per_second)
        max_rl_workers = int(machine.max_rl_workers if machine.max_rl_workers is not None else (machine.max_workers if "all_on_rl" in allowed else 0))
        rl_rate = machine.rl_effective_steps_per_second
        if "all_on_rl" in allowed and rl_rate is None:
            rl_rate = float(machine.effective_steps_per_second)
        profiles.append(MachineAllocationProfile(
            machine_id=machine.machine_id,
            max_workers=int(machine.max_workers),
            allowed_treatments=allowed,
            eligible_stages=tuple(int(stage) for stage in machine.eligible_stages),
            reference_effective_steps_per_second=reference_rate,
            rl_effective_steps_per_second=float(rl_rate) if rl_rate is not None else None,
            max_rl_workers=max_rl_workers,
            rl_rate_source=machine.rl_rate_source,
        ))
    return profiles


def allocation_config_rows(machines: list[Machine]) -> list[dict[str, Any]]:
    return treatment_allocation_config_rows(machine_allocation_profiles(machines))


def generate_allocation_patch_id(machines: list[Machine], rl_overhead_multiplier: float) -> str:
    payload = {
        "allocation_patch_label": ALLOCATION_PATCH_LABEL,
        "allocation_algorithm": "treatment_aware_constrained_lpt.v1",
        "machine_allocation_config": allocation_config_rows(machines),
        "condition_cost_model": {
            "base_backend_steps": BACKEND_STEPS_PER_RUN,
            "rl_overhead_multiplier": float(rl_overhead_multiplier),
            "robot_count_adjustment_per_5": 0.015,
            "order_rate_adjustment_per_100": 0.02,
        },
    }
    return f"{ALLOCATION_PATCH_LABEL}_{stable_json_hash(payload)}"


def hash_machine_config(machines: list[Machine]) -> str:
    return stable_json_hash(allocation_config_rows(machines))


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
    strict: bool = True,
) -> None:
    canonical_pps = relative_to_repo(DEFAULT_PPS_INFERENCE_ARTIFACT)
    canonical_rts = relative_to_repo(DEFAULT_RTS_CHECKPOINT_DIR)
    if Path(pps_relative_path).as_posix() != canonical_pps:
        raise RuntimeError(f"distributed sensitivity requires canonical PPS model {canonical_pps}, got {pps_relative_path}")
    if Path(rts_checkpoint_relative_dir).as_posix() != canonical_rts:
        raise RuntimeError(f"distributed sensitivity requires canonical RTS checkpoint {canonical_rts}, got {rts_checkpoint_relative_dir}")

    # 1. PPS validation
    pps_path = repo_root / pps_relative_path
    if not pps_path.exists():
        raise FileNotFoundError(f"PPS model path does not exist: {pps_path}")
    pps_sha = sha256_file(pps_path)
    if expected_pps_sha256 and pps_sha != expected_pps_sha256:
        message = f"PPS model hash mismatch: got {pps_sha}, expected {expected_pps_sha256}"
        if strict:
            raise RuntimeError(message)
        validation_warning(message)
    
    # Strictly load and verify the compact PPS inference policy (no fallback).
    load_pps_rl_inference_policy(pps_path, expected_sha256=pps_sha)
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
        message = f"RTS model.pt hash mismatch: got {model_sha}, expected {expected_rts_model_sha256}"
        if strict:
            raise RuntimeError(message)
        validation_warning(message)
    if expected_rts_metadata_sha256 and metadata_sha != expected_rts_metadata_sha256:
        message = f"RTS metadata.json hash mismatch: got {metadata_sha}, expected {expected_rts_metadata_sha256}"
        if strict:
            raise RuntimeError(message)
        validation_warning(message)
    if expected_rts_feature_schema_sha256 and schema_sha != expected_rts_feature_schema_sha256:
        message = f"RTS feature_schema.json hash mismatch: got {schema_sha}, expected {expected_rts_feature_schema_sha256}"
        if strict:
            raise RuntimeError(message)
        validation_warning(message)

    # Load RTS policy to verify feature schema loads successfully
    loaded = load_policy_from_checkpoint(checkpoint_dir, device="cpu")
    checkpoint_id = resolve_policy_checkpoint_id(checkpoint_dir)
    if loaded.policy_checkpoint_id != checkpoint_id:
        message = f"RTS checkpoint ID mismatch after strict load: loaded={loaded.policy_checkpoint_id}, dir={checkpoint_id}"
        if strict:
            raise RuntimeError(message)
        validation_warning(message)
    if expected_rts_checkpoint_id and checkpoint_id != expected_rts_checkpoint_id:
        message = f"RTS checkpoint ID mismatch: got {checkpoint_id}, expected {expected_rts_checkpoint_id}"
        if strict:
            raise RuntimeError(message)
        validation_warning(message)
    if checkpoint_id != CANONICAL_RTS_CHECKPOINT_ID:
        message = f"distributed sensitivity expected RTS checkpoint ID {CANONICAL_RTS_CHECKPOINT_ID}, got {checkpoint_id}"
        if strict:
            raise RuntimeError(message)
        validation_warning(message)

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
    pps_path = Path(args.pps_model_path or DEFAULT_PPS_INFERENCE_ARTIFACT)
    if not pps_path.is_absolute():
        pps_path = REPO_ROOT / pps_path
    
    checkpoint_dir, training_root, lineage, source = resolve_rts_checkpoint(
        explicit_checkpoint_dir=args.rts_checkpoint_dir,
        training_artifact=args.rts_training_artifact,
    )

    # Verify that required assets are readable and loadable.  Provenance hash
    # differences are reported by the validator but do not block a host.
    validate_assets_strict(
        repo_root=REPO_ROOT,
        pps_relative_path=relative_to_repo(pps_path),
        rts_checkpoint_relative_dir=relative_to_repo(checkpoint_dir),
        strict=False,
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
        pps_observation_schema=pps_observation_schema(load_pps_rl_inference_policy(pps_path)),
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
        # Primary treatments generate and hash their charger configuration in
        # the condition workspace.  The historical static legacy_union JSON is
        # intentionally not a required campaign asset.
        charging_config_relative_path="",
        charging_config_sha256="",
    )


def seed_for_replication(replication: int) -> int:
    return SEED_BASE_MINUS_ONE + int(replication)


def condition_key(policy: str, robot_count: int, order_rate: int, replication: int) -> str:
    return f"{policy}|robots={robot_count}|order_rate={order_rate}|rep={replication}"


def paired_group_id(robot_count: int, order_rate: int, replication: int) -> str:
    """Treatment-independent identity shared by an all_off/all_on_rl pair.

    Seed is a pure function of replication (see seed_for_replication), so both
    members of a pair carry the same seed; the pair id encodes it explicitly.
    """
    return (
        f"robots={int(robot_count)}|rate={int(order_rate)}"
        f"|rep={int(replication)}|seed={seed_for_replication(int(replication))}"
    )


def _stage_candidate_tuples(stage: int) -> list[tuple[str, int, int, int]]:
    """Full candidate set targeted by a stage (before cumulative subtraction)."""
    if stage == 1:
        return [
            (t, rc, orr, 1)
            for t in TREATMENTS for rc in ROBOT_COUNTS for orr in ORDER_RATES
        ]
    if stage == 2:
        return [
            ("all_off", CENTRAL_ROBOT_COUNT, CENTRAL_ORDER_RATE, rep)
            for rep in REPLICATIONS
        ]
    if stage == 3:
        return [
            ("all_on_rl", CENTRAL_ROBOT_COUNT, CENTRAL_ORDER_RATE, rep)
            for rep in REPLICATIONS
        ]
    if stage == 4:
        return [
            (t, rc, orr, rep)
            for t in TREATMENTS for rc in ROBOT_COUNTS
            for orr in ORDER_RATES for rep in REPLICATIONS
        ]
    raise ValueError(f"unknown stage: {stage}")


def condition_charging_identity(
    policy: str, robot_count: int, seed: int, layout_path: Path
) -> dict[str, Any]:
    """Deterministic treatment-local charging metadata (both treatments enabled).

    all_off  -> reference package (build_baseline_random, 10 chargers, 20/90/100).
    all_on_rl -> Salsa adaptive package (build_adaptive_hybrid, rho=0.6, 18/60/50).
    Config is computed in-memory here only for identity/hashing; the actual
    JSON is generated into the temporary run workspace at execution time and
    deleted after the KPI row is durably stored (never unioned with grid/static).
    """
    layout_sha = sha256_file(layout_path)

    def _thresholds(cfg):
        return {"low": cfg.get("battery_low_pct"), "charged": cfg.get("battery_charged_pct"),
                "interrupt": cfg.get("battery_interrupt_pct")}

    if policy == "all_off":
        cache_key = ("reference", str(Path(layout_path).resolve()), int(seed))
        cached = _SALSA_CONFIG_CACHE.get(cache_key)
        if cached is not None:
            return dict(cached)
        grid = load_salsa_grid(layout_path)
        config = build_reference_config(grid, num_chargers=10, seed=int(seed))
        canonical = (json.dumps(config, indent=2, sort_keys=True) + "\n").encode("utf-8")
        value = {
            "charging_placement_source": "generated_reference", "charging_enabled": True,
            "config": config, "config_sha256": hashlib.sha256(canonical).hexdigest(),
            "declared_count": int(config["num_chargers"]), "realized_layout_sha256": layout_sha,
            "generator": "scripts/data/build_baseline_random.py",
            "generator_parameters": {"num_chargers": 10, "seed": int(seed)},
            "thresholds": _thresholds(config),
        }
        _SALSA_CONFIG_CACHE[cache_key] = dict(value)
        return dict(value)

    if policy == "all_on_rl":
        cache_key = ("salsa", str(Path(layout_path).resolve()), int(robot_count))
        cached = _SALSA_CONFIG_CACHE.get(cache_key)
        if cached is not None:
            return dict(cached)
        grid = load_salsa_grid(layout_path)
        config, _picker, _depot = build_salsa_adaptive_config(grid, int(robot_count), rho=0.6)
        canonical = (json.dumps(config, indent=2, sort_keys=True) + "\n").encode("utf-8")
        value = {
            "charging_placement_source": "generated_salsa_adaptive", "charging_enabled": True,
            "config": config, "config_sha256": hashlib.sha256(canonical).hexdigest(),
            "declared_count": int(config["num_chargers"]), "realized_layout_sha256": layout_sha,
            "generator": "scripts/data/build_adaptive_hybrid.py",
            "generator_parameters": {"rho": 0.6, "robot_count": int(robot_count)},
            "thresholds": _thresholds(config),
        }
        _SALSA_CONFIG_CACHE[cache_key] = dict(value)
        return dict(value)

    raise ValueError(f"unknown primary treatment: {policy!r}")


def base_condition_rows(stage: int, already_requested: set[str]) -> list[dict[str, Any]]:
    """Cumulative-stage condition generation: emit only NEW conditions this stage
    (those not already requested/completed in an earlier stage)."""
    candidates = _stage_candidate_tuples(stage)
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
            "paired_group_id": paired_group_id(robot_count, order_rate, replication),
            "stage_first_requested": int(stage),
            "charging": condition_charging_identity(
                policy, int(robot_count), int(seed_for_replication(replication)),
                REPO_ROOT / INPUT_ROOT_RELATIVE / "generated_pod.csv"
            ),
        })
    return rows


def estimate_condition_steps(row: dict[str, Any], *, rl_overhead_multiplier: float) -> float:
    multiplier = 1.0
    if row["policy_configuration"] == "all_on_rl":
        multiplier *= float(rl_overhead_multiplier)
    multiplier *= 1.0 + 0.015 * ((int(row["robot_count"]) - CENTRAL_ROBOT_COUNT) / 5.0)
    multiplier *= 1.0 + 0.02 * ((int(row["order_rate"]) - CENTRAL_ORDER_RATE) / 100.0)
    return BACKEND_STEPS_PER_RUN * max(0.2, multiplier)


def build_input_identity(input_meta: dict[str, Any]) -> dict[str, Any]:
    file_identities = {}
    for rel_path in sorted(input_meta.get("file_digests", {})):
        file_identities[rel_path] = tracked_text_identity(REPO_ROOT / INPUT_ROOT_RELATIVE / rel_path)
    required = {"generated_pod.csv", "items.csv", "pods.csv", "raw_order.csv"}
    missing = sorted(required - set(file_identities))
    if missing:
        raise RuntimeError(f"scientific input identity missing executed input files: {missing}")
    return {
        "input_root_relative": INPUT_ROOT_RELATIVE.as_posix(),
        "tracked_file_identities": file_identities,
        "layout_identity": file_identities["generated_pod.csv"],
    }


def treatment_execution_contract(policy: str, assets: AssetBundle | dict[str, Any]) -> dict[str, Any]:
    asset_dict = asdict(assets) if isinstance(assets, AssetBundle) else dict(assets)
    common = {
        "order_generation_mode": "shuffled_historical_cycle",
        "full_raw_order_replay": False,
        "pod_location_mode": "randomize_slots",
        "persist_final_state": False,
        "detail_db": False,
        "timing": False,
        "robot_task_allocator": "legacy_nearest",
        "regret_k": None,
        "task_allocator_scope": "active_job_queue",
        "worker_status_cadence": 1000,
    }
    if policy == "all_on_rl":
        # Integrated proposed treatment: RTS RL + PPO PPS + Salsa adaptive charging.
        return {
            **common,
            "rts_policy_mode": "rts_rl_explicit",
            "rts_rollout_enabled": True,
            "rts_rollout_write_disk": False,
            "rts_zone_selection_mode": "auto",
            "rts_zone_ids": ["auto"],
            "rts_checkpoint_id": asset_dict["rts_checkpoint_id"],
            "rts_policy_action_mode": "greedy",
            "rts_policy_device": "cpu",
            "rts_feature_ablation": "full",
            "rts_state_capture_mode": "full",
            "pps_mode": "ppo",
            "pps_policy": "ppo",
            "charging_enabled": True,
            "charging_package": "salsa_adaptive",
            "charging_generator": "scripts/data/build_adaptive_hybrid.py",
            "charging_placement_source": "generated_salsa_adaptive",
            "active_charging_enabled": True,
            "drive_by_charging_enabled": True,
            "committed_next_reservations_enabled": True,
        }
    if policy == "all_off":
        # Reference bundled treatment: heuristic RTS + heuristic PPS + reference charging.
        return {
            **common,
            "rts_policy_mode": "current",
            "rts_rollout_enabled": False,
            "rts_rollout_write_disk": False,
            "rts_checkpoint_id": "not_applicable",
            "rts_policy_action_mode": None,
            "rts_policy_device": None,
            "rts_feature_ablation": None,
            "rts_state_capture_mode": "auto",
            "pps_mode": "heuristic",
            "pps_policy": "heuristic",
            "charging_enabled": True,
            "charging_package": "reference_baseline",
            "charging_generator": "scripts/data/build_baseline_random.py",
            "charging_placement_source": "generated_reference",
            "active_charging_enabled": True,
            "drive_by_charging_enabled": True,
            "committed_next_reservations_enabled": False,
        }
    raise ValueError(f"unknown primary treatment: {policy}")


def treatment_execution_contracts(assets: AssetBundle | dict[str, Any]) -> dict[str, Any]:
    return {
        policy: treatment_execution_contract(policy, assets)
        for policy in POLICY_CONFIGURATIONS
    }


def build_scientific_identity(
    *,
    assets: AssetBundle,
    input_identity: dict[str, Any],
) -> dict[str, Any]:
    rts_dir = REPO_ROOT / assets.rts_checkpoint_relative_dir
    return {
        "schema_version": CAMPAIGN_SCHEMA_VERSION,
        "simulation_semantics_id": SIMULATION_SEMANTICS_ID,
        "kpi_schema_version": FULL_KPI_V3_SCHEMA_VERSION,
        "policy_configurations": list(POLICY_CONFIGURATIONS),
        "robot_counts": list(ROBOT_COUNTS),
        "order_rates": list(ORDER_RATES),
        "picking_stations": PICKER_COUNT,
        "replenishment_stations": REPLENISHMENT_COUNT,
        "replication_range": {"first": 1, "last": 40},
        "seed_formula": "seed = 41 + replication",
        "seed_base_minus_one": SEED_BASE_MINUS_ONE,
        "simulated_seconds": SIMULATED_SECONDS,
        "backend_steps_per_full_run": BACKEND_STEPS_PER_RUN,
        "tick_to_second": TICK_TO_SECOND,
        "source_tree_hash": SourceIdentity.compute(REPO_ROOT).source_tree_hash,
        "input_identity": input_identity,
        "scenario_layout_identity": {
            "scenario_hash": short_hash(input_identity),
            "layout_hash": input_identity["layout_identity"],
        },
        "assets": {
            "pps_model_relative_path": assets.pps_model_relative_path,
            "pps_model_sha256": assets.pps_model_sha256,
            "pps_observation_schema": assets.pps_observation_schema,
            "rts_checkpoint_relative_dir": assets.rts_checkpoint_relative_dir,
            "rts_checkpoint_id": assets.rts_checkpoint_id,
            "rts_model_sha256": assets.rts_model_sha256,
            "rts_metadata_canonical_sha256": canonical_json_sha256(rts_dir / "metadata.json"),
            "rts_feature_schema_canonical_sha256": canonical_json_sha256(rts_dir / "feature_schema.json"),
            "rts_feature_schema_id": assets.rts_feature_schema_id,
        },
        "run_profile": {
            "order_generation_mode": "shuffled_historical_cycle",
            "full_raw_order_replay": False,
            "pod_location_mode": "randomize_slots",
            "robot_task_allocator": "legacy_nearest",
            "task_allocator_scope": "active_job_queue",
            "detail_db": False,
            "persist_final_state": False,
            "charging_semantics": "both_enabled__all_off_reference__all_on_rl_salsa_adaptive",
        },
        "treatment_execution_contracts": treatment_execution_contracts(assets),
    }


def generate_campaign_id_from_identity(scientific_identity: dict[str, Any]) -> str:
    return f"sensitivity_full_kpi_v3_{stable_json_hash(scientific_identity)}"


def allocate_stage(
    rows: list[dict[str, Any]],
    *,
    stage: int,
    machines: list[Machine],
    rl_overhead_multiplier: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    allocated, summary = treatment_allocate_stage(
        rows,
        stage=stage,
        profiles=machine_allocation_profiles(machines),
        estimate_steps=lambda row: estimate_condition_steps(row, rl_overhead_multiplier=rl_overhead_multiplier),
    )
    for row in allocated:
        row["estimated_cost_model"] = {
            "base_backend_steps": BACKEND_STEPS_PER_RUN,
            "rl_overhead_multiplier": rl_overhead_multiplier,
            "effective_steps_per_second_semantics": "aggregate_machine_throughput_at_configured_worker_count",
            "slot_rate_steps_per_second": row["allocation_rate_steps_per_second"],
            "rate_source": row["allocation_rate_source"],
        }
    return allocated, {**summary, "name": STAGE_NAMES[stage]}


def add_run_identity(
    row: dict[str, Any],
    *,
    campaign_id: str,
    allocation_patch_id: str,
    scientific_identity: dict[str, Any],
    branch: str | None,
    commit: str | None,
    ticks: int,
    assets: AssetBundle,
    scenario_hash: str,
    layout_hash: str,
) -> dict[str, Any]:
    identity = {
        "campaign_id": campaign_id,
        "simulation_semantics_id": SIMULATION_SEMANTICS_ID,
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
        "source_tree_hash": scientific_identity["source_tree_hash"],
        "kpi_schema_version": FULL_KPI_V3_SCHEMA_VERSION,
        "rts_checkpoint_id": assets.rts_checkpoint_id,
        "rts_checkpoint_sha256": assets.rts_model_sha256,
        "rts_metadata_canonical_sha256": scientific_identity["assets"]["rts_metadata_canonical_sha256"],
        "rts_feature_schema_canonical_sha256": scientific_identity["assets"]["rts_feature_schema_canonical_sha256"],
        "pps_model_sha256": assets.pps_model_sha256,
        "charging_placement_source": row["charging"]["charging_placement_source"],
        "charging_config_sha256": row["charging"]["config_sha256"],
        "realized_layout_sha256": row["charging"]["realized_layout_sha256"],
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
        "allocation_patch_id": allocation_patch_id,
        "simulation_semantics_id": SIMULATION_SEMANTICS_ID,
        "scenario_id": f"scenario_{scenario_hash}",
        "scenario_hash": scenario_hash,
        "layout_hash": layout_hash,
        "source_tree_hash": scientific_identity["source_tree_hash"],
        "run_spec_identity": {
            "run_id": run_id,
            "campaign_id": campaign_id,
            "simulation_semantics_id": SIMULATION_SEMANTICS_ID,
            "kpi_schema_version": FULL_KPI_V3_SCHEMA_VERSION,
            "policy_configuration": row["policy_configuration"],
            "robot_count": row["robot_count"],
            "order_rate_per_hour": row["order_rate"],
            "replication": row["replication"],
            "campaign_seed": row["seed"],
            "netlogo_steps_requested": ticks,
            "tick_to_second": TICK_TO_SECOND,
            "rts_checkpoint_sha256": assets.rts_model_sha256,
            "pps_model_sha256": assets.pps_model_sha256,
            "source_tree_hash": scientific_identity["source_tree_hash"],
        },
    }


def generate_campaign_id(
    machines: list[Machine],
    assets: AssetBundle,
    rl_overhead_multiplier: float,
) -> str:
    input_meta = validate_input_root(REPO_ROOT / INPUT_ROOT_RELATIVE)
    scientific_identity = build_scientific_identity(
        assets=assets,
        input_identity=build_input_identity(input_meta),
    )
    return generate_campaign_id_from_identity(scientific_identity)


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
    input_identity = build_input_identity(input_meta)
    scientific_identity = build_scientific_identity(assets=assets, input_identity=input_identity)
    expected_campaign_id = generate_campaign_id_from_identity(scientific_identity)
    if campaign_id != expected_campaign_id:
        raise RuntimeError(f"campaign_id mismatch: got {campaign_id}, expected {expected_campaign_id}")
    allocation_patch_id = generate_allocation_patch_id(machines, rl_overhead_multiplier)
    scenario_hash = scientific_identity["scenario_layout_identity"]["scenario_hash"]
    layout_hash = scientific_identity["scenario_layout_identity"]["layout_hash"]
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
                allocation_patch_id=allocation_patch_id,
                scientific_identity=scientific_identity,
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
        "simulation_semantics_id": SIMULATION_SEMANTICS_ID,
        "scientific_identity": scientific_identity,
        "allocation_patch_id": allocation_patch_id,
        "allocation_patch_label": ALLOCATION_PATCH_LABEL,
        "allocation_config": {
            "algorithm": "treatment_aware_constrained_lpt.v1",
            "machine_allocation_config": allocation_config_rows(machines),
            "effective_steps_per_second_semantics": "aggregate_machine_throughput_at_configured_worker_count",
            "allocation_waves": {"critical_stages": [1, 2, 3], "best_effort_stages": [4]},
        },
        "created_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "source_tree_hash": SourceIdentity.compute(REPO_ROOT).source_tree_hash,
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
        "kpi_schema_version": FULL_KPI_V3_SCHEMA_VERSION,
        "machines": [asdict(machine) for machine in machines],
        "machine_config_hash": hash_machine_config(machines),
        "machine_rates_used": {
            machine.machine_id: {
                "reference_effective_steps_per_second": machine.reference_effective_steps_per_second or machine.effective_steps_per_second,
                "rl_effective_steps_per_second": machine.rl_effective_steps_per_second or (machine.effective_steps_per_second if "all_on_rl" in machine.allowed_treatments else None),
                "rl_rate_source": machine.rl_rate_source,
            }
            for machine in machines
        },
        "assets": {
            key: value for key, value in asdict(assets).items()
            if key not in {"charging_config_relative_path", "charging_config_sha256"}
        },
        "treatment_contracts": {
            policy: treatment_execution_contract(policy, assets)
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
    stages = (1, 2, 3, 4)
    by_stage = {s: [run for run in runs if run["stage_first_requested"] == s] for s in stages}
    by_machine_total = {
        machine_id: sum(1 for run in runs if run["machine_id"] == machine_id)
        for machine_id in machine_ids
    }
    rl_hashes = {
        (run["identity"]["rts_checkpoint_sha256"], run["identity"]["pps_model_sha256"])
        for run in runs
        if run["policy_configuration"] == "all_on_rl"
    }

    # ── Pairing validation (Section 5): group by paired_group_id ──
    pairs: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for run in runs:
        pid = run["paired_group_id"]
        pairs.setdefault(pid, {"all_off": [], "all_on_rl": []})[run["policy_configuration"]].append(run)
    pair_defects = []
    seed_mismatch = layout_mismatch = duplicate_member = 0
    for pid, members in pairs.items():
        off, on = members["all_off"], members["all_on_rl"]
        if len(off) != 1 or len(on) != 1:
            # incomplete/duplicate pair (allowed to exist mid-run, but flagged)
            if len(off) > 1 or len(on) > 1:
                duplicate_member += 1
            pair_defects.append(pid)
            continue
        if off[0]["seed"] != on[0]["seed"]:
            seed_mismatch += 1
        if off[0]["identity"]["layout_hash"] != on[0]["identity"]["layout_hash"]:
            layout_mismatch += 1
    complete_pairs = [p for p, m in pairs.items() if len(m["all_off"]) == 1 and len(m["all_on_rl"]) == 1]

    assertions = {
        "machine_count": len(machines),
        "stage_new_runs": {str(s): len(by_stage[s]) for s in stages},
        "total_unique_fresh_runs": len({run["condition_key"] for run in runs}),
        "total_fresh_runs_by_machine": by_machine_total,
        "replication_seeds": {"1": seed_for_replication(1), "40": seed_for_replication(40)},
        "unique_run_ids": len({run["run_id"] for run in runs}),
        "unique_condition_keys": len({run["condition_key"] for run in runs}),
        "duplicate_run_ids": len(runs) - len({run["run_id"] for run in runs}),
        "duplicate_condition_keys": len(runs) - len({run["condition_key"] for run in runs}),
        "old_capacity_study_roots_contribute_completions": 0,
        "rts_rl_unique_asset_hash_pairs": len(rl_hashes),
        "primary_treatments": sorted({run["policy_configuration"] for run in runs}),
        "pair_count": len(pairs),
        "complete_pair_count": len(complete_pairs),
        "pair_seed_mismatches": seed_mismatch,
        "pair_layout_mismatches": layout_mismatch,
        "pair_duplicate_members": duplicate_member,
        "critical_wave_runs": sum(1 for run in runs if run.get("allocation_wave") == "critical"),
        "best_effort_wave_runs": sum(1 for run in runs if run.get("allocation_wave") == "best_effort"),
    }
    # ── Hard assertions (Sections 3, 4, 5) ──
    if assertions["stage_new_runs"] != {"1": 18, "2": 39, "3": 39, "4": 624}:
        raise AssertionError(f"stage counts: {assertions['stage_new_runs']}")
    if assertions["total_unique_fresh_runs"] != 720:
        raise AssertionError(f"total unique conditions must be 720, got {assertions['total_unique_fresh_runs']}")
    if assertions["replication_seeds"] != {"1": 42, "40": 81}:
        raise AssertionError(assertions["replication_seeds"])
    if assertions["duplicate_run_ids"] or assertions["duplicate_condition_keys"]:
        raise AssertionError("duplicate campaign identities detected")
    if assertions["rts_rl_unique_asset_hash_pairs"] != 1:
        raise AssertionError("all_on_rl runs must pin exactly one RTS/PPS hash pair")
    if assertions["primary_treatments"] != ["all_off", "all_on_rl"]:
        raise AssertionError(assertions["primary_treatments"])
    # Every condition belongs to a pair; a full 720 matrix has 360 complete pairs.
    if assertions["pair_count"] != 360:
        raise AssertionError(f"expected 360 paired groups, got {assertions['pair_count']}")
    if assertions["complete_pair_count"] != 360:
        raise AssertionError(f"expected 360 complete pairs, got {assertions['complete_pair_count']}")
    if assertions["pair_seed_mismatches"] or assertions["pair_layout_mismatches"] or assertions["pair_duplicate_members"]:
        raise AssertionError("paired seed/layout/member invariant violated")
    if assertions["critical_wave_runs"] != 96 or assertions["best_effort_wave_runs"] != 624:
        raise AssertionError("allocation wave counts must be 96 critical / 624 best-effort")
    for machine_id in ("alisha_pc", "dewa_macbook"):
        if machine_id not in machine_ids:
            continue
        if any(run["machine_id"] == machine_id and run["policy_configuration"] == "all_on_rl" for run in runs):
            raise AssertionError(f"{machine_id} received an ineligible all_on_rl run")
        if not any(run["machine_id"] == machine_id and run["policy_configuration"] == "all_off" and run.get("allocation_wave") == "critical" for run in runs):
            raise AssertionError(f"{machine_id} lacks a critical compatible run")
    return assertions


def machine_map(manifest: dict[str, Any]) -> dict[str, Machine]:
    return {row["machine_id"]: Machine(**row) for row in manifest["machines"]}


def campaign_root(repo_root: Path, manifest: dict[str, Any]) -> Path:
    return repo_root / manifest["campaign_root_relative"]


def write_launchers(root: Path, manifest: dict[str, Any]) -> dict[str, str]:
    launchers: dict[str, str] = {}
    for row in manifest["machines"]:
        machine = Machine(**row)
        if machine.os.lower().startswith("win"):
            path = root / "launchers" / f"run_{machine.machine_id}.ps1"
            content = "\n".join([
                "$ErrorActionPreference = \"Stop\"",
                f"Set-Location -LiteralPath \"{machine.repository}\"",
                (
                    f"& \"{machine.python}\" \"scripts\\experiments\\distributed_sensitivity_campaign.py\" "
                    f"--machine-id \"{machine.machine_id}\" --launch-host --progress"
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
                    f"--machine-id {json.dumps(machine.machine_id)} --launch-host --progress"
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
    assignment_paths = {}
    for machine in manifest["machines"]:
        machine_id = machine["machine_id"]
        assigned_runs = [
            run for run in manifest["runs"]
            if run["machine_id"] == machine_id
        ]
        paths = {}
        for wave, stages in (("critical", (1, 2, 3)), ("best_effort", (4,))):
            assignment = {
                "schema_version": CAMPAIGN_SCHEMA_VERSION,
                "campaign_id": manifest["campaign_id"],
                "allocation_patch_id": manifest["allocation_patch_id"],
                "simulation_semantics_id": manifest["simulation_semantics_id"],
                "machine_id": machine_id,
                "allocation_wave": wave,
                "assigned_conditions": [run for run in assigned_runs if int(run["stage_first_requested"]) in stages],
                "stage_counts": {
                    str(stage): sum(1 for run in assigned_runs if run["stage_first_requested"] == stage)
                    for stage in stages
                },
            }
            path = root / "host_assignments" / f"{machine_id}_{wave}.json"
            write_json(path, assignment)
            paths[wave] = path.relative_to(root).as_posix()
        assignment_paths[machine_id] = paths
    manifest = dict(manifest)
    manifest["host_assignments"] = assignment_paths
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


def validate_local_assets(manifest: dict[str, Any], repo_root: Path, *, strict: bool = True) -> None:
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
        strict=strict,
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
    charging = dict(condition.get("charging", {}))
    generated_charging_path = None
    if charging.get("charging_enabled"):
        generated_charging_path = condition_runtime_root(repo_root, manifest, machine.machine_id, condition["run_id"]) / "input_snapshot" / "salsa_adaptive_charging.json"
        write_json(generated_charging_path, charging["config"])
    contract = treatment_execution_contract(condition["policy_configuration"], assets)
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
        "persist_final_state": bool(contract["persist_final_state"]),
        "keep_runtime_artifacts": False,
        "detail_db": bool(contract["detail_db"]),
        "timing": bool(contract["timing"]),
        "worker_status_cadence": int(contract["worker_status_cadence"]),
        "run_profile": "training",
        "run_horizon_ticks": int(condition["ticks"]),
        "demand_horizon_ticks": int(condition["ticks"]) + 1000,
        "demand_buffer_ticks": 1000,
        "order_generation_mode": str(contract["order_generation_mode"]),
        "full_raw_order_replay": bool(contract["full_raw_order_replay"]),
        "order_rate_per_hour": int(condition["order_rate"]),
        "pod_location_mode": str(contract["pod_location_mode"]),
        "pod_location_seed": seed,
        "experiment_id": manifest["campaign_id"],
        "scenario_id": condition["scenario_id"],
        "artifact_label": condition["run_id"],
        "batch_id": int(condition["stage_first_requested"]),
        "worker_id": int(condition["replication"]),
        "robot_task_allocator": str(contract["robot_task_allocator"]),
        "regret_k": contract["regret_k"],
        "task_allocator_scope": str(contract["task_allocator_scope"]),
        "rts_torch_threads": 1,
        "rts_torch_interop_threads": 1,
        "campaign_id": manifest["campaign_id"],
        "allocation_patch_id": manifest["allocation_patch_id"],
        "simulation_semantics_id": manifest["simulation_semantics_id"],
        "machine_id": machine.machine_id,
        "stage_first_requested": int(condition["stage_first_requested"]),
        "kpi_schema_version": manifest["kpi_schema_version"],
        "policy_configuration": condition["policy_configuration"],
        "replication": int(condition["replication"]),
        "campaign_seed": seed,
        "source_tree_hash": manifest.get("source_tree_hash"),
        "charging_placement_source": str(contract["charging_placement_source"]),
        "charging_config_sha256": charging.get("config_sha256"),
        "charging_realized_layout_sha256": charging.get("realized_layout_sha256"),
        "charging_declared_count": charging.get("declared_count"),
        "rts_checkpoint_sha256": assets["rts_model_sha256"],
        "pps_model_sha256": assets["pps_model_sha256"],
    }
    if str(contract["rts_policy_mode"]) == "rts_rl_explicit":
        return RunSpec(
            **common,
            rts_policy_mode=str(contract["rts_policy_mode"]),
            rts_rollout_enabled=bool(contract["rts_rollout_enabled"]),
            rts_rollout_write_disk=bool(contract["rts_rollout_write_disk"]),
            rts_zone_ids=list(contract["rts_zone_ids"]),
            rts_policy_checkpoint_dir=str(local_asset_path(repo_root, assets["rts_checkpoint_relative_dir"])),
            rts_policy_checkpoint_id=str(contract["rts_checkpoint_id"]),
            rts_policy_action_mode=str(contract["rts_policy_action_mode"]),
            rts_policy_device=str(contract["rts_policy_device"]),
            rts_feature_ablation=str(contract["rts_feature_ablation"]),
            rts_state_capture_mode=str(contract["rts_state_capture_mode"]),
            pps_mode=str(contract["pps_mode"]),
            # all_on_rl loads the compact policy-only inference artifact (routed to
            # the inference loader by basename); all_off passes no PPS model.
            pps_model_path=str(local_asset_path(repo_root, assets["pps_model_relative_path"])),
            charging_enabled=bool(contract["charging_enabled"]),
            charging_config_path=str(generated_charging_path) if generated_charging_path else None,
            committed_next_reservations_enabled=bool(contract["committed_next_reservations_enabled"]),
        )
    return RunSpec(
        **common,
        rts_policy_mode=str(contract["rts_policy_mode"]),
        rts_rollout_enabled=bool(contract["rts_rollout_enabled"]),
        rts_rollout_write_disk=bool(contract["rts_rollout_write_disk"]),
        rts_policy_checkpoint_id=str(contract["rts_checkpoint_id"]),
        rts_state_capture_mode=str(contract["rts_state_capture_mode"]),
        pps_mode=str(contract["pps_mode"]),
        charging_enabled=bool(contract["charging_enabled"]),
        charging_config_path=str(generated_charging_path) if generated_charging_path else None,
        committed_next_reservations_enabled=bool(contract["committed_next_reservations_enabled"]),
    )


RELAXED_COMPLETION_IDENTITY_FIELDS = {
    "allocation_patch_id",
    "machine_id",
    "repo_commit",
    "rts_checkpoint_id",
    "rts_checkpoint_sha256",
    "pps_model_sha256",
    "rts_metadata_canonical_sha256",
    "rts_feature_schema_canonical_sha256",
    "charging_config_canonical_sha256",
}


def run_complete_for_campaign(condition: dict[str, Any], spec: RunSpec, manifest: dict[str, Any], *, relaxed: bool = False) -> bool:
    spec_path = spec.runtime_root / "run_spec.json"
    summary_path = spec.runtime_root / "worker_summary.json"
    if not spec_path.exists() or not summary_path.exists():
        return False
    try:
        previous_spec = read_json(spec_path)
        summary = load_worker_summary(spec.runtime_root)
    except Exception:
        return False
    required_identity = condition["run_spec_identity"]
    summary_key_map = {
        "campaign_seed": "seed",
        "robot_count": "requested_robot_count",
    }
    for key, expected in required_identity.items():
        if relaxed and key in RELAXED_COMPLETION_IDENTITY_FIELDS:
            continue
        if previous_spec.get(key) != expected:
            return False
        summary_key = summary_key_map.get(key, key)
        if key in {"order_rate_per_hour", "netlogo_steps_requested", "tick_to_second"}:
            continue
        if relaxed and summary_key in RELAXED_COMPLETION_IDENTITY_FIELDS:
            continue
        if summary.get(summary_key) != expected:
            return False
    if summary.get("seed") != condition["seed"]:
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


def completion_failure_summary(
    condition: dict[str, Any],
    spec: RunSpec,
    return_codes: dict[str, int],
) -> dict[str, Any]:
    summary = load_worker_summary(spec.runtime_root)
    return {
        "run_id": condition["run_id"],
        "condition_key": condition["condition_key"],
        "stage": condition["stage_first_requested"],
        "return_code": return_codes.get(condition["run_id"]),
        "status": summary.get("status", "missing"),
        "error_type": summary.get("error_type", ""),
        "error_message": summary.get("error_message", ""),
    }


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


RUN_OUTCOME_BASE_FIELDS = (
    "campaign_id",
    "allocation_patch_id",
    "simulation_semantics_id",
    "run_id",
    "condition_key",
    "policy_configuration",
    "robot_count",
    "order_rate",
    "replication",
    "seed",
    "stage_first_requested",
    "status",
    "termination_reason",
    "source_machine_id",
    "repo_commit",
    "error_type",
    "error_message",
    "row_sha256",
)
RUN_OUTCOME_HASH_FIELDS = (
    "scenario_hash",
    "layout_hash",
    "source_tree_hash",
    "manifest_sha256",
    "paired_group_id",
    "charging_placement_source",
    "charging_config_sha256",
    "realized_layout_sha256",
    "effective_charger_coordinate_hash",
    "rts_checkpoint_id",
    "rts_checkpoint_sha256",
    "rts_metadata_canonical_sha256",
    "rts_feature_schema_canonical_sha256",
    "pps_model_sha256",
    "charging_config_canonical_sha256",
)
RUN_OUTCOME_FIELDS = tuple(dict.fromkeys((*RUN_OUTCOME_BASE_FIELDS, *RUN_OUTCOME_HASH_FIELDS, *FULL_KPI_V3_FIELDS)))
FAILED_CONDITION_FIELDS = tuple(dict.fromkeys((*RUN_OUTCOME_BASE_FIELDS, *RUN_OUTCOME_HASH_FIELDS, "attempt_count")))


def _row_sha256(row: dict[str, Any], fields: tuple[str, ...]) -> str:
    payload = {field: row.get(field, "") for field in fields if field != "row_sha256"}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")).hexdigest()


def _condition_outcome_row(
    *,
    manifest: dict[str, Any],
    machine_id: str,
    condition: dict[str, Any],
    summary: dict[str, Any],
    status: str,
    manifest_sha256: str = "",
    error_type: str = "",
    error_message: str = "",
) -> dict[str, Any]:
    """Build the self-contained, v3-only record retained after workspace removal."""
    identity = dict(condition.get("identity", {}))
    charging = dict(condition.get("charging", {}))
    row = {field: "" for field in RUN_OUTCOME_FIELDS}
    row.update({
        "campaign_id": manifest["campaign_id"],
        "allocation_patch_id": manifest.get("allocation_patch_id", ""),
        "simulation_semantics_id": manifest.get("simulation_semantics_id", ""),
        "run_id": condition["run_id"],
        "condition_key": condition["condition_key"],
        "policy_configuration": condition["policy_configuration"],
        "robot_count": condition["robot_count"],
        "order_rate": condition["order_rate"],
        "replication": condition["replication"],
        "seed": condition["seed"],
        "stage_first_requested": condition["stage_first_requested"],
        "status": status,
        "termination_reason": summary.get("termination_reason", summary.get("finalization", {}).get("reason", "")),
        "source_machine_id": machine_id,
        "repo_commit": summary.get("repo_commit", manifest.get("commit", "")),
        "error_type": error_type or summary.get("error_type", ""),
        "error_message": error_message or summary.get("error_message", ""),
        "scenario_hash": condition.get("scenario_hash", ""),
        "layout_hash": condition.get("layout_hash", ""),
        "source_tree_hash": condition.get("source_tree_hash", manifest.get("source_tree_hash", "")),
        "manifest_sha256": manifest_sha256,
        "paired_group_id": condition.get("paired_group_id", ""),
        "charging_placement_source": charging.get("charging_placement_source", identity.get("charging_placement_source", "")),
        "charging_config_sha256": charging.get("config_sha256", identity.get("charging_config_sha256", "")),
        "realized_layout_sha256": charging.get("realized_layout_sha256", identity.get("realized_layout_sha256", "")),
        "effective_charger_coordinate_hash": summary.get("effective_charger_coordinate_hash", ""),
        "rts_checkpoint_id": identity.get("rts_checkpoint_id", ""),
        "rts_checkpoint_sha256": identity.get("rts_checkpoint_sha256", ""),
        "rts_metadata_canonical_sha256": identity.get("rts_metadata_canonical_sha256", ""),
        "rts_feature_schema_canonical_sha256": identity.get("rts_feature_schema_canonical_sha256", ""),
        "pps_model_sha256": identity.get("pps_model_sha256", ""),
        "charging_config_canonical_sha256": "",
    })
    kpi = summary.get("kpi", summary)
    for field in FULL_KPI_V3_FIELDS:
        if field in kpi and kpi.get(field) is not None:
            row[field] = kpi[field]
    row["row_sha256"] = _row_sha256(row, RUN_OUTCOME_FIELDS)
    return row


def _upsert_csv_row(path: Path, fields: tuple[str, ...], row: dict[str, Any], *, key: str = "run_id") -> str:
    """Atomically persist one terminal record and return its verified row hash."""
    records: dict[str, dict[str, str]] = {}
    if path.exists():
        with path.open(newline="", encoding="utf-8") as fh:
            records = {str(existing.get(key, "")): existing for existing in csv.DictReader(fh) if existing.get(key)}
    records[str(row[key])] = {field: row.get(field, "") for field in fields}
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    with tmp_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records[k] for k in sorted(records))
    replace_with_retry(tmp_path, path)
    with path.open(newline="", encoding="utf-8") as fh:
        persisted = next((item for item in csv.DictReader(fh) if item.get(key) == str(row[key])), None)
    if not persisted or persisted.get("row_sha256") != row.get("row_sha256"):
        raise RuntimeError(f"durable CSV verification failed for {key}={row[key]}")
    return str(row["row_sha256"])


def rebuild_run_outcomes_csv(*, manifest: dict[str, Any], machine: Machine, repo_root: Path) -> Path:
    machine_root = campaign_root(repo_root, manifest) / machine.machine_id
    conditions = {
        run["run_id"]: run
        for run in manifest["runs"]
        if run["machine_id"] == machine.machine_id
    }
    outcomes_path = machine_root / "run_outcomes.csv"
    # Section 8: preserve rows already durably written for runs whose per-run
    # workspace was deleted after finalization (append-persist, not rebuild-only).
    preserved: dict[str, dict[str, str]] = {}
    if outcomes_path.exists():
        try:
            with outcomes_path.open(newline="", encoding="utf-8") as fh:
                for existing in csv.DictReader(fh):
                    rid = existing.get("run_id")
                    if rid:
                        preserved[rid] = existing
        except Exception:
            preserved = {}
    fresh: dict[str, dict[str, Any]] = {}
    for run_id, condition in sorted(conditions.items(), key=lambda item: condition_sort_key(item[1])):
        summary_path = machine_root / "runs" / run_id / "worker_summary.json"
        if not summary_path.exists():
            continue
        try:
            summary = read_json(summary_path)
        except Exception:
            continue
        row = {field: "" for field in RUN_OUTCOME_FIELDS}
        row.update({
            "campaign_id": manifest["campaign_id"],
            "allocation_patch_id": manifest["allocation_patch_id"],
            "simulation_semantics_id": manifest["simulation_semantics_id"],
            "run_id": run_id,
            "condition_key": condition["condition_key"],
            "policy_configuration": condition["policy_configuration"],
            "robot_count": condition["robot_count"],
            "order_rate": condition["order_rate"],
            "replication": condition["replication"],
            "seed": condition["seed"],
            "stage_first_requested": condition["stage_first_requested"],
            "status": summary.get("status", ""),
            "termination_reason": summary.get("termination_reason", summary.get("finalization", {}).get("reason", "")),
            "source_machine_id": machine.machine_id,
            "repo_commit": summary.get("repo_commit", manifest.get("commit", "")),
            "error_type": summary.get("error_type", ""),
            "error_message": summary.get("error_message", ""),
            "scenario_hash": condition.get("scenario_hash", ""),
            "layout_hash": condition.get("layout_hash", ""),
            "rts_metadata_canonical_sha256": condition["identity"].get("rts_metadata_canonical_sha256", ""),
            "rts_feature_schema_canonical_sha256": condition["identity"].get("rts_feature_schema_canonical_sha256", ""),
            "charging_config_canonical_sha256": condition["identity"].get("charging_config_canonical_sha256", ""),
        })
        kpi_fields_set = SENSITIVITY_KPI_FIELDS
        if manifest.get("kpi_schema_version") == "full_kpi_v3":
            kpi_fields_set = FULL_KPI_V3_FIELDS
        for field in kpi_fields_set:
            if field in RUN_OUTCOME_FIELDS:
                value = summary.get(field, summary.get("kpi", {}).get(field, ""))
                if value not in ("", None) or row.get(field, "") == "":
                    row[field] = value
        for field in RUN_OUTCOME_HASH_FIELDS:
            if row.get(field, "") == "":
                row[field] = summary.get(field, "")
        fresh[run_id] = row

    # Merge: a freshly-scanned dir wins; otherwise keep the durably-preserved row.
    merged = dict(preserved)
    merged.update(fresh)
    ordered_ids = [rid for rid in conditions if rid in merged]
    ordered_ids += [rid for rid in merged if rid not in conditions]  # any extras last
    rows = [merged[rid] for rid in ordered_ids]

    path = outcomes_path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    with tmp_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(RUN_OUTCOME_FIELDS), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    replace_with_retry(tmp_path, path)
    return path


def run_outcomes_contains(machine_root: Path, run_id: str) -> bool:
    """Section 8: confirm a run's KPI row is durably present before deleting its workspace."""
    path = machine_root / "run_outcomes.csv"
    if not path.exists():
        return False
    try:
        with path.open(newline="", encoding="utf-8") as fh:
            return any(r.get("run_id") == run_id for r in csv.DictReader(fh))
    except Exception:
        return False


def condition_sort_key(run: dict[str, Any]) -> tuple[Any, ...]:
    return (
        int(run["stage_first_requested"]),
        run["policy_configuration"],
        int(run["robot_count"]),
        int(run["order_rate"]),
        int(run["replication"]),
        run["run_id"],
    )


def min_free_disk_bytes_from_env() -> int:
    raw = os.environ.get("RMFS_SENSITIVITY_MIN_FREE_GB")
    try:
        gb = float(raw) if raw is not None else DEFAULT_MIN_FREE_DISK_GB
    except (TypeError, ValueError):
        gb = DEFAULT_MIN_FREE_DISK_GB
    return int(max(0.0, gb) * 1024 * 1024 * 1024)


def free_disk_bytes(path: Path) -> int:
    target = Path(path)
    while not target.exists() and target != target.parent:
        target = target.parent
    return int(shutil.disk_usage(target).free)


def merge_reclaim_stats(target: dict[str, Any], addition: dict[str, Any]) -> dict[str, Any]:
    target["bytes"] = int(target.get("bytes", 0) or 0) + int(addition.get("bytes", 0) or 0)
    by_filename = dict(target.get("by_filename", {}) or {})
    for name, size in dict(addition.get("by_filename", {}) or {}).items():
        by_filename[str(name)] = by_filename.get(str(name), 0) + int(size)
    target["by_filename"] = by_filename
    return target


def execute_machine(
    *,
    manifest_path: Path,
    machine_id: str,
    stages: list[int],
    resume: bool,
    progress: bool,
    keep_run_artifacts: bool = False,
) -> int:
    ensure_clean_tracked_scientific_files(REPO_ROOT)
    manifest = read_json(manifest_path)
    machines = machine_map(manifest)
    if machine_id not in machines:
        raise SystemExit(f"unknown machine id {machine_id}; expected one of {sorted(machines)}")
    machine = machines[machine_id]
    validate_local_assets(manifest, REPO_ROOT, strict=False)
    assigned_runs = [run for run in manifest["runs"] if run["machine_id"] == machine_id]
    cleanup_stats: dict[str, Any] = {"bytes": 0, "by_filename": {}}
    startup_reclaimed_runs = 0
    if not keep_run_artifacts:
        for condition in assigned_runs:
            run_root = condition_runtime_root(REPO_ROOT, manifest, machine_id, condition["run_id"])
            if run_complete_for_campaign(
                condition,
                build_run_spec_from_condition(condition, manifest=manifest, machine=machine, repo_root=REPO_ROOT),
                manifest,
            ):
                stats = reclaim_completed_run_artifacts_with_stats(run_root)
            else:
                stats = reclaim_interrupted_run_artifacts_with_stats(run_root)
            if int(stats.get("bytes", 0) or 0) > 0:
                startup_reclaimed_runs += 1
                merge_reclaim_stats(cleanup_stats, stats)
        if int(cleanup_stats.get("bytes", 0) or 0) > 0:
            print(
                f"[sensitivity] startup sweep: reclaimed {int(cleanup_stats['bytes']) / 1_000_000:.0f} MB "
                f"from {startup_reclaimed_runs} run(s)"
            )

    selected = sorted(
        [run for run in assigned_runs if int(run["stage_first_requested"]) in {int(stage) for stage in stages}],
        key=condition_sort_key,
    )
    specs = []
    skipped = []
    for condition in selected:
        spec = build_run_spec_from_condition(condition, manifest=manifest, machine=machine, repo_root=REPO_ROOT)
        if resume and run_complete_for_campaign(condition, spec, manifest):
            skipped.append(condition["run_id"])
            continue
        specs.append(spec)

    min_free = min_free_disk_bytes_from_env()
    campaign_root_path = campaign_root(REPO_ROOT, manifest)
    peak_runtime_directory_bytes = 0

    def _before_launch(spec: RunSpec, active_count: int) -> bool:
        available = free_disk_bytes(campaign_root_path)
        if available >= min_free:
            return True
        if active_count > 0:
            print(
                "[sensitivity] free disk below floor; pausing new launches until active workers finish: "
                f"path={campaign_root_path} free={available} required={min_free}"
            )
            return False
        raise RuntimeError(
            "insufficient free disk before launching sensitivity worker: "
            f"path={campaign_root_path} free={available} required={min_free}"
        )

    def _reclaim_on_complete(spec: RunSpec, return_code: int) -> None:
        nonlocal peak_runtime_directory_bytes
        peak_runtime_directory_bytes = max(peak_runtime_directory_bytes, directory_size_bytes(spec.runtime_root))
        if keep_run_artifacts:
            return
        try:
            summary = load_worker_summary(spec.runtime_root)
            if int(return_code) == 0 and summary.get("status") == "success":
                stats = reclaim_completed_run_artifacts_with_stats(spec.runtime_root)
            else:
                stats = reclaim_failed_run_artifacts_with_stats(spec.runtime_root)
            merge_reclaim_stats(cleanup_stats, stats)
        except Exception as exc:
            print(
                f"[sensitivity] warning: cleanup failed for {spec.run_id}: "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
            )

    previous_log_cap = os.environ.get("RMFS_WORKER_LOG_CAP_MB")
    os.environ["RMFS_WORKER_LOG_CAP_MB"] = "5"
    start = time.perf_counter()
    completed = []
    run_return_codes: dict[str, int] = {}
    try:
        if specs:
            print(
                f"[sensitivity] starting combined stage-priority queue: {len(specs)} runs, "
                f"max_workers={machine.max_workers}"
            )
            completed = run_specs(
                specs,
                max_workers=int(machine.max_workers),
                progress=progress,
                on_run_complete=_reclaim_on_complete,
                before_launch=_before_launch,
            )
            try:
                rebuild_run_outcomes_csv(manifest=manifest, machine=machine, repo_root=REPO_ROOT)
            except Exception as exc:
                print(
                    "[sensitivity] warning: run_outcomes.csv rebuild failed after worker queue: "
                    f"{type(exc).__name__}: {exc}",
                    file=sys.stderr,
                )
    finally:
        if previous_log_cap is None:
            os.environ.pop("RMFS_WORKER_LOG_CAP_MB", None)
        else:
            os.environ["RMFS_WORKER_LOG_CAP_MB"] = previous_log_cap

    for item in completed:
        run_return_codes[item["spec"].run_id] = int(item.get("return_code", -1))
        peak_runtime_directory_bytes = max(
            peak_runtime_directory_bytes,
            int(item.get("peak_runtime_directory_bytes", 0) or 0),
        )
    elapsed = time.perf_counter() - start
    rebuild_run_outcomes_csv(manifest=manifest, machine=machine, repo_root=REPO_ROOT)
    write_machine_summary(
        manifest=manifest,
        machine=machine,
        repo_root=REPO_ROOT,
        stage=None if stages == [1, 2, 3, 4] else min(stages) if len(stages) == 1 else None,
        launched=[spec.run_id for spec in specs],
        skipped=skipped,
        elapsed_seconds=elapsed,
    )
    machine_summary_path = campaign_root_path / machine.machine_id / "machine_summary.json"
    if machine_summary_path.exists():
        summary = read_json(machine_summary_path)
        summary.update({
            "stages_requested": [int(stage) for stage in stages],
            "allocation_patch_id": manifest["allocation_patch_id"],
            "simulation_semantics_id": manifest["simulation_semantics_id"],
            "cleanup_reclaimed_total_bytes": cleanup_stats.get("bytes", 0),
            "cleanup_reclaimed_bytes_by_filename": cleanup_stats.get("by_filename", {}),
            "persist_final_state_enabled": False,
            "peak_runtime_directory_bytes": peak_runtime_directory_bytes,
            "min_free_disk_bytes": min_free,
        })
        write_json(machine_summary_path, summary)
    if int(cleanup_stats.get("bytes", 0) or 0) > 0:
        print(f"[sensitivity] reclaimed {int(cleanup_stats['bytes']) / 1_000_000:.0f} MB during this invocation")

    invalid = []
    for condition in selected:
        spec = build_run_spec_from_condition(condition, manifest=manifest, machine=machine, repo_root=REPO_ROOT)
        if not run_complete_for_campaign(condition, spec, manifest):
            invalid.append(completion_failure_summary(condition, spec, run_return_codes))
    if invalid:
        try:
            rebuild_run_outcomes_csv(manifest=manifest, machine=machine, repo_root=REPO_ROOT)
        except Exception as exc:
            print(f"[sensitivity] warning: failed to rebuild run_outcomes.csv after invalid runs: {type(exc).__name__}: {exc}")
        print(f"[sensitivity] {len(invalid)} selected run(s) did not satisfy strict completion:")
        for row in invalid[:20]:
            print(
                " - "
                f"run_id={row['run_id']} condition={row['condition_key']} stage={row['stage']} "
                f"return_code={row['return_code']} status={row['status']} "
                f"error={row['error_type']}: {row['error_message']}"
            )
        if len(invalid) > 20:
            print(f" - ... {len(invalid) - 20} more")
        return 1
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


def execute_host(
    *,
    manifest_path: Path,
    machine_id: str,
    stages: list[int],
    resume: bool,
    progress: bool,
    max_retries: int = 2,
    host_data_root: Path | None = None,
    keep_run_artifacts: bool = False,
) -> int:
    from src.rmfs.orchestration.host_ledger import HostLedger
    from src.rmfs.orchestration.source_identity import SourceIdentity

    manifest = read_json(manifest_path)
    try:
        source_ident = validate_source_identity(manifest, REPO_ROOT)
    except Exception as exc:
        print(f"\n[CRITICAL ERROR] Source identity validation failed: {exc}\n", file=sys.stderr)
        raise SystemExit(f"campaign stopped: source identity invalid: {exc}")

    all_dirty = dirty_tracked_files(REPO_ROOT)
    non_scientific_dirty = [p for p in all_dirty if p not in SCIENTIFIC_TRACKED_PATHS]
    if non_scientific_dirty:
        print("[sensitivity] warning: non-scientific tracked files differ from Git; portable source hash remains authoritative")
    machines = machine_map(manifest)
    if machine_id not in machines:
        raise SystemExit(f"unknown machine id {machine_id}; expected one of {sorted(machines)}")
    machine = machines[machine_id]

    try:
        # Missing or unloadable canonical assets still stop this host.  Hash,
        # metadata, and checkpoint provenance mismatches are diagnostics only.
        validate_local_assets(manifest, REPO_ROOT, strict=False)
    except Exception as exc:
        print(f"\n[CRITICAL ERROR] Campaign-wide stop: asset or base input validation failed: {exc}\n", file=sys.stderr)
        raise SystemExit(f"campaign stopped: validation failed: {exc}")

    if host_data_root is None:
        campaign_id = manifest["campaign_id"]
        host_data_root = REPO_ROOT / "data" / "runtime" / "distributed_sensitivity" / campaign_id / machine_id
    
    host_data_root.mkdir(parents=True, exist_ok=True)
    ledger_path = host_data_root / "host_ledger.json"
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()

    immutable_hashes = {
        key: str(value)
        for key, value in manifest.get("assets", {}).items()
        if key.endswith("sha256") and value
    }
    if ledger_path.exists():
        print(f"[sensitivity] Loading existing HostLedger from {ledger_path}")
        ledger = HostLedger.load(ledger_path)
        if ledger.campaign_id != manifest["campaign_id"]:
            raise RuntimeError(f"Ledger campaign_id {ledger.campaign_id} does not match manifest {manifest['campaign_id']}")
        if ledger.manifest_sha256 != manifest_sha:
            raise RuntimeError("host ledger manifest hash differs from the loaded manifest")
        if ledger.source_tree_hash and ledger.source_tree_hash != source_ident.source_tree_hash:
            raise RuntimeError("host source-tree hash differs from the assigned source identity")
        if ledger.immutable_hashes and ledger.immutable_hashes != immutable_hashes:
            raise RuntimeError("host immutable asset/input hashes differ from the assigned identity")
        if ledger.kpi_schema_version and ledger.kpi_schema_version != manifest.get("kpi_schema_version"):
            raise RuntimeError("host ledger KPI schema differs from the loaded manifest")
    else:
        print(f"[sensitivity] Creating new HostLedger under {host_data_root}")
        assigned = [
            run for run in manifest["runs"]
            if run["machine_id"] == machine_id
        ]
        ledger = HostLedger.create(
            host_id=machine_id,
            campaign_id=manifest["campaign_id"],
            manifest_sha256=manifest_sha,
            assigned_conditions=assigned,
            kpi_schema_version=manifest.get("kpi_schema_version"),
            source_tree_hash=source_ident.source_tree_hash,
            immutable_hashes=immutable_hashes,
        )
        ledger.save(ledger_path)

    runs_dir = host_data_root / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    
    def classify_result(condition, spec_dict, summary):
        """Classify a terminal result consistently for launch and restart."""
        if not spec_dict or not isinstance(summary, dict):
            return "quarantined"
        for key in ("policy_configuration", "robot_count", "order_rate_per_hour", "replication"):
            expected = condition.get("order_rate" if key == "order_rate_per_hour" else key)
            if spec_dict.get(key) != expected:
                return "quarantined"
        expected_seed = int(condition["seed"])
        if spec_dict.get("campaign_seed", spec_dict.get("rts_random_seed")) != expected_seed:
            return "quarantined"
        if int(spec_dict.get("ticks", -1)) != int(condition["ticks"]):
            return "quarantined"
        if summary.get("status") != "success" or not summary.get("finalization", {}).get("finalized", False):
            return "failed"
        if summary.get("kpi_schema_version") != manifest.get("kpi_schema_version"):
            return "quarantined"
        if not summary.get("kpi_complete", False):
            # A result with all required fields but missing optional diagnostics
            # remains useful.  Never elevate an incomplete KPI payload to strict.
            required = FULL_KPI_V3_REQUIRED_FIELDS if manifest.get("kpi_schema_version") == "full_kpi_v3" else SENSITIVITY_KPI_FIELDS
            payload = summary.get("kpi", summary)
            if all(payload.get(field) is not None for field in required if field != "kpi_complete"):
                return "completed_with_warnings"
            return "failed"
        return "completed_strict"

    print("[sensitivity] Reconciling ledger with local results directory...")
    rec_result = ledger.reconcile_with_outputs(runs_dir, classify_fn=classify_result)
    if rec_result["reconciled"]:
        print(f"[sensitivity] Reconciled {len(rec_result['reconciled'])} already completed conditions.")
        ledger.save(ledger_path)

    cleanup_stats: dict[str, Any] = {"bytes": 0, "by_filename": {}}
    min_free = min_free_disk_bytes_from_env()

    def _before_launch(spec: RunSpec, active_count: int) -> bool:
        available = free_disk_bytes(host_data_root)
        if available >= min_free:
            return True
        if active_count > 0:
            print(f"[sensitivity] free disk below floor; pausing: free={available} required={min_free}")
            return False
        raise RuntimeError(f"insufficient free disk: free={available} required={min_free}")

    previous_log_cap = os.environ.get("RMFS_WORKER_LOG_CAP_MB")
    os.environ["RMFS_WORKER_LOG_CAP_MB"] = "5"

    eligible_stages = {int(stage) for stage in stages}
    stage_order = tuple(stage for stage in (1, 2, 3, 4) if stage in eligible_stages)
    try:
        while True:
            # A host drains one local stage before it admits the next.  This
            # keeps the best-effort Stage 4 wave out of worker slots until
            # this machine's Stage 3 allocation has reached a terminal state.
            active_stage = next(
                (
                    stage
                    for stage in stage_order
                    if ledger.next_condition(eligible_stages={stage}) is not None
                ),
                None,
            )
            if active_stage is None:
                break
            batch: list[tuple[dict[str, Any], RunSpec, float]] = []
            while len(batch) < int(machine.max_workers):
                cond = ledger.next_condition(eligible_stages={active_stage})
                if cond is None:
                    break
                spec = build_run_spec_from_condition(cond, manifest=manifest, machine=machine, repo_root=REPO_ROOT)
                object.__setattr__(spec, "runtime_root", runs_dir / cond["run_id"])
                ledger.start_condition(str(cond["condition_key"]))
                batch.append((cond, spec, time.perf_counter()))
            ledger.save(ledger_path)
            print(
                f"[sensitivity] launching stage {active_stage}: {len(batch)} independent conditions "
                f"(worker budget={machine.max_workers})"
            )
            completed_by_run: dict[str, dict[str, Any]] = {}
            try:
                completed = run_specs(
                    [spec for _cond, spec, _started in batch], max_workers=int(machine.max_workers),
                    progress=progress, before_launch=_before_launch,
                )
                completed_by_run = {item["spec"].run_id: item for item in completed}
            except Exception as exc:
                # run_specs drains active children before raising.  Reconciliation below
                # records each condition independently, preserving healthy siblings.
                print(f"[sensitivity] worker batch callback/executor exception: {exc}", file=sys.stderr)

            for cond, spec, started in batch:
                item = completed_by_run.get(spec.run_id, {})
                return_code = int(item.get("return_code", -1))
                # run_specs captured this before any host cleanup.  Do not
                # reread the workspace: it is deliberately removed once the
                # CSV and ledger records below are verified.
                summary = item.get("worker_summary") if isinstance(item.get("worker_summary"), dict) else {
                    "status": "failure", "error_type": "missing_terminal_result",
                    "error_message": "executor did not return a terminal worker summary",
                }
                classification = classify_result(cond, spec.to_json_dict(), summary)
                elapsed = time.perf_counter() - started
                if classification in {"completed_strict", "completed_with_warnings"}:
                    try:
                        row = _condition_outcome_row(
                            manifest=manifest, machine_id=machine_id, condition=cond, summary=summary,
                            status=classification, manifest_sha256=manifest_sha,
                        )
                        row_hash = _upsert_csv_row(host_data_root / "run_outcomes.csv", RUN_OUTCOME_FIELDS, row)
                        ledger.mark_completed(str(cond["condition_key"]), {
                            "run_id": spec.run_id,
                            "result_path": f"run_outcomes.csv#{spec.run_id}",
                            "row_sha256": row_hash,
                            "return_code": return_code,
                            "elapsed_seconds": elapsed,
                        }, warning_count=int(classification == "completed_with_warnings"))
                        ledger.save(ledger_path)
                    except Exception as exc:
                        # Never lose a result to a host persistence failure.
                        ledger.mark_failed(str(cond["condition_key"]), {
                            "return_code": return_code, "elapsed_seconds": elapsed,
                            "error_type": "host_persistence_failure", "error_message": str(exc),
                        }, max_retries=max_retries, quarantined=True)
                        ledger.save(ledger_path)
                        continue
                    if not keep_run_artifacts:
                        stats = reclaim_completed_run_artifacts_with_stats(spec.runtime_root)
                        merge_reclaim_stats(cleanup_stats, stats)
                        if spec.runtime_root.exists():
                            shutil.rmtree(spec.runtime_root, ignore_errors=False)
                else:
                    ledger.mark_failed(str(cond["condition_key"]), {
                        "return_code": return_code, "elapsed_seconds": elapsed,
                        "status": summary.get("status", "failure"),
                        "error_type": summary.get("error_type", classification),
                        "error_message": summary.get("error_message", classification),
                    }, max_retries=max_retries, quarantined=classification == "quarantined")
                    state = ledger.state_for(str(cond["condition_key"]))
                    if state.get("status") in {"failed_final", "quarantined"}:
                        failure = _condition_outcome_row(
                            manifest=manifest, machine_id=machine_id, condition=cond, summary=summary,
                            status=state["status"], manifest_sha256=manifest_sha,
                            error_type=str(summary.get("error_type", classification)),
                            error_message=str(summary.get("error_message", classification)),
                        )
                        failure["attempt_count"] = str(state.get("attempt_count", 0))
                        failure = {field: failure.get(field, "") for field in FAILED_CONDITION_FIELDS}
                        failure["row_sha256"] = _row_sha256(failure, FAILED_CONDITION_FIELDS)
                        try:
                            row_hash = _upsert_csv_row(host_data_root / "failed_conditions.csv", FAILED_CONDITION_FIELDS, failure)
                            state["terminal_failure_row_sha256"] = row_hash
                            ledger.save(ledger_path)
                            if not keep_run_artifacts:
                                stats = reclaim_failed_run_artifacts_with_stats(spec.runtime_root)
                                merge_reclaim_stats(cleanup_stats, stats)
                                if spec.runtime_root.exists():
                                    shutil.rmtree(spec.runtime_root, ignore_errors=False)
                        except Exception as exc:
                            # Keep the complete workspace for recovery if the
                            # compact terminal failure record cannot be proven.
                            state["failure_reason"] = f"host_persistence_failure: {exc}"
                            ledger.save(ledger_path)
            ledger.save(ledger_path)

    finally:
        if previous_log_cap is None:
            os.environ.pop("RMFS_WORKER_LOG_CAP_MB", None)
        else:
            os.environ["RMFS_WORKER_LOG_CAP_MB"] = previous_log_cap

    write_machine_summary(
        manifest=manifest,
        machine=machine,
        repo_root=REPO_ROOT,
        stage=None if stages == [1, 2, 3, 4] else min(stages) if len(stages) == 1 else None,
        launched=[cond["run_id"] for cond in ledger.assigned_conditions],
        skipped=[],
        elapsed_seconds=0.0,
    )
    
    machine_summary_path = host_data_root / "machine_summary.json"
    if machine_summary_path.exists():
        msum = read_json(machine_summary_path)
        msum.update({
            "ledger_summary": ledger.summary(),
            "updated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        })
        write_json(machine_summary_path, msum)

    terminal_failures = [
        key for key, state in ledger.condition_states.items()
        if state.get("status") in {"failed_final", "quarantined"}
        and int((ledger._condition_by_key(key) or {}).get("stage_first_requested", 0)) in eligible_stages
    ]
    if terminal_failures:
        print(f"[sensitivity] {len(terminal_failures)} selected conditions remain unusable.", file=sys.stderr)
        return 1
    return 0


def prepare_host_assignment(
    *,
    manifest_path: Path,
    machine_id: str,
    output_dir: Path,
) -> Path:
    from src.rmfs.orchestration.host_ledger import HostLedger
    from src.rmfs.orchestration.source_identity import SourceIdentity

    manifest = read_json(manifest_path)
    machines = machine_map(manifest)
    if machine_id not in machines:
        raise SystemExit(f"unknown machine id {machine_id}; expected one of {sorted(machines)}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    shard_runs = [
        run for run in manifest["runs"]
        if run["machine_id"] == machine_id
    ]
    
    write_json(output_dir / "manifest.json", manifest)
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    source_ident = SourceIdentity.compute(REPO_ROOT)
    
    ledger = HostLedger.create(
        host_id=machine_id,
        campaign_id=manifest["campaign_id"],
        manifest_sha256=manifest_sha,
        assigned_conditions=shard_runs,
        kpi_schema_version=manifest.get("kpi_schema_version"),
        source_tree_hash=source_ident.source_tree_hash,
    )
    ledger.save(output_dir / "host_ledger.json")
    
    print(f"[sensitivity] Prepared host assignment folder under: {output_dir}")
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare or run the distributed RMFS sensitivity campaign.")
    parser.add_argument("--prepare-campaign", action="store_true", default=False)
    parser.add_argument("--machine-id", choices=MACHINE_IDS)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--run-continuously", action="store_true", default=False)
    parser.add_argument("--stage", action="append", type=int, choices=(1, 2, 3, 4))
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--progress", action="store_true", default=False)
    parser.add_argument("--rebalance-future-stages", action="store_true", default=False)
    parser.add_argument("--dry-run", action="store_true", default=False)
    parser.add_argument("--rts-checkpoint-dir", default=None)
    parser.add_argument("--rts-training-artifact", default=None)
    parser.add_argument("--pps-model-path", default=str(DEFAULT_PPS_INFERENCE_ARTIFACT))
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
    parser.add_argument("--execute-host", action="store_true", default=False)
    parser.add_argument(
        "--launch-host",
        action="store_true",
        default=False,
        help="Materialize or reuse this host's ignored local plan, then run stages 1-4 in order.",
    )
    parser.add_argument("--prepare-host-assignment", action="store_true", default=False)
    parser.add_argument("--host-id", help="Override host machine ID")
    parser.add_argument("--max-retries", type=int, default=2, help="Retry budget per condition")
    parser.add_argument("--host-data-root", default=None, help="Root folder for local run results & ledger")
    parser.add_argument("--output-dir", default=None, help="Output folder for prepared host assignment")
    return parser


def materialize_local_campaign_plan(args: argparse.Namespace) -> tuple[Path, dict[str, Any]]:
    """Create a local ignored plan once, or reuse the one this host created.

    This deliberately does not compare local plan provenance with another
    machine's checkout.  Every host pulls the same code, then materializes its
    own plan from the files it will actually execute.
    """
    machines = default_machines(REPO_ROOT)
    assets = resolve_assets(args)
    campaign_id = generate_campaign_id(machines, assets, float(args.rl_overhead_multiplier))
    manifest = build_campaign_plan(
        campaign_id=campaign_id,
        machines=machines,
        assets=assets,
        rl_overhead_multiplier=float(args.rl_overhead_multiplier),
    )
    root = campaign_root(REPO_ROOT, manifest)
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        validation_warning(f"reusing local campaign plan: {manifest_path}")
        return manifest_path, read_json(manifest_path)
    if root.exists():
        raise RuntimeError(
            f"local campaign directory exists without a manifest: {root}; "
            "preserve it for inspection and choose a clean runtime root before launching"
        )
    print(f"[sensitivity] Materializing local campaign plan under: {root}")
    write_campaign_files(manifest, dry_run=False)
    return manifest_path, manifest


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.prepare_campaign or args.validate_only or args.run_continuously or args.stage or args.execute_host or args.launch_host or args.prepare_host_assignment:
        ensure_clean_tracked_scientific_files(REPO_ROOT)

    if args.prepare_host_assignment:
        if not args.manifest:
            raise SystemExit("--prepare-host-assignment requires --manifest")
        if not args.machine_id:
            raise SystemExit("--prepare-host-assignment requires --machine-id")
        if not args.output_dir:
            raise SystemExit("--prepare-host-assignment requires --output-dir")
        prepare_host_assignment(
            manifest_path=manifest_path_from_arg(args.manifest),
            machine_id=args.machine_id,
            output_dir=Path(args.output_dir),
        )
        return 0

    if args.execute_host:
        if not args.manifest:
            raise SystemExit("--execute-host requires --manifest")
        if not args.machine_id:
            raise SystemExit("--execute-host requires --machine-id")
        stages = args.stage or [1, 2, 3, 4]
        return execute_host(
            manifest_path=manifest_path_from_arg(args.manifest),
            machine_id=args.machine_id,
            stages=stages,
            resume=bool(args.resume),
            progress=bool(args.progress),
            max_retries=int(args.max_retries),
            host_data_root=Path(args.host_data_root) if args.host_data_root else None,
            keep_run_artifacts=bool(args.keep_run_artifacts),
        )

    if args.launch_host:
        if not args.machine_id:
            raise SystemExit("--launch-host requires --machine-id")
        manifest_path, _manifest = materialize_local_campaign_plan(args)
        return execute_machine(
            manifest_path=manifest_path,
            machine_id=args.machine_id,
            stages=[1, 2, 3, 4],
            resume=False,
            progress=bool(args.progress),
            keep_run_artifacts=bool(args.keep_run_artifacts),
        )

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
                
                if existing_manifest.get("allocation_patch_id") != new_manifest.get("allocation_patch_id"):
                    mismatches.append(
                        "allocation_patch_id mismatch: "
                        f"existing={existing_manifest.get('allocation_patch_id')}, expected={new_manifest.get('allocation_patch_id')}"
                    )
                if existing_manifest.get("simulation_semantics_id") != SIMULATION_SEMANTICS_ID:
                    mismatches.append("simulation_semantics_id mismatch")
                if existing_manifest.get("allocation_config", {}).get("machine_allocation_config") != new_manifest.get("allocation_config", {}).get("machine_allocation_config"):
                    mismatches.append("machine allocation configuration mismatch")
                     
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
            validate_source_identity(manifest, REPO_ROOT)
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
        
        # 2. Check bundled two-treatment identities (720 unique paired runs)
        runs = manifest["runs"]
        if len(runs) != 720:
            raise AssertionError(f"Expected 720 runs, got {len(runs)}")

        run_ids = [run["run_id"] for run in runs]
        condition_keys = [run["condition_key"] for run in runs]
        if len(set(run_ids)) != 720:
            raise AssertionError(f"Duplicate run IDs found, unique count: {len(set(run_ids))}")
        if len(set(condition_keys)) != 720:
            raise AssertionError(f"Duplicate condition keys found, unique count: {len(set(condition_keys))}")

        # 3. Check cumulative stage counts and seeds
        expected_stage_counts = {"1": 18, "2": 39, "3": 39, "4": 624}
        actual_stage_counts = manifest["assertions"]["stage_new_runs"]
        if actual_stage_counts != expected_stage_counts:
            raise AssertionError(f"Stage new runs count mismatch: expected {expected_stage_counts}, got {actual_stage_counts}")

        # Verify seeds
        for replication in (1, 40):
            expected_seed = seed_for_replication(replication)
            for run in runs:
                if run["replication"] == replication:
                    if run["seed"] != expected_seed:
                        raise AssertionError(f"Seed mismatch for rep {replication}: expected {expected_seed}, got {run['seed']}")
                        
        # 4. Construct every RunSpec
        machine_by_id = machine_map(manifest)
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
            if spec.persist_final_state is not False:
                raise AssertionError(f"RunSpec {run['run_id']} unexpectedly persists final state")
            if spec.rts_policy_checkpoint_id != (manifest["assets"]["rts_checkpoint_id"] if run["policy_configuration"] == "all_on_rl" else "not_applicable"):
                raise AssertionError(f"RunSpec {run['run_id']} has wrong RTS checkpoint ID")
        print(json.dumps({
            "campaign_id": manifest["campaign_id"],
            "allocation_patch_id": manifest["allocation_patch_id"],
            "simulation_semantics_id": manifest["simulation_semantics_id"],
            "stage_counts": actual_stage_counts,
            "total_runs": len(runs),
            "complete_pairs": manifest["assertions"]["complete_pair_count"],
            "clean_tracked_scientific_inputs": True,
            "strict_completion_required": True,
            "cleanup_errors_stop_worker_queue": False,
            "treatment_execution_contracts": manifest["scientific_identity"]["treatment_execution_contracts"],
            "sensitivity_persist_final_state": False,
            "expected_worker_files_without_state": [
                "assign_order.csv",
                "pod_info.csv",
                "skus_data.csv",
                "sorted_skus_data.csv",
                "worker_summary.json",
            ],
            "successful_run_retained_files": [],
            "successful_run_durable_records": ["run_outcomes.csv", "machine_summary.json"],
            "successful_run_deleted_files": list(SUCCESS_RECLAIMABLE_RUN_ARTIFACTS),
            "failed_run_deleted_files": list(FAILED_RECLAIMABLE_RUN_ARTIFACTS),
            "min_free_disk_bytes": min_free_disk_bytes_from_env(),
        }, indent=2, sort_keys=True))
        
        print("[sensitivity] Validation successful! All checks passed.")
        return 0

    if args.prepare_campaign:
        if args.dry_run:
            machines = default_machines(REPO_ROOT)
            assets = resolve_assets(args)
            campaign_id = generate_campaign_id(machines, assets, float(args.rl_overhead_multiplier))
            manifest = build_campaign_plan(
                campaign_id=campaign_id,
                machines=machines,
                assets=assets,
                rl_overhead_multiplier=float(args.rl_overhead_multiplier),
            )
            root = campaign_root(REPO_ROOT, manifest)
        else:
            manifest_path, manifest = materialize_local_campaign_plan(args)
            campaign_id = manifest["campaign_id"]
            root = manifest_path.parent
        print(json.dumps({
            "campaign_id": campaign_id,
            "allocation_patch_id": manifest["allocation_patch_id"],
            "simulation_semantics_id": manifest["simulation_semantics_id"],
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
