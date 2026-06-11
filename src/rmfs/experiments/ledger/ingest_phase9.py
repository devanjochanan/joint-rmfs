"""Ingest Phase 9 RTS training run outputs into the SQLite ledger."""

from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Any

from src.rmfs.experiments.feature_flags import default_rika_rts_rl_feature_flags
from src.rmfs.experiments.identity import make_experiment_run_id, make_scenario_id

from .writer import (
    json_text,
    upsert_checkpoint,
    upsert_experiment,
    upsert_training_batch,
    upsert_worker_rollout,
)


def read_json(path: Path, default: Any = None) -> Any:
    if not Path(path).exists():
        return default
    with Path(path).open() as fh:
        return json.load(fh)


def ingest_phase9_run(run_root: Path, db_path: Path) -> dict[str, Any]:
    run_root = Path(run_root)
    config = read_json(run_root / "training_config.json")
    if config is None:
        raise FileNotFoundError(run_root / "training_config.json")
    feature_flags = config.get("feature_flags") or default_rika_rts_rl_feature_flags()
    scenario_id = config.get("scenario_id") or make_scenario_id({"config": config, "feature_flags": feature_flags})

    # Derive experiment_id using config parameters but without path-dependent run_root
    artifact_label = config.get("artifact_label") or run_root.name
    experiment_id = config.get("experiment_id") or make_experiment_run_id({
        "artifact_label": artifact_label,
        "seed": config.get("seed"),
        "scenario_id": scenario_id,
        "repo_commit": config.get("commit"),
        "netlogo_steps_per_run": config.get("netlogo_steps_per_run"),
        "workers": config.get("workers"),
    })

    upsert_experiment(
        db_path,
        {
            "experiment_id": experiment_id,
            "scenario_id": scenario_id,
            "artifact_label": artifact_label,
            "phase": "phase9",
            "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "repo_branch": config.get("branch"),
            "repo_commit": config.get("commit"),
            "python_executable": config.get("python_executable"),
            "seed_base": config.get("seed"),
            "output_root": str(run_root),
            "status": "ingested",
            "config_json": json_text(config),
            "feature_flags_json": json_text(feature_flags),
        },
    )
    latest = read_json(run_root / "latest.json", {}) or {}
    latest_dir = latest.get("checkpoint_dir")
    for batch_dir in sorted(run_root.glob("batch_*")):
        if not batch_dir.is_dir():
            continue
        batch_id = int(batch_dir.name.split("_")[-1])
        summary = read_json(batch_dir / "batch_summary.json", {}) or {}
        dataset = summary.get("dataset_summary", {}) or {}
        ppo = summary.get("ppo_update", {}) or {}
        upsert_training_batch(
            db_path,
            {
                "experiment_id": experiment_id,
                "batch_id": batch_id,
                "status": summary.get("status"),
                "policy_checkpoint_id_before": summary.get("active_checkpoint_id"),
                "policy_checkpoint_id_after": summary.get("active_checkpoint_id") if summary.get("latest_updated") else None,
                "netlogo_steps_per_run": summary.get("netlogo_steps_per_run") or config.get("netlogo_steps_per_run"),
                "workers": summary.get("workers") or config.get("workers"),
                "trainable_step_count": summary.get("trainable_step_count") or dataset.get("trainable_step_count"),
                "avg_reward": dataset.get("avg_reward"),
                "ppo_total_loss": ppo.get("total_loss_mean"),
                "policy_loss": ppo.get("policy_loss_mean"),
                "value_loss": ppo.get("value_loss_mean"),
                "entropy": ppo.get("entropy_mean"),
                "approx_kl": ppo.get("approx_kl_mean"),
                "clip_fraction": ppo.get("clip_fraction_mean"),
                "latest_updated": 1 if summary.get("latest_updated") else 0,
                "batch_summary_path": str(batch_dir / "batch_summary.json"),
                "dataset_summary_json": json_text(dataset),
                "ppo_update_json": json_text(ppo),
            },
        )
        checkpoint_dir = batch_dir / "checkpoint"
        if checkpoint_dir.exists():
            checkpoint_id = batch_dir.name
            upsert_checkpoint(
                db_path,
                {
                    "experiment_id": experiment_id,
                    "batch_id": batch_id,
                    "policy_checkpoint_id": checkpoint_id,
                    "checkpoint_dir": str(checkpoint_dir),
                    "metadata_path": str(checkpoint_dir / "metadata.json"),
                    "feature_schema_path": str(checkpoint_dir / "feature_schema.json"),
                    "cycle_reference_path": str(checkpoint_dir / "cycle_reference.json"),
                    "is_latest": 1 if latest_dir and Path(latest_dir) == checkpoint_dir else 0,
                    "is_best": 0,
                    "created_at": None,
                    "checkpoint_json": json_text(read_json(checkpoint_dir / "metadata.json", {}) or {}),
                },
            )
        for worker_dir in sorted((batch_dir / "workers").glob("run_*")):
            worker_summary = read_json(worker_dir / "worker_summary.json", {}) or {}
            rollout_summary = read_json(worker_dir / "rts_rollout_summary.json", {}) or {}
            upsert_worker_rollout(
                db_path,
                {
                    "experiment_id": experiment_id,
                    "batch_id": batch_id,
                    "worker_run_id": worker_dir.name,
                    "derived_seed": None,
                    "netlogo_steps_requested": worker_summary.get("netlogo_steps_requested", config.get("netlogo_steps_per_run")),
                    "netlogo_steps_completed": worker_summary.get("netlogo_steps_completed", worker_summary.get("ticks_completed")),
                    "warehouse_time_start": worker_summary.get("warehouse_time_start"),
                    "warehouse_time_end": worker_summary.get("warehouse_time_end"),
                    "warehouse_time_elapsed": worker_summary.get("warehouse_time_elapsed"),
                    "tick_to_second": worker_summary.get("tick_to_second"),
                    "status": worker_summary.get("status"),
                    "worker_summary_path": str(worker_dir / "worker_summary.json"),
                    "rollout_path": str(worker_dir / "rts_rollout.jsonl"),
                    "summary_json": json_text(rollout_summary),
                },
            )
    return {"experiment_id": experiment_id, "scenario_id": scenario_id, "run_root": str(run_root)}
