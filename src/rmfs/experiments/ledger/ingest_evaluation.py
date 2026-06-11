"""Ingestion logic for evaluation summaries into the SQLite experiment ledger."""

from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Any

from src.rmfs.experiments.identity import short_hash
from src.rmfs.experiments.ledger.writer import upsert_evaluation


def ingest_evaluation_summary(
    summary_path: Path, db_path: Path, *, experiment_id: str | None = None
) -> dict[str, Any]:
    """Read eval_summary.json and upsert it into the evaluations table."""
    summary_path = Path(summary_path)
    with summary_path.open() as fh:
        summary = json.load(fh)

    # Preserve eval_run_id from summary if present
    eval_run_id = summary.get("eval_run_id") or f"eval_{short_hash(summary)}"

    # If experiment_id is not supplied, derive a stable placeholder from summary fields
    derived_exp_id = experiment_id or summary.get("experiment_id")
    if not derived_exp_id:
        placeholder_payload = {
            "policy_checkpoint_id": summary.get("policy_checkpoint_id"),
            "seed_pack_id": summary.get("seed_pack_id"),
            "policy_action_mode": summary.get("policy_action_mode"),
            "netlogo_steps_per_run": summary.get("netlogo_steps_per_run"),
            "replications": summary.get("replications"),
        }
        derived_exp_id = f"exp_placeholder_{short_hash(placeholder_payload)}"

    policy_checkpoint_id = summary.get("policy_checkpoint_id")
    policy_action_mode = summary.get("policy_action_mode", "greedy")

    # If summary lacks policy_mode, use rts_rl_explicit for checkpoint summaries
    policy_mode = summary.get("policy_mode")
    if not policy_mode:
        if policy_checkpoint_id:
            policy_mode = "rts_rl_explicit"
        else:
            policy_mode = "unknown"

    seed_pack_id = summary.get("seed_pack_id")
    netlogo_steps_per_run = summary.get("netlogo_steps_per_run")
    replications = summary.get("replications")
    status = summary.get("status", "completed")
    created_at = summary.get("created_at") or datetime.datetime.now(datetime.timezone.utc).isoformat()

    # Metrics parsing
    metrics = summary.get("metrics")
    if metrics is None:
        metric_keys = [
            "avg_order_cycle_time", "avg_order_cycle_time_mean",
            "orders_completed", "orders_completed_mean",
            "congestion_rate", "congestion_rate_mean",
            "energy_per_order", "energy_per_order_mean"
        ]
        metrics = {k: summary[k] for k in metric_keys if k in summary}

    metrics_str = json.dumps(metrics) if metrics else "{}"

    row = {
        "eval_run_id": eval_run_id,
        "experiment_id": derived_exp_id,
        "policy_checkpoint_id": policy_checkpoint_id,
        "policy_mode": policy_mode,
        "policy_action_mode": policy_action_mode,
        "seed_pack_id": seed_pack_id,
        "netlogo_steps_per_run": netlogo_steps_per_run,
        "replications": replications,
        "status": status,
        "created_at": created_at,
        "summary_path": str(summary_path.resolve()),
        "metrics_json": metrics_str,
    }

    upsert_evaluation(db_path, row)
    return row
