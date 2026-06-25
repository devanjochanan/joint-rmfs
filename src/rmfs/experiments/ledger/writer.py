"""Writer helpers for the SQLite experiment ledger."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .schema import connect, init_schema


def json_text(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, default=str)


def upsert_experiment(db_path: Path, row: Mapping[str, Any]) -> None:
    init_schema(db_path)
    with connect(db_path) as conn, conn:
        conn.execute(
            """INSERT OR REPLACE INTO experiments VALUES
            (:experiment_id,:scenario_id,:artifact_label,:phase,:created_at,:repo_branch,:repo_commit,
             :python_executable,:seed_base,:output_root,:status,:config_json,:feature_flags_json)""",
            dict(row),
        )


def upsert_training_batch(db_path: Path, row: Mapping[str, Any]) -> None:
    init_schema(db_path)
    with connect(db_path) as conn, conn:
        conn.execute(
            """INSERT OR REPLACE INTO training_batches VALUES
            (:experiment_id,:batch_id,:status,:policy_checkpoint_id_before,:policy_checkpoint_id_after,
             :netlogo_steps_per_run,:workers,:trainable_step_count,:avg_reward,:ppo_total_loss,
             :policy_loss,:value_loss,:entropy,:approx_kl,:clip_fraction,:latest_updated,
             :batch_summary_path,:dataset_summary_json,:ppo_update_json)""",
            dict(row),
        )


def upsert_checkpoint(db_path: Path, row: Mapping[str, Any]) -> None:
    init_schema(db_path)
    with connect(db_path) as conn, conn:
        conn.execute(
            """INSERT OR REPLACE INTO checkpoints VALUES
            (:experiment_id,:batch_id,:policy_checkpoint_id,:checkpoint_dir,:metadata_path,:feature_schema_path,
             :cycle_reference_path,:is_latest,:is_best,:created_at,:checkpoint_json)""",
            dict(row),
        )


def upsert_worker_rollout(db_path: Path, row: Mapping[str, Any]) -> None:
    init_schema(db_path)
    with connect(db_path) as conn, conn:
        conn.execute(
            """INSERT OR REPLACE INTO worker_rollouts VALUES
            (:experiment_id,:batch_id,:worker_run_id,:derived_seed,:netlogo_steps_requested,
             :netlogo_steps_completed,:warehouse_time_start,:warehouse_time_end,:warehouse_time_elapsed,
             :tick_to_second,:status,:worker_summary_path,:rollout_path,:summary_json)""",
            dict(row),
        )


def upsert_evaluation(db_path: Path, row: Mapping[str, Any]) -> None:
    init_schema(db_path)
    with connect(db_path) as conn, conn:
        conn.execute(
            """INSERT OR REPLACE INTO evaluations VALUES
            (:eval_run_id,:experiment_id,:policy_checkpoint_id,:policy_mode,:policy_action_mode,
             :seed_pack_id,:netlogo_steps_per_run,:replications,:status,:created_at,:summary_path,:metrics_json)""",
            dict(row),
        )


def upsert_cycle_proposal(db_path: Path, row: Mapping[str, Any]) -> None:
    init_schema(db_path)
    with connect(db_path) as conn, conn:
        conn.execute(
            """INSERT OR REPLACE INTO cycle_proposals VALUES
            (:proposal_id,:experiment_id,:source_eval_run_id,:source_checkpoint_id,:decision,
             :requires_manual_approval,:proposal_path,:current_reference_json,:candidate_reference_json,:created_at)""",
            dict(row),
        )

