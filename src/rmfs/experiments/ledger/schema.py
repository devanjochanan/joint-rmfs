"""SQLite schema for the RMFS experiment ledger."""

from __future__ import annotations

from pathlib import Path
import sqlite3


DEFAULT_LEDGER_PATH = Path("data/output/rmfs_experiments.sqlite")


TABLES = (
    "experiments",
    "training_batches",
    "checkpoints",
    "worker_rollouts",
    "evaluations",
    "cycle_proposals",
)


DDL = [
    """CREATE TABLE IF NOT EXISTS experiments (
        experiment_id TEXT PRIMARY KEY,
        scenario_id TEXT NOT NULL,
        artifact_label TEXT,
        phase TEXT,
        created_at TEXT,
        repo_branch TEXT,
        repo_commit TEXT,
        python_executable TEXT,
        seed_base INTEGER,
        output_root TEXT,
        status TEXT,
        config_json TEXT,
        feature_flags_json TEXT
    )""",
    """CREATE TABLE IF NOT EXISTS training_batches (
        experiment_id TEXT,
        batch_id INTEGER,
        status TEXT,
        policy_checkpoint_id_before TEXT,
        policy_checkpoint_id_after TEXT,
        netlogo_steps_per_run INTEGER,
        workers INTEGER,
        trainable_step_count INTEGER,
        avg_reward REAL,
        ppo_total_loss REAL,
        policy_loss REAL,
        value_loss REAL,
        entropy REAL,
        approx_kl REAL,
        clip_fraction REAL,
        latest_updated INTEGER,
        batch_summary_path TEXT,
        dataset_summary_json TEXT,
        ppo_update_json TEXT,
        PRIMARY KEY (experiment_id, batch_id)
    )""",
    """CREATE TABLE IF NOT EXISTS checkpoints (
        experiment_id TEXT,
        batch_id INTEGER,
        policy_checkpoint_id TEXT,
        checkpoint_dir TEXT,
        metadata_path TEXT,
        feature_schema_path TEXT,
        cycle_reference_path TEXT,
        is_latest INTEGER DEFAULT 0,
        is_best INTEGER DEFAULT 0,
        created_at TEXT,
        checkpoint_json TEXT,
        PRIMARY KEY (experiment_id, policy_checkpoint_id)
    )""",
    """CREATE TABLE IF NOT EXISTS worker_rollouts (
        experiment_id TEXT,
        batch_id INTEGER,
        worker_run_id TEXT,
        derived_seed INTEGER,
        netlogo_steps_requested INTEGER,
        netlogo_steps_completed INTEGER,
        warehouse_time_start REAL,
        warehouse_time_end REAL,
        warehouse_time_elapsed REAL,
        tick_to_second REAL,
        status TEXT,
        worker_summary_path TEXT,
        rollout_path TEXT,
        summary_json TEXT,
        PRIMARY KEY (experiment_id, batch_id, worker_run_id)
    )""",
    """CREATE TABLE IF NOT EXISTS evaluations (
        eval_run_id TEXT PRIMARY KEY,
        experiment_id TEXT,
        policy_checkpoint_id TEXT,
        policy_mode TEXT,
        policy_action_mode TEXT,
        seed_pack_id TEXT,
        netlogo_steps_per_run INTEGER,
        replications INTEGER,
        status TEXT,
        created_at TEXT,
        summary_path TEXT,
        metrics_json TEXT
    )""",
    """CREATE TABLE IF NOT EXISTS cycle_proposals (
        proposal_id TEXT PRIMARY KEY,
        experiment_id TEXT,
        source_eval_run_id TEXT,
        source_checkpoint_id TEXT,
        decision TEXT,
        requires_manual_approval INTEGER,
        proposal_path TEXT,
        current_reference_json TEXT,
        candidate_reference_json TEXT,
        created_at TEXT
    )""",
]


def connect(db_path: Path = DEFAULT_LEDGER_PATH) -> sqlite3.Connection:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(db_path)


def init_schema(db_path: Path = DEFAULT_LEDGER_PATH) -> int:
    with connect(db_path) as conn:
        with conn:
            for ddl in DDL:
                conn.execute(ddl)
        return len(TABLES)

