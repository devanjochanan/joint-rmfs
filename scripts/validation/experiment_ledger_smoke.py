#!/usr/bin/env python3
"""Smoke test for the SQLite experiment ledger."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import sqlite3
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.rmfs.experiments.ledger.schema import TABLES, init_schema
from src.rmfs.experiments.ledger.writer import json_text, upsert_experiment


def main():
    root = REPO_ROOT / "data" / "runtime" / "phase10_ledger_smoke"
    shutil.rmtree(root, ignore_errors=True)
    db_path = root / "ledger.sqlite"
    assert init_schema(db_path) == len(TABLES)
    upsert_experiment(
        db_path,
        {
            "experiment_id": "rtsrl_test",
            "scenario_id": "scenario_test",
            "artifact_label": "smoke",
            "phase": "phase10",
            "created_at": "now",
            "repo_branch": None,
            "repo_commit": None,
            "python_executable": sys.executable,
            "seed_base": 42,
            "output_root": str(root),
            "status": "ok",
            "config_json": json_text({"a": 1}),
            "feature_flags_json": json_text({"rts_rl_enabled": True}),
        },
    )
    with sqlite3.connect(db_path) as conn:
        names = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert set(TABLES).issubset(names)
        config = conn.execute("SELECT config_json FROM experiments WHERE experiment_id='rtsrl_test'").fetchone()[0]
        assert json.loads(config)["a"] == 1
    shutil.rmtree(root)
    print("experiment ledger smoke ok")


if __name__ == "__main__":
    main()

