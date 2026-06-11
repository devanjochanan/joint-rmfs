#!/usr/bin/env python3
"""Export evaluation summary CSVs from SQLite."""

from __future__ import annotations

import argparse
from pathlib import Path
import sqlite3
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.experiments.export_training_summary import export_query
from src.rmfs.experiments.ledger.schema import DEFAULT_LEDGER_PATH


def main(argv=None):
    parser = argparse.ArgumentParser(description="Export eval CSV summaries.")
    parser.add_argument("--db-path", default=str(DEFAULT_LEDGER_PATH))
    parser.add_argument("--export-dir", default="data/output/exports")
    args = parser.parse_args(argv)
    export_dir = Path(args.export_dir)
    with sqlite3.connect(args.db_path) as conn:
        export_query(conn, "SELECT * FROM evaluations ORDER BY experiment_id,eval_run_id", export_dir / "eval_checkpoint_summary.csv")
        export_query(conn, "SELECT eval_run_id, metrics_json FROM evaluations ORDER BY eval_run_id", export_dir / "eval_metrics_long.csv")
    print(export_dir)


if __name__ == "__main__":
    raise SystemExit(main())

