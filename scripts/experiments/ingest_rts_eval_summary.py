#!/usr/bin/env python3
"""CLI script to ingest evaluation summaries into the SQLite experiment ledger."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.rmfs.experiments.ledger.ingest_evaluation import ingest_evaluation_summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Ingest evaluation summary into SQLite ledger.")
    parser.add_argument("--summary-path", required=True, help="Path to eval_summary.json")
    parser.add_argument("--db-path", default="data/output/rmfs_experiments.sqlite", help="Path to SQLite ledger file")
    parser.add_argument("--experiment-id", default=None, help="Optional experiment ID override")
    
    args = parser.parse_args(argv)
    
    summary_path = Path(args.summary_path)
    db_path = Path(args.db_path)
    
    if not summary_path.exists():
        print(f"Error: summary path does not exist: {summary_path}", file=sys.stderr)
        return 1
        
    row = ingest_evaluation_summary(summary_path, db_path, experiment_id=args.experiment_id)
    print(f"Ingested evaluation '{row['eval_run_id']}' under experiment '{row['experiment_id']}' successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
