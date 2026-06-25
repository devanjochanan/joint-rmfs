#!/usr/bin/env python3
"""Export training summary CSVs from SQLite."""

from __future__ import annotations

import argparse
from pathlib import Path
import sqlite3
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.rmfs.experiments.ledger.schema import DEFAULT_LEDGER_PATH


def export_query(conn, query: str, path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import pandas as pd

        df = pd.read_sql_query(query, conn)
        df.to_csv(path, index=False)
        return len(df)
    except Exception:
        import csv

        rows = conn.execute(query).fetchall()
        names = [d[0] for d in conn.execute(query).description]
        with path.open("w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(names)
            writer.writerows(rows)
        return len(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Export training CSV summaries.")
    parser.add_argument("--db-path", default=str(DEFAULT_LEDGER_PATH))
    parser.add_argument("--export-dir", default="data/output/exports")
    args = parser.parse_args(argv)
    export_dir = Path(args.export_dir)
    with sqlite3.connect(args.db_path) as conn:
        export_query(conn, "SELECT * FROM training_batches ORDER BY experiment_id,batch_id", export_dir / "training_batches.csv")
        export_query(conn, "SELECT * FROM checkpoints ORDER BY experiment_id,batch_id,policy_checkpoint_id", export_dir / "checkpoint_index.csv")
    print(export_dir)


if __name__ == "__main__":
    raise SystemExit(main())

