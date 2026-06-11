#!/usr/bin/env python3
"""Initialize the RMFS SQLite experiment ledger."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.rmfs.experiments.ledger.schema import DEFAULT_LEDGER_PATH, init_schema


def main(argv=None):
    parser = argparse.ArgumentParser(description="Initialize RMFS experiment ledger.")
    parser.add_argument("--db-path", default=str(DEFAULT_LEDGER_PATH))
    args = parser.parse_args(argv)
    table_count = init_schema(Path(args.db_path))
    print(f"{Path(args.db_path)} {table_count}")


if __name__ == "__main__":
    raise SystemExit(main())

