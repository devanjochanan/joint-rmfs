#!/usr/bin/env python3
"""Ingest a Phase 9 RTS training run into SQLite."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.rmfs.experiments.ledger.ingest_phase9 import ingest_phase9_run
from src.rmfs.experiments.ledger.schema import DEFAULT_LEDGER_PATH


def main(argv=None):
    parser = argparse.ArgumentParser(description="Ingest RTS training run.")
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--db-path", default=str(DEFAULT_LEDGER_PATH))
    args = parser.parse_args(argv)
    result = ingest_phase9_run(Path(args.run_root), Path(args.db_path))
    print(result["experiment_id"])


if __name__ == "__main__":
    raise SystemExit(main())

