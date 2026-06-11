#!/usr/bin/env python3
"""Select best RTS checkpoint pointer from evaluation summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.rmfs.experiments.evaluation.selection import select_best_checkpoint, write_best_checkpoint


def main(argv=None):
    parser = argparse.ArgumentParser(description="Select best RTS checkpoint pointer.")
    parser.add_argument("summaries", nargs="+")
    parser.add_argument("--output", default="best_checkpoint.json")
    args = parser.parse_args(argv)
    rows = []
    for path in args.summaries:
        with Path(path).open() as fh:
            rows.append(json.load(fh))
    pointer = select_best_checkpoint(rows)
    write_best_checkpoint(Path(args.output), pointer)
    print(pointer["selected_checkpoint_id"])


if __name__ == "__main__":
    raise SystemExit(main())

