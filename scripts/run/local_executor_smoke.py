#!/usr/bin/env python3
"""CLI wrapper for the local isolated executor smoke."""

import argparse
from pathlib import Path
import sys


def main():
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))

    parser = argparse.ArgumentParser(description="Run local isolated RMFS executor smoke.")
    parser.add_argument("--runs", type=int, default=4)
    parser.add_argument("--ticks", type=int, default=3)
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()

    if args.runs < 1:
        parser.error("--runs must be >= 1")
    if args.ticks < 0:
        parser.error("--ticks must be >= 0")
    if args.max_workers < 1:
        parser.error("--max-workers must be >= 1")

    from src.rmfs.orchestration.local_executor import run_controller

    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = repo_root / output_root

    run_controller(
        repo_root=repo_root,
        output_root=output_root.resolve(),
        runs=args.runs,
        ticks=args.ticks,
        max_workers=args.max_workers,
        python_executable=sys.executable,
    )


if __name__ == "__main__":
    main()
