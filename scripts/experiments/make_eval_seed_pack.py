#!/usr/bin/env python3
"""Create a deterministic evaluation seed pack."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.rmfs.experiments.evaluation.seed_pack import build_seed_pack, write_seed_pack


def main(argv=None):
    parser = argparse.ArgumentParser(description="Create RMFS eval seed pack.")
    parser.add_argument("--seed-base", type=int, default=42)
    parser.add_argument("--replications", type=int, default=5)
    parser.add_argument("--netlogo-steps-per-run", type=int, default=1000)
    parser.add_argument("--purpose", default="short_smoke_eval")
    parser.add_argument("--output-dir", default="data/runtime/eval_seed_packs")
    parser.add_argument("--force", action="store_true", default=False)
    args = parser.parse_args(argv)
    pack = build_seed_pack(seed_base=args.seed_base, replications=args.replications, netlogo_steps_per_run=args.netlogo_steps_per_run, purpose=args.purpose)
    path = Path(args.output_dir) / f"{pack['seed_pack_id']}.json"
    write_seed_pack(path, pack, force=args.force)
    print(path)


if __name__ == "__main__":
    raise SystemExit(main())

