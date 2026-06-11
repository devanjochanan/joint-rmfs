#!/usr/bin/env python3
"""Smoke test deterministic eval seed packs."""

from __future__ import annotations

from pathlib import Path
import shutil
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.rmfs.experiments.evaluation.seed_pack import build_seed_pack, read_seed_pack, write_seed_pack


def main():
    root = REPO_ROOT / "data" / "runtime" / "phase10_seed_pack_smoke"
    shutil.rmtree(root, ignore_errors=True)
    pack = build_seed_pack(seed_base=42, replications=3, netlogo_steps_per_run=5, purpose="smoke")
    assert pack["seed_pack_id"] == build_seed_pack(seed_base=42, replications=3, netlogo_steps_per_run=5, purpose="smoke")["seed_pack_id"]
    path = root / f"{pack['seed_pack_id']}.json"
    write_seed_pack(path, pack)
    assert read_seed_pack(path)["seed_pack_id"] == pack["seed_pack_id"]
    assert len(pack["seeds"]) == 3
    shutil.rmtree(root)
    print("eval seed pack smoke ok")


if __name__ == "__main__":
    main()

