"""Deterministic evaluation seed-pack generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.rmfs.experiments.identity import short_hash
from src.rmfs.rl.rts.training.seeding import derive_worker_seed


SCHEMA_VERSION = "eval_seed_pack.v1"


def build_seed_pack(*, seed_base: int, replications: int, netlogo_steps_per_run: int, purpose: str) -> dict[str, Any]:
    if int(replications) < 1:
        raise ValueError("replications must be >= 1")
    if int(netlogo_steps_per_run) < 1:
        raise ValueError("netlogo_steps_per_run must be >= 1")
    payload = {
        "seed_base": int(seed_base),
        "replications": int(replications),
        "netlogo_steps_per_run": int(netlogo_steps_per_run),
        "purpose": str(purpose),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "seed_pack_id": f"eval_pack_{short_hash(payload)}",
        **payload,
        "seeds": [
            {"replication": index, "seed": derive_worker_seed(seed_base, 0, index)}
            for index in range(1, int(replications) + 1)
        ],
    }


def write_seed_pack(path: Path, pack: dict[str, Any], *, force: bool = False) -> None:
    path = Path(path)
    if path.exists() and not force:
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        json.dump(pack, fh, indent=2)


def read_seed_pack(path: Path) -> dict[str, Any]:
    with Path(path).open() as fh:
        return json.load(fh)

