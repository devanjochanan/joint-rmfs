"""Dry-run RTS checkpoint evaluation controller."""

from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Any

from src.rmfs.experiments.identity import short_hash


def build_eval_run_id(config: dict[str, Any]) -> str:
    return f"eval_{short_hash(config)}"


def write_eval_dry_run(
    *,
    checkpoint_dir: Path,
    policy_checkpoint_id: str,
    zone_ids: tuple[str, ...],
    seed_pack_path: Path,
    output_root: Path,
    policy_action_mode: str = "greedy",
) -> dict[str, Any]:
    with Path(seed_pack_path).open() as fh:
        seed_pack = json.load(fh)
    config = {
        "checkpoint_dir": str(checkpoint_dir),
        "policy_checkpoint_id": policy_checkpoint_id,
        "zone_ids": list(zone_ids),
        "seed_pack_id": seed_pack["seed_pack_id"],
        "netlogo_steps_per_run": seed_pack["netlogo_steps_per_run"],
        "replications": seed_pack["replications"],
        "policy_action_mode": policy_action_mode,
    }
    eval_run_id = build_eval_run_id(config)
    run_root = Path(output_root) / eval_run_id
    run_root.mkdir(parents=True, exist_ok=True)
    worker_specs = [
        {
            "worker_run_id": f"eval_{seed['replication']:03d}",
            "seed": seed["seed"],
            "netlogo_steps_per_run": seed_pack["netlogo_steps_per_run"],
            "policy_checkpoint_id": policy_checkpoint_id,
            "policy_action_mode": policy_action_mode,
            "zone_ids": list(zone_ids),
        }
        for seed in seed_pack["seeds"]
    ]
    summary = {
        "status": "dry_run",
        "eval_run_id": eval_run_id,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        **config,
        "worker_count": len(worker_specs),
    }
    for name, payload in (("eval_config.json", config), ("worker_specs.json", worker_specs), ("eval_summary.json", summary)):
        with (run_root / name).open("w") as fh:
            json.dump(payload, fh, indent=2)
    return summary

