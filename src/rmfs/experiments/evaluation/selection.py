"""Best-checkpoint metadata pointer selection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence


def select_best_checkpoint(candidates: Sequence[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in candidates if row.get("status") in {"success", "completed"} or row.get("valid")]
    if not valid:
        raise ValueError("no valid evaluation candidates")

    def key(row):
        metrics = row.get("metrics", row)
        return (
            float(metrics.get("avg_order_cycle_time", float("inf"))),
            -float(metrics.get("orders_completed", 0.0)),
            float(metrics.get("congestion_rate", float("inf"))),
            float(metrics.get("energy_per_order", float("inf"))),
            str(row.get("policy_checkpoint_id", "")),
        )

    selected = sorted(valid, key=key)[0]
    metrics = selected.get("metrics", selected)
    return {
        "schema_version": "rts_best_checkpoint.v1",
        "selection_rule": "eval_cycle_time_then_orders_then_congestion",
        "selected_checkpoint_id": selected["policy_checkpoint_id"],
        "selected_checkpoint_dir": selected["checkpoint_dir"],
        "eval_run_id": selected.get("eval_run_id"),
        "metrics": metrics,
    }


def write_best_checkpoint(path: Path, pointer: dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w") as fh:
        json.dump(pointer, fh, indent=2)

