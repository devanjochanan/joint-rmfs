"""Best-checkpoint metadata pointer selection."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any, Sequence


def checkpoint_sort_index(policy_checkpoint_id: str) -> int:
    """Extract numeric index from checkpoint ID (e.g. batch_000074 -> 74). Return -1 if not found."""
    if not policy_checkpoint_id:
        return -1
    match = re.search(r"\d+", str(policy_checkpoint_id))
    if match:
        return int(match.group(0))
    return -1


def select_best_checkpoint(candidates: Sequence[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in candidates if row.get("status") in {"success", "completed"} or row.get("valid")]
    if not valid:
        raise ValueError("no valid evaluation candidates")

    def get_metric(metrics: dict[str, Any], key_base: str, default: Any) -> Any:
        val = metrics.get(key_base)
        if val is None:
            val = metrics.get(f"{key_base}_mean")
        if val is None:
            return default
        return val

    def key(row):
        metrics = row.get("metrics", row)
        if metrics is None:
            metrics = {}

        avg_cycle = get_metric(metrics, "avg_order_cycle_time", float("inf"))
        orders = get_metric(metrics, "orders_completed", 0.0)
        congestion = get_metric(metrics, "congestion_rate", float("inf"))
        energy = get_metric(metrics, "energy_per_order", float("inf"))

        chk_id = row.get("policy_checkpoint_id", "")
        chk_idx = checkpoint_sort_index(chk_id)

        return (
            float(avg_cycle) if avg_cycle is not None else float("inf"),
            -float(orders) if orders is not None else 0.0,
            float(congestion) if congestion is not None else float("inf"),
            float(energy) if energy is not None else float("inf"),
            -chk_idx,
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
