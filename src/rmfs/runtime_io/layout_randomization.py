"""Seeded pod-location randomization helpers."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd


def read_pod_storage_slots(generated_pod_path: str | Path) -> list[tuple[int, int]]:
    """Return row/column storage slots marked as active pods in scan order."""
    frame = pd.read_csv(generated_pod_path, header=None)
    slots: list[tuple[int, int]] = []
    for row_index, row in frame.iterrows():
        for col_index, value in row.items():
            if int(value) == 1:
                slots.append((int(row_index), int(col_index)))
    return slots


def build_pod_location_assignment(
    generated_pod_path: str | Path,
    seed: int,
    mode: str = "randomize_slots",
) -> list[dict[str, int]]:
    """Build a pod_id-to-slot assignment without changing the slot set."""
    slots = read_pod_storage_slots(generated_pod_path)
    pod_ids = np.arange(len(slots), dtype=int)
    if mode in {"fixed", "off", "none"}:
        assigned_ids = pod_ids
    elif mode == "randomize_slots":
        rng = np.random.default_rng(int(seed))
        assigned_ids = rng.permutation(pod_ids)
    else:
        raise ValueError(f"Unsupported pod-location randomization mode: {mode}")

    rows = []
    for slot_index, ((row, col), pod_id) in enumerate(zip(slots, assigned_ids)):
        rows.append(
            {
                "slot_index": int(slot_index),
                "pod_id": int(pod_id),
                "row": int(row),
                "col": int(col),
            }
        )
    return rows


def slot_index_to_pod_id(
    generated_pod_path: str | Path,
    seed: int,
    mode: str = "randomize_slots",
) -> dict[int, int]:
    """Return slot-index to pod-id mapping for NetLogo bridge setup."""
    return {
        row["slot_index"]: row["pod_id"]
        for row in build_pod_location_assignment(generated_pod_path, seed, mode=mode)
    }


def randomize_pod_locations(
    generated_pod_path: str | Path,
    output_path: str | Path,
    seed: int,
    mode: str = "randomize_slots",
) -> Path:
    """Write a deterministic pod-location assignment CSV.

    The generated layout, item master, and pod-SKU allocation files are not
    modified. The CSV maps pod IDs onto the existing storage slot coordinates.
    """
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = build_pod_location_assignment(generated_pod_path, seed, mode=mode)
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["slot_index", "pod_id", "row", "col"])
        writer.writeheader()
        writer.writerows(rows)
    return output
