"""Deterministic seed derivation for RTS training workers."""

from __future__ import annotations

import hashlib


def derive_worker_seed(seed_base: int, batch_id: int, worker_index: int, run_index: int = 0) -> int:
    payload = f"{int(seed_base)}:{int(batch_id)}:{int(worker_index)}:{int(run_index)}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:4], "big", signed=False)

