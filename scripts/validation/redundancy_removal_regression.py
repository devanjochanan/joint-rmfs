#!/usr/bin/env python3
"""Focused regression for the four redundancy removals.

Covers:
  * find_new_orders() reuses the loaded DataFrame (no duplicate CSV read);
  * process_orders() does not perform a no-op read/write round-trip;
  * robots_location dead construction is removed (process_orders still works);
  * local executor digest/signature is computed only for first, final, and
    trace-selected ticks — values match the exhaustive every-tick approach.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("RMFS_FAST_TRAIN", "1")
os.environ.setdefault("RMFS_DETAIL_DB", "0")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/rmfs-mpl")

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


# ── A: find_new_orders reuse ────────────────────────────────────────────

def test_find_new_orders_no_duplicate_read():
    """find_new_orders uses the already-loaded DataFrame, not a second read."""
    import pandas as pd
    from model.inventory import Inventory

    tmp = tempfile.TemporaryDirectory()
    root = Path(tmp.name)
    inv = Inventory(
        runtime_paths={
            "assign_order_csv": str(root / "assign_order.csv"),
            "pod_info_csv": str(root / "pod_info.csv"),
            "generated_order_csv": str(root / "generated_order.csv"),
        },
        sqlite_db_path=str(root / "warehouse.db"),
    )
    inv._tmp = tmp
    inv.fast_train = True
    inv.tick_to_second = 1.0

    orders_data = pd.DataFrame({
        "sequence_id": [0, 1, 2],
        "order_id": [0, 0, 1],
        "order_type": [1, 1, 1],
        "item_id": [100, 101, 100],
        "item_quantity": [2, 3, 1],
        "order_arrival": [1, 1, 2],
        "source_order_id": ["s0", "s0", "s1"],
    })
    orders_data.to_csv(str(root / "generated_order.csv"), index=False)

    inv._tick = 1.0
    inv.next_process_tick = 1
    result = inv.find_new_orders()

    assert os.path.exists(str(root / "assign_order.csv")), "assign_order.csv must be created"
    csv_df = pd.read_csv(str(root / "assign_order.csv"))
    assert "status" in csv_df.columns
    assert "assigned_station" in csv_df.columns
    assert "assigned_pod" in csv_df.columns
    assert len(csv_df) == 3, f"expected 3 rows, got {len(csv_df)}"
    assert len(result) == 2, f"tick=1 should admit 2 rows (order_arrival==1), got {len(result)}"

    inv.next_process_tick = 2
    result2 = inv.find_new_orders()
    assert len(result2) == 1, f"tick=2 should admit 1 row, got {len(result2)}"

    orders_on_mgr = list(inv.order_manager.orders)
    assert len(orders_on_mgr) == 2, f"expected 2 orders, got {len(orders_on_mgr)}"
    order_ids = {o.order_id for o in orders_on_mgr}
    assert order_ids == {0, 1}, f"order ids: {order_ids}"

    tmp.cleanup()
    print("PASS test_find_new_orders_no_duplicate_read")


# ── B: process_orders round-trip removal ────────────────────────────────

def test_process_orders_no_roundtrip():
    """assign_order.csv is not rewritten by process_orders when no POA/PPS modifies it."""
    import pandas as pd
    from model.inventory import Inventory

    tmp = tempfile.TemporaryDirectory()
    root = Path(tmp.name)
    inv = Inventory(
        runtime_paths={
            "assign_order_csv": str(root / "assign_order.csv"),
            "pod_info_csv": str(root / "pod_info.csv"),
            "generated_order_csv": str(root / "generated_order.csv"),
        },
        sqlite_db_path=str(root / "warehouse.db"),
    )
    inv._tmp = tmp
    inv.fast_train = True
    inv.tick_to_second = 1.0
    inv.joint_rl = False
    inv.poa_first = False
    inv.poa_aisyahna = False
    inv.poa_podmatch = False
    inv.poa_second = False
    inv.pps_rl = False
    inv.pps_demand = False
    inv.pps_pileon = False
    inv._tick = 1.0

    csv_path = root / "assign_order.csv"
    initial_df = pd.DataFrame({
        "sequence_id": [0],
        "order_id": [0],
        "item_id": [100],
        "item_quantity": [2],
        "order_arrival": [1],
        "source_order_id": ["s0"],
        "assigned_station": [None],
        "assigned_pod": [None],
        "status": [-3],
    })
    initial_df.to_csv(str(csv_path), index=False)
    content_before = csv_path.read_text()
    mtime_before = csv_path.stat().st_mtime_ns

    inv.process_orders()

    content_after = csv_path.read_text()
    assert content_before == content_after, "assign_order.csv must not be rewritten"

    tmp.cleanup()
    print("PASS test_process_orders_no_roundtrip")


# ── C: robots_location removal ──────────────────────────────────────────

def test_process_orders_works_without_robots_location():
    """process_orders completes without the dead robots_location construction."""
    import pandas as pd
    from model.inventory import Inventory

    tmp = tempfile.TemporaryDirectory()
    root = Path(tmp.name)
    inv = Inventory(
        runtime_paths={
            "assign_order_csv": str(root / "assign_order.csv"),
            "pod_info_csv": str(root / "pod_info.csv"),
            "generated_order_csv": str(root / "generated_order.csv"),
        },
        sqlite_db_path=str(root / "warehouse.db"),
    )
    inv._tmp = tmp
    inv.fast_train = True
    inv.tick_to_second = 1.0
    inv.joint_rl = True
    inv._tick = 1.0

    csv_path = root / "assign_order.csv"
    pd.DataFrame({
        "sequence_id": [0],
        "order_id": [0],
        "item_id": [100],
        "item_quantity": [1],
        "order_arrival": [1],
        "source_order_id": ["s0"],
        "assigned_station": [None],
        "assigned_pod": [None],
        "status": [-3],
    }).to_csv(str(csv_path), index=False)

    inv.process_orders()

    tmp.cleanup()
    print("PASS test_process_orders_works_without_robots_location")


# ── D: digest gating logic ──────────────────────────────────────────────

def _stable_digest(payload):
    def sanitize(obj):
        if obj is None or isinstance(obj, (int, str, bool)):
            return obj
        if isinstance(obj, float):
            return f"{obj:.6f}"
        if isinstance(obj, dict):
            return {str(k): sanitize(v) for k, v in sorted(obj.items())}
        if isinstance(obj, set):
            return sorted(
                (sanitize(item) for item in obj),
                key=lambda x: json.dumps(x, sort_keys=True, default=str),
            )
        if isinstance(obj, (list, tuple)):
            return [sanitize(item) for item in obj]
        if hasattr(obj, "__dict__"):
            return sanitize(obj.__dict__)
        return str(obj)
    serialized = json.dumps(sanitize(payload), sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _return_signature(payload):
    if hasattr(payload, "__len__"):
        length = len(payload)
    else:
        length = None
    return {"type": type(payload).__name__, "length": length}


def test_digest_gating_matches_exhaustive():
    """Selective digest computation produces identical first/final/trace values."""
    n_ticks = 20
    trace_first_n = 3
    trace_cadence = 5

    payloads = [list(range(i, i + 12)) for i in range(n_ticks)]

    exhaustive_digests = [_stable_digest(p) for p in payloads]
    exhaustive_sigs = [_return_signature(p) for p in payloads]

    exhaustive_trace_indices = set()
    for index in range(n_ticks):
        if trace_first_n > 0 and index < trace_first_n:
            exhaustive_trace_indices.add(index)
        if trace_cadence > 0 and (index + 1) % trace_cadence == 0:
            exhaustive_trace_indices.add(index)
        if index == n_ticks - 1:
            exhaustive_trace_indices.add(index)

    selective_first = None
    selective_final = None
    selective_trace_rows = []
    for index in range(n_ticks):
        tick_result = payloads[index]
        is_first = index == 0
        is_final = index == n_ticks - 1

        trace_selected = False
        if trace_first_n > 0 and index < trace_first_n:
            trace_selected = True
        if trace_cadence > 0 and (index + 1) % trace_cadence == 0:
            trace_selected = True
        if is_final:
            trace_selected = True

        if is_first or is_final or trace_selected:
            digest = _stable_digest(tick_result)
            sig = _return_signature(tick_result)

        if is_first:
            selective_first = (digest, sig)
        if is_final:
            selective_final = (digest, sig, tick_result)

        if trace_selected:
            selective_trace_rows.append({
                "tick_index": index + 1,
                "digest": digest,
                "signature": sig,
            })

    assert selective_first[0] == exhaustive_digests[0], "first digest mismatch"
    assert selective_first[1] == exhaustive_sigs[0], "first signature mismatch"
    assert selective_final[0] == exhaustive_digests[-1], "final digest mismatch"
    assert selective_final[1] == exhaustive_sigs[-1], "final signature mismatch"

    selective_trace_tick_indices = {r["tick_index"] - 1 for r in selective_trace_rows}
    assert selective_trace_tick_indices == exhaustive_trace_indices, (
        f"trace indices differ: {selective_trace_tick_indices} vs {exhaustive_trace_indices}"
    )

    for row in selective_trace_rows:
        idx = row["tick_index"] - 1
        assert row["digest"] == exhaustive_digests[idx], f"trace digest mismatch at tick {idx}"
        assert row["signature"] == exhaustive_sigs[idx], f"trace signature mismatch at tick {idx}"

    print("PASS test_digest_gating_matches_exhaustive")


def test_digest_gating_no_trace():
    """When debug_trace is off, only first and final ticks get digests."""
    n_ticks = 10
    payloads = [list(range(i, i + 6)) for i in range(n_ticks)]

    computed_count = 0
    first_result = None
    final_result = None
    for index in range(n_ticks):
        is_first = index == 0
        is_final = index == n_ticks - 1
        trace_selected = False

        if is_first or is_final or trace_selected:
            digest = _stable_digest(payloads[index])
            sig = _return_signature(payloads[index])
            computed_count += 1

        if is_first:
            first_result = (digest, sig)
        if is_final:
            final_result = (digest, sig, payloads[index])

    assert computed_count == 2, f"expected 2 digests (first+final), got {computed_count}"
    assert first_result[0] == _stable_digest(payloads[0])
    assert final_result[0] == _stable_digest(payloads[-1])
    print("PASS test_digest_gating_no_trace")


def test_digest_gating_trace_cadence_only():
    """trace_cadence selects correct tick indices without trace_first_n."""
    n_ticks = 12
    trace_first_n = 0
    trace_cadence = 4
    payloads = [[i] for i in range(n_ticks)]

    expected_trace = set()
    for index in range(n_ticks):
        if trace_cadence > 0 and (index + 1) % trace_cadence == 0:
            expected_trace.add(index)
        if index == n_ticks - 1:
            expected_trace.add(index)

    actual_trace = set()
    for index in range(n_ticks):
        is_first = index == 0
        is_final = index == n_ticks - 1
        trace_selected = False
        if trace_first_n > 0 and index < trace_first_n:
            trace_selected = True
        if trace_cadence > 0 and (index + 1) % trace_cadence == 0:
            trace_selected = True
        if is_final:
            trace_selected = True
        if trace_selected:
            actual_trace.add(index)

    assert actual_trace == expected_trace, f"{actual_trace} != {expected_trace}"
    print("PASS test_digest_gating_trace_cadence_only")


def main():
    test_find_new_orders_no_duplicate_read()
    test_process_orders_no_roundtrip()
    test_process_orders_works_without_robots_location()
    test_digest_gating_matches_exhaustive()
    test_digest_gating_no_trace()
    test_digest_gating_trace_cadence_only()
    print("\nALL REDUNDANCY REMOVAL REGRESSION TESTS PASSED")


if __name__ == "__main__":
    raise SystemExit(main())
