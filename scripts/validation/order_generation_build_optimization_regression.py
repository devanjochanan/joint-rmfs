"""Regression test for _build_generated_orders optimization.

Compares the optimized groupby-based builder against a small literal reference
implementation that uses the previous per-order DataFrame filtering.  Covers:
  - shuffled and repeated source IDs
  - multi-line orders
  - duplicate lines within an order
  - non-numeric source_order_id values
  - exact DataFrame equality including row order and column order

Also includes a bounded timing report using the repository's actual historical
input data when available (timing is informational, not a brittle pass/fail).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.rmfs.order_generation.bootstrap import _build_generated_orders


# ---------------------------------------------------------------------------
# Reference implementation: previous per-order DataFrame filtering
# ---------------------------------------------------------------------------
def _build_generated_orders_reference(sampled_order_ids, raw_orders, sampled_arrivals):
    """Exact reproduction of the pre-optimization implementation."""
    order_lines = []
    sequence_id = 0

    for generated_order_id, (source_order_id, order_arrival) in enumerate(
        zip(sampled_order_ids, sampled_arrivals)
    ):
        source_lines = raw_orders.loc[
            raw_orders["source_order_id"] == source_order_id,
            ["item_id", "item_quantity"],
        ]
        for line in source_lines.itertuples(index=False):
            order_lines.append(
                {
                    "sequence_id": int(sequence_id),
                    "order_id": int(generated_order_id),
                    "order_type": 1,
                    "item_id": int(line.item_id),
                    "item_quantity": int(line.item_quantity),
                    "order_arrival": int(order_arrival),
                    "source_order_id": str(source_order_id),
                }
            )
            sequence_id += 1

    generated_order = pd.DataFrame(order_lines)
    if generated_order.empty:
        raise ValueError("Bootstrap generation produced no order lines.")
    return generated_order


# ---------------------------------------------------------------------------
# Fixture: synthetic raw_orders with edge cases
# ---------------------------------------------------------------------------
def _make_test_raw_orders() -> pd.DataFrame:
    """Build a small raw_orders DataFrame exercising all required edge cases.

    Coverage:
      - Multi-line order (ORD-A has 3 lines)
      - Single-line order (ORD-B)
      - Duplicate lines within an order (X99 has two identical lines)
      - Non-numeric source_order_id values (all are strings)
    """
    return pd.DataFrame(
        {
            "source_order_id": ["ORD-A", "ORD-A", "ORD-A", "ORD-B", "X99", "X99"],
            "item_id": [10, 20, 30, 40, 50, 50],
            "item_quantity": [1, 2, 3, 4, 5, 5],
        }
    )


def test_exact_equivalence():
    """Exact DataFrame equality between optimized and reference implementation."""
    raw_orders = _make_test_raw_orders()

    # Shuffled and repeated source IDs
    sampled_ids = np.array(["X99", "ORD-A", "ORD-B", "X99", "ORD-A"], dtype=object)
    sampled_arrivals = np.array([100, 200, 300, 400, 500], dtype=np.int64)

    optimized = _build_generated_orders(sampled_ids, raw_orders, sampled_arrivals)
    reference = _build_generated_orders_reference(sampled_ids, raw_orders, sampled_arrivals)

    pd.testing.assert_frame_equal(
        optimized,
        reference,
        check_dtype=True,
        check_exact=True,
        obj="optimized vs reference _build_generated_orders",
    )

    # Sanity checks on output shape
    expected_lines = (
        2  # X99 (2 lines)
        + 3  # ORD-A (3 lines)
        + 1  # ORD-B (1 line)
        + 2  # X99 again (2 lines)
        + 3  # ORD-A again (3 lines)
    )
    assert len(optimized) == expected_lines, (
        f"Expected {expected_lines} lines, got {len(optimized)}"
    )

    # Verify sequence_id is contiguous
    assert list(optimized["sequence_id"]) == list(range(expected_lines))

    # Verify order_id assignment
    assert list(optimized["order_id"].unique()) == [0, 1, 2, 3, 4]

    # Verify duplicate lines are preserved (X99 at order_id 0)
    x99_first = optimized[optimized["order_id"] == 0]
    assert len(x99_first) == 2
    assert list(x99_first["item_id"]) == [50, 50]
    assert list(x99_first["item_quantity"]) == [5, 5]

    # Verify source-line order within multi-line order (ORD-A at order_id 1)
    orda_first = optimized[optimized["order_id"] == 1]
    assert len(orda_first) == 3
    assert list(orda_first["item_id"]) == [10, 20, 30]
    assert list(orda_first["item_quantity"]) == [1, 2, 3]

    # Verify column order matches reference
    assert list(optimized.columns) == list(reference.columns)

    print("  exact_equivalence:  PASS")


def test_missing_source_order():
    """Orders referencing a source_order_id not in raw_orders produce no lines."""
    raw_orders = _make_test_raw_orders()
    sampled_ids = np.array(["ORD-B", "MISSING", "ORD-A"], dtype=object)
    sampled_arrivals = np.array([10, 20, 30], dtype=np.int64)

    optimized = _build_generated_orders(sampled_ids, raw_orders, sampled_arrivals)
    reference = _build_generated_orders_reference(sampled_ids, raw_orders, sampled_arrivals)

    pd.testing.assert_frame_equal(optimized, reference, check_dtype=True, check_exact=True)
    # MISSING should produce no lines, so order_id=1 should not appear
    assert 1 not in optimized["order_id"].values
    print("  missing_source_order: PASS")


def test_single_order():
    """Single sampled order."""
    raw_orders = _make_test_raw_orders()
    sampled_ids = np.array(["ORD-A"], dtype=object)
    sampled_arrivals = np.array([0], dtype=np.int64)

    optimized = _build_generated_orders(sampled_ids, raw_orders, sampled_arrivals)
    reference = _build_generated_orders_reference(sampled_ids, raw_orders, sampled_arrivals)

    pd.testing.assert_frame_equal(optimized, reference, check_dtype=True, check_exact=True)
    assert len(optimized) == 3
    print("  single_order:       PASS")


# ---------------------------------------------------------------------------
# Bounded performance timing (informational, not pass/fail)
# ---------------------------------------------------------------------------
def bounded_performance_timing():
    """Report timing using the repository's actual raw_order.csv if available.

    Compares optimized vs reference on a realistic workload.  Wall-clock
    timing is printed but never used as a brittle pass/fail assertion.
    """
    base = REPO_ROOT / "data" / "input" / "base"
    raw_csv = base / "raw_order.csv"
    items_csv = base / "items.csv"

    if not raw_csv.exists() or not items_csv.exists():
        print("  performance_timing: SKIP (actual data not found)")
        return

    # Import loader to prepare raw_orders the same way the pipeline does
    from src.rmfs.order_generation.bootstrap import _load_raw_orders

    raw_orders, _, _, _ = _load_raw_orders(raw_csv, items_csv_path=items_csv)
    source_ids = raw_orders["source_order_id"].unique()
    n_source = len(source_ids)

    # Create a realistic sampled workload: sample with replacement up to
    # 2× source order count, capped at 5000 to stay bounded.
    rng = np.random.default_rng(42)
    sample_size = min(n_source * 2, 5000)
    sampled_ids = rng.choice(source_ids, size=sample_size, replace=True)
    sampled_arrivals = np.arange(sample_size, dtype=np.int64) * 10

    # Time optimized version
    t0 = time.perf_counter()
    optimized = _build_generated_orders(sampled_ids, raw_orders, sampled_arrivals)
    t_optimized = time.perf_counter() - t0

    # Time reference version
    t0 = time.perf_counter()
    reference = _build_generated_orders_reference(sampled_ids, raw_orders, sampled_arrivals)
    t_reference = time.perf_counter() - t0

    # Verify identical output
    pd.testing.assert_frame_equal(optimized, reference, check_dtype=True, check_exact=True)

    speedup = t_reference / t_optimized if t_optimized > 0 else float("inf")
    print(f"  performance_timing: PASS (output identical)")
    print(f"    source orders:     {n_source:,}")
    print(f"    sampled orders:    {sample_size:,}")
    print(f"    generated lines:   {len(optimized):,}")
    print(f"    reference time:    {t_reference:.4f}s")
    print(f"    optimized time:    {t_optimized:.4f}s")
    print(f"    speedup:           {speedup:.1f}x")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    print("order_generation_build_optimization_regression")
    try:
        test_exact_equivalence()
        test_missing_source_order()
        test_single_order()
        bounded_performance_timing()
        print("order generation build optimization regression OK")
        return 0
    except Exception as e:
        print(f"FAIL: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
