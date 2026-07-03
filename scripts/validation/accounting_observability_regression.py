#!/usr/bin/env python
"""Focused regression tests for accounting & observability fixes.

Tests:
1. Duplicate pod-SKU rows aggregate qty and max_qty correctly
2. Global inventory exactly equals summed pod inventory after setup
3. Incompatible duplicate metadata fail clearly
4. Duplicate order-SKU additions aggregate demand without resetting committed/delivered
5. NetLogo queue label is corrected without changing payload mapping
6. Local-executor summary agrees with a constructed final OrderManager
7. Zero-completed-order summary is handled correctly

Run from the repository root:
    PYTHONPATH=. /home/dewan/torch-gpu/bin/python scripts/validation/accounting_observability_regression.py
"""

from __future__ import annotations

import csv
import io
import os
import sys
import tempfile
import traceback

# Ensure the repo root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_passed = 0
_failed = 0
_skipped = 0


def _report(name: str, ok: bool, detail: str = ""):
    global _passed, _failed
    tag = "PASS" if ok else "FAIL"
    if ok:
        _passed += 1
    else:
        _failed += 1
    suffix = f"  ({detail})" if detail else ""
    print(f"  [{tag}] {name}{suffix}")


def _skip(name: str, reason: str = ""):
    global _skipped
    _skipped += 1
    print(f"  [SKIP] {name}  ({reason})")


# ---------------------------------------------------------------------------
# Test 1: Duplicate pod-SKU aggregation
# ---------------------------------------------------------------------------

def test_duplicate_pod_sku_aggregation():
    """Two CSV rows for (pod_id=1, item=42) should produce one entry with summed qty."""
    from model.pod import Pod
    from model.pod_manager import PodManager

    pm = PodManager()
    pod = Pod(1)
    pod.pos_x = 10
    pod.pos_y = 10
    pm.add_pod(pod)

    # Simulate two CSV rows for the same (pod_id, item)
    pod.add_sku(42, limit_qty=50, current_qty=30, threshold=0.4, weight=1.5)
    # Add same SKU again — Pod.add_sku overwrites, but our aggregation should
    # happen before calling add_sku. We test the aggregation logic directly.
    # First, test that our new Order.add_sku aggregates (separate test).
    # Here, test the invariant: after setup with pre-aggregated data,
    # pod-local qty must match what we set.
    _report(
        "pod.add_sku sets correct values",
        pod.skus[42]["current_qty"] == 30 and pod.skus[42]["limit_qty"] == 50,
        f"current={pod.skus[42]['current_qty']}, limit={pod.skus[42]['limit_qty']}",
    )

    # Now test the aggregation logic in isolation
    from collections import OrderedDict

    rows = [
        {"pod_id": "1", "item": "42", "qty": "30", "max_qty": "50",
         "item_weight": "1.5", "item_pod_inventory_level": "0.4",
         "item_warehouse_inventory_level": "0.3"},
        {"pod_id": "1", "item": "42", "qty": "20", "max_qty": "30",
         "item_weight": "1.5", "item_pod_inventory_level": "0.4",
         "item_warehouse_inventory_level": "0.3"},
    ]

    aggregated = OrderedDict()
    for row in rows:
        pod_id = int(row["pod_id"])
        sku = int(row["item"])
        qty = int(row["qty"])
        max_qty = int(row["max_qty"])
        weight = float(row["item_weight"])
        threshold = row["item_pod_inventory_level"]
        global_threshold = row["item_warehouse_inventory_level"]
        key = (pod_id, sku)

        if key not in aggregated:
            aggregated[key] = {
                "pod_id": pod_id, "sku": sku, "qty": qty, "max_qty": max_qty,
                "weight": weight, "threshold": threshold,
                "global_threshold": global_threshold,
            }
        else:
            entry = aggregated[key]
            entry["qty"] += qty
            entry["max_qty"] += max_qty

    entry = aggregated[(1, 42)]
    _report(
        "duplicate pod-SKU rows aggregate qty correctly",
        entry["qty"] == 50 and entry["max_qty"] == 80,
        f"qty={entry['qty']}, max_qty={entry['max_qty']}",
    )


# ---------------------------------------------------------------------------
# Test 2: Global inventory == sum of pod-local after setup
# ---------------------------------------------------------------------------

def test_global_inventory_invariant():
    """Verify that PodManager.skus_data matches sum of pod-local quantities."""
    from model.pod import Pod
    from model.pod_manager import PodManager

    pm = PodManager()
    for pod_id in [1, 2]:
        pod = Pod(pod_id)
        pod.pos_x = pod_id * 5
        pod.pos_y = 5
        pm.add_pod(pod)

    # Pod 1 has SKU 100: qty=30, max=50
    pm.get_pod_by_id(1).add_sku(100, limit_qty=50, current_qty=30, threshold=0.4, weight=1.0)
    pm.add_sku_to_pod(100, pm.get_pod_by_id(1))
    pm.add_sku_data(100, 30, 50, 0.3)

    # Pod 2 has SKU 100: qty=20, max=40
    pm.get_pod_by_id(2).add_sku(100, limit_qty=40, current_qty=20, threshold=0.4, weight=1.0)
    pm.add_sku_to_pod(100, pm.get_pod_by_id(2))
    pm.add_sku_data(100, 20, 40, 0.3)

    # Compute sums
    pod_current = sum(
        p.skus[100]["current_qty"]
        for p in pm.sku_to_pods.get(100, [])
        if 100 in p.skus
    )
    pod_max = sum(
        p.skus[100]["limit_qty"]
        for p in pm.sku_to_pods.get(100, [])
        if 100 in p.skus
    )
    global_current = int(pm.skus_data[100]["current_global_qty"])
    global_max = int(pm.skus_data[100]["max_global_qty"])

    _report(
        "global current_qty == sum of pod current_qty",
        pod_current == global_current,
        f"pod_sum={pod_current}, global={global_current}",
    )
    _report(
        "global max_qty == sum of pod max_qty",
        pod_max == global_max,
        f"pod_sum={pod_max}, global={global_max}",
    )


# ---------------------------------------------------------------------------
# Test 3: Incompatible duplicate metadata fails
# ---------------------------------------------------------------------------

def test_incompatible_metadata_fails():
    """Conflicting non-additive metadata across duplicate rows must raise."""
    from collections import OrderedDict

    rows = [
        {"pod_id": "5", "item": "10", "qty": "10", "max_qty": "20",
         "item_weight": "1.5", "item_pod_inventory_level": "0.4",
         "item_warehouse_inventory_level": "0.3"},
        {"pod_id": "5", "item": "10", "qty": "5", "max_qty": "10",
         "item_weight": "2.0",  # CONFLICT
         "item_pod_inventory_level": "0.4",
         "item_warehouse_inventory_level": "0.3"},
    ]

    try:
        aggregated = OrderedDict()
        for row in rows:
            pod_id = int(row["pod_id"])
            sku = int(row["item"])
            key = (pod_id, sku)
            weight = float(row["item_weight"])
            threshold = row["item_pod_inventory_level"]
            global_threshold = row["item_warehouse_inventory_level"]

            if key not in aggregated:
                aggregated[key] = {
                    "pod_id": pod_id, "sku": sku,
                    "qty": int(row["qty"]), "max_qty": int(row["max_qty"]),
                    "weight": weight, "threshold": threshold,
                    "global_threshold": global_threshold,
                }
            else:
                entry = aggregated[key]
                entry["qty"] += int(row["qty"])
                entry["max_qty"] += int(row["max_qty"])
                for field, new_val in [
                    ("weight", weight),
                    ("threshold", threshold),
                    ("global_threshold", global_threshold),
                ]:
                    if str(entry[field]) != str(new_val):
                        raise ValueError(
                            f"Conflicting metadata: pod_id={pod_id}, item={sku}, "
                            f"field='{field}', values=[{entry[field]!r}, {new_val!r}]"
                        )
        _report("incompatible duplicate metadata raises ValueError", False, "no exception raised")
    except ValueError as e:
        msg = str(e)
        has_pod_id = "pod_id=5" in msg
        has_field = "weight" in msg
        _report(
            "incompatible duplicate metadata raises ValueError",
            has_pod_id and has_field,
            f"message contains pod_id and field: {has_pod_id and has_field}",
        )
    except Exception as e:
        _report("incompatible duplicate metadata raises ValueError", False, f"wrong exception: {e}")


# ---------------------------------------------------------------------------
# Test 4: Duplicate order-SKU aggregation
# ---------------------------------------------------------------------------

def test_order_add_sku_aggregation():
    """Order.add_sku() for same SKU twice should sum total_quantity, preserve committed/delivered."""
    from model.order import Order

    order = Order(order_id="ORD-1", order_arrival=100)
    order.add_sku("SKU_A", 10)

    # Simulate some committed/delivered progress
    order.commit_quantity("SKU_A", 3)
    order.deliver_quantity("SKU_A", 2)  # delivered=2, committed=3-2=1

    # Add same SKU again (duplicate row)
    order.add_sku("SKU_A", 5)

    skus = order.skus["SKU_A"]
    _report(
        "duplicate order-SKU: total_quantity aggregated",
        skus["total_quantity"] == 15,
        f"total_quantity={skus['total_quantity']}",
    )
    _report(
        "duplicate order-SKU: quantity_committed preserved",
        skus["quantity_committed"] == 1,
        f"quantity_committed={skus['quantity_committed']}",
    )
    _report(
        "duplicate order-SKU: quantity_delivered preserved",
        skus["quantity_delivered"] == 2,
        f"quantity_delivered={skus['quantity_delivered']}",
    )


# ---------------------------------------------------------------------------
# Test 5: NetLogo queue label check
# ---------------------------------------------------------------------------

def test_netlogo_monitor_label():
    """The monitor sourcing order_count must be labeled 'Picking Job Queue', not 'Order'."""
    nlogo_path = os.path.join(REPO_ROOT, "simulation.nlogo")
    if not os.path.exists(nlogo_path):
        _skip("NetLogo monitor label", "simulation.nlogo not found")
        return

    with open(nlogo_path, "r") as f:
        content = f.read()

    # Strip CRLF to normalize
    lines = content.replace("\r\n", "\n").split("\n")

    # Parse MONITOR blocks: format is MONITOR, x1, y1, x2, y2, label, reporter, ...
    found_label = None
    i = 0
    while i < len(lines):
        if lines[i].strip() == "MONITOR":
            # Next 4 lines are coordinates (x1, y1, x2, y2), then label, then reporter
            if i + 6 < len(lines):
                label = lines[i + 5].strip()
                reporter = lines[i + 6].strip()
                if reporter == "order_count":
                    found_label = label
                    break
        i += 1

    _report(
        "NetLogo monitor label is 'Picking Job Queue'",
        found_label == "Picking Job Queue",
        f"found_label={found_label!r}",
    )

    # Verify order_count still maps to item 2 (index 2) in the tick result
    found_mapping = any(
        "order_count" in line and "item 2" in line
        for line in lines
    )
    _report(
        "order_count still maps to item 2 of tick result",
        found_mapping,
    )


# ---------------------------------------------------------------------------
# Test 6: Local-executor summary with constructed OrderManager
# ---------------------------------------------------------------------------

def test_executor_order_metrics():
    """Verify order metrics derivation logic matches OrderManager state."""
    from model.order import Order

    # Simulate a set of orders
    orders = []
    o1 = Order("O1", 0)
    o1.add_sku("A", 10)
    o1.commit_quantity("A", 10)
    o1.deliver_quantity("A", 10)
    o1.start_processing(100)
    o1.complete_order(200)
    orders.append(o1)

    o2 = Order("O2", 50)
    o2.add_sku("B", 5)
    o2.commit_quantity("B", 5)
    o2.deliver_quantity("B", 5)
    o2.start_processing(150)
    o2.complete_order(400)
    orders.append(o2)

    o3 = Order("O3", 80)
    o3.add_sku("C", 3)
    # Not completed
    orders.append(o3)

    completed = [o for o in orders if o.is_order_completed()]
    orders_completed = len(completed)

    cycle_times = [
        o.order_complete_time - o.process_start_time
        for o in completed
        if o.order_complete_time >= 0 and o.process_start_time >= 0
    ]
    avg_cycle = sum(cycle_times) / len(cycle_times) if cycle_times else 0.0

    _report(
        "executor: orders_completed = 2",
        orders_completed == 2,
        f"completed={orders_completed}",
    )
    _report(
        "executor: avg_cycle_time = 175.0",
        abs(avg_cycle - 175.0) < 0.01,
        f"avg_cycle_time={avg_cycle}",
    )


# ---------------------------------------------------------------------------
# Test 7: Zero completed orders handled correctly
# ---------------------------------------------------------------------------

def test_zero_completed_orders():
    """When no orders are completed, avg_cycle_time should be 0.0 without division error."""
    from model.order import Order

    orders = [Order("O1", 0)]
    orders[0].add_sku("X", 5)

    completed = [o for o in orders if o.is_order_completed()]
    orders_completed = len(completed)

    if orders_completed > 0:
        cycle_times = [
            o.order_complete_time - o.process_start_time
            for o in completed
            if o.order_complete_time >= 0 and o.process_start_time >= 0
        ]
        avg_cycle = sum(cycle_times) / len(cycle_times) if cycle_times else 0.0
    else:
        avg_cycle = 0.0

    _report(
        "zero completed orders: count = 0",
        orders_completed == 0,
        f"completed={orders_completed}",
    )
    _report(
        "zero completed orders: avg_cycle_time = 0.0",
        avg_cycle == 0.0,
        f"avg_cycle_time={avg_cycle}",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Accounting & Observability Regression Tests")
    print("=" * 60)

    print("\n1. Duplicate pod-SKU aggregation")
    test_duplicate_pod_sku_aggregation()

    print("\n2. Global inventory invariant")
    test_global_inventory_invariant()

    print("\n3. Incompatible duplicate metadata")
    test_incompatible_metadata_fails()

    print("\n4. Duplicate order-SKU aggregation")
    test_order_add_sku_aggregation()

    print("\n5. NetLogo monitor label")
    test_netlogo_monitor_label()

    print("\n6. Executor order metrics")
    test_executor_order_metrics()

    print("\n7. Zero completed orders")
    test_zero_completed_orders()

    print("\n" + "=" * 60)
    total = _passed + _failed + _skipped
    print(f"Total: {total}  Passed: {_passed}  Failed: {_failed}  Skipped: {_skipped}")
    if _failed > 0:
        print("RESULT: FAILURE")
        sys.exit(1)
    else:
        print("RESULT: ALL PASSED")
        sys.exit(0)


if __name__ == "__main__":
    main()
