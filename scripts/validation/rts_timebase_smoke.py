#!/usr/bin/env python3
"""Pure smoke for RTS training timebase helpers."""

from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.rmfs.rl.rts.training.seeding import derive_worker_seed
from src.rmfs.rl.rts.training.timebase import (
    RMFSTimebase,
    netlogo_steps_to_warehouse_time,
    validate_tick_to_second,
    warehouse_time_to_netlogo_steps,
)


def assert_raises(fn, expected):
    try:
        fn()
    except expected:
        return
    raise AssertionError(f"expected {expected.__name__}")


def main():
    assert validate_tick_to_second(0.5) == 0.5
    assert netlogo_steps_to_warehouse_time(4, 0.5) == 2.0
    assert warehouse_time_to_netlogo_steps(2.0, 0.5) == 4
    tb = RMFSTimebase(
        netlogo_steps_completed=4,
        tick_to_second=0.5,
        warehouse_time_start=10.0,
        warehouse_time_end=12.0,
    )
    assert tb.warehouse_time_elapsed == 2.0
    assert tb.expected_warehouse_time_elapsed == 2.0
    assert derive_worker_seed(42, 1, 2) == derive_worker_seed(42, 1, 2)
    assert derive_worker_seed(42, 1, 2) != derive_worker_seed(42, 1, 3)
    assert_raises(lambda: validate_tick_to_second(0), ValueError)
    assert_raises(lambda: netlogo_steps_to_warehouse_time(-1, 0.5), ValueError)
    print("rts timebase smoke ok")


if __name__ == "__main__":
    main()

