"""
eval/run_baseline.py
────────────────────
Straw-man baseline run for RMFS charging infrastructure evaluation.

Configuration
─────────────
  Placement : 10 chargers chosen uniformly at random from empty storage
              cells (value=0 in generated_pod.csv).  No optimisation.
  Policy    : Pure threshold-based charging — go charge when SoC < t_low,
              stop when SoC > t_up.  Interrupt heuristic is DISABLED so
              this is a clean two-threshold baseline.
  Count     : 10 chargers (matches the Phase-1 "budget = 10" comparator).

Policy thresholds (t_low=20 %, t_up=90 %)
  • 20 % low trigger — standard for AGV fleets to prevent deep-discharge
    damage; used in Mönch et al. (2018) Int. J. Prod. Res. 56(1), 588-606
    and Zou et al. (2018) IEEE Trans. Ind. Electron. 65(9), 7249-7258.
  • 90 % upper target — avoids lithium overcharge; also used in
    Roodbergen & Vis (2009) Eur. J. Oper. Res. 194(2), 343-362 survey.
  These values coincide with robot.py BATTERY_LOW_PCT / BATTERY_CHARGED_PCT
  defaults, so no model-level change is needed — only the interrupt is
  disabled to keep the policy as simple as possible.

Outputs (eval/runs/baseline/)
  charging_config.json  — overlay applied (charger coords + policy params)
  order-finished.csv    — fulfilled-orders log (copy from output/)
  netlogo.state         — pickled final universe (for post-hoc analysis)
  per_robot.json        — per-robot energy consumption + battery level
  run_summary.json      — headline metrics + timing

Usage
─────
    python eval/run_baseline.py
    python eval/run_baseline.py --horizon 100000 --seed 1
    python eval/run_baseline.py --num-chargers 10 --seed 42

Transfer note
─────────────
  Copy this file to the target project's eval/ folder unchanged.
  The only dependencies are:
    • generated_pod.csv  (warehouse grid — must already exist)
    • netlogo.py + model/ + engine/  (project simulation modules)
  Nothing from the Phase 1-3 analysis pipeline is required.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import netlogo          # noqa: E402
from model.robot import Robot  # noqa: E402


# ── Baseline policy constants ───────────────────────────────────────────────
# Mönch et al. (2018), Zou et al. (2018): 20 % low / 90 % upper thresholds.
# Set interrupt to 100 % (unreachable) → effectively disables the heuristic.
T_LOW = 20.0    # % SoC — go charge when below this
T_UP  = 90.0    # % SoC — stop charging when above this
T_INT = 100.0   # % SoC — interrupt threshold; 100 % disables it entirely


# ── Grid helpers ────────────────────────────────────────────────────────────

def load_grid(path: Path) -> list[list[int]]:
    rows = []
    with open(path, newline="") as f:
        for r in csv.reader(f):
            rows.append([int(x) for x in r])
    return rows


def find_empty_storage_cells(grid: list[list[int]]) -> list[tuple[int, int]]:
    """Return (row, col) for every cell with value 0 (empty storage slot)."""
    cells = []
    for r, row in enumerate(grid):
        for c, val in enumerate(row):
            if val == 0:
                cells.append((r, c))
    return cells


# ── Config writer ────────────────────────────────────────────────────────────

def write_charging_config(workdir: Path, charger_positions: list[list[int]]) -> dict:
    """Write charging_config.json with the baseline overlay."""
    config = {
        "pipeline": 0,
        "description": "baseline — random placement, 20/90 threshold, no interrupt",
        "num_chargers": len(charger_positions),
        "charger_positions": charger_positions,
        "battery_low_pct":       T_LOW,
        "battery_charged_pct":   T_UP,
        "battery_interrupt_pct": T_INT,
        "disable_active_charging": False,
    }
    with open(workdir / "charging_config.json", "w") as f:
        json.dump(config, f, indent=2)
    return config


# ── Transient-state cleanup (same as run_one.py) ────────────────────────────

def clear_transient_state(workdir: Path) -> None:
    for rel in (
        "netlogo.state",
        "assign_order.csv",
        "pod_info.csv",
        "output/order-finished.csv",
    ):
        p = workdir / rel
        if p.exists():
            p.unlink()


# ── Output helpers ───────────────────────────────────────────────────────────

def snapshot_outputs(workdir: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for rel in (
        "charging_config.json",
        "output/order-finished.csv",
        "netlogo.state",
    ):
        src = workdir / rel
        if src.exists():
            shutil.copy2(src, dest / Path(rel).name)


def extract_per_robot_snapshot(state_path: Path) -> list[dict]:
    """Load the final universe state and pull per-robot energy + battery."""
    with open(state_path, "rb") as f:
        universe = pickle.load(f)
    rows = []
    for obj in universe._objects:
        if getattr(obj, "object_type", None) != "robot":
            continue
        rows.append({
            "robot_id":           getattr(obj, "id", None),
            "battery_level_j":    float(getattr(obj, "battery_level_j", 0.0)),
            "battery_pct":        float(getattr(obj, "battery_pct", 0.0)),
            "energy_consumption_j": float(getattr(obj, "energy_consumption", 0.0)),
            "current_state":      getattr(obj, "current_state", None),
            "is_charging":        bool(getattr(obj, "is_charging", False)),
        })
    return rows


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Baseline: random charger placement + 20/90 threshold policy."
    )
    parser.add_argument(
        "--horizon", type=int, default=100_000,
        help="Max simulation ticks (default 100 000 — matches Phase-1 horizon).",
    )
    parser.add_argument("--heartbeat", type=int, default=5000,
                        help="Print progress every N ticks (0 = silent).")
    parser.add_argument(
        "--num-chargers", type=int, default=10,
        help="Number of chargers to place at random (default 10).",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help=(
            "RNG seed applied before setup().  Controls BOTH random charger "
            "selection and initial robot placement (netlogo.initRobots)."
        ),
    )
    parser.add_argument("--out-root", type=str, default="eval/runs")
    args = parser.parse_args()

    # ── 0. Seed RNGs ─────────────────────────────────────────────────────
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        print(f"[baseline] RNG seeded: {args.seed}")

    workdir = ROOT
    dest    = ROOT / args.out_root / "baseline"

    # ── 1. Find empty storage cells and pick charger positions ───────────
    grid        = load_grid(workdir / "generated_pod.csv")
    empty_cells = find_empty_storage_cells(grid)
    print(f"[baseline] Grid: {len(grid)} rows × {len(grid[0])} cols")
    print(f"[baseline] Empty storage cells available: {len(empty_cells)}")

    if len(empty_cells) < args.num_chargers:
        print(f"[baseline] ERROR: only {len(empty_cells)} empty cells, "
              f"cannot place {args.num_chargers} chargers.")
        return 1

    # random.sample uses the already-seeded global RNG
    chosen = random.sample(empty_cells, args.num_chargers)
    charger_positions = [[r, c] for r, c in chosen]
    print(f"[baseline] Selected {len(charger_positions)} random charger positions:")
    for pos in charger_positions:
        print(f"           row={pos[0]:3d}  col={pos[1]:3d}")

    # ── 2. Write charging config ─────────────────────────────────────────
    clear_transient_state(workdir)
    cfg = write_charging_config(workdir, charger_positions)
    print(
        f"[baseline] charging_config.json written  "
        f"(t_low={T_LOW}%  t_up={T_UP}%  interrupt=disabled)"
    )

    # ── 3. Apply policy overrides to Robot class ─────────────────────────
    Robot.BATTERY_LOW_PCT       = T_LOW
    Robot.BATTERY_CHARGED_PCT   = T_UP
    Robot.BATTERY_INTERRUPT_PCT = T_INT
    print(
        f"[baseline] Robot policy: "
        f"LOW={Robot.BATTERY_LOW_PCT}%  CHARGED={Robot.BATTERY_CHARGED_PCT}%  "
        f"INTERRUPT={Robot.BATTERY_INTERRUPT_PCT}% (disabled)"
    )

    # ── 4. Setup simulation ───────────────────────────────────────────────
    t0 = time.time()
    setup_result = netlogo.setup()
    t_setup = time.time() - t0
    if isinstance(setup_result, str) and setup_result.startswith("An error"):
        print("[baseline] SETUP FAILED:", setup_result)
        return 1
    print(f"[baseline] setup() done in {t_setup:.1f}s")

    # ── 5. Run ticks ──────────────────────────────────────────────────────
    t1 = time.time()
    tick_result = netlogo.console_tick(
        max_ticks=args.horizon,
        heartbeat_every=args.heartbeat,
    )
    t_tick = time.time() - t1
    print(f"[baseline] console_tick() done in {t_tick:.1f}s  result={tick_result!r}")

    # ── 6. Snapshot outputs ───────────────────────────────────────────────
    snapshot_outputs(workdir, dest)
    per_robot = extract_per_robot_snapshot(dest / "netlogo.state")

    with open(dest / "per_robot.json", "w") as f:
        json.dump(per_robot, f, indent=2)

    # ── 7. Aggregate metrics ──────────────────────────────────────────────
    dead             = sum(1 for r in per_robot if r["current_state"] == "dead")
    total_energy_j   = sum(r["energy_consumption_j"] for r in per_robot)
    total_energy_kwh = total_energy_j / 3_600_000.0
    mean_batt_pct    = (
        sum(r["battery_pct"] for r in per_robot) / len(per_robot)
        if per_robot else 0.0
    )

    # Count fulfilled orders from order-finished.csv
    orders_done = 0
    order_file  = dest / "order-finished.csv"
    if order_file.exists():
        with open(order_file, newline="") as f:
            orders_done = sum(1 for _ in csv.reader(f)) - 1  # subtract header

    summary = {
        "config":              "baseline",
        "horizon":             args.horizon,
        "seed":                args.seed,
        "num_chargers":        args.num_chargers,
        "charger_positions":   charger_positions,
        "t_low_pct":           T_LOW,
        "t_up_pct":            T_UP,
        "t_interrupt_pct":     T_INT,
        "setup_seconds":       round(t_setup, 2),
        "tick_seconds":        round(t_tick, 2),
        "tick_result":         tick_result,
        "num_robots":          len(per_robot),
        "dead_robots":         dead,
        "orders_finished":     orders_done,
        "total_energy_j":      round(total_energy_j, 2),
        "total_energy_kwh":    round(total_energy_kwh, 6),
        "mean_final_batt_pct": round(mean_batt_pct, 2),
    }
    with open(dest / "run_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # ── 8. Console summary ────────────────────────────────────────────────
    print(f"\n{'='*52}")
    print(f"  BASELINE RESULTS  (seed={args.seed}  horizon={args.horizon})")
    print(f"{'='*52}")
    print(f"  Robots total    : {len(per_robot)}")
    print(f"  Dead at end     : {dead}")
    print(f"  Orders finished : {orders_done}")
    print(f"  Total energy    : {total_energy_kwh:.6f} kWh  ({total_energy_j:.0f} J)")
    print(f"  Mean final SoC  : {mean_batt_pct:.1f} %")
    print(f"  Outputs         : {dest}")
    print(f"{'='*52}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())