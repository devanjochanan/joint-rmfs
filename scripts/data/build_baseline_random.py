"""
eval/build_baseline_random.py
─────────────────────────────
GENERALIZABLE baseline (straw-man comparator) config generator:
  Placement : N chargers chosen uniformly at random from empty cells (value 0).
  Policy    : 20 / 90 thresholds, interrupt DISABLED (battery_interrupt_pct=100).
  Count     : 10 chargers (default).

Adapts to ANY grid (reads generated_pod.csv) — parallel to
build_adaptive_hybrid.py (the solution). Generate BOTH configs on the same grid
and run each to compare baseline vs solution. NOT instance-specific: no fixed
coordinates or thresholds-for-one-layout; positions are re-drawn per grid.

    python eval/build_baseline_random.py --num-chargers 10 --seed 1 \
        --grid generated_pod.csv --out charging_config.json
"""
from __future__ import annotations
import sys, json, csv, random, argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]   # scripts/data/<file> -> repo root
sys.path.insert(0, str(ROOT))

# Baseline policy (Monch et al. 2018; Zou et al. 2018): 20% low / 90% upper;
# interrupt at 100% => should_interrupt never fires => heuristic disabled.
T_LOW, T_UP, T_INT = 20.0, 90.0, 100.0


def load_grid(path) -> list[list[int]]:
    with open(path, newline="") as f:
        return [[int(x) for x in r] for r in csv.reader(f)]


def build(grid: list[list[int]], num_chargers: int = 10, seed: int = 1) -> dict:
    rng = random.Random(seed)
    empty = [[r, c] for r, row in enumerate(grid)
             for c, v in enumerate(row) if v == 0]
    if len(empty) < num_chargers:
        raise ValueError(f"only {len(empty)} empty cells, need {num_chargers}")
    positions = rng.sample(empty, num_chargers)
    return {
        "pipeline": 0,
        "description": "baseline -- random placement, 20/90 threshold, no interrupt",
        "num_chargers": len(positions),
        "charger_positions": positions,
        "battery_low_pct": T_LOW,
        "battery_charged_pct": T_UP,
        "battery_interrupt_pct": T_INT,
        "disable_active_charging": False,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-chargers", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--grid", default=str(ROOT / "generated_pod.csv"))
    ap.add_argument("--out", default=str(ROOT / "charging_config.json"))
    a = ap.parse_args()
    grid = load_grid(a.grid)
    cfg = build(grid, a.num_chargers, a.seed)
    Path(a.out).write_text(json.dumps(cfg, indent=2))
    print(f"[baseline] grid {len(grid)}x{len(grid[0])} -> "
          f"{cfg['num_chargers']} random chargers, policy 20/90 (interrupt off)")
    print(f"[baseline] wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
