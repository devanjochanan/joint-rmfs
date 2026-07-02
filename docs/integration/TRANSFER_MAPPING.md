# Transfer Mapping — charging contribution → team repo (`devanjochanan/joint-rmfs`, branch `charging_policy_x_placement`)

**Goal:** transfer the **generalizable** charging machinery (P1–P4 + adaptive
generator + mechanism + DoE analysis), NOT the instance-specific numbers. The
team's warehouse changes to **3 picking stations + different pods/SKUs**, so the
12 hardcoded coordinates currently in their repo (`salsa_charging_config.json`)
are for the OLD 5-station grid and must be **regenerated**, not reused.

> Do NOT do this in this folder (84 uncommitted files + divergent history).
> Work in a **separate clone/worktree** of the team branch (see "Commit" below).

## What the team already has
- `data/input/charging/salsa_charging_config.json` — your **instance-specific** config (12 coords, 6/6, 18/60/50). ← will be wrong for 3 stations.
- `scripts/data/build_charging_solution.py` — the **hardcoded** generator (P3 cells hardcoded).
- `src/rmfs/app/charging_bridge.py` — opt-in Stage-3 accounting bridge.
- `run_baseline.py` — the random-10 / 20/90 baseline (already ported, generalizable).
- Pending (their doc, §6): charging **mechanism** in `model/robot.py`, charger state in `engine/universe.py`, overlay+policy in `netlogo.py`.

## File mapping

| Your file | Team destination | Action |
|---|---|---|
| `model/charging_layout_generator.py` (**P1–P4**) | `model/charging_layout_generator.py` *(new)* | **ADD** ⭐ the missing generalizable core |
| `eval/build_adaptive_hybrid.py` (adaptive |C|/split) | `scripts/data/build_adaptive_hybrid.py` *(new)* | **ADD** — replaces the hardcoded build script |
| `eval/build_baseline_random.py` (baseline config gen) | `scripts/data/build_baseline_random.py` *(new)* | **ADD** — baseline comparator, config-driven |
| charging methods in `model/robot.py` (`_apply_drive_by_charging`, `_start_charging_trip`, `_should_interrupt_charging`, energy model, `BATTERY_*_PCT`, `CORRECTED_ENERGY_MODEL`) | team `model/robot.py` | **MERGE per-function** (their robot.py is refactored — do not overwrite) — their touchpoint #1 |
| charger state (`charger_cells`, `active_charger_cells`, `occupied_chargers`) | team `engine/universe.py` | **MERGE** — their touchpoint #2 |
| `netlogo.py` `draw_storage_from_generated_file()` overlay parsing + **policy auto-apply** + dynamic `pod_num` | team `netlogo.py` | **MERGE** — their touchpoints #3 & **#4 (already solved by the policy-auto-apply fix)** |
| `eval/run_one.py`, `eval/run_doe_phase*.py`, `eval/analyze_*.py` | `scripts/experiments/` *(new)* | **ADD** (optional) — the DoE analysis to find optimal placement+policy on a new layout |

## Leave behind (instance-specific — angka fix)
- `final_solution/charging_config.json` and team's `salsa_charging_config.json` → **regenerate** for the 3-station grid via the generator; keep old only as an *example*.
- `final_solution/build_charging_solution.py` and team's `scripts/data/build_charging_solution.py` (hardcoded) → superseded by `build_adaptive_hybrid.py`.

## On the team's layout (3 stations) — regenerate, don't reuse
```bash
# inside the team clone, with their generated_pod.csv (3-station grid):
python scripts/data/build_adaptive_hybrid.py --n 20 --rho 0.6 --out data/input/charging/salsa_charging_config.json
python scripts/data/build_baseline_random.py --num-chargers 10 --seed 1 --out data/input/charging/baseline_config.json
# then run each through the team's setup/tick path and compare run_summary metrics.
```
Verified here: the P1–P4 + adaptive-hybrid pipeline RUNS on 3-station / different-pod / smaller-grid layouts (adaptive split auto-becomes 6/6 for 3 stations).
