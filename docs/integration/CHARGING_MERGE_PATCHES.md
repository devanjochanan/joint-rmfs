# Charging mechanism — MERGE patches for the team repo (`engine/` + `model/`)

These are the three **MERGE** touchpoints (config + bridge alone are inert — see
`salsa_charging_pending.md` §6). The flow you are wiring in:

```
engine/universe.py  Universe.tick()  ->  o.move()           # already exists
model/robot.py      Robot.move()     ->  [base drain] [charge trigger] [drive-by]   # ADD these
model/robot.py      Robot.drawNextPosition() -> [motion drain]                       # ADD one line
netlogo.py          draw_storage_from_generated_file() -> register charger_cells + apply policy  # ADD
```

> ⚠️ **Adapt + test, don't blind-paste.** These are adapted to the team's
> structure from reading their files via git, but I cannot run their code.
> **Verify the marked attributes exist** in the team's `Robot`/`Universe`, then
> run a smoke (setup + a few hundred ticks) before relying on it. The long
> method bodies are in your `model/robot.py` — copy them verbatim and fix only
> the marked dependencies.

---

## Dependency checklist (verify in the team's classes)
The charging code references these. Confirm each exists (the team's Robot already
has most — it has `self.universe`, `pos_x/pos_y`, `current_state`, `calculateEnergy`,
`NetLogoCoordinate`, `landscape`):

| Reference in charging code | Team equivalent — verify |
|---|---|
| `self.universe.tick_to_second` | ✔ exists (used in `calculateEnergy`) |
| `self.pos_x`, `self.pos_y` | ✔ exists (`get_robot_by_coord`) |
| `self.current_state`, `self.job` | ✔ exists |
| `self.set_move(dest, graph=...)`, `self.universe.graph` | verify name (routing call) |
| `self.movementPlan()` | verify the team's per-tick movement call name |
| `self.warehouse.job_queue` | team uses `self.universe` — **change to `self.universe.job_queue`** (or their queue) |
| `self._release_charger()` | NEW helper (below) |
| `self.is_charging`, `self.idle_time` | add `is_charging=False` in `__init__`; `idle_time` verify |

---

## PATCH A — `model/robot.py` (Robot class)

### A1. Class attributes (battery spec + policy) — paste into the `class Robot` body
```python
    # ── Battery spec (commercial AGV; see paper Energy Model) ──────────────
    BATTERY_CAPACITY_J: float = 6_480_000.0     # 1.8 kWh
    BASE_DRAIN_RATE_PER_S: float = 90.0         # electronics drain, J/s (5%/hr)
    CHARGE_POWER_W: float = 397.6               # 28 A x 14.2 V
    INITIAL_BATTERY_FRAC: float = 1.0           # start-of-shift SoC fraction
    # ── Charging policy (config-driven; netlogo overwrites these at setup) ──
    BATTERY_LOW_PCT: float = 20.0               # go charge below this
    BATTERY_CHARGED_PCT: float = 90.0           # stop charging above this
    BATTERY_INTERRUPT_PCT: float = 50.0         # interrupt threshold
    CORRECTED_ENERGY_MODEL: bool = False        # R1/R16 physics fix (opt-in)
```

### A2. In `Robot.__init__` — add the battery + charging state
```python
        self.battery_level_j: float = self.BATTERY_CAPACITY_J * self.INITIAL_BATTERY_FRAC
        self.is_charging: bool = False
```

### A3. `battery_pct` property — paste as-is (your `robot.py:217-220`)
```python
    @property
    def battery_pct(self) -> float:
        return (self.battery_level_j / self.BATTERY_CAPACITY_J) * 100.0
```

### A4. The charging methods — copy VERBATIM from your `model/robot.py`, fix marked deps
- `_apply_drive_by_charging`  (your lines ~222-242) — paste as-is.
- `_start_charging_trip`       (your lines ~244-~330) — paste; uses `self.set_move`, `self.universe.graph`, `NetLogoCoordinate`, `universe.charger_cells/active_charger_cells/occupied_chargers`.
- `_should_interrupt_charging` (your lines ~560-~590).
- `_release_charger`           (your helper; clears the robot's claim in `universe.occupied_chargers`).

`_apply_drive_by_charging` (short — reproduced so you can sanity-check the shape):
```python
    def _apply_drive_by_charging(self) -> None:
        charger_cells = getattr(self.universe, 'charger_cells', set())
        grid_pos = (round(self.pos_x), round(self.pos_y))
        if grid_pos in charger_cells:
            charge_j = self.CHARGE_POWER_W * self.universe.tick_to_second
            self.battery_level_j = min(self.BATTERY_CAPACITY_J,
                                       self.battery_level_j + charge_j)
            self.is_charging = True
        else:
            self.is_charging = False
```

### A5. Inject into the team's `Robot.move()` (their line ~775)
At the **top** of `move()` (before movement/path logic):
```python
        # Dead-robot handling
        if self.battery_level_j <= 0.0 and self.current_state != "dead":
            self._release_charger(); self.velocity = 0; self.acceleration = 0
            self.route_stop_points = []; self.current_state = "dead"
            return
        if self.current_state == "dead":
            return
        # Base operational drain (skipped while charging)
        if not self.is_charging:
            self.battery_level_j = max(0.0, self.battery_level_j
                                       - self.BASE_DRAIN_RATE_PER_S * self.universe.tick_to_second)
        # Charge trigger: idle + low battery -> nearest charger
        if (self.current_state == "idle"
                and (self.job is None or getattr(self.job, "is_finished", True))
                and self.battery_pct < self.BATTERY_LOW_PCT
                and not getattr(self.universe, "disable_active_charging", False)):
            self._start_charging_trip()
        # Preemptive: abort taking_pod if low (return job to queue first)
        elif (self.current_state == "taking_pod" and self.battery_pct < self.BATTERY_LOW_PCT
              and not self.is_charging
              and not getattr(self.universe, "disable_active_charging", False)):
            if self.job is not None and not getattr(self.job, "is_finished", False):
                self.universe.job_queue.append(self.job)   # <-- team queue name
            self.job = None; self.route_stop_points = []
            self.velocity = 0; self.acceleration = 0; self.current_state = "idle"
            self._start_charging_trip()
```
At the **bottom** of `move()` (AFTER the team's movement call, so it fires every tick):
```python
        self._apply_drive_by_charging()
```
Also add the "interrupt on arrival" check where the team detects a robot has
reached its charger destination (your `robot.py` does this in `move()` after
`movementPlan`): release + go idle when `battery_pct >= BATTERY_CHARGED_PCT` or
`_should_interrupt_charging()` returns True.

### A6. In the team's `Robot.drawNextPosition()` (motion-energy step) — add battery motion drain
Right after they compute `energy = self.calculateEnergy(...)` and `self.energy_consumption += energy`:
```python
        self.battery_level_j = max(0.0, self.battery_level_j - energy)
```

---

## PATCH B — `engine/universe.py` (shared charger state)
Add to the `Universe` class body (class attrs) so `netlogo` can populate them and
`Robot` can read them:
```python
    charger_cells = set()            # all charger coords (drive-by eligible)
    active_charger_cells = set()     # active-dispatch targets (empty => global)
    occupied_chargers = {}           # cell -> robot id (one claim per cell)
    disable_active_charging = False  # True => opportunity-only (drive-by only)
```
> If `Universe` uses instance state, initialise these in `__init__` instead
> (`self.charger_cells = set()` …) to avoid shared-class-attr aliasing.

---

## PATCH C — `netlogo.py` `draw_storage_from_generated_file()`
Two additions inside the grid-parsing function:

**C1. Register charger cells** — wherever the team builds the grid, for cells that
are chargers (value `2`, and the overlay coordinates from the config), do:
```python
        universe.charger_cells.add((x, y))          # (x=col, y=row)
        # if the config has active_charger_positions:
        #   universe.active_charger_cells.add((x, y))
```
and read the overlay + structure flags from the config at the top of the function:
```python
    import os, json
    _cfg = {}
    if os.path.exists('charging_config.json'):
        _cfg = json.load(open('charging_config.json'))
    for pos in _cfg.get('charger_positions', []):
        universe.charger_cells.add((int(pos[1]), int(pos[0])))   # [row,col] -> (x,y)
    for pos in _cfg.get('active_charger_positions', []):
        universe.active_charger_cells.add((int(pos[1]), int(pos[0])))
    universe.disable_active_charging = bool(_cfg.get('disable_active_charging', False))
```

**C2. Apply policy to the Robot class BEFORE robots are created** (their touchpoint
#4 — this is the integration fix). Place it in `draw_storage` *before* `initRobots`
runs (or at the top of `setup()`):
```python
    from model.robot import Robot
    for _k, _a in (("battery_low_pct","BATTERY_LOW_PCT"),
                   ("battery_charged_pct","BATTERY_CHARGED_PCT"),
                   ("battery_interrupt_pct","BATTERY_INTERRUPT_PCT"),
                   ("initial_battery_frac","INITIAL_BATTERY_FRAC")):
        if _k in _cfg:
            setattr(Robot, _a, float(_cfg[_k]))
    if "corrected_energy_model" in _cfg:
        Robot.CORRECTED_ENERGY_MODEL = bool(_cfg["corrected_energy_model"])
```

---

## Verify-and-test checklist (in the team clone)
1. `python -c "import model.robot, netlogo"` — imports clean.
2. Run a smoke (setup + ~500 ticks) with the generated config; confirm in a per-robot
   dump that `battery_pct` changes over time and some robots enter `going_to_charge`.
3. Confirm `universe.charger_cells` is non-empty after setup (placement registered).
4. Only then run the full comparison (baseline vs adaptive-hybrid).
