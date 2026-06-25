# Rika RMFS Time Units Analysis

This document compiles the known simulation time steps, tick parameters, and schedules observed from code inspection.

---

## 1. Time Semantics Policy
> [!NOTE]
> **No Time Semantics Modification**
> Phase 2 does not alter or re-implement time semantics or simulation step calculations. The details below are documented solely for diagnostic and future planning purposes.

---

## 2. Suspected / Observed Time Concepts

Based on code inspection of `netlogo.py` and `model/inventory.py`, the following timing parameters are observed in the codebase:

1. **Simulation Time Step (`tick_to_second`)**:
   * Initialized as `0.15` in `model/inventory.py` (`self.tick_to_second = 0.15`).
   * Current setup code also assigns `tick_to_second` to `0.15` during setup.
   * Each simulation tick step is interpreted as representing **0.15 seconds** of simulated time.
2. **Simulation Horizon**:
   * Evaluated inside `netlogo.py` (`tick()` method).
   * The stop condition uses `universe._tick > 28800` where the loop stops (returns `IndexError`).
   * Under the assumed tick scaling, **28,800 simulated seconds** is interpreted as representing **8 hours** of warehouse operations (8 hours * 3600 seconds/hour = 28800). This interpretation should be preserved and verified during refactor.
3. **Order Process Interval**:
   * Handled in `model/inventory.py` (`tick()` method).
   * Checks for new orders when `int(self._tick) == self.next_process_tick` (integer increments of seconds).
   * The process ticks update by `1` simulated second every `1 / 0.15 = 6.67` ticks.
4. **Robot Task Delay Coefficients**:
   * Inside `model/robot.py`, a base delay coefficient is defined: `self.delay_per_task = 10` ticks.
   * **Turning Delay**: `self.turning_delay += self.delay_per_task * angular_change` (where angular change is the count of 90-degree rotations needed).
   * **Taking Pod / Lifting Delay**: `self.taking_pod_delay += self.delay_per_task`.
   * Under the assumed tick scaling, a delay of `10` ticks equates to `1.5 seconds` of simulated lift/turn operations. However, this is an observed code value rather than a validated simulation semantic.

> [!IMPORTANT]
> **Refactoring Constraints & Guidelines**
> * **Do not change time semantics in Phase 3**: Keep the current timing logic intact without modification.
> * **Do not claim validated timing behavior** without a targeted equivalence check. All details above are observed code facts rather than fully validated simulation semantics.


---

## 3. Recommended Verification Areas for Future Phases

Future reviewers and developers should verify the following files to ensure timing semantics are maintained:

* **Tick Duration & Speed**: Verify `netlogo.py` lines 857 and 887 to check that `universe.tick_to_second` remains `0.15` and the limit is `28800`.
* **Order Process Intervals**: Verify `model/inventory.py` lines 123, 216, 217, and 349–355 to verify order arrivals check only on integer second boundaries.
* **Kinematics & Acceleration**: Check `model/robot.py` lines 113–128 (`calculateEnergy`) and lines 791–798 (`update_motion_parameters` using deceleration distance math) to ensure physical velocity-to-time ratios are correct.
* **Wait Time Metric Aggregations**: Verify how `total_idle` is scaled in `model/inventory.py` line 170 (`total_idle += (o.total_idle * 0.15)`) to match actual idle time tracking.

---

## 4. RTS-RL Paper-Cycle Time Semantics

The RTS-RL semantic recovery patch does not change simulation time semantics. It records additional RTS rollout timestamps using the same warehouse time source (`Inventory._tick`) and stores `tick_to_second` for derived NetLogo-step metadata.

* **Paper-cycle start**: RTS decision tick (`paper_cycle_start_tick` / `return_start_tick`).
* **Return arrival**: storage-return completion tick (`paper_cycle_storage_arrival_tick`), logged as a pending diagnostic outcome with `return_duration`.
* **Paper-cycle completion**: same robot's next picking-station arrival tick (`paper_cycle_next_station_arrival_tick`).
* **Training target**: `paper_cycle_duration = paper_cycle_next_station_arrival_tick - paper_cycle_start_tick`.

Return duration is auxiliary telemetry and is not used as the PPO reward horizon. `netlogo_step` and `netlogo_steps_elapsed_since_decision` are derived from warehouse time and `tick_to_second`; no fixed conversion constant is introduced in RTS-RL logging or training.
