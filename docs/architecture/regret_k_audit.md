# Regret-k Scheduling Audit

This document audits the status of regret-k task allocation scheduling in the Rika refactored repository (`Rika's Version`) compared to the mature reference repository (`netlogo-rmfs`).

---

## 1. Classification
* **Status**: `implemented_active_job_queue`
* **Default allocator**: `regret_k`
* **Default k**: `2`
* **Scope**: active visible `Inventory.job_queue` assignments to idle robots.
* **Deferred mature behavior**: committed-next reservations, future job lookahead, and broader station-pressure scheduling feedback remain out of scope.

---

## 2. Findings

### Mature Repo Implementation
In `netlogo-rmfs`, the class `RobotTaskAllocator` under `world/managers/robot_task_allocator.py` implemented a regret-based task allocation algorithm (`assign_active_grouped_tasks_regret_k` and `_assign_tasks_with_regret`). This algorithm computed scheduling priorities and assigned pods/robots to pick/replenish stations based on regret scores.

### New Repo Status
In `Rika's Version`, active job-queue regret-k task allocation is implemented in `src/rmfs/decisions/task_allocation/regret_k.py` and wired into `model/inventory.py`.

The scheduler selects one assignment per tick from currently visible jobs and currently idle robots. Under `regret_k`, each feasible job is scored by the distance gap between its cheapest robot and its next-best robots up to k. The selected assignment prefers higher regret, then lower cheapest cost, then stable queue/job/robot tie-breakers. The legacy behavior remains available as `legacy_nearest`, which assigns the first feasible queue job to its nearest idle robot.

This is a targeted recovery patch, not a full port of the mature scheduler. It does not introduce committed-next task reservations or lookahead beyond the active queue contents.

---

## 3. RTS-RL Observability of Scheduling Context

The RTS-RL action features define explicit placeholders to observe scheduling context:
* `estimated_queue_time`: Intended to observe station queue delay. (Currently `hardcoded_zero`).
* `selected_replenishment_station_logical_load`: Intended to observe replenishment station assignment pressure. (Currently dynamically computed based on robot destinations).
* `picking_station_count` and `replenishment_station_count`: Currently fully implemented and grounded.

Since only active job-queue regret-k is implemented, some advanced queue time feedback features remain default/unavailable or hardcoded to zero. RTS-RL artifacts now record scheduler metadata (`robot_task_allocator`, `regret_k`, `task_allocator_scope`, and `committed_next_reservations_enabled`) so dry-run/training outputs can be audited against the allocator used by workers.

No performance, throughput, congestion, paper-fidelity, or full mature-scheduler equivalence claim is made from this patch.
