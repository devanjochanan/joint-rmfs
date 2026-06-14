# Regret-k Scheduling Audit

This document audits the status of regret-k task allocation scheduling in the Rika refactored repository (`Rika's Version`) compared to the mature reference repository (`netlogo-rmfs`).

---

## 1. Classification
* **Status**: `deferred`

---

## 2. Findings

### Mature Repo Implementation
In `netlogo-rmfs`, the class `RobotTaskAllocator` under `world/managers/robot_task_allocator.py` implemented a regret-based task allocation algorithm (`assign_active_grouped_tasks_regret_k` and `_assign_tasks_with_regret`). This algorithm computed scheduling priorities and assigned pods/robots to pick/replenish stations based on regret scores.

### New Repo Status
In `Rika's Version`, regret-k is not implemented or wired. The only mention of it is a placeholder note in `src/rmfs/decisions/rts/README.md`. The task allocation in the Rika-host currently relies on the default sequential assignment scheduler.

---

## 3. RTS-RL Observability of Scheduling Context

The RTS-RL action features define explicit placeholders to observe scheduling context:
* `estimated_queue_time`: Intended to observe station queue delay. (Currently `hardcoded_zero`).
* `selected_replenishment_station_logical_load`: Intended to observe replenishment station assignment pressure. (Currently dynamically computed based on robot destinations).
* `picking_station_count` and `replenishment_station_count`: Currently fully implemented and grounded.

Since regret-k task scheduling is deferred, some advanced queue time feedback features remain default/unavailable or hardcoded to zero. However, RTS-RL has the necessary feature slots ready to observe these context metrics once a regret-k scheduling system is implemented.
