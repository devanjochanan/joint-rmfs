# PPS (Pick Pod Selection) Module Ownership Profile

This profile documents the ownership details, current code mappings, and plans for the PPS module.

---

## 1. Module Overview
* **Owner**: Devan
* **Responsibility**: PPS (Pick Pod Selection) matching policies that decide which pods to bring to active stations based on current queues.
* **Future Folder Location**: `src/rmfs/decisions/pps/`

---

## 2. Refactoring Phase Status

* **Status**: Scaffold placeholder only.
* **Restrictions**:
  * Do not write execution code in the scaffold directories.
  * Do not add PPO or PPS-RL models yet.
  * Do not edit or modify POA, RTS, charging, or order generation internals.
  * Preserving current baseline simulation behavior is paramount.

---

## 3. Behavior Source of Truth
The active behavior logic remains housed in:
* `model/inventory.py`: Implements pod retrieval policies (specifically `process_orders`, `find_best_pod`, `find_pod_with_the_highest_pile_on`, and `find_pod_with_the_highest_demand`).

---

## 4. Migration Risks & Verification Targets
Refactoring PPS logic in future phases affects:
* **Inventory mutation**: PPS decides which SKUs are picked from which pods, updating the remaining quantities. Miscalculations cause inventory mismatches.
* **Station demand tracking**: Pod selections rely on order queues at pickers. Broken cues result in robots fetching incorrect pods.
* **Pod availability registry**: Pods must be correctly flagged as `idle` or `busy` (`mark_pod_available`, `mark_pod_not_available` in `model/pod_manager.py`).
