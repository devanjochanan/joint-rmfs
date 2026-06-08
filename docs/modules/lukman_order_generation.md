# Order Generation & SKU Allocation Module Ownership Profile

This profile documents the ownership details, current code mappings, and plans for the order generation and pod stocking module.

---

## 1. Module Overview
* **Owner**: Lukman
* **Responsibility**: Designing and orchestrating customer order streaming batches and initial allocations of item inventory levels inside the pod layouts.
* **Future Folder Location**: `src/rmfs/order_generation/`

---

## 2. Refactoring Phase Status

* **Status**: Scaffold placeholder only.
* **Restrictions**:
  * Do not write execution code in the scaffold directories.
  * Do not add imported-scenario playback mechanisms yet.
  * Do not change baseline CSV formats or layout matrix dimensions.
  * Preserving current baseline simulation behavior is paramount.

---

## 3. Behavior Source of Truth
The active behavior logic remains housed in:
* `model/order_generator.py`: Generates the order lists (`generated_order.csv`) based on SKU classifications (A/B/C) and quantity limits.
* `model/pod_generator.py` and `model/item_pod_generator.py`: Stocks SKU units into pods (`pods.csv`) based on frequency statistics.

---

## 4. Migration Risks & Verification Targets
Refactoring these generators affects:
* **Grid loading files**: Any issues with matrix structures (`generated_pod.csv`) crash the graph parsing engine in `netlogo.py`.
* **SKU inventory levels**: Inconsistencies between `pods.csv` and global catalogues (`items.csv`) result in pickers failing to retrieve requested stock units.
* **Batch processing timelines**: Changing the arrival timestamps in `generated_order.csv` disrupts simulated queues and alters throughput results.
