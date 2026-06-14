# RTS State-Feature Gap Map

This document catalogs the gap between the features defined in the mature reference repo (`netlogo-rmfs`) and the refactored Rika-host repo (`Rika's Version`). It also records which Phase 12 items were implemented from the gap map.

---

## 1. Feature Classifications

Each feature family is classified under one of the following statuses:

* **implemented**: Fully calculated using available Rika-host object data.
* **approx_repo_grounded**: Calculated using approximation strategies grounded in the new object model.
* **default_unavailable**: No data source in the Rika-host object model; defaults to standard baseline values.
* **hardcoded_zero**: Explicitly hardcoded to `0.0` or `0` in Python.
* **missing**: Exists in the mature repo's contract but is not defined in the new repo.
* **do_not_port**: Defer or ignore due to current design/scope constraints.
* **needs_user_decision**: Under review by the user.

---

## 2. Feature Families Gap Analysis

### 1. Next Retrieval Zone
* **Features**:
  * `next_retrieval_zone_known`
  * `next_retrieval_zone`
  * `next_retrieval_zone_one_hot__*`
* **Status**: `default_unavailable` / `hardcoded_zero`
* **Details**: Hardcoded to `0`, `""`, or all zeros in current `state.py`. Mature repo computed these from the NetLogo order pipeline when a future retrieval zone was known.

### 2. Committed Next Task
* **Features**:
  * `estimated_queue_time`
* **Status**: `approx_repo_grounded`
* **Details**: Phase 12 estimates logical load from robots currently associated with, or headed near, the selected replenishment station.

### 3. Source Station Context
* **Features**:
  * `source_station_is_picking`
  * `source_station_is_replenishment`
  * `picking_station_count`
  * `replenishment_station_count`
* **Status**: `implemented`
* **Details**: Grounded in current Rika-host `station` and `warehouse.station_manager` objects.

### 4. Station Coordinates/Location
* **Features**:
  * `source_station_x_norm`
  * `source_station_y_norm`
* **Status**: `implemented`
* **Details**: Calculated by normalizing current `station.pos_x` and `station.pos_y` in `state.py`.

### 5. Station/Replenishment Queue Pressure
* **Features**:
  * `selected_replenishment_station_logical_load`
* **Status**: `approx_repo_grounded`
* **Details**: Phase 12 estimates logical load from robots currently associated with, or headed near, the selected replenishment station.

### 6. Zone Occupancy/Free Slots
* **Features**:
  * `occupation_level`
  * `free_slot_count`
* **Status**: `implemented`
* **Details**: Calculated from `storage_manager.storages` and empty/assigned pods in `zone_features.py`.

### 7. Zone Traffic/Robot Density
* **Features**:
  * `zone_present_robot_count`
  * `neighbor_zone_present_robot_count`
  * `superzone_present_robot_count`
* **Status**: `approx_repo_grounded`
* **Details**: Phase 12 counts present robots by inferred zone, adjacent zones, and superzone grouping.

### 8. Destination Robot Counts
* **Features**:
  * `zone_destination_robot_count`
  * `neighbor_zone_destination_robot_count`
  * `superzone_destination_robot_count`
* **Status**: `approx_repo_grounded`
* **Details**: Phase 12 counts robot destinations by inferred zone, adjacent zones, and superzone grouping.

### 9. Cycle-Time Estimates
* **Features**:
  * `storage_cycle_time_estimate`
  * `replenish_cycle_time_estimate`
  * `arrival_rate_order_cycle_time`
* **Status**: `hardcoded_zero`
* **Details**: Hardcoded to `0.0` in `state.py` and `zone_features.py`.

### 10. SKU Similarity/Usefulness
* **Features**:
  * `sku_similarity`
* **Status**: `approx_repo_grounded`
* **Details**: Phase 12 estimates SKU similarity from overlap between the current pod SKUs and SKUs already assigned to pods in each candidate zone.

### 11. Replenishment Station Context
* **Features**:
  * `selected_replenishment_station_x_norm`
  * `selected_replenishment_station_y_norm`
  * `candidate_zone_to_selected_replenishment_station_distance`
  * `candidate_zone_to_nearest_replenishment_station_distance`
* **Status**: `approx_repo_grounded`
* **Details**: Phase 12 computes selected/nearest replenishment station coordinates and candidate-zone distances from current station and storage objects when available.

### 12. Action Validity Masks
* **Features**:
  * `store_action_valid`
  * `replenish_store_action_valid`
* **Status**: `implemented`
* **Details**: Fully active and computed from zone availability and stock below threshold ratio in `zone_features.py`.

### 13. Derived Stock-Risk Features
* **Features**:
  * Action features: `pod_fill_ratio`, `pod_below_threshold_ratio`, `replenishment_signal_active`, `pod_has_zero_and_global_low_sku`, `zero_global_low_sku_count`, `zero_global_low_sku_ratio`, `below_threshold_sku_count`, `below_threshold_sku_ratio`, `min_sku_fill_ratio`, `mean_sku_fill_ratio`, `shortage_depth`, `global_low_depth`
  * Stock features: `current_qty`, `limit_qty`, `fill_ratio`, `pod_below_threshold`, `below_threshold`, `is_zero_qty`, `is_zero_and_global_low`, `shortage_depth`, `global_low_depth`
  * Mature-only missing action features: `paper_replenishment_threshold`
  * Mature-only missing stock features: `threshold`, `global_inv_level`, `global_threshold_inv_level`
* **Status**: Mixed (`implemented` / `missing`)
* **Details**: Basic stats are fully implemented, but mature threshold and global inventory metadata are currently missing from the new repo's stock data representation.

---

## 3. Conclusions and Next Steps

Phase 12 resolved the selected replenishment station context, zone traffic/destination pressure, SKU similarity, and replenishment-distance gaps with current-repo-grounded approximations.

Remaining deferred/defaulted feature families are next retrieval context, committed next task/queue time, cycle-time estimates, and mature-only stock metadata that is not present in the current Rika-host data model.
