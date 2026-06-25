# RMFS RTS (Return-to-Storage) Module

This folder contains the RTS decision seam introduced in Phase 5B.

* **Status**: Active — behavior-preserving seam.
* **Owner**: Dewa
* **Rules**:
  * Default policy (`CurrentRTSPolicy`) preserves existing fixed/nearest behavior.
  * No RL hooks, PPO training, or reward functions live here yet.
  * Do not import POA/PPS/charging/order-generation logic from this package.
* **Purpose**: RTS decisions (Return-to-Storage placement algorithm).
  The current implementation wraps the original inline fixed/nearest
  logic from `model/robot.py` behind a strategy interface so that
  future policies (RTS-RL, regret-k, etc.) can be swapped in without
  modifying the robot controller.

## Package contents

| File | Purpose |
|---|---|
| `__init__.py` | Public surface: re-exports types, protocol, and default policy |
| `types.py` | `RTSDestinationContext` (input) and `RTSDecision` (output) dataclasses |
| `policy.py` | `RTSPolicy` protocol (strategy interface) |
| `nearest_policy.py` | `CurrentRTSPolicy` — behavior-preserving default implementation |

## Integration points

* `model/inventory.py` instantiates `CurrentRTSPolicy()` as `self.rts_policy`.
* `model/robot.py` calls `self.warehouse.rts_policy.select_destination(context)`
  inside `handle_pod_return()`, then applies all side effects locally.
