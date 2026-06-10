# RTS Decision Seam — Phase 5B

This document describes the behavior-preserving RTS (Return-to-Storage)
decision seam introduced in Phase 5B on the `dewa_rts` branch.

---

## 1. Where the seam sits

The RTS seam is a strategy pattern inserted into the robot's
return-to-storage flow:

```
Robot.advance_state_if_needed()
  └─ elif current_state == "returning_pod"
       └─ self.handle_pod_return(station)          ← NEW METHOD
            └─ self.warehouse.rts_policy.select_destination(context)
```

The policy package lives at:

```
src/rmfs/decisions/rts/
├── __init__.py           # Public re-exports
├── types.py              # RTSDestinationContext, RTSDecision
├── policy.py             # RTSPolicy protocol
└── nearest_policy.py     # CurrentRTSPolicy (default)
```

---

## 2. What behavior it preserves

The default `CurrentRTSPolicy` wraps the **exact same** decision logic
that was previously inline in `model/robot.py` lines 730–758
(commit `d690c77`):

- `return_fix=True` preserves fixed return
- `return_nearest=True` preserves nearest return
- `return_nearest` with no empty storage preserves original fallback to current pod position
- neither `return_fix` nor `return_nearest` is treated as invalid configuration and raises explicitly

All side effects are executed **in `model/robot.py`**, not in the policy:

- `self.destination` assignment
- `self.job.pod_return_coordinate` recording
- `self.job.writePodReturnReport(...)` invocation
- `self.set_move(...)` path planning
- `nearest_storage.setStoragePod(...)` reservation
- `storage_manager.pods_to_storage[...]` update
- `self.initial_w1 = False`
- `upsert_pod_travel(...)` SQLite telemetry

---

## 3. What it does NOT change

- **POA / Future-Aware logic** — intentionally untouched (Devan's area)
- **PPS / Station-Match scoring** — intentionally untouched (Devan's area)
- **Order generation / pod-SKU allocation** — intentionally untouched (Lukman's area)
- **Charging / energy logic** — intentionally untouched (Salsa's area)
- **Path planning** — `set_move()`, `dijkstra()`, graph structures unchanged
- **Storage manager** — `getNearestEmptyStorageToLocation()` unchanged
- **Database schema** — no table/column changes
- **CSV format** — no schema changes
- **setup/tick return shapes** — unchanged
- **simulation.nlogo / netlogo.py** — not edited

---

## 4. How this differs from RTS-RL hooks

This seam is a **delegation wrapper**, not an RL integration:

- No reward function
- No observation vector
- No action space
- No PPO/DQN training loop
- No checkpoint loading
- No TensorBoard logging

Future RTS-RL work (Phase 6+) will create a new `RTSPolicy`
implementation that plugs into the same interface.  The seam
guarantees that swapping policies does not require modifying the
robot controller or any side-effect code.

---

## 5. Validation run

Phase 5B validation:

```bash
# 1. Compile check
/home/dewan/torch-gpu/bin/python -m py_compile \
  model/robot.py model/inventory.py \
  src/rmfs/decisions/rts/types.py \
  src/rmfs/decisions/rts/policy.py \
  src/rmfs/decisions/rts/nearest_policy.py

# 2. Import integrity
PYTHONPATH=. /home/dewan/torch-gpu/bin/python -c "import netlogo; print('netlogo import ok')"

# 3. Short executor smoke
/home/dewan/torch-gpu/bin/python scripts/run/local_executor_smoke.py \
  --runs 2 --ticks 3 --max-workers 2 --snapshot-inputs \
  --output-root data/runtime/local_executor_smoke/phase5_rts_seam_smoke
```

---

## 6. What remains for Phase 6

- Create `RLRTSPolicy` implementing `RTSPolicy` with observation/action/reward
- Add policy selection via configuration or constructor argument
- Add RTS-specific telemetry/logging hooks (optional decorator pattern)
- Wire RL training loop to call policies during episodes
- Golden trace comparison against Phase 5B baseline (1000+ ticks)
