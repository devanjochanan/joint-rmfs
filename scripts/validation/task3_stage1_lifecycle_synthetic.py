#!/usr/bin/env python3
"""Task 3 Stage 1 synthetic validation: charging lifecycle (C), committed-next
exact-once (D), and replenishment-cap integrity (E). No simulation horizon.

Run:  /home/dewan/torch-gpu/bin/python scripts/validation/task3_stage1_lifecycle_synthetic.py
"""
from __future__ import annotations
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from model.robot import Robot
from model.inventory import Inventory
from src.rmfs.decisions.task_allocation.committed_next import (
    CommittedNextRegistry, CommittedNextReservation, STATUS_CANCELLED,
)

results = []
def check(name, cond, mechanism):
    results.append((name, bool(cond), mechanism))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name} :: {mechanism}")

# ───────────────────────── shared fixtures ─────────────────────────
class FakeLandscape:
    def setObject(self, *a, **k): pass

class FakeUniverse:
    def __init__(self, charger_cells):
        self.charger_cells = set(charger_cells)
        self.active_charger_cells = set()
        self.occupied_chargers = {}
        self.charging_enabled = True
        self.disable_active_charging = False
        self._tick = 0
        self.tick_to_second = 0.15
        self.graph = object()
        self.job_queue = []
        self.charging_counters = {}
        self.landscape = FakeLandscape()
    def get_movable_objects(self): return []

def make_robot(uni, x=5, y=5, soc_pct=15.0, rid=1):
    r = Robot.__new__(Robot)
    r.universe = uni; r.warehouse = uni; r.id = rid
    r.pos_x = float(x); r.pos_y = float(y)
    r.velocity = 0.0; r.acceleration = 0.0; r.heading = 0; r.load_mass = 0.0
    r.battery_level_j = Robot.BATTERY_CAPACITY_J * (soc_pct / 100.0)
    r.is_charging = False; r._claimed_charger = None; r.charge_after_current_task = False
    r._waiting_for_charger = False; r._claim_created_tick = None
    r._claim_progress_pos = None; r._drive_by_charged_last = False
    r.current_state = "idle"; r.job = None; r.route_stop_points = []
    r.fixed_load_energy_consumption = 0.0; r.latest_tick = 0
    return r

def set_move_ok(dest, graph=None): return None
def fail_for(bad):
    def _sm(dest, graph=None):
        if (int(round(dest.x)), int(round(dest.y))) in bad: raise RuntimeError("route fail")
    return _sm

class FakePod:
    def __init__(self, pid):
        self.pod_id = pid
        self.committed_next_owner_robot_id = None
        self.committed_next_reservation_id = None
        self.is_awaiting_replenishment = False
        self.rts_return_in_progress = False
        self.has_pending_replenishment_dispatch = True
        self.must_replenish_before_pick = True

class FakeJob:
    def __init__(self, jid, pod):
        self.my_id = jid; self.pod = pod; self.is_finished = False
        self.orders = [("o1", "s1", 1)]; self.station_id = "picker-1"
        self.pod_coordinate = (1, 1)
        self.committed_next_owner_robot_id = None
        self.committed_next_reservation_id = None

# ═════════════════════════ Part C (charging lifecycle) ═════════════════════════
print("=== Part C: charging lifecycle ===")
u = FakeUniverse({(10,5),(20,5)}); r = make_robot(u, 5,5,15); r.set_move = set_move_ok
ok = r._start_charging_trip()
check("C_valid_trip", ok and r._claimed_charger==(10,5) and u.occupied_chargers.get((10,5))==r.id,
      "nearest charger claimed")

u = FakeUniverse({(10,5)}); u.occupied_chargers[(10,5)]=999; r=make_robot(u,soc_pct=15); r.set_move=set_move_ok
ok = r._start_charging_trip()
check("C_no_charger_waiting", (not ok) and r._waiting_for_charger and r._claimed_charger is None,
      "no false claim; explicit waiting")

u = FakeUniverse({(10,5),(20,5)}); r=make_robot(u,5,5,15); r.set_move=fail_for({(10,5)})
ok = r._start_charging_trip()
check("C_first_fail_second_ok", ok and r._claimed_charger==(20,5) and (10,5) in u._unroutable_charger_cells,
      "distance-order retry; mark first invalid")

u = FakeUniverse({(10,5),(20,5)}); r=make_robot(u,soc_pct=15); r.set_move=fail_for({(10,5),(20,5)})
ok = r._start_charging_trip()
check("C_all_fail_waiting", (not ok) and r._waiting_for_charger and r._claimed_charger is None,
      "all routes fail -> waiting, no claim")

u = FakeUniverse({(10,5)}); r=make_robot(u,soc_pct=0.0); r.set_move=set_move_ok
r._start_charging_trip(); r.battery_level_j=0.0
u.committed_next_registry=None; u.rts_rollout_runtime=None
dead = r._charging_pre_move()
check("C_release_on_death", dead and r.current_state=="dead" and r._claimed_charger is None,
      "death releases claim")

u = FakeUniverse({(5,5)}); r=make_robot(u,5,5,55); r.set_move=set_move_ok
r._apply_drive_by_charging()
check("C_incidental_available", (not r.is_unavailable_for_work_due_to_charging()) and r._claimed_charger is None,
      "incidental drive-by: energy only, stays available, no claim")

u = FakeUniverse({(10,5)}); r=make_robot(u,soc_pct=55); r.set_move=set_move_ok; r._start_charging_trip()
check("C_deliberate_unavailable", r.is_unavailable_for_work_due_to_charging(),
      "deliberate trip still unavailable (regression guard)")

# ═════════════════════════ Part D (committed-next exact-once) ═════════════════════════
print("=== Part D: committed-next exact-once ===")
def build_reservation(reg, robot_id="1", job_id="J1", pod_id="7"):
    pod = FakePod(pod_id); job = FakeJob(job_id, pod)
    res = CommittedNextReservation(reservation_id="R1", owner_robot_id=robot_id, job=job,
        job_id=job_id, pod_id=pod_id, picking_station_id="picker-1",
        original_queue_index=0, created_time_seconds=0.0)
    reg.reservations_by_id["R1"]=res
    reg.robot_id_to_reservation[robot_id]="R1"
    reg.pod_id_to_reservation[pod_id]="R1"
    reg.job_id_to_reservation[job_id]="R1"
    reg._set_markers(res)
    return res, job, pod

reg = CommittedNextRegistry()
res, job, pod = build_reservation(reg)
queue = [job]                     # committed reservation keeps job in queue
out = reg.cancel_reservation(res, "reservation_cancelled_charging")
check("D_cancel_clears_markers",
      job.committed_next_reservation_id is None and job.committed_next_owner_robot_id is None
      and pod.committed_next_reservation_id is None and pod.committed_next_owner_robot_id is None,
      "cancel clears job+pod committed-next markers")
check("D_exact_once_restore", queue.count(job) == 1 and out.status == STATUS_CANCELLED
      and reg.lifecycle_counters.get("committed_next_jobs_restored") == 1,
      "job present exactly once in queue; restored counted once")
check("D_charging_reason_bucketed",
      reg.lifecycle_counters.get("committed_next_reservations_cancelled_charging") == 1,
      "charging cancellation reason bucketed")
# double cancel must NOT duplicate
reg.cancel_reservation(res, "reservation_cancelled_charging")
check("D_no_duplicate_restore", reg.lifecycle_counters.get("committed_next_jobs_restored") == 1,
      "terminal reservation re-cancel does not duplicate restoration")

# robot death wiring: cancels committed-next + preserves RTS censor reason
class SpyRegistry:
    def __init__(self): self.cancelled=[]
    def cancel_for_robot(self, robot, reason): self.cancelled.append((robot.id, reason))
class SpyRuntime:
    def __init__(self): self.censored=[]
    def censor_pending_for_robot(self, *, robot, status, reason): self.censored.append((status, reason))
u = FakeUniverse({(10,5)}); u.committed_next_registry=SpyRegistry(); u.rts_rollout_runtime=SpyRuntime()
r = make_robot(u, soc_pct=0.0); r.set_move=set_move_ok; r.battery_level_j=0.0
dead = r._charging_pre_move()
check("D_death_cancels_committed_next", dead and u.committed_next_registry.cancelled==[(r.id,"reservation_cancelled_robot_death")],
      "death path cancels committed-next reservation")
check("D_death_censors_rts", u.rts_rollout_runtime.censored==[("censored_robot_death","robot_battery_death")],
      "death path censors RTS transition (no fabricated completion)")

# ═════════════════════════ Part E (replenishment cap) ═════════════════════════
print("=== Part E: replenishment-cap integrity ===")
class FakePM:
    def __init__(self, pods): self.pods={str(p.pod_id):p for p in pods}
    def get_pod_by_id(self, pid): return self.pods.get(str(pid))

inv = Inventory.__new__(Inventory)
inv._tick = 5000
inv.replenishment_pending_stale_ticks = 3000
inv.replenishment_counters = {}
inv.replenishment_hard_cap = 11
podA = FakePod("1"); podB = FakePod("2")
inv.pod_manager = FakePM([podA, podB])
req_valid   = {"pod_id":"1","skus_to_replenish":[10],"created_tick":4900}   # age 100, valid
req_stale   = {"pod_id":"2","skus_to_replenish":[10],"created_tick":100}    # age 4900 > 3000
req_missing = {"pod_id":"999","skus_to_replenish":[10],"created_tick":4900} # pod gone
req_nosku   = {"pod_id":"1","skus_to_replenish":[],"created_tick":4900}     # no sku
inv.pending_replenishment_dispatches = [req_valid, req_stale, req_missing, req_nosku]
removed = inv.prune_stale_pending_replenishment()
check("E_prune_removes_invalid", removed == 3 and inv.pending_replenishment_dispatches == [req_valid],
      "stale-age + missing-pod + no-sku pending pruned; valid kept")
check("E_prune_release_once", inv.replenishment_counters.get("replenishment_pending_pruned") == 3,
      "each stale entry releases cap exactly once")
# new request admitted after release: composition reflects the single remaining pending
inv.job_queue = []
inv._iter_robots = lambda: []          # shadow method: no active robots in this fixture
comp = inv.replenishment_cap_composition()
check("E_composition_sums", comp["pending"] == 1 and comp["total"] == 1 and comp["hard_cap"] == 11
      and comp["active"] == 0 and comp["queued"] == 0,
      "cap composition exposes active/queued/pending; sums to total")

# cap-block accounting (nonsemantic): hard cap blocks each source, counted by source
inv.replenishment_counters = {}
inv.replenishment_commitments_by_pod = lambda **k: {i: "x" for i in range(11)}  # at cap
b1 = inv.can_admit_replenishment("proactive")
b2 = inv.can_admit_replenishment("rts")
b3 = inv.can_admit_replenishment("post_pick")
check("E_cap_blocks_counted",
      (not b1) and (not b2) and (not b3)
      and inv.replenishment_counters.get("replenishment_cap_blocks_proactive") == 1
      and inv.replenishment_counters.get("replenishment_cap_blocks_rts") == 1
      and inv.replenishment_counters.get("replenishment_cap_blocks_post_pick") == 1,
      "hard-cap blocks counted by source (cap value unchanged)")

n = sum(1 for _, ok, _ in results if ok)
print(f"\n=== {n}/{len(results)} Stage-1 (C+D+E) checks passed ===")
sys.exit(0 if n == len(results) else 1)
