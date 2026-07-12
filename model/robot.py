import math
import numpy as np
from dataclasses import replace
from typing import Optional, List, TYPE_CHECKING

from engine.heading import Heading
from engine.netlogo_coordinate import NetLogoCoordinate
from engine.object import Object
from engine.util import *
from .intersection import Intersection
from .robot_job import RobotJob
from .station import Station
from .traffic_policy import TrafficPolicy
from .zone import Zone
from .pod import Pod
from .tools.write_record import write_record_to
from .tools.pod_location import upsert_pod_location
from .tools.pod_travel import upsert_pod_travel
from src.rmfs.runtime_io.logging import debug_print

if TYPE_CHECKING:
    from model.inventory import Inventory
    from model.storage import Storage
    from src.rmfs.decisions.rts.types import RTSDestinationContext

ACTIVATE_NEAREST = True

MOVEMENT_STATE_PRIORITY = {
    'station_processing': 3,
    'delivering_pod': 3,
    'returning_pod': 2,
    'taking_pod': 1,
    # Charging lifecycle states.  A robot leaving a charger should be allowed
    # to clear the charger cell, while waiting/plugged-in robots should not
    # outrank production movement.
    'leaving_charger': 1,
    'going_to_charge': 0,
    'waiting_for_charger': 0,
    'physically_charging': 0,
    # Defensive aliases for charging lifecycle internals that may leak into a
    # neighbor state dictionary in future changes.
    'waiting_fifo': 0,
    'charger_claimed': 0,
    'charging': 0,
    'dead': 0,
    'idle': 0,
}

DEFAULT_MOVEMENT_STATE_PRIORITY = 0


class Robot(Object):
    # netlogo related
    shape = 'turtle-2'
    object_type = 'robot'
    _id = 0

    # movement related
    coordinate = None
    maximum_speed = 1.5
    current_state = 'idle'
    energy_consumption = 0
    travel_distances = 0
    turning = 0
    heading = 0
    suspend_movement = 0
    route_stop_points = []
    total_idle = 0
    # Mutually-exclusive per-tick state accumulators (ticks; the warehouse loop
    # scales to seconds). Every current_state maps to exactly one of these so
    # the fleet-time conservation identity holds. total_available is a secondary
    # non-exclusive flag (work-eligible idle), excluded from the conservation sum.
    total_working = 0
    total_going_to_charge = 0
    total_waiting_for_charger = 0
    total_physically_charging = 0
    total_dead = 0
    total_available = 0
    # routing related
    latest_rotation = ''

    # traffic policy related
    traffic_policy = []
    latest_tick = 0

    lift = False
    # energy consumption related
    mass = 100 # kg
    load_mass = 0
    _gravity = 9.8
    _friction = 0.02
    _impact_resistance = 0.15
    _robot_width = 0.6 # m
    _robot_radius = _robot_width / 2
    _length = 0.75 # m
    _lift_coef = 0.2
    
    e_accel = 0
    e_decel = 0
    e_const = 0
    e_rot = 0
    e_krot = 0
    e_frot = 0
    e_lift = 0
    e_lu = 0

    # ── Battery spec (commercial AGV; see paper Energy Model) — PATCH A1 ──
    BATTERY_CAPACITY_J = 6_480_000.0     # 1.8 kWh
    BASE_DRAIN_RATE_PER_S = 90.0         # electronics drain, J/s
    CHARGE_POWER_W = 397.6               # 28 A x 14.2 V
    INITIAL_BATTERY_FRAC = 1.0           # start-of-shift SoC fraction
    # ── Charging policy (config-driven; netlogo overwrites these at setup) ──
    BATTERY_LOW_PCT = 20.0               # go charge below this %
    BATTERY_CHARGED_PCT = 90.0           # stop charging above this %
    BATTERY_INTERRUPT_PCT = 50.0         # interrupt threshold %
    CORRECTED_ENERGY_MODEL = False
    # Integration-owned (NOT a Salsa threshold): bounded no-progress claim cleanup.
    # A robot holding a charger claim while not physically charging and making no
    # route progress for this many ticks releases the claim and re-enters waiting.
    STALE_CLAIM_MAX_TICKS = 600
    # A charged robot must clear the charger exit within this many simulated
    # seconds.  Normal departures are unchanged; this only turns an otherwise
    # endless blockage into a clear failed run.
    LEAVING_CHARGER_MAX_SECONDS = 600.0
    
    
    

    def __init__(self, universe: "Inventory"):
        self.id = None
        self.traffic_policy = []
        self.job: RobotJob = None
        self.turning_delay = 0
        self.taking_pod_delay = 0
        self.delay_per_task = 10
        self.idle_time = 0
        self.current_intersection_id = None
        self.future_intersection_id = None
        self.previous_intersection_id = None
        self.current_intersection_energy_consumption = 0
        self.current_intersection_stop_and_go = 0
        self.current_intersection_start_time = None
        self.current_intersection_finish_time = None
        self.loaded_travel_distance = 0.0
        self.empty_travel_distance = 0.0
        self.completed_cycle_duration_sum = 0.0
        self.completed_cycle_count = 0
        self.current_cycle_start_tick = None
        self.fixed_load_energy_consumption = 0.0
        self.warehouse = universe
        # ── Battery + charging state — PATCH A2 ──
        self.battery_level_j = self.BATTERY_CAPACITY_J * self.INITIAL_BATTERY_FRAC
        self.is_charging = False
        self._claimed_charger = None
        self.charge_after_current_task = False
        # ── Integration lifecycle state (Task 3 Part C) ──
        self._waiting_for_charger = False       # unresolved: needs charge, none obtainable
        self._claim_created_tick = None         # tick a claim was created (for no-progress check)
        self._claim_progress_pos = None         # last position while holding a claim
        self._leaving_started_at = None
        self._leaving_progress_at = None
        self._leaving_progress_pos = None
        self._drive_by_charged_last = False     # incidental drive-by transferred energy last tick
        self._charging_episode_id = None
        self._charging_queue_entered_at = None
        self._charging_lifecycle_state = None
        self._charging_unavailable_recorded = False
        self.return_fix = False
        self.return_nearest = True 
        # self.zone_boundary =[]
        # self.zone: Optional[Zone] = None
        super().__init__()

    @property
    def battery_pct(self):
        return (self.battery_level_j / self.BATTERY_CAPACITY_J) * 100.0

    @property
    def is_charging_pending(self):
        if not getattr(self.universe, "charging_enabled", False):
            return False
        if getattr(self.universe, "disable_active_charging", False):
            return False
        return bool(
            getattr(self, "charge_after_current_task", False)
            or self._battery_requires_work_safety_charging()
            or getattr(self, "_charging_lifecycle_state", None) in {
                "waiting_fifo", "charger_claimed", "going_to_charge",
                "physically_charging", "leaving_charger",
            }
        )

    def is_unavailable_for_work_due_to_charging(self):
        if not getattr(self.universe, "charging_enabled", False):
            return False
        if getattr(self.universe, "disable_active_charging", False):
            return False
        if self._battery_requires_work_safety_charging():
            return True
        if self.battery_pct >= self.BATTERY_CHARGED_PCT and not getattr(self, "_charging_lifecycle_state", None):
            return False
        # Task 3 Part C5: unavailability requires a *deliberate* charging commitment
        # (an active trip, a live claim, or a pending deferred request). Incidental
        # drive-by energy transfer (is_charging while merely passing over a charger
        # overlay cell, with no claim/trip) no longer removes the robot from work.
        return bool(
            self.current_state == "going_to_charge"
            or self.current_state in {"waiting_for_charger", "physically_charging", "leaving_charger"}
            or getattr(self, "is_charging_pending", False)
            or getattr(self, "_claimed_charger", None) is not None
        )

    def _battery_work_safety_pct(self):
        """Battery percentage below which a robot must not accept new work."""
        return max(float(self.BATTERY_LOW_PCT), float(self.BATTERY_INTERRUPT_PCT))

    def _battery_requires_work_safety_charging(self):
        """True when active charging should take priority over new work."""
        if not getattr(self.universe, "charging_enabled", False):
            return False
        if getattr(self.universe, "disable_active_charging", False):
            return False
        if self.current_state in {
            "waiting_for_charger", "going_to_charge", "physically_charging",
            "leaving_charger", "dead",
        }:
            return False
        return self.battery_pct <= self._battery_work_safety_pct()

    def _incr_charging_counter(self, name, n=1):
        """Nonsemantic integration telemetry (Task 3 Part C/G). Aggregated by KPI wiring."""
        wh = getattr(self, "universe", None)
        if wh is None:
            return
        ctr = getattr(wh, "charging_counters", None)
        if ctr is None:
            ctr = wh.charging_counters = {}
        ctr[name] = ctr.get(name, 0) + n

    def _enter_waiting_for_charger(self, reason):
        """Task 3 Part C2/C6: explicit unresolved state; visible; no false claim."""
        if str(reason) != "waiting_fifo" and not self._charging_unavailable_recorded:
            self._incr_charging_counter("charger_unavailable_events")
            self._charging_unavailable_recorded = True
        self._waiting_for_charger = True
        self._waiting_reason = str(reason)
        self.current_state = "waiting_for_charger"

    def _apply_drive_by_charging(self):
        charger_cells = getattr(self.universe, 'charger_cells', set())
        grid_pos = (round(self.pos_x), round(self.pos_y))
        if grid_pos in charger_cells:
            charge_j = self.CHARGE_POWER_W * self.universe.tick_to_second
            self.battery_level_j = min(self.BATTERY_CAPACITY_J,
                                       self.battery_level_j + charge_j)
            self._incr_charging_counter("charging_energy_added_j", charge_j)
            self.is_charging = True
            # Task 3 Part C5: count an incidental drive-by event only on entry
            # (robot was not charging last tick and is not on a deliberate trip).
            if not self._drive_by_charged_last and self._claimed_charger is None \
                    and self.current_state != "going_to_charge":
                self._incr_charging_counter("drive_by_charging_events")
            self._drive_by_charged_last = True
        else:
            self.is_charging = False
            self._drive_by_charged_last = False

    def _cancel_committed_next_for_charging(self):
        registry = getattr(self.warehouse, "committed_next_registry", None)
        reservation = registry.get_for_robot(self) if registry is not None else None
        if reservation is None:
            return
        registry.cancel_for_robot(self, "reservation_cancelled_charging")
        self._incr_charging_counter("committed_next_cancelled_for_charging")
        runtime = getattr(self.warehouse, "rts_rollout_runtime", None)
        if runtime is not None:
            runtime.censor_pending_for_robot(
                robot=self,
                status="censored_next_task_charging",
                reason="reservation_cancelled_charging",
            )

    def _queue_for_charging(self, *, preserve_timestamp=False):
        dispatcher = getattr(self.universe, "charging_dispatcher", None)
        if dispatcher is None:
            raise RuntimeError("active charging requires a warehouse charging dispatcher")
        if self._charging_episode_id is None:
            # This is the sole normal operational seam that may cancel a
            # committed-next reservation: the previous physical job is already
            # complete and the robot is about to enter the charging FIFO.
            self._cancel_committed_next_for_charging()
        dispatcher.enqueue(self, preserve_timestamp=preserve_timestamp)

    def _start_charging_trip(self):
        """Compatibility entry point: create one FIFO episode, never poll."""
        self._queue_for_charging()
        return self.current_state == "going_to_charge"

    def _charger_physically_available(self, cell):
        for obj in self.universe.get_movable_objects():
            if obj is self or getattr(obj, "object_type", None) != "robot":
                continue
            if (round(getattr(obj, "pos_x", -1)), round(getattr(obj, "pos_y", -1))) == cell:
                return False
        station = getattr(self.universe, "charger_station_by_cell", {}).get(cell)
        if station is not None and getattr(station, "robot_ids", {}):
            return False
        return True

    def _assign_charger_from_fifo(self):
        """Try Salsa's nearest-free ordering once for the current event."""
        wh = self.universe
        active = getattr(wh, 'active_charger_cells', None) or set()
        charger_cells = active if active else getattr(wh, 'charger_cells', set())
        if not charger_cells:
            self._enter_waiting_for_charger("no_chargers_configured")
            return False
        occupied = getattr(wh, 'occupied_chargers', {})
        available = [
            c for c in charger_cells
            if c not in occupied and self._charger_physically_available(c)
        ]
        # Diagnostic only: count candidates excluded because a charger cell is
        # already claimed or physically occupied. Does not change selection.
        occupied_count = len(charger_cells) - len(available)
        if occupied_count > 0:
            self._incr_charging_counter("charger_candidate_occupied", occupied_count)
        if not available:
            self._incr_charging_counter("charger_assignment_no_feasible_candidate")
            self._enter_waiting_for_charger("no_available_charger")
            return False
        available.sort(key=lambda c: (
            (c[0] - self.pos_x) ** 2 + (c[1] - self.pos_y) ** 2,
            int(c[0]), int(c[1]),
        ))
        for nearest in available:
            dest = NetLogoCoordinate(nearest[0], nearest[1])
            graph_name = getattr(wh, "charger_route_graph_by_cell", {}).get(nearest, "standard")
            graph = wh.graph_pod if graph_name == "pod" else wh.graph
            try:
                estimated_energy = self._estimate_charger_route_energy_j(dest, graph)
                if estimated_energy is None:
                    raise ValueError(f"No route available to charger {nearest}")
                if float(self.battery_level_j) < estimated_energy:
                    self._incr_charging_counter("charger_energy_infeasible_candidates")
                    self._incr_charging_counter("charger_candidate_insufficient_energy")
                    continue
                self.set_move(dest, graph=graph)
                if not self.route_stop_points and not self.close_enough(dest):
                    # A turn-free graph route needs an explicit final stop; the
                    # legacy path compactor otherwise emits an empty list.
                    self.route_stop_points = [dest]
            except Exception:
                self._incr_charging_counter("charger_route_failures")
                self._incr_charging_counter("charger_candidate_route_rejections")
                self._incr_charging_counter("charger_candidate_unroutable")
                continue
            self.current_state = "going_to_charge"
            self._charging_lifecycle_state = "going_to_charge"
            occupied[nearest] = self.id                     # claim only after a valid route
            self._claimed_charger = nearest
            self._claim_created_tick = int(getattr(wh, "_tick", 0))
            self._claim_progress_pos = (self.pos_x, self.pos_y)
            self._waiting_for_charger = False
            self.charge_after_current_task = False
            self._incr_charging_counter("charger_claims_created")
            self._incr_charging_counter("charger_assignment_successes")
            return True
        # every candidate route failed (unroutable or energy-infeasible)
        self._incr_charging_counter("charger_assignment_no_feasible_candidate")
        self._enter_waiting_for_charger("all_routes_failed")
        return False

    def _estimate_charger_route_energy_j(self, dest, graph):
        """Conservatively estimate route energy from the existing model."""
        start = self.coordinate_to_string_key(round(self.pos_x), round(self.pos_y))
        end = self.coordinate_to_string_key(round(dest.x), round(dest.y))
        route = graph.dijkstra(start, end)
        if route is None:
            return None
        hops = max(0, len(route) - 1)
        if hops == 0:
            return 0.0
        seconds = hops / max(float(self.maximum_speed), 1e-9)
        steady_motion = self.mass * self._gravity * self._friction * hops
        fixed_load = self.BASE_DRAIN_RATE_PER_S * seconds
        # Full acceleration and deceleration at every hop is intentionally
        # conservative while remaining derived from the active energy model.
        accel_per_hop = self.calculateEnergy(self.maximum_speed, self.maximum_speed)
        decel_per_hop = self.calculateEnergy(self.maximum_speed, -self.maximum_speed)
        return float(steady_motion + fixed_load + hops * (accel_per_hop + decel_per_hop))

    def _should_interrupt_charging(self):
        if self.battery_pct < self.BATTERY_INTERRUPT_PCT:
            return False
        if not len(getattr(self.universe, 'job_queue', [])):
            return False
        for o in self.universe.get_movable_objects():
            if (getattr(o, 'object_type', None) == 'robot' and o is not self
                    and o.current_state == 'idle'
                    and getattr(o, 'battery_level_j', 0) > 0):
                return False
        return True

    def _release_charger(self, *, notify=True):
        cell = getattr(self, '_claimed_charger', None)
        if cell is None:
            return
        occupied = getattr(self.universe, 'occupied_chargers', {})
        if occupied.get(cell) == self.id:
            del occupied[cell]
        self._claimed_charger = None
        self._claim_created_tick = None
        self._claim_progress_pos = None
        self._incr_charging_counter("charger_claims_released")
        if notify:
            dispatcher = getattr(self.universe, "charging_dispatcher", None)
            if dispatcher is not None:
                dispatcher.notify_availability_changed("charger_claim_released")

    def _release_stale_charger_claim_and_route(self):
        """Release ownership before clearing every charging-specific movement cue."""
        self._release_charger(notify=False)
        self.route_stop_points = []
        self.destination = None
        self.movement_plan = None
        self.charge_after_current_task = False

    def _complete_charging_departure(self):
        # Keep the claim until the robot has physically cleared the charger
        # exit.  Releasing it earlier can send another robot into the same
        # one-lane entrance while this robot is still leaving.
        self._release_charger(notify=False)
        self.current_state = "idle"
        self._charging_lifecycle_state = None
        self._charging_episode_id = None
        self._charging_queue_entered_at = None
        self._waiting_for_charger = False
        self._charging_unavailable_recorded = False
        self._leaving_started_at = None
        self._leaving_progress_at = None
        self._leaving_progress_pos = None
        self.is_charging = False
        dispatcher = getattr(self.universe, "charging_dispatcher", None)
        if dispatcher is not None:
            dispatcher.notify_availability_changed("charging_departure_completed")

    def _finish_physical_charging(self, *, interrupted):
        if interrupted:
            self._incr_charging_counter("charging_sessions_interrupted")
            self._charging_lifecycle_state = "interrupted"
        else:
            self._incr_charging_counter("charging_sessions_completed")
            self._charging_lifecycle_state = "completed"
        cell = self._claimed_charger
        self.is_charging = False
        exit_cell = getattr(self.universe, "charger_exit_cell_by_cell", {}).get(cell)
        if exit_cell is None or exit_cell == (round(self.pos_x), round(self.pos_y)):
            self._complete_charging_departure()
            return
        graph_name = getattr(self.universe, "charger_route_graph_by_cell", {}).get(cell, "standard")
        graph = self.universe.graph_pod if graph_name == "pod" else self.universe.graph
        self.set_move(NetLogoCoordinate(*exit_cell), graph=graph)
        if not self.route_stop_points and (round(self.pos_x), round(self.pos_y)) != exit_cell:
            self.route_stop_points = [NetLogoCoordinate(*exit_cell)]
        self.current_state = "leaving_charger"
        self._charging_lifecycle_state = "leaving_charger"
        now = float(getattr(self.universe, "_tick", 0.0))
        self._leaving_started_at = now
        self._leaving_progress_at = now
        self._leaving_progress_pos = (self.pos_x, self.pos_y)

    def _retry_charging_departure_after_no_progress(self, *, now):
        """Keep the run alive when a charger exit is blocked for a long time."""
        self._incr_charging_counter("charging_departure_no_progress_retries")
        cell = getattr(self, "_claimed_charger", None)
        exit_cell = getattr(self.universe, "charger_exit_cell_by_cell", {}).get(cell)
        here = (round(self.pos_x), round(self.pos_y))
        if exit_cell is None or here == exit_cell:
            self._complete_charging_departure()
            return

        graph_name = getattr(self.universe, "charger_route_graph_by_cell", {}).get(cell, "standard")
        graph = self.universe.graph_pod if graph_name == "pod" else self.universe.graph
        try:
            self.set_move(NetLogoCoordinate(*exit_cell), graph=graph)
        except Exception:
            self.route_stop_points = [NetLogoCoordinate(*exit_cell)]
        if not self.route_stop_points:
            self.route_stop_points = [NetLogoCoordinate(*exit_cell)]
        self.current_state = "leaving_charger"
        self._charging_lifecycle_state = "leaving_charger"
        self._leaving_progress_at = now
        self._leaving_progress_pos = (self.pos_x, self.pos_y)

    @staticmethod
    def _checkMovementDirection(p1, p2):
        if p1.y == p2.y:
            # horizontal
            if p1.x < p2.x:
                return 90
            if p1.x > p2.x:
                return 270

        if p1.x == p2.x:
            # vertical
            if p1.y < p2.y:
                return 0
            if p1.y > p2.y:
                return 180

        return None

    def calculateEnergy(self, velocity, acceleration):
        tick_unit = self.universe.tick_to_second
       
        if acceleration > 0 and velocity != 0:
            # average_speed = 2 * velocity + (acceleration * tick_unit)
            e_accel = (self.mass + self.load_mass) * (acceleration * self._impact_resistance + self._gravity * self._friction) * velocity * tick_unit
            return e_accel
        elif acceleration < 0 and velocity != 0:
            # average_speed = 2 * velocity + (acceleration * tick_unit)
            e_decel = abs((self.mass + self.load_mass) * (acceleration * self._impact_resistance - self._gravity * self._friction) * velocity * tick_unit)
            return e_decel
        elif velocity != 0:
            e_const = (self.mass + self.load_mass) * self._gravity * self._friction * velocity * tick_unit
            return e_const
       
        return 0

    def setPath(self, path):
        current_heading = self.heading
        route_stop_points = []

        # convert path list to route_stop_points (list of NetLogoCoordinate where the robot should stop and Heading to turn the robot)
        for i in range(1, len(path), 1):
            p1 = NetLogoCoordinate(path[i - 1][0], path[i - 1][1])
            p2 = NetLogoCoordinate(path[i][0], path[i][1])
            heading = self.getHeading(p1, p2)
            if current_heading != heading:
                current_heading = heading
                route_stop_points.append(NetLogoCoordinate(path[i - 1][0], path[i - 1][1]))
                route_stop_points.append(Heading(heading))
        if len(route_stop_points) > 0 and isinstance(route_stop_points[len(route_stop_points) - 1],
                                                     NetLogoCoordinate) == False:
            now = path[len(path) - 1]
            route_stop_points.append(NetLogoCoordinate(now[0], now[1]))
        self.route_stop_points = route_stop_points

    def changeColorByState(self):
        if self.current_state == "taking_pod": # occupied 
            self.color = 57  # green
        elif self.current_state == "delivering_pod":
            station: Station = self.universe.station_manager.get_station_by_id(self.job.station_id)
            if station.is_replenishment_station():
                self.color = 138
            else:
                self.color = 15  # red # have pod and go to picking station 
        elif self.current_state == "returning_pod":
            self.color = 46  # yellow
        elif self.current_state == "station_processing": #inclue queue
            self.color = 94  # blue
        elif self.current_state == "idle":
            self.color = 0  # black

    _WORKING_STATES = ("taking_pod", "delivering_pod", "station_processing", "returning_pod")

    def _accumulate_state_seconds(self):
        """Classify this tick into exactly one mutually-exclusive state bucket.

        A robot waiting in the charging FIFO, travelling to a charger, or
        physically charging is NOT counted as ordinary idle. ``total_idle`` keeps
        its historical meaning (genuine idle ticks) so ``total_robot_idle`` is
        unchanged for runs without charging.
        """
        state = self.current_state
        if state == "idle":
            self.total_idle += 1
            # 'available' = idle and genuinely work-eligible (not charging-committed).
            if not self.is_unavailable_for_work_due_to_charging():
                self.total_available += 1
        elif state in self._WORKING_STATES:
            self.total_working += 1
        elif state in ("going_to_charge", "leaving_charger"):
            self.total_going_to_charge += 1
        elif state == "waiting_for_charger":
            self.total_waiting_for_charger += 1
        elif state == "physically_charging":
            self.total_physically_charging += 1
        elif state == "dead":
            self.total_dead += 1
        else:
            # Unknown/unmapped state: fall back to idle so the conservation
            # identity (sum of buckets == fleet-seconds) still holds exactly.
            self.total_idle += 1

    def advance_state(self):
        # print(f"current_state: {self.current_state}")
        if self.current_state == "taking_pod":
            self.taking_pod_delay += self.delay_per_task
            if self.job is not None:
                pod: Pod = self.job.pod
                self.load_mass = pod.mass
            e_li = self.load_mass * self._gravity * self._lift_coef
            self.energy_consumption += e_li
            self.current_state = "delivering_pod"
            upsert_pod_travel(
                self.job.my_id,
                self.robotID(self.robotName()),
                str(self.job.pod.pod_id),
                "taking_pod",
                finish_time=self.universe._tick
            )
        elif self.current_state == "delivering_pod":
            self.current_state = "station_processing"
            station: Station = self.universe.station_manager.get_station_by_id(self.job.station_id)
            self.warehouse.rts_rollout_runtime.on_station_arrival(robot=self, station=station)
            upsert_pod_travel(
                self.job.my_id,
                self.robotID(self.robotName()),
                str(self.job.pod.pod_id),
                "delivering_pod",
                finish_time=self.universe._tick
            )
        elif self.current_state == "station_processing":
            self.current_state = "returning_pod"
        elif self.current_state == "returning_pod":
            # print(f"[DEBUG] finish returning pod {self.job.pod} to {self.destination}")
            # print(f"[DEBUG] current pod position {self.job.pod.coordinate}")
            # print(f"[DEBUG] pos_x pos_y {self.job.pod.pos_x} {self.job.pod.pos_y}")
            # print(f"{self.warehouse.pod_manager.get_pod_by_id(self.job.pod.pod_id).coordinate}")
            # input()
            # raise AssertionError
            completed_job = self.job
            upsert_pod_location(completed_job.pod.pod_id, completed_job.pod.pos_x, completed_job.pod.pos_y)
            self.warehouse.rts_rollout_runtime.on_return_completed(robot=self)
            if getattr(completed_job, "rts_branch", None) == "store":
                self.warehouse.completed_post_pick_store_actions += 1
            self._clear_rts_return_ownership()
            # Finalize the just-returned pod (mark available, clear pod.station,
            # drop stale station incoming membership) BEFORE any committed-next
            # activation replaces robot.job — otherwise the completed pod's
            # cleanup would be bypassed and it would be left unavailable/unowned.
            self.warehouse.finalize_completed_return(completed_job)
            # A proactive replenishment pod is physically back at its retained
            # origin storage: keep the normal pod-to-storage assignment intact and
            # clear only the temporary proactive-origin reservation metadata.
            if getattr(completed_job, "proactive_origin_reserved", False):
                completed_job.clear_proactive_origin_reservation()
            if self.current_cycle_start_tick is not None:
                self.completed_cycle_duration_sum += max(0.0, self.universe._tick - self.current_cycle_start_tick)
                self.completed_cycle_count += 1
                self.current_cycle_start_tick = None
            e_li = self.load_mass * self._gravity * self._lift_coef
            self.energy_consumption += e_li
            self.load_mass = 0
            self.taking_pod_delay += self.delay_per_task
            upsert_pod_travel(
                completed_job.my_id,
                self.robotID(self.robotName()),
                str(completed_job.pod.pod_id),
                "returning_pod",
                finish_time=self.universe._tick
            )
            charging_queued = (getattr(self, "charge_after_current_task", False) or self.battery_pct <= self.BATTERY_LOW_PCT) and not getattr(self.universe, "disable_active_charging", False)
            if getattr(self.universe, "charging_enabled", False) and charging_queued:
                # The return is already fully finalized above.  Do not carry its
                # finished job record into the charging lifecycle: the pod is
                # available again and may legitimately receive a new job.
                self.job = None
                self.current_state = "idle"
                self._queue_for_charging()
            else:
                if self.warehouse.activate_committed_next_after_return(self):
                    return
                self.current_state = "idle"
        # print(f"become {self.current_state}")

    def decideCollision(self, collision_block, o, collide_distance):
        will_collide = False
        selected_label = ""
        object_heading = 0
        distance = math.sqrt((o['x'] - self.pos_x) ** 2 + (o['y'] - self.pos_y) ** 2)
        if collision_block is not None:
            self_distance = self._calculateTwoPoint(NetLogoCoordinate(self.pos_x, self.pos_y),
                                                    NetLogoCoordinate(collision_block[0], collision_block[1]))
            if (self.velocity ** 2) / 2 >= self_distance or distance < collide_distance:
                will_collide = True
                selected_label = o['label']
                object_heading = o['heading']
        elif distance < collide_distance:
            will_collide = True
            selected_label = o['label']
            object_heading = o['heading']
        return will_collide, selected_label, object_heading

    def get_nearest_robot_conflict_candidate(self, next_step_coords, search_area):
        neighbor_candidates = []

        neighbors = self.universe.landscape.getNeighborObject(round(self.pos_x), round(self.pos_y), search_area)
        if neighbors:
            for neighbor in neighbors:
                neighbor_robot = self.get_robot_by_name(neighbor['label'])
                if neighbor_robot == self:
                    continue

                # if neighbor['velocity'] == 0:
                #     continue

                neighbor_next_step_coords = self._calculate_next_blocks(round(neighbor['x']), round(neighbor['y']),
                                                                        neighbor['heading'], search_area,
                                                                        include_self=True)
                meeting_coordinate = self._getIntersectionBlock(next_step_coords, neighbor_next_step_coords)
                if meeting_coordinate and self.is_collision_candidate(neighbor):
                    neighbor_candidates.append((neighbor, meeting_coordinate))

        return neighbor_candidates

    def get_priority_diff(self, object):
        is_replenish = False
        if self.job is not None:
            station: Station = self.universe.station_manager.get_station_by_id(self.job.station_id)
            is_replenish = station.is_replenishment_station()

        self_priority = MOVEMENT_STATE_PRIORITY.get(
            self.current_state,
            DEFAULT_MOVEMENT_STATE_PRIORITY,
        )
        if self.job is not None and is_replenish:
            self_priority = MOVEMENT_STATE_PRIORITY['returning_pod']

        other_priority = MOVEMENT_STATE_PRIORITY.get(
            object.get('state'),
            DEFAULT_MOVEMENT_STATE_PRIORITY,
        )
        
        return self_priority - other_priority 

    def is_collision_candidate(self, obj):
        # Check for collision candidate based on relative positions and headings
        relative_x = obj['x'] - self.pos_x
        relative_y = obj['y'] - self.pos_y

        if self.heading == 0:
            return relative_y >= 0
        if self.heading == 180:
            return relative_y <= 0
        if self.heading == 90:
            return relative_x >= 0
        if self.heading == 270:
            return relative_x <= 0

    def get_robot_by_name(self, robot_name):
        for o in self.universe._objects:
            if o.object_type == "robot" and o.robotName() == robot_name:
                return o

    def get_robot_by_coord(self, x, y):
        for o in self.universe._objects:
            if o.object_type == "robot" and o.pos_x == x and o.pos_y == y:
                return o

    def appliedTrafficPolicyKeys(self):
        result = []
        for tp in self.traffic_policy:
            result.append(tp.prioritized_robot)
        return result

    def hasTrafficPolicyFor(self, robot_name):
        for robot in self.appliedTrafficPolicyKeys():
            if robot == robot_name:
                return True
        return False

    def addTrafficPolicy(self, traffic_policy: TrafficPolicy):
        if not self.hasTrafficPolicyFor(traffic_policy.prioritized_robot):
            self.traffic_policy.append(traffic_policy)

    def deadlockPrevention(self):
        self_next_blocks = self._calculate_next_blocks(round(self.pos_x), round(self.pos_y), self.heading, 5)
        for p in self_next_blocks:
            if self.universe.deadlock_prevention_manager.coordinateBooked(NetLogoCoordinate(p[0], p[1])):
                self.acceleration = -1
                return True
        return False

    def should_remove_policy(self, prioritized_robot, self_blocks, other_blocks):
        if self.heading == prioritized_robot.heading:
            return self.velocity < prioritized_robot.velocity
        else:
            return self._getIntersectionBlock(self_blocks, other_blocks) is None

    @staticmethod
    def _transformRouteToList(path):
        path_int = []
        for p in path:
            l = p.split(',')
            path_int.append([int(l[0]), int(l[1])])
        return path_int

    @staticmethod
    def transform_coords_to_list(coords: List[NetLogoCoordinate]):
        path_int = []
        for coord in coords:
            path_int.append([coord.x, coord.y])

        return path_int

    def neutralizeRobotState(self):
        self.route_stop_points = []

    def update_current_position(self):
        self.coordinate = NetLogoCoordinate(round(self.pos_x), round(self.pos_y))
        self.pos_x = round(self.pos_x)
        self.pos_y = round(self.pos_y)

    def picking_item_in_pod(self):
        if self.job is not None and self.is_being_process_on_station():
            self.job.decrement_delay()
            return True

    def is_in_station_path(self):
        if self.job is not None:
            station: Station = self.universe.station_manager.get_station_by_id(self.job.station_id)
            for coord in station.get_path():
                if round(self.pos_x) == coord.x and round(self.pos_y) == coord.y:
                    return True

    def is_being_process_on_station(self):
        station: Station = self.universe.station_manager.get_station_by_id(self.job.station_id)
        return self.job.is_being_processed() and self.close_enough(
            station.coordinate, 0.1)

    def movementPlan(self):
        if self.current_state == "going_to_charge" and not self.route_stop_points:
            claimed = getattr(self, "_claimed_charger", None)
            here = (round(self.pos_x), round(self.pos_y))
            if claimed is None or here != claimed:
                raise RuntimeError(f"charging arrival mismatch: robot={self.id} here={here} claim={claimed}")
            self.current_state = "physically_charging"
            self._charging_lifecycle_state = "physically_charging"
            self.is_charging = True
            self._incr_charging_counter("charging_sessions_started")
            self._incr_charging_counter("charger_arrivals")
            return
        if self.current_state == "physically_charging":
            if self.battery_pct >= self.BATTERY_CHARGED_PCT:
                self._finish_physical_charging(interrupted=False)
            elif self._should_interrupt_charging():
                self._finish_physical_charging(interrupted=True)
            return
        if self.current_state == "leaving_charger" and not self.route_stop_points:
            self._complete_charging_departure()
            return
        if self.picking_item_in_pod():
            # print(f"robot {self.id} picking_item_in_pod")
            # print(f"robot job {self.job} and is_being_process {self.is_being_process_on_station()}")
            # print(f"delay picking {self.job.picking_delay} delay replenishment {self.job.replenishment_delay}")
            return
        # print(f"robot {self.id} has route_stop_points {self.route_stop_points}")
        if not self.route_stop_points:
            # print("called from movementPlan")
            self.advance_state_if_needed()
            return

        if self.eligible_to_reroute():
            if self.current_state == "taking_pod":
                self.set_move(self.route_stop_points[-1], self.universe.graph, avoid_side=True)
            elif self.current_state == "delivering_pod" or self.current_state == "returning_pod":
                self.set_move(self.route_stop_points[-1], self.universe.graph_pod, avoid_side=True)
            elif self.current_state == "station_processing":
                station: Station = self.universe.station_manager.get_station_by_id(self.job.station_id)
                station.update_robot_route_type(self.robotName())
                path = station.get_sub_path(self.robotName(), round(self.pos_x), round(self.pos_y))
                self.setPath(self.transform_coords_to_list(path))

            self.idle_time = 0

        if not self.route_stop_points:
            return

        next_destination_coordinate = self.route_stop_points[0]

        if isinstance(next_destination_coordinate, Heading):
            self.handle_directional(next_destination_coordinate)
            return

        if self.not_able_to_move(next_destination_coordinate):
            self.update_idle_state()
            return

        candidate_conflict_coordinate = self.handle_conflicts(next_destination_coordinate)

        self.execute_move(candidate_conflict_coordinate, next_destination_coordinate)

    def update_idle_state(self):
        self.idle_time += 1
        self.velocity = 0
        self.acceleration = 0
        self.universe.landscape.setObject(self.robotName(), self.pos_x, self.pos_y, self.velocity,
                                          self.acceleration, self.heading, self.current_state, self.load_mass)

        if self.current_intersection_id:
            self.current_intersection_stop_and_go += 1

    def handle_conflicts(self, next_destination_coordinate):
        candidate_conflict_coordinate = None
        if isinstance(next_destination_coordinate, NetLogoCoordinate) and not self.is_in_station_path():
            self_coord = NetLogoCoordinate(self.pos_x, self.pos_y)
            search_area = self.calculate_search_area(next_destination_coordinate)

            next_step_coordinates = self._calculate_next_blocks(round(self.pos_x), round(self.pos_y),
                                                                self.heading, search_area)
            nearest_conflict_candidates = self.get_nearest_robot_conflict_candidate(next_step_coordinates, search_area)
            if nearest_conflict_candidates is not None:
                for candidate, meeting_coordinate in nearest_conflict_candidates:
                    if (candidate['state'] == "station_processing" or candidate['state'] == 'idle'
                            or candidate['velocity'] == 0):
                        continue

                    neighbor_coord = NetLogoCoordinate(candidate['x'], candidate['y'])
                    meeting_coordinate = NetLogoCoordinate(meeting_coordinate[0], meeting_coordinate[1])
                    self_distance_to_meeting_block = self._calculateTwoPoint(self_coord, meeting_coordinate)
                    neighbor_distance_to_meeting_block = self._calculateTwoPoint(neighbor_coord, meeting_coordinate)

                    priority_diff = self.get_priority_diff(candidate)
                    if candidate['heading'] == self.heading:
                        for x, y in next_step_coordinates:
                            if round(candidate['x']) == x and round(candidate['y']) == y:
                                candidate_conflict_coordinate = self.calculate_next_movement_from_conflict(
                                    meeting_coordinate, next_destination_coordinate)
                                break

                    elif priority_diff < 0:
                        candidate_conflict_coordinate = self.calculate_next_movement_from_conflict(meeting_coordinate,
                                                                                                   next_destination_coordinate)

                    elif priority_diff == 0:
                        if self_distance_to_meeting_block > neighbor_distance_to_meeting_block:
                            candidate_conflict_coordinate = self.calculate_next_movement_from_conflict(
                                meeting_coordinate, next_destination_coordinate)

        return candidate_conflict_coordinate

    def execute_move(self, candidate_conflict_coordinate, next_destination_coordinate):
        self.idle_time = 0
        if candidate_conflict_coordinate and candidate_conflict_coordinate != next_destination_coordinate:
            self.handle_next_movement(candidate_conflict_coordinate, is_next_route_stop=False)
        else:
            self.handle_next_movement(next_destination_coordinate, is_next_route_stop=True)

        self.drawNextPosition()
        
    def check_in_zone(self, x, y):
        for index, (bottom_left, top_right) in enumerate(self.zone_boundary):
            if bottom_left[0] <= x <= top_right[0] and bottom_left[1] <= y <= top_right[1]:
                return index + 1  # Return zone index starting from 1
        return -1
    
    def check_on_boundary(self, x, y):
        for index, (bottom_left, top_right) in enumerate(self.zone_boundary):
            # Extract coordinates
            x1, y1 = bottom_left
            x2, y2 = top_right
            
            # Check if the point (x, y) is on the boundary
            if ((x == x1 or x == x2) and y1 <= y <= y2) or ((y == y1 or y == y2) and x1 <= x <= x2):
                return index  # Return zone index starting from 1
        return -1

    def eligible_to_reroute(self):
         # if got into a zone, and the zone is full now
        # print(f"Masuk kesini per modulo 30 detik {self.universe._tick}")
        if self.idle_time <= 50 or self.current_state == "delivering_pod":
            return False

        if self.is_in_station_path():
            station: Station = self.universe.station_manager.get_station_by_id(self.job.station_id)

            if self.current_state == "station_processing" and station.has_route_changed(self.robotName()):
                return True
            else:
                return False
        
        # Calculate next step coordinates
        next_step_coordinates = self._calculate_next_blocks(
            round(self.pos_x), round(self.pos_y), self.heading, 1, include_self=False)
        robot_front = self.universe.landscape.get_neighbor_object(*next_step_coordinates[0])

        # Check if there is no robot in front
        if not robot_front:
            return False

        # Check if the robot in front is idle
        if robot_front['state'] == "idle":
            return True

        # Check if the robot in front is heading in the same direction
        if robot_front['heading'] == self.heading:
            return False

        # Compare priorities
        priority_diff = self.get_priority_diff(robot_front)
        if priority_diff > 0:
            return False
        elif priority_diff < 0:
            return True

        # Resolve ties by ID
        return self.robotID(self.robotName()) < self.robotID(robot_front['label'])

    def calculate_next_movement_from_conflict(self, conflict_coordinate: NetLogoCoordinate,
                                              next_destination_coordinate: NetLogoCoordinate):
        potential_next = None
        if self.heading == 0:
            potential_next = NetLogoCoordinate(conflict_coordinate.x, conflict_coordinate.y - 1)
        elif self.heading == 180:
            potential_next = NetLogoCoordinate(conflict_coordinate.x, conflict_coordinate.y + 1)
        elif self.heading == 90:
            potential_next = NetLogoCoordinate(conflict_coordinate.x - 1, conflict_coordinate.y)
        elif self.heading == 270:
            potential_next = NetLogoCoordinate(conflict_coordinate.x + 1, conflict_coordinate.y)

        if self.heading in [0, 180]:
            if abs(potential_next.y - next_destination_coordinate.y) < abs(
                    conflict_coordinate.y - next_destination_coordinate.y):
                return next_destination_coordinate
        else:
            if abs(potential_next.x - next_destination_coordinate.x) < abs(
                    conflict_coordinate.x - next_destination_coordinate.x):
                return next_destination_coordinate

        return potential_next

    def calculate_search_area(self, next_destination_coordinate: NetLogoCoordinate):
        if self.is_in_station_path():
            return 1

        return 3

    def not_able_to_move(self, next_destination_coordinate: NetLogoCoordinate):
        if self.turning_delay > 0:
            self.turning_delay -= 1
            return True

        if self.taking_pod_delay > 0:
            self.taking_pod_delay -= 1
            return True

        if next_destination_coordinate.x == round(self.pos_x) and next_destination_coordinate.y == round(self.pos_y):
            return False

        return self.path_blocked()

    def path_blocked(self):
        next_step_coordinates = self._calculate_next_blocks(round(self.pos_x), round(self.pos_y),
                                                            self.heading, 1, include_self=False)

        if not self.is_aligned_with_heading(next_step_coordinates):
            return False

        return (self.path_blocked_by_intersection(next_step_coordinates)
                or self.path_blocked_by_robot(next_step_coordinates))

    def path_blocked_by_intersection(self, next_step_coordinates):
        for next_x, next_y in next_step_coordinates:
            intersection = self.universe.intersection_manager.get_intersection_by_coordinate(next_x, next_y)
            if (intersection and self.close_enough(intersection.intersection_coordinate, 1)
                    and not intersection.is_allowed_to_move(self.heading)):
                return True
        return False

    def path_blocked_by_robot(self, next_step_coordinates):
        neighbors = self.universe.landscape.getNeighborObject(round(self.pos_x), round(self.pos_y), 2)
        for neighbor in neighbors:
            if self.get_robot_by_name(neighbor['label']) == self:
                continue

            near_robot_coord = NetLogoCoordinate(neighbor['x'], neighbor['y'])

            for next_x, next_y in next_step_coordinates:
                x_difference = abs(neighbor['x'] - next_x)
                y_difference = abs(neighbor['y'] - next_y)
                if x_difference < 1 and y_difference < 1 and self.close_enough(near_robot_coord, 1):
                    self_distance_to_conflict = abs(self.pos_x - next_x) + abs(self.pos_y - next_y)
                    neighbor_robot_distance_to_conflict = abs(neighbor['x'] - next_x) + abs(neighbor['y'] - next_y)

                    if neighbor_robot_distance_to_conflict < self_distance_to_conflict:
                        return True
                    else:
                        continue

        return False

    def is_in_path(self, destination, steps):
        # Assuming steps is a list of tuples (x, y) coordinates
        current_x, current_y = round(self.pos_x), round(self.pos_y)
        dest_x, dest_y = destination.x, destination.y

        # Check if destination is on the path between current and next step
        for step_x, step_y in steps:
            if self.heading == 0:  # Moving up along the y-axis
                if current_y <= dest_y and current_x == dest_x:
                    return current_x == step_x and current_y <= step_y <= dest_y
            elif self.heading == 180:  # Moving down along the y-axis
                if current_y >= dest_y and current_x == dest_x:
                    return current_x == step_x and current_y >= step_y >= dest_y
            elif self.heading == 90:  # Moving right along the x-axis
                if current_x <= dest_x and current_y == dest_y:
                    return current_y == step_y and current_x <= step_x <= dest_x
            elif self.heading == 270:  # Moving left along the x-axis
                if current_x >= dest_x and current_y == dest_y:
                    return current_y == step_y and current_x >= step_x >= dest_x
        return False

    def is_aligned_with_heading(self, steps):
        for step_x, step_y in steps:
            if self.heading in (0, 180):  # Vertical movement
                return round(self.pos_x) == step_x
            elif self.heading in (90, 270):  # Horizontal movement
                return round(self.pos_y) == step_y

    def get_robots_by_coords(self, coords):
        robots = []
        for coord in coords:
            robot = self.get_robot_by_coord(coord[0], coord[1])
            if robot:
                robots.append(robot)
        return robots

    def handle_directional(self, heading):
        angular_change = min(abs(heading.getHeading() - self.heading),
                             360 - abs(heading.getHeading() - self.heading)) // 90
        self.turning_delay += self.delay_per_task * angular_change

        self.heading = heading.getHeading()
        rotation_deg = 0
        if angular_change == 1:
            rotation_deg = 90
        else:
            rotation_deg = 180
        rotation = np.radians(rotation_deg)
        e_krot = 1/6 * (self.mass + self.load_mass) * (self._length + self._robot_width**2) * (rotation**2/2.5**2)
        e_frot = (self.mass + self.load_mass) * self._gravity * self._friction * self._robot_radius * rotation
        e_rot = e_krot + e_frot
        self.energy_consumption += e_rot
        self.turning += 1
        self.route_stop_points.pop(0)

    def handle_next_movement(self, next_destination_coordinate, is_next_route_stop=True):
        current_coord = NetLogoCoordinate(self.pos_x, self.pos_y)

        if self.close_enough(next_destination_coordinate, 0.3):
            self.update_position(next_destination_coordinate)
            if is_next_route_stop:
                self.route_stop_points.pop(0)
        else:
            self.update_motion_parameters(current_coord, next_destination_coordinate)

    def update_position(self, coordinate):
        # Update robot's position and movement parameters to match the intersection_coordinate
        self.pos_x = round(coordinate.x)
        self.pos_y = round(coordinate.y)
        self.coordinate = NetLogoCoordinate(self.pos_x, self.pos_y)
        self.velocity = 0
        self.acceleration = 0

    def advance_state_if_needed(self):
        # Check if all route points are done and handle state transitions
        if not self.route_stop_points:
            # print("called from advance_state_if_needed")
            self.advance_state()
            self.update_current_position()
            if self.current_state == "delivering_pod":
                # station: Station = self.universe.station_manager.get_station_by_id(self.job.station_id)
                # self.set_move_to_station_gate()
                # """
                pod = self.job.pod
                storage_manager = self.warehouse.storage_manager
                storage = storage_manager.pods_to_storage.get(pod)
                self.job.record_delivery(self.universe._tick)
                # Proactive replenishment keeps its origin reserved for the whole
                # trip: do NOT release the pod-to-storage mapping or mark the slot
                # empty when the pod is collected. Every other job type (ordinary
                # picking, post-pick replenishment, RTS-selected returns) releases
                # the origin here and selects a new destination on return.
                if storage and not getattr(self.job, "proactive_origin_reserved", False):
                    storage.removeStoragePod()
                    del storage_manager.pods_to_storage[pod]
                    storage_manager.setStorageAvailable(NetLogoCoordinate(storage.pos_x, storage.pos_y))

                self.set_move_to_station_gate()
                upsert_pod_travel(
                    self.job.my_id,
                    self.robotID(self.robotName()),
                    str(self.job.pod.pod_id),
                    "delivering_pod",
                    str((self.job.pod.pos_x, self.job.pod.pos_y)),
                    str(
                        (
                            self.universe.station_manager.get_station_by_id(self.job.station_id).coordinate.x,
                            self.universe.station_manager.get_station_by_id(self.job.station_id).coordinate.y
                        )
                    ),
                    self.universe._tick
                )
                # """
            elif self.current_state == "returning_pod":
                # station: Station = self.universe.station_manager.get_station_by_id(self.job.station_id)
                # station.remove_robot(self.robotName())
                
                # self.set_move(self.job.pod_coordinate, self.universe.graph_pod, need_neutralize_robot=True)
                # """
                station: Station = self.universe.station_manager.get_station_by_id(self.job.station_id)
                station.remove_robot(self.robotName())

                self.handle_pod_return(station)

                self.initial_w1 = False
                upsert_pod_travel(
                    self.job.my_id,
                    self.robotID(self.robotName()),
                    str(self.job.pod.pod_id),
                    "returning_pod",
                    str((self.job.pod.pos_x, self.job.pod.pos_y)),
                    str(
                        (
                            self.destination.x,
                            self.destination.y
                        )
                    ),
                    self.universe._tick
                )
                # """
            elif self.current_state == "station_processing":
                # station: Station = self.universe.station_manager.get_station_by_id(self.job.station_id)
                # if station.is_replenishment_station():
                #     print(f"Gets into replenishment")
                # station.add_robot(self.robotName())
                # self.setPath(self.transform_coords_to_list(station.get_robot_route(self.robotName())))
                # """
                station: Station = self.universe.station_manager.get_station_by_id(self.job.station_id)
                station.add_robot(self.robotName())
                self.setPath(self.transform_coords_to_list(station.get_robot_route(self.robotName())))
                # """

        self.universe.landscape.setObject(self.robotName(), self.pos_x, self.pos_y, self.velocity, self.acceleration,
                                          self.heading, self.current_state, self.load_mass)

    def update_motion_parameters(self, current_coord, next_destination_coordinate):
        # Adjust robot's acceleration based on proximity to the next intersection_coordinate
        self.acceleration = 1
        deceleration_buffer = 0.5
        distance_to_stop = self._calculateTwoPoint(current_coord, next_destination_coordinate)
        if (self.velocity ** 2) / (2 * deceleration_buffer) >= distance_to_stop:
            self.acceleration = -1

    def move(self):
        self.changeColorByState()
        self._accumulate_state_seconds()

        # ── PATCH A5: charging lifecycle — OPT-IN (universe.charging_enabled) ──
        # When charging is disabled (team default), none of this runs and the
        # robot behaves exactly as before (no battery drain, no charging).
        if getattr(self.universe, "charging_enabled", False):
            if self._charging_pre_move():
                return   # robot is dead/parked this tick -> skip movement

        try:
            self.movementPlan()
        except Exception as e:
            print(f"Error in robot {self.id} with robotjob {self.job}")
            raise e

        if getattr(self.universe, "charging_enabled", False):
            self._apply_drive_by_charging()   # drive-by charge, every tick

        self.latest_tick += 1

    def _charging_pre_move(self):
        """Battery drain + charge triggers (run only when charging enabled).
        Returns True if the robot is dead/parked this tick so move() skips
        the movement step."""
        # Dead-robot handling: battery fully drained -> park in place.
        if self.battery_level_j <= 0.0 and self.current_state != "dead":
            if getattr(self, "_charging_episode_id", None) is None:
                self._incr_charging_counter("robots_died_without_charge_request")
            dispatcher = getattr(self.universe, "charging_dispatcher", None)
            if dispatcher is not None:
                dispatcher.terminal_remove(self)
            self._release_charger()
            # Task 3 Part D2/D3: a dying robot must not strand its committed-next
            # job/pod (which would block re-allocation forever) or leave a
            # fabricated RTS outcome. Cancel the reservation (restores the job to
            # the queue exactly once, clears all markers) and censor any pending
            # RTS transition with an explicit reason (no fabricated completion).
            registry = getattr(self.universe, "committed_next_registry", None)
            if registry is not None:
                registry.cancel_for_robot(self, "reservation_cancelled_robot_death")
            runtime = getattr(self.universe, "rts_rollout_runtime", None)
            if runtime is not None:
                runtime.censor_pending_for_robot(
                    robot=self,
                    status="censored_robot_death",
                    reason="robot_battery_death",
                )
            self.velocity = 0; self.acceleration = 0
            self.route_stop_points = []
            self.current_state = "dead"
            self._incr_charging_counter("robots_died_from_battery")
            self.universe.landscape.setObject(
                self.robotName(), self.pos_x, self.pos_y,
                self.velocity, self.acceleration, self.heading,
                self.current_state, self.load_mass)
            return True
        if self.current_state == "dead":
            return True
        # Task 3 Part C4: bounded stale-claim / no-progress cleanup. A robot holding
        # a claim on a charging trip but not physically charging and not moving for
        # STALE_CLAIM_MAX_TICKS releases the claim and re-enters waiting (retryable).
        if self._claimed_charger is not None and self.current_state == "going_to_charge" \
                and not self.is_charging:
            pos = (self.pos_x, self.pos_y)
            now = int(getattr(self.universe, "_tick", 0))
            if self._claim_progress_pos is not None and pos != self._claim_progress_pos:
                self._claim_progress_pos = pos
                self._claim_created_tick = now          # made progress; reset the clock
            elif self._claim_created_tick is not None and \
                    (now - int(self._claim_created_tick)) >= self.STALE_CLAIM_MAX_TICKS:
                self._incr_charging_counter("stale_charger_claims_detected")
                self._release_stale_charger_claim_and_route()
                self._enter_waiting_for_charger("stale_claim_no_progress")
                self._queue_for_charging(preserve_timestamp=True)
        if self.current_state == "leaving_charger":
            pos = (self.pos_x, self.pos_y)
            now = float(getattr(self.universe, "_tick", 0.0))
            if self._leaving_progress_pos is None or pos != self._leaving_progress_pos:
                self._leaving_progress_pos = pos
                self._leaving_progress_at = now
            elif self._leaving_progress_at is None:
                self._leaving_progress_at = now
            elif now - float(self._leaving_progress_at) >= self.LEAVING_CHARGER_MAX_SECONDS:
                self._retry_charging_departure_after_no_progress(now=now)
        # Base operational drain (skipped while plugged in / charging).
        if not self.is_charging:
            fixed_load_energy = self.BASE_DRAIN_RATE_PER_S * self.universe.tick_to_second
            self.fixed_load_energy_consumption += fixed_load_energy
            self.battery_level_j = max(0.0, self.battery_level_j - fixed_load_energy)
        # Deferred charging check: if busy and battery is unsafe for new work,
        # queue the charging request as soon as the current physical task ends.
        # This uses the existing interrupt threshold as a safety threshold so a
        # robot cannot keep accepting work until it silently drains to zero.
        is_busy = (self.current_state in {"taking_pod", "delivering_pod", "station_processing", "returning_pod"}) or (self.job is not None and not getattr(self.job, "is_finished", False))
        if is_busy and self._battery_requires_work_safety_charging():
            if not getattr(self, "charge_after_current_task", False):
                self._incr_charging_counter("forced_charging_requests")
            self.charge_after_current_task = True

        # Task 3 Part C6: a recovered robot must not stay hidden in waiting.
        if self._waiting_for_charger and self.battery_pct >= self._battery_work_safety_pct() \
                and not getattr(self, "charge_after_current_task", False):
            self._waiting_for_charger = False
        # Trigger: idle + (unsafe battery or deferred charging queued) -> go to a charger.
        is_idle = (self.current_state == "idle") and (self.job is None or getattr(self.job, "is_finished", True))
        if is_idle and (self._battery_requires_work_safety_charging() or getattr(self, "charge_after_current_task", False)) and not getattr(self.universe, "disable_active_charging", False):
            if self._charging_episode_id is None:
                if self.battery_pct > self.BATTERY_LOW_PCT:
                    self._incr_charging_counter("forced_charging_requests")
                self._incr_charging_counter("robots_sent_to_charge_before_job")
                self._queue_for_charging()
            return self.current_state in {
                "waiting_for_charger", "going_to_charge", "physically_charging", "leaving_charger"
            }
        return False

    def drawNextPosition(self):
        initial_velocity = self.velocity
        initial_acceleration = self.acceleration

        energy = self.calculateEnergy(initial_velocity, initial_acceleration)
        self.energy_consumption += energy
        # ── PATCH A6: drain battery by motion energy this tick (opt-in) ──
        if getattr(self.universe, "charging_enabled", False):
            self.battery_level_j = max(0.0, self.battery_level_j - energy)

        if self.velocity != 0:
            distance_delta = self.velocity * self.universe.tick_to_second
            self.travel_distances += distance_delta
            if self.load_mass > 0 or self.current_state in {"delivering_pod", "station_processing", "returning_pod"}:
                self.loaded_travel_distance += distance_delta
            else:
                self.empty_travel_distance += distance_delta
            if self.heading == 0:
                self.pos_y += distance_delta
            elif self.heading == 180:
                self.pos_y -= distance_delta
            elif self.heading == 90:
                self.pos_x += distance_delta
            elif self.heading == 270:
                self.pos_x -= distance_delta
        self.coordinate = NetLogoCoordinate(round(self.pos_x), round(self.pos_y))

        if self.acceleration != 0:
            self.velocity += (self.acceleration * self.universe.tick_to_second)
            self.velocity = max(0, min(self.maximum_speed, self.velocity))

        if ACTIVATE_NEAREST:
            if self.current_state in ['delivering_pod', 'station_processing', 'returning_pod']:
                self.job.pod.pos_x = self.pos_x
                self.job.pod.pos_y = self.pos_y
                self.job.pod.velocity = self.velocity
                self.job.pod.acceleration = self.acceleration

                pod_id = self.job.pod.pod_id

                self.warehouse.pod_manager.get_pod_by_id(pod_id).pos_x = self.pos_x
                self.warehouse.pod_manager.get_pod_by_id(pod_id).pos_y = self.pos_y
                self.warehouse.pod_manager.get_pod_by_id(pod_id).velocity = self.velocity
                self.warehouse.pod_manager.get_pod_by_id(pod_id).acceleration = self.acceleration

        # for traffic policy purposes, report states to the manager
        self.universe.landscape.setObject(self.robotName(), self.pos_x, self.pos_y, self.velocity, self.acceleration,
                                          self.heading, self.current_state, self.load_mass)
        # self.update_intersection_information(energy)

    def update_intersection_information(self, energy):
        intersection_id = self.universe.intersection_manager.find_intersection_by_path_coordinate(round(self.pos_x),
                                                                                                  round(self.pos_y))
        if intersection_id:
            if self.current_intersection_id == intersection_id:
                self.update_current_intersection(energy)
            else:
                self.finalize_current_intersection()

                self.start_new_intersection(intersection_id)
        else:
            self.finalize_current_intersection()

            self.reset_intersection_tracking()

    def update_current_intersection(self, energy):
        self.current_intersection_energy_consumption += energy

        intersection: Intersection = self.universe.intersection_manager.find_intersection_by_id(
            self.current_intersection_id)
        intersection.update_robot(self)

    def finalize_current_intersection(self):
        if self.current_intersection_id is None:
            return

        # Mark the finish time for the current intersection
        self.current_intersection_finish_time = self.universe._tick
        # Log the intersection information to CSV

        intersection: Intersection = self.universe.intersection_manager.find_intersection_by_id(
            self.current_intersection_id)

        # if intersection.should_save_robot_info():
            # self.insert_robot_intersection_information_to_csv(intersection)

        intersection.remove_robot(self)

    def start_new_intersection(self, intersection_id):
        # Set the new intersection ID and reset the energy consumption
        self.current_intersection_id = intersection_id
        self.current_intersection_energy_consumption = 0
        self.current_intersection_start_time = self.universe._tick

        intersection: Intersection = self.universe.intersection_manager.find_intersection_by_id(
            self.current_intersection_id)
        intersection.add_robot(self)

    def reset_intersection_tracking(self):
        # Reset all intersection-related data
        self.current_intersection_id = None
        self.current_intersection_energy_consumption = 0
        self.current_intersection_stop_and_go = 0
        self.current_intersection_start_time = 0
        self.current_intersection_finish_time = 0

    def insert_robot_intersection_information_to_csv(self, intersection: Intersection):
        header = ["robot_name", "robot_state", "robot_destination", "intersection_start_time",
                  "intersection_finish_time", "intersection_id",
                  "energy_consumption_intersection", "queueing_robot"]
        data = [self.robotName(), self.current_state, self.route_stop_points[-1], self.current_intersection_start_time,
                self.current_intersection_finish_time, self.current_intersection_id,
                self.current_intersection_energy_consumption,
                intersection.robot_count()]

        write_to_csv("intersection-energy-consumption.csv", header, data,
                     self.universe.landscape.current_date_string)

    def handle_pod_return(self, station: "Station"):
        """Select a storage destination via the RTS policy seam, then
        execute all side effects (storage reservation, path planning,
        report writing).  Introduced in Phase 5B.

        The policy only *selects* a destination; all mutation stays here
        so that pathing, telemetry, and storage bookkeeping are unchanged.
        """
        from src.rmfs.decisions.rts.types import RTSDestinationContext

        if getattr(self.job, "rts_continuation_active", False):
            self._continue_rts_return_after_replenishment()
            return

        # Proactive replenishment never consults RTS on return: it routes the pod
        # straight back to the exact storage it left, which was kept reserved for
        # the whole trip.
        if getattr(self.job, "proactive_origin_reserved", False):
            self._return_proactive_pod_to_origin()
            return

        context = RTSDestinationContext(
            warehouse=self.warehouse,
            robot=self,
            pod=self.job.pod,
            station=station,
        )
        decision = self.warehouse.rts_policy.select_destination(context)
        if decision.mode == "rl":
            self._validate_rts_rl_decision(decision)

        if decision.mode == "fixed":
            self._record_rts_decision(context, decision)
            # --- Fixed return: identical to old self.return_fix branch ---
            self.job.writePodReturnReport(-1)
            self.destination = decision.destination
            self.set_move(self.destination, self.warehouse.graph_pod, need_neutralize_robot=False)

        elif decision.mode == "nearest":
            self._record_rts_decision(context, decision)
            # --- Nearest return: identical to old self.return_nearest branch (storage found) ---
            nearest_storage = decision.storage
            print(f"[DEBUG] nearest_storage: {nearest_storage}")
            print(f"[DEBUG] {vars(nearest_storage)}")

            self.destination = decision.destination
            debug_print(f"[DEBUG] destination: {self.destination}")
            self.job.pod_return_coordinate = self.destination
            self.job.writePodReturnReport(
                calculateManhattanDistance(
                    (self.job.pod_return_coordinate.x, self.job.pod_return_coordinate.y),
                    (self.job.pod_coordinate.x, self.job.pod_coordinate.y),
                )
            )

            self.set_move(self.destination, self.universe.graph_pod, need_neutralize_robot=False)
            self.warehouse.storage_manager.addPodToStorage(self.job.pod, nearest_storage)

        elif decision.mode == "rl":
            if decision.branch == "store":
                self._start_rts_store_return(decision, station, context)
            elif decision.branch == "replenish_store":
                self._start_rts_replenish_store_return(decision, station, context)

        elif decision.mode in ("nearest_fallback", "none"):
            self._record_rts_decision(context, decision)
            # --- Fallback: no empty storage, or neither flag set ---
            self.job.writePodReturnReport(-1)
            self.destination = decision.destination
            self.set_move(self.destination, self.warehouse.graph_pod, need_neutralize_robot=False)

    def _return_proactive_pod_to_origin(self):
        """Route a proactive replenishment pod back to its exact original storage
        without invoking RTS.

        The origin was kept reserved (via the StorageManager ownership model) for
        the whole trip. Ownership is re-verified immediately before routing; a
        conflict is a hard error with no nearest-storage fallback and no RTS zone
        selection.
        """
        pod = self.job.pod
        origin = self.job.proactive_origin_storage
        storage_manager = self.warehouse.storage_manager
        if origin is None or not storage_manager.storage_owned_by_pod(origin, pod):
            raise RuntimeError(
                "proactive replenishment origin ownership conflict for pod "
                f"{getattr(pod, 'pod_id', None)}: recorded origin "
                f"{self.job.proactive_origin_storage_id} is no longer owned by the pod"
            )
        self.destination = NetLogoCoordinate(origin.pos_x, origin.pos_y)
        self.job.pod_return_coordinate = self.destination
        self.job.writePodReturnReport(
            calculateManhattanDistance(
                (self.job.pod_return_coordinate.x, self.job.pod_return_coordinate.y),
                (self.job.pod_coordinate.x, self.job.pod_coordinate.y),
            )
        )
        self.set_move(self.destination, self.universe.graph_pod, need_neutralize_robot=False)

    def _record_rts_decision(self, context, decision):
        decision_event_id = self.warehouse.rts_rollout_runtime.on_decision(
            robot=self,
            context=context,
            decision=decision,
        )
        self.warehouse.link_committed_next_decision(self, decision_event_id)
        return decision_event_id

    def _capture_rts_precommit_decision_state_if_needed(self, context, decision):
        runtime = getattr(self.warehouse, "rts_rollout_runtime", None)
        capture = getattr(runtime, "capture_precommit_decision_state", None)
        if capture is None:
            return decision
        return capture(robot=self, context=context, decision=decision)

    def _validate_rts_rl_decision(self, decision):
        if decision.storage is None:
            raise ValueError("RTS policy returned 'rl' mode but decision.storage is None")
        if decision.destination is None:
            raise ValueError("RTS policy returned 'rl' mode but decision.destination is None")
        if decision.branch not in {"store", "replenish_store"}:
            raise ValueError(f"RTS policy returned unsupported rl branch: {decision.branch!r}")
        if decision.zone_id is None or not str(decision.zone_id).strip():
            raise ValueError("RTS policy returned 'rl' mode without a zone_id")
        if getattr(decision, "storage_id", None) and str(decision.storage_id) != self._storage_id(decision.storage):
            raise ValueError("RTS policy returned mismatched storage_id and storage object")
        if decision.branch == "replenish_store":
            if getattr(decision, "replenishment_station", None) is None:
                raise ValueError("RTS replenish_store decision is missing replenishment_station")
            station_id = str(getattr(decision.replenishment_station, "station_id", ""))
            if getattr(decision, "replenishment_station_id", None) and str(decision.replenishment_station_id) != station_id:
                raise ValueError("RTS policy returned mismatched replenishment_station_id and station object")

    def _commit_selected_next_for_rts_decision(self, decision):
        if not getattr(self.warehouse, "committed_next_reservations_enabled", False):
            return decision
        metadata = dict(getattr(decision, "metadata", {}) or {})
        try:
            reservation = self.warehouse.commit_committed_next_decision(self, decision)
        except Exception as exc:
            metadata["committed_next_commit_failed"] = str(exc)
            return replace(decision, metadata=metadata)
        if reservation is None:
            metadata["committed_next_reservation_id"] = None
            return replace(decision, metadata=metadata, committed_next_reservation_id=None)
        metadata["committed_next_reservation_id"] = getattr(reservation, "reservation_id", None)
        return replace(
            decision,
            committed_next_reservation_id=getattr(reservation, "reservation_id", None),
            metadata=metadata,
        )

    def _reserve_rts_storage(self, decision):
        storage = decision.storage
        storage_manager = self.warehouse.storage_manager
        storage_manager.reserveStorageForPod(self.job.pod, storage)
        return storage

    def _begin_rts_return_ownership(self, *, decision, storage, station, stage):
        pod = self.job.pod
        pod.rts_return_in_progress = True
        pod.rts_return_branch = decision.branch
        pod.rts_return_zone_id = str(decision.zone_id)
        self.job.rts_continuation_active = True
        self.job.rts_decision_identity = (
            f"{decision.policy_name}:{self.robotName()}:{self.job.my_id}:"
            f"{pod.pod_id}:{self.warehouse._tick}"
        )
        self.job.rts_branch = decision.branch
        self.job.rts_zone_id = str(decision.zone_id)
        self.job.rts_final_storage = storage
        self.job.rts_final_storage_id = getattr(decision, "storage_id", None) or self._storage_id(storage)
        self.job.rts_final_destination = decision.destination
        self.job.rts_action_index = getattr(decision, "action_index", None)
        self.job.rts_action_context_id = getattr(decision, "action_context_id", None)
        self.job.rts_action_context_version = getattr(decision, "action_context_version", None)
        self.job.rts_next_job_proposal_id = getattr(decision, "next_job_proposal_id", None)
        self.job.rts_committed_next_reservation_id = getattr(decision, "committed_next_reservation_id", None)
        self.job.rts_source_station_id = station.station_id
        self.job.rts_stage = stage
        self.job.rts_storage_reserved = True

    def _start_rts_store_return(self, decision, station: "Station", context):
        storage = None
        try:
            storage = self._reserve_rts_storage(decision)
            decision = self._capture_rts_precommit_decision_state_if_needed(context, decision)
            decision = self._commit_selected_next_for_rts_decision(decision)
            self._begin_rts_return_ownership(
                decision=decision,
                storage=storage,
                station=station,
                stage="to_storage",
            )
            self.destination = decision.destination
            self.job.pod_return_coordinate = self.destination
            self.job.writePodReturnReport(
                calculateManhattanDistance(
                    (self.job.pod_return_coordinate.x, self.job.pod_return_coordinate.y),
                    (self.job.pod_coordinate.x, self.job.pod_coordinate.y),
                )
            )
            self.set_move(self.destination, self.universe.graph_pod, need_neutralize_robot=False)
            self._record_rts_decision(context, decision)
        except Exception as exc:
            self._rollback_rts_return(
                reason=f"failed to start RTS store return: {exc}",
                reserved_storage=storage,
            )
            raise

    def _start_rts_replenish_store_return(self, decision, station: "Station", context):
        if self.warehouse.replenishment_hard_cap_reached():
            raise RuntimeError("replenishment_hard_cap_reached")
        storage = None
        replenishment_station = None
        try:
            storage = self._reserve_rts_storage(decision)
            decision = self._capture_rts_precommit_decision_state_if_needed(context, decision)
            decision = self._commit_selected_next_for_rts_decision(decision)
            self._begin_rts_return_ownership(
                decision=decision,
                storage=storage,
                station=station,
                stage="to_replenishment",
            )
            replenishment_station = decision.replenishment_station
            if replenishment_station is None:
                raise RuntimeError("RTS replenish_store decision is missing replenishment station")
            skus_to_replenish, _ = self.warehouse.get_replenishment_skus_for_pod(self.job.pod)
            if not skus_to_replenish:
                # The RTS policy selected replenish_store for this pod. Physical
                # restoration is always full-pod, so fall back to the pod's full
                # compartment set as the (diagnostic) trigger list rather than
                # aborting the policy-chosen action.
                skus_to_replenish = sorted(self.job.pod.skus.keys())

            pending_request = self.warehouse.get_pending_replenishment_dispatch(self.job.pod.pod_id)
            self.job.rts_pending_replenishment_request_snapshot = (
                dict(pending_request) if pending_request is not None else None
            )
            if pending_request is not None:
                self.warehouse.remove_pending_replenishment_dispatch(self.job.pod.pod_id)

            self.job.station_id = replenishment_station.station_id
            self.job.replenishment_skus = []
            self.job.replenishment_delay = 0
            self.job.add_replenishment_task(self.job.pod, skus_to_replenish, source="rts")
            self.job.is_finished = False
            self.job.rts_replenishment_station = replenishment_station
            self.job.rts_replenishment_station_id = getattr(decision, "replenishment_station_id", None) or str(
                getattr(replenishment_station, "station_id", "")
            )
            replenishment_station.add_pod(self.job.pod.pod_id)
            self.job.pod.station = replenishment_station
            self.job.pod.is_awaiting_replenishment = True
            self.job.pod.has_pending_replenishment_dispatch = False
            self.job.pod.must_replenish_before_pick = False
            self.warehouse.pod_manager.mark_pod_not_available(self.job.pod)

            self.current_state = "delivering_pod"
            self.destination = replenishment_station.get_path()[0]
            self.set_move(self.destination, self.universe.graph_pod, need_neutralize_robot=False)
            self._record_rts_decision(context, decision)
        except Exception as exc:
            self._rollback_rts_return(
                reason=f"failed to start RTS replenish_store return: {exc}",
                reserved_storage=storage,
                replenishment_station=replenishment_station,
            )
            raise

    @staticmethod
    def _storage_id(storage):
        if storage is None:
            return ""
        if getattr(storage, "storage_id", None) is not None:
            return str(getattr(storage, "storage_id"))
        return f"{getattr(storage, 'pos_x', '')}:{getattr(storage, 'pos_y', '')}"

    def _continue_rts_return_after_replenishment(self):
        if getattr(self.job, "rts_stage", None) != "post_replenishment_to_storage":
            raise RuntimeError(
                f"RTS continuation reached return handling in invalid stage {self.job.rts_stage!r}"
            )
        if self.job.rts_final_storage is None or self.job.rts_final_destination is None:
            raise RuntimeError("RTS continuation is missing its final storage destination")
        try:
            self.destination = self.job.rts_final_destination
            self.job.pod_return_coordinate = self.destination
            self.job.writePodReturnReport(
                calculateManhattanDistance(
                    (self.job.pod_return_coordinate.x, self.job.pod_return_coordinate.y),
                    (self.job.pod_coordinate.x, self.job.pod_coordinate.y),
                )
            )
            self.set_move(self.destination, self.universe.graph_pod, need_neutralize_robot=False)
            self.job.rts_stage = "to_storage"
        except Exception as exc:
            self._rollback_rts_return(
                reason=f"failed to continue RTS return after replenishment: {exc}",
                reserved_storage=self.job.rts_final_storage,
            )
            raise

    def _rollback_rts_return(self, *, reason, reserved_storage=None, replenishment_station=None):
        pod = self.job.pod if self.job is not None else None
        storage = reserved_storage or getattr(self.job, "rts_final_storage", None)
        registry = getattr(self.warehouse, "committed_next_registry", None)
        if registry is not None and registry.get_for_robot(self) is not None:
            from src.rmfs.decisions.task_allocation.committed_next import CommittedNextInvariantError
            raise CommittedNextInvariantError(
                "RTS return rollback attempted to disrupt a committed-next reservation"
            )
        if pod is not None and storage is not None:
            self.warehouse.storage_manager.releaseStorageReservation(pod, storage)
        if pod is not None:
            if replenishment_station is not None:
                replenishment_station.remove_pod(pod.pod_id)
            pod.rts_return_in_progress = False
            pod.rts_return_branch = None
            pod.rts_return_zone_id = None
            pod.is_awaiting_replenishment = False
            pod.station = None
        snapshot = getattr(self.job, "rts_pending_replenishment_request_snapshot", None)
        if snapshot is not None and pod is not None:
            if self.warehouse.get_pending_replenishment_dispatch(pod.pod_id) is None:
                self.warehouse.pending_replenishment_dispatches.append(dict(snapshot))
                pod.has_pending_replenishment_dispatch = True
                pod.must_replenish_before_pick = bool(snapshot.get("guaranteed_on_release", False))
        if self.job is not None:
            self.job.replenishment_skus = []
            self.job.replenishment_delay = 0
            self.job.clear_rts_continuation()
        raise RuntimeError(reason)

    def _clear_rts_return_ownership(self):
        if self.job is None or not getattr(self.job, "rts_continuation_active", False):
            return
        pod = self.job.pod
        pod.rts_return_in_progress = False
        pod.rts_return_branch = None
        pod.rts_return_zone_id = None
        pod.is_awaiting_replenishment = False
        pending_request = self.warehouse.get_pending_replenishment_dispatch(pod.pod_id)
        pod.has_pending_replenishment_dispatch = pending_request is not None
        pod.must_replenish_before_pick = bool(
            pending_request and pending_request.get("guaranteed_on_release", False)
        )
        self.job.clear_rts_continuation()

    def assign_job_and_set_move_to_take_pod(self, job: RobotJob):
        self.job = job
        self.current_cycle_start_tick = self.universe._tick

        self.set_move_to_take_pod()

    def assign_job_and_set_move_to_station(self, job: RobotJob):
        self.job = job
        self.current_cycle_start_tick = self.universe._tick
        self.current_state = "taking_pod"
        self.route_stop_points = None
        # print("called from assign_jobn_and_set_move_to_station")
        self.advance_state_if_needed()
        self.taking_pod_delay = 0

    def set_move_to_take_pod(self):
        # self.set_move(self.job.pod_coordinate, graph=self.universe.graph, need_neutralize_robot=False)
        # self.current_state = "taking_pod"
        pod_id = self.job.pod.pod_id
        x = self.warehouse.pod_manager.get_pod_by_id(pod_id).pos_x
        y = self.warehouse.pod_manager.get_pod_by_id(pod_id).pos_y
        self.w1 = NetLogoCoordinate(x, y)
        self.destination = NetLogoCoordinate(x, y)

        # self.w1 = NetLogoCoordinate(self.job.pod.pos_x, self.job.pod.pos_y)
        # self.destination = NetLogoCoordinate(self.job.pod.pos_x, self.job.pod.pos_y)
        debug_print(f"[DEBUG] destination: {self.destination}")

        if self.close_enough(self.destination):
            # self.job.pod.pos_x = self.pos_x
            # self.job.pod.pos_y = self.pos_y
            # self.job.pod.velocity = 0
            # self.job.pod.acceleration = 0
            self.warehouse.pod_manager.get_pod_by_id(pod_id).pos_x = self.pos_x
            self.warehouse.pod_manager.get_pod_by_id(pod_id).pos_y = self.pos_y
            self.warehouse.pod_manager.get_pod_by_id(pod_id).velocity = 0
            self.warehouse.pod_manager.get_pod_by_id(pod_id).acceleration = 0
            self.job.pod.pos_x = self.pos_x
            self.job.pod.pos_y = self.pos_y
            self.job.pod.velocity = 0
            self.job.pod.acceleration = 0
            self.destination = NetLogoCoordinate(round(self.pos_x), round(self.pos_y))
        try:
            self.set_move(self.destination, graph=self.warehouse.graph, need_neutralize_robot=False)
        except Exception as primary_error:
            try:
                self.set_move(self.destination, graph=self.warehouse.graph_pod, need_neutralize_robot=False)
                debug_print(
                    f"[WARN] fallback to graph_pod for pickup route on pod {self.job.pod.pod_id} "
                    f"from robot {self.robotName()}"
                )
            except Exception as e:
                print(f"[ERROR] move pod {self.job.pod.pod_id} location {(self.job.pod.pos_x, self.job.pod.pos_y)} destination {self.destination}")
                print(f"[ERROR] current robot {self.robotName()} job {self.job.job_id}")
                print(f"[ERROR] other ongoing robots")
                for o in self.warehouse.get_movable_objects():
                    if isinstance(o, Robot):
                        job_desc = "no-job"
                        if o.job is not None:
                            job_desc = f"job {o.job.job_id} pod {o.job.pod.pod_id}"
                        print(f" >>> robot {o.robotName()} {job_desc}")
                        if o.job is not None and o.job.pod.pod_id == self.job.pod.pod_id:
                            print("[CRITICAL] double job for this pod")
                raise e from primary_error
        self.current_state = "taking_pod"
        upsert_pod_travel(
            self.job.my_id,
            self.robotID(self.robotName()),
            str(self.job.pod.pod_id),
            "taking_pod",
            str((self.pos_x, self.pos_y)),
            str((self.job.pod.pos_x, self.job.pod.pos_y)),
            self.universe._tick
        )

    def set_move_to_station_gate(self):
        station: Station = self.universe.station_manager.get_station_by_id(self.job.station_id)
        self.set_move(station.get_path()[0], graph=self.universe.graph_pod, need_neutralize_robot=False)

    def set_move(self, dest: NetLogoCoordinate, graph, need_neutralize_robot: bool = False, avoid_side: bool = False):
        start = self.coordinate_to_string_key(round(self.pos_x), round(self.pos_y))
        end = self.coordinate_to_string_key(round(dest.x), round(dest.y))
        # print(f"[DEBUG] set move start: {start} end: {end}")
        if need_neutralize_robot:
            self.neutralizeRobotState()

        nodes_to_avoid = []
        if avoid_side:
            avoid_coords = self.calculate_all_directions_next_blocks(round(self.pos_x), round(self.pos_y), 1,
                                                                     include_self=False)
            for avoid_coord in avoid_coords:
                if self.universe.landscape.get_neighbor_object(*avoid_coord) is None:
                    continue

                nodes_to_avoid.append(self.coordinate_to_string_key(*avoid_coord))

        node_routes = None
        if self.universe.zoning:
            # print(f"[DEBUG] universe.zoning == True")
            zone_boundary, penalties = self.create_zone(method="kmeans")
            node_routes = graph.dijkstra_modified(start,end, penalties, zone_boundary, nodes_to_avoid)
        else:
            # print(f"[DEBUG] universe.zoning == False")
            node_routes = graph.dijkstra(start, end, nodes_to_avoid) # This one is baseline
        if node_routes is None:
            graph_name = getattr(graph, "key", "") or "warehouse"
            raise ValueError(
                f"No route found in {graph_name} graph from {start} to {end}"
            )
        try:
            self.setPath(self._transformRouteToList(node_routes))
        except Exception as e:
            print(f"[ERROR] failed to setPath from {start} to {end} with route {node_routes}")
            raise e

    def create_zone(self, method):
        robot_objects = self.universe.landscape.get_robot_object()
        robots_location = [[info['x'], info['y']] for info in robot_objects.values() if info['state'] != 'station_processing']
        robots_idle_time = []
        robot_list = []
        if len(robots_location) > 0:
            robot_list = self.get_robots_by_coords(robots_location)

        for robot in robot_list:
            robots_idle_time.append(robot.idle_time)
        
        zones = Zone(robots_location, self.universe.get_warehouse_size(), methods=method)
        penalties = zones.calculate_penalty(robots_location, robots_idle_time, self.universe.get_warehouse_size(), threshold=5)
        zone_boundary = self.zone.get_boundary()
        return zone_boundary, penalties

    # utility functions
    
    @staticmethod
    def calculate_all_directions_next_blocks(x, y, block_count=5, include_self=False):
        headings = [0, 90, 180, 270]
        result = []

        for heading in headings:
            blocks = Robot._calculate_next_blocks(x, y, heading, block_count, include_self)
            for block in blocks:
                result.append(block)

        return result
    
    @staticmethod
    def getHeading(p1: NetLogoCoordinate, p2: NetLogoCoordinate):
        if p1.x == p2.x:
            if p1.y > p2.y:
                return 180
            else:
                return 0
        elif p1.y == p2.y:
            if p1.x > p2.x:
                return 270
            else:
                return 90

    @staticmethod
    def _calculateTwoPoint(p1: NetLogoCoordinate, p2: NetLogoCoordinate):
        return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)

    def close_enough(self, p: NetLogoCoordinate, precision=0.25):
        self_coord = NetLogoCoordinate(self.pos_x, self.pos_y)
        return self._calculateTwoPoint(self_coord, p) < precision

    @staticmethod
    def ensure_coordinate(number):
        if isinstance(number, int):
            print(f"{number} is an integer.")
        elif isinstance(number, float) and number.is_integer():
            print(f"{number} is a float with 0 precision.")
        else:
            print(f"{number} is not a valid integer or float with 0 precision.")

    @staticmethod
    def get_decimal(number):
        subtractor = int(number)
        return number - subtractor

    def robotName(self):
        return f"robot-{self._id}"

    @staticmethod
    def robotID(robot_name):
        return int(robot_name.split('-')[1])

    @staticmethod
    def _calculate_next_blocks(x, y, heading, block_count=5, include_self=False):
        x_difference = 0
        y_difference = 0

        if heading == 0:
            y_difference = 1
        if heading == 90:
            x_difference = 1
        if heading == 270:
            x_difference = -1
        if heading == 180:
            y_difference = -1

        result = []

        if include_self:
            result.append([x, y])
        for i in range(block_count):
            x += x_difference
            y += y_difference

            result.append([x, y])

        return result

    @staticmethod
    def coordinate_to_string_key(x: int, y: int):
        return "{},{}".format(x, y)

    @staticmethod
    def _getIntersectionBlock(blocks_1, blocks_2):
        for p in blocks_1:
            if p in blocks_2:
                return p


def calculateManhattanDistance(u, v):
    return abs(u[0] - v[0]) + abs(u[1] - v[1])
