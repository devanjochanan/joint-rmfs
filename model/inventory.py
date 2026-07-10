from typing import Optional, List
import builtins
import csv
import os
import math
import threading
import tempfile
import time
import re
from collections import defaultdict, deque
import ast
import json
import pandas as pd
from datetime import datetime

from engine.landscape import Landscape
from engine.universe import Universe
from engine.util import *
from .intersection_manager import IntersectionManager
from .order import Order
from .order_manager import OrderManager
from .pod import Pod
from .pod_manager import PodManager
from .robot import Robot
from .robot_job import RobotJob
from .station_manager import StationManager
from .station import Station
from .storage_manager import StorageManager
from .storage import Storage
from .tools.write_record import write_record_to
# DB
from .tools.pod_location import get_pod_location
from .tools.order_history import upsert_order_history
from .tools.job_task import upsert_job_task, update_job_task
from .tools.pre_assign import initialize_pre_assign_table, clear_pre_assign_table, insert_pre_assign
# from .live_advanced_table import start_gui
# RTS decision seam (Phase 5B)
from src.rmfs.decisions.task_allocation import (
    CommittedNextRegistry,
    DEFAULT_REGRET_K,
    DEFAULT_ROBOT_TASK_ALLOCATOR,
    TASK_ALLOCATOR_SCOPE,
    select_active_job_queue_assignment,
)
from src.rmfs.decisions.rts import CurrentRTSPolicy
from src.rmfs.rl.rts.outcome_tracker import NoopRTSRolloutRuntime
from src.rmfs.rl.rts.runtime_install import install_rts_runtime
from src.rmfs.rl.rts.runtime_registry import get_rts_runtime_config, get_rts_runtime_root
from src.rmfs.runtime_io.logging import debug_print as _rmfs_debug_print


def print(*args, **kwargs):  # noqa: A001
    """Gate legacy model debug chatter while preserving warnings/errors."""
    first = str(args[0]) if args else ""
    if first.startswith(("[ERROR]", "[WARN]")):
        return builtins.print(*args, **kwargs)
    return _rmfs_debug_print(*args, **kwargs)

# Show full column content
pd.set_option('display.max_colwidth', None)

# Show all columns without truncation
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)  # Let it auto-expand

class Inventory(Universe):
    dimension = 60
    map = []
    landscape = None
    stop_and_go = 0
    total_energy = 0
    total_pod = 0
    total_turning = 0
    total_robot_idle = 0
    movement_channel = {}
    graph = None
    graph_pod = None

    def __init__(self, runtime_paths=None, sqlite_db_path="warehouse.db"):
        self._tick = 0  #current counter
        self.runtime_paths = runtime_paths or {}
        self.sqlite_db_path = sqlite_db_path
        # self.ignored_types = ["pod", "station", "way-direction"]
        self.ignored_types = ["station", "way-direction"]
        self.tick_to_second = 0.15
        self.fast_train = (
            os.environ.get("RMFS_FAST_TRAIN", "0").strip().lower()
            in {"1", "true", "yes", "on"}
        )
        self.dynamic_job_update_enabled = (
            os.environ.get("RMFS_DYNAMIC_JOB_UPDATE", "1").strip().lower()
            not in {"0", "false", "no", "off"}
        )
        self._fast_pod_info_records = []
        self._fast_finished_orders = []
        self.job_queue: list[RobotJob] = []
        # ── Charging state (Salsa charging integration; PATCH B) ──────────
        self.charging_enabled = True          # ON by default; RMFS_CHARGING_ENABLED=0 disables
        self.charger_cells = set()            # all charger coords (drive-by eligible)
        self.active_charger_cells = set()     # active-dispatch targets (empty => global)
        self.occupied_chargers = {}           # cell -> robot id (one claim per cell)
        self.disable_active_charging = False  # True => opportunity-only (drive-by only)
        # Instance-level mutable state (prevents cross-instance contamination)
        self.map = []
        self.movement_channel = {}
        self.stop_and_go = 0
        self.total_energy = 0
        self.total_pod = 0
        self.total_turning = 0
        self.total_robot_idle = 0
        self.graph = None
        self.graph_pod = None
        self.landscape = Landscape(self.dimension)
        self.pod_manager = PodManager()
        self.station_manager = StationManager()
        self.storage_manager = StorageManager(self)
        self.order_manager = OrderManager()
        self.next_process_tick = 0
        self.intersection_manager = IntersectionManager(self.landscape.current_date_string)
        self.update_intersection_using_RL = False
        self.zoning = False
        self.robot_queue_order = {}
        self.preassign_dict = {}
        self.last_order = {}
        
        self.preassign_per_station = defaultdict(deque)
        # self.currently_picking = {}
        # # Shared wrapper for the DataFrame
        # self.shared_data = {"df": pd.DataFrame()}

        # # Start GUI in a thread
        # self.gui_thread = threading.Thread(target=start_gui, args=(self.shared_data,), daemon=True)
        # self.gui_thread.start()
        self.poa_podmatch = False
        self.poa_first = False  # preasign2 gajelas nih / F3
        self.poa_second = True  # Rika's Future-aware POA (no batching)
        self.poa_aisyahna = False  # Similarity-based order batching + POA

        self.pps_pileon = True
        self.pps_demand = False
        self.pps_rl = False       # When True, PPS is controlled by RL agent (PPSEnv)
        self.pps_picked_quantity = 0
        self.pps_pod_visits = 0
        self.joint_rl = False     # When True, both POA and PPS are controlled by JointEnv
        self.replenishment_count = 0
        self.replenishment_trips = 0
        self.completed_post_pick_replenishment_actions = 0
        self.completed_rts_replenishment_actions = 0
        self.completed_proactive_replenishment_actions = 0
        self.completed_post_pick_store_actions = 0
        self.job_queue_cumulative_sum = 0
        self.job_queue_sample_count = 0
        self.peak_job_queue = 0
        self.global_critical_skus = set()
        self.pending_replenishment_dispatches = []
        self.rts_controls_post_pick_replenishment = False
        self.replenishment_dispatch_aging_ticks = int(
            os.environ.get("RMFS_REPLENISHMENT_AGING_TICKS", "300")
        )
        self.pod_replenishment_threshold = float(
            os.environ.get("RMFS_POD_REPLENISHMENT_THRESHOLD", "0.4")
        )
        # Approved replenishment capacity policy.
        #   soft cap: at most 3 proactive replenishment robot commitments during
        #             busy operation (bypassed only for genuinely idle robots
        #             that have no assignable picking job).
        #   hard cap: at most 11 total unique replenishment commitments across
        #             all sources; load is counted by unique committed pods.
        self.replenishment_soft_cap = int(
            os.environ.get("RMFS_REPLENISHMENT_SOFT_CAP", "3")
        )
        self.replenishment_hard_cap = int(
            os.environ.get("RMFS_REPLENISHMENT_HARD_CAP", "11")
        )
        self.proactive_replenishment_enabled = (
            os.environ.get("RMFS_PROACTIVE_REPLENISHMENT", "1").strip().lower()
            not in {"0", "false", "no", "off"}
        )
        # Task 3 Part E: bounded age after which a still-undispatched pending
        # replenishment request is considered stale and released from the cap.
        # Cap VALUE is unchanged; this only prunes entries that can never dispatch.
        self.replenishment_pending_stale_ticks = int(
            os.environ.get("RMFS_REPLENISHMENT_PENDING_STALE_TICKS", "3000")
        )
        self.replenishment_counters: dict = {}   # nonsemantic Part E/G accounting

        self.priority_order = False
        self.robot_task_allocator = DEFAULT_ROBOT_TASK_ALLOCATOR
        self.regret_k = DEFAULT_REGRET_K
        self.task_allocator_scope = TASK_ALLOCATOR_SCOPE
        self.committed_next_reservations_enabled = False
        self.committed_next_registry = CommittedNextRegistry()

        # Aisyahna's similarity-based batching parameters
        self.aisyahna_batch_interval = 5   # fire every t ticks
        self._aisyahna_last_batch_tick = 0  # last tick the batching fired


        if self.poa_second and not self.fast_train:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            initialize_pre_assign_table(timestamp, db_path=self.sqlite_db_path)
            clear_pre_assign_table(db_path=self.sqlite_db_path)
        # RTS decision seam (Phase 5B): default policy preserves current behavior
        self.rts_policy = CurrentRTSPolicy()
        self.rts_rollout_runtime = NoopRTSRolloutRuntime()
        install_rts_runtime(self, get_rts_runtime_config(), get_rts_runtime_root())
        super().__init__()

    def runtime_path(self, key, default):
        return self.runtime_paths.get(key, default)

    @property
    def assign_order_csv(self):
        return self.runtime_path("assign_order_csv", "assign_order.csv")

    @property
    def pod_info_csv(self):
        return self.runtime_path("pod_info_csv", "pod_info.csv")

    @property
    def generated_order_csv(self):
        return self.runtime_path("generated_order_csv", "generated_order.csv")

    @staticmethod
    def _assign_order_retryable(exc):
        if isinstance(exc, PermissionError):
            return True
        if isinstance(exc, (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError)):
            return True
        if isinstance(exc, OSError):
            winerror = getattr(exc, "winerror", None)
            errno = getattr(exc, "errno", None)
            return winerror in {32, 33} or errno in {13, 16}
        return False

    def _with_assign_order_retry(self, action, operation):
        attempts = 7
        delay = 0.05
        last_exc = None
        for attempt in range(attempts):
            try:
                return operation()
            except Exception as exc:  # retry only transient file/share and parse states below
                if not self._assign_order_retryable(exc):
                    raise
                last_exc = exc
                if attempt >= attempts - 1:
                    break
                time.sleep(delay)
                delay = min(delay * 2.0, 1.0)
        raise RuntimeError(
            f"assign_order.csv {action} failed after {attempts} attempts: "
            f"{type(last_exc).__name__}: {last_exc}"
        ) from last_exc

    def _read_assign_order_csv(self):
        return self._with_assign_order_retry(
            "read",
            lambda: pd.read_csv(self.assign_order_csv),
        )

    def _write_assign_order_csv(self, df):
        path = os.path.abspath(self.assign_order_csv)
        directory = os.path.dirname(path) or "."
        basename = os.path.basename(path)

        def write_once():
            os.makedirs(directory, exist_ok=True)
            fd, tmp_path = tempfile.mkstemp(prefix=f".{basename}.", suffix=".tmp", dir=directory, text=True)
            try:
                with os.fdopen(fd, "w", newline="", encoding="utf-8") as fh:
                    df.to_csv(fh, index=False)
                    fh.flush()
                    os.fsync(fh.fileno())
                os.replace(tmp_path, path)
            finally:
                try:
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)
                except OSError:
                    pass

        return self._with_assign_order_retry("atomic write", write_once)

    @staticmethod
    def _normalize_sku_qty_dict(sku_qty):
        if not isinstance(sku_qty, dict):
            return {}

        normalized = {}
        for sku, qty in sku_qty.items():
            try:
                sku_key = int(sku)
            except (TypeError, ValueError):
                sku_key = sku
            try:
                qty_value = int(qty)
            except (TypeError, ValueError):
                qty_value = qty
            normalized[sku_key] = qty_value
        return normalized

    def _job_ready_to_finish(self, robot: Robot) -> bool:
        job = getattr(robot, "job", None)
        if job is None or job.is_finished or robot.current_state != "station_processing":
            return False

        job_station = self.station_manager.get_station_by_id(job.station_id)
        if job_station.is_picker_station():
            return job.picking_delay <= 0 and not job.is_being_processed()
        if job_station.is_replenishment_station():
            return job.replenishment_delay <= 0 and not job.is_being_processed()
        return False

    def get_below_reorder_skus_for_pod(self, pod: Pod) -> list[int]:
        if pod is None or not pod.skus:
            return []

        critical_skus = []
        for sku_id in pod.skus.keys():
            _, needs_replenishment = self.pod_manager.is_sku_need_replenished(sku_id)
            if needs_replenishment:
                self.global_critical_skus.add(sku_id)
                critical_skus.append(sku_id)
        return sorted(set(critical_skus))

    def get_local_replenishment_skus_for_pod(self, pod: Pod) -> list[int]:
        if pod is None or not pod.skus:
            return []
        local_skus = []
        for sku_id, details in pod.skus.items():
            limit_qty = details.get("limit_qty", 0)
            threshold = details.get("threshold", 0)
            if limit_qty <= 0:
                continue
            if float(details.get("current_qty", 0)) / float(limit_qty) <= float(threshold):
                local_skus.append(sku_id)
        return sorted(set(local_skus))

    def get_pod_critical_fill_score(self, pod: Pod, critical_skus=None) -> float:
        if pod is None or not pod.skus:
            return 1.0
        critical_skus = list(critical_skus or self.get_below_reorder_skus_for_pod(pod))
        if not critical_skus:
            return 1.0

        fill_ratios = []
        for sku_id in critical_skus:
            details = pod.skus.get(sku_id)
            if not details:
                continue
            limit_qty = details.get("limit_qty", 0)
            if limit_qty <= 0:
                continue
            fill_ratios.append(
                max(0.0, min(1.0, details.get("current_qty", 0) / limit_qty))
            )
        if not fill_ratios:
            return 1.0
        return float(sum(fill_ratios) / len(fill_ratios))

    def get_pod_local_fill_ratio(self, pod: Pod):
        """Aggregate local fill ratio over the pod's valid SKU compartments.

        Computed as the mean of ``current_qty / limit_qty`` across compartments
        with a strictly positive capacity. Invalid (zero/negative capacity)
        compartments are ignored rather than dividing by zero. Returns ``None``
        when the pod has no valid compartment. This aggregate is independent of
        the globally-low SKU subset.
        """
        if pod is None or not pod.skus:
            return None
        ratios = []
        for details in pod.skus.values():
            limit_qty = details.get("limit_qty", 0)
            if limit_qty is None or limit_qty <= 0:
                continue
            current_qty = details.get("current_qty", 0)
            ratios.append(max(0.0, float(current_qty) / float(limit_qty)))
        if not ratios:
            return None
        return float(sum(ratios) / len(ratios))

    def get_globally_low_refillable_skus_for_pod(self, pod: Pod) -> list[int]:
        """SKUs on this pod that are globally low AND locally refillable.

        A globally-low SKU is refillable on the pod only when the SKU exists on
        that pod and its ``current_qty < limit_qty`` (there is headroom to add).
        """
        if pod is None or not pod.skus:
            return []
        result = []
        for sku_id, details in pod.skus.items():
            limit_qty = details.get("limit_qty", 0)
            current_qty = details.get("current_qty", 0)
            if limit_qty is None or current_qty >= limit_qty:
                continue
            _, needs_replenishment = self.pod_manager.is_sku_need_replenished(sku_id)
            if needs_replenishment:
                self.global_critical_skus.add(sku_id)
                result.append(sku_id)
        return sorted(set(result))

    def evaluate_pod_replenishment_eligibility(self, pod: Pod) -> dict:
        """Independent OR eligibility: local_trigger OR global_trigger.

        Keeps trigger diagnostics separate: the local aggregate fill, the set of
        globally-low refillable SKUs, and which branch(es) fired.
        """
        local_fill = self.get_pod_local_fill_ratio(pod)
        local_trigger = (
            local_fill is not None and local_fill < self.pod_replenishment_threshold
        )
        global_low_refillable = self.get_globally_low_refillable_skus_for_pod(pod)
        global_trigger = len(global_low_refillable) > 0
        branches = []
        if local_trigger:
            branches.append("local")
        if global_trigger:
            branches.append("global")
        trigger_skus = sorted(
            set(self.get_local_replenishment_skus_for_pod(pod)) | set(global_low_refillable)
        )
        return {
            "eligible": bool(local_trigger or global_trigger),
            "local_fill": local_fill,
            "local_trigger": bool(local_trigger),
            "global_low_refillable_skus": global_low_refillable,
            "global_trigger": bool(global_trigger),
            "trigger_skus": trigger_skus,
            "branches": branches,
        }

    def is_pod_replenishment_eligible(self, pod: Pod) -> bool:
        return bool(self.evaluate_pod_replenishment_eligibility(pod)["eligible"])

    def get_replenishment_skus_for_pod(self, pod: Pod) -> tuple[list[int], float]:
        if pod is None or not pod.skus:
            return [], 1.0
        plan = self.evaluate_pod_replenishment_eligibility(pod)
        local_fill = plan["local_fill"] if plan["local_fill"] is not None else 1.0
        if not plan["eligible"]:
            return [], local_fill
        # Restoration is always full-pod; the returned trigger list is diagnostic
        # only. When the pod is eligible purely on the aggregate-local trigger and
        # no individual SKU is flagged, fall back to the full compartment set so
        # downstream non-empty checks do not treat an eligible pod as ineligible.
        trigger_skus = plan["trigger_skus"] or sorted(pod.skus.keys())
        return trigger_skus, local_fill

    def get_pending_replenishment_dispatch(self, pod_id: int):
        for request in self.pending_replenishment_dispatches:
            if int(request["pod_id"]) == int(pod_id):
                return request
        return None

    def should_guarantee_replenishment_request(self, request, current_tick=None) -> bool:
        if request is None:
            return False
        current_tick = int(self._tick) if current_tick is None else int(current_tick)
        if bool(request.get("guaranteed_on_release", False)):
            return True
        wait_time = current_tick - int(request.get("created_tick", current_tick))
        return wait_time >= self.replenishment_dispatch_aging_ticks

    def _iter_robots(self):
        for obj in self.get_movable_objects():
            if getattr(obj, "object_type", None) == "robot":
                yield obj

    def replenishment_commitments_by_pod(self, *, include_pending: bool = True) -> dict[int, str]:
        """Unique committed replenishment pods → source.

        Capacity is accounted by *unique committed pods* so that a single
        commitment which flows pending → queued → active is counted exactly
        once. The commitment sources are pending replenishment requests, queued
        replenishment jobs, active robot replenishment jobs, returning
        replenishment robots, and RTS replenish-store continuations.
        """
        commitments: dict[int, str] = {}
        # Active/returning robot replenishment jobs (authoritative source label).
        # Do not drop a job merely because station service set is_finished=True:
        # the commitment remains live until the pod physically returns to storage.
        for robot in self._iter_robots():
            job = getattr(robot, "job", None)
            if job is None:
                continue
            state = getattr(robot, "current_state", None)
            if state == "idle":
                continue
            if not self._is_replenishment_commitment_job(job):
                continue
            pod = getattr(job, "pod", None)
            if pod is None:
                continue
            commitments[int(pod.pod_id)] = getattr(job, "replenishment_source", None) or "post_pick"
        # Queued replenishment jobs.
        for job in self.job_queue:
            if not getattr(job, "is_replenishment_job", False):
                continue
            pod = getattr(job, "pod", None)
            if pod is None:
                continue
            commitments.setdefault(
                int(pod.pod_id),
                getattr(job, "replenishment_source", None) or "post_pick",
            )
        # Pending replenishment requests.
        if include_pending:
            for request in self.pending_replenishment_dispatches:
                pid = int(request["pod_id"])
                commitments.setdefault(pid, request.get("source", "post_pick"))
        return commitments

    def _is_replenishment_commitment_job(self, job) -> bool:
        return bool(
            getattr(job, "is_replenishment_job", False)
            or getattr(job, "rts_continuation_active", False)
            or getattr(job, "rts_branch", None) == "replenish_store"
        )

    def total_replenishment_load(self) -> int:
        return len(self.replenishment_commitments_by_pod())

    def _bump_repl_counter(self, name: str, n: int = 1) -> None:
        """Task 3 Part E/G nonsemantic accounting."""
        self.replenishment_counters[name] = self.replenishment_counters.get(name, 0) + n

    def replenishment_cap_composition(self) -> dict:
        """Task 3 Part E1: hard-cap load broken down by stage (unique pods, priority
        active > queued > pending). Sums to total_replenishment_load()."""
        seen: dict[int, str] = {}
        active = queued = pending = rts_cont = 0
        for robot in self._iter_robots():
            job = getattr(robot, "job", None)
            if job is None or getattr(robot, "current_state", None) == "idle":
                continue
            if not self._is_replenishment_commitment_job(job):
                continue
            pod = getattr(job, "pod", None)
            if pod is None:
                continue
            pid = int(pod.pod_id)
            if pid in seen:
                continue
            seen[pid] = "active"
            active += 1
            if getattr(job, "rts_continuation_active", False):
                rts_cont += 1
        for job in self.job_queue:
            if not getattr(job, "is_replenishment_job", False):
                continue
            pod = getattr(job, "pod", None)
            if pod is None:
                continue
            pid = int(pod.pod_id)
            if pid in seen:
                continue
            seen[pid] = "queued"
            queued += 1
        for req in self.pending_replenishment_dispatches:
            pid = int(req["pod_id"])
            if pid in seen:
                continue
            seen[pid] = "pending"
            pending += 1
        return {
            "active": active, "queued": queued, "pending": pending,
            "rts_continuation": rts_cont, "total": len(seen),
            "hard_cap": self.replenishment_hard_cap,
        }

    def prune_stale_pending_replenishment(self) -> int:
        """Task 3 Part E2/E3: drop pending replenishment requests that can no longer
        validly consume the cap (missing/ineligible pod, no valid SKU, or exceeded
        bounded age), releasing each entry's cap contribution exactly once. The hard
        cap VALUE is unchanged; this only removes entries that can never dispatch."""
        if not self.pending_replenishment_dispatches:
            return 0
        now = int(self._tick)
        kept = []
        removed = 0
        for req in self.pending_replenishment_dispatches:
            pod = self.pod_manager.get_pod_by_id(int(req["pod_id"]))
            skus = req.get("skus_to_replenish") or []
            age = now - int(req.get("created_tick", now))
            invalid = (
                pod is None
                or getattr(pod, "is_awaiting_replenishment", False)
                or getattr(pod, "rts_return_in_progress", False)
                or getattr(pod, "committed_next_owner_robot_id", None)
                or not skus
                or age > self.replenishment_pending_stale_ticks
            )
            if invalid:
                removed += 1
                if pod is not None:
                    pod.has_pending_replenishment_dispatch = False
                    pod.must_replenish_before_pick = False
                self._bump_repl_counter("replenishment_pending_pruned")
            else:
                kept.append(req)
        self.pending_replenishment_dispatches = kept
        return removed

    def proactive_replenishment_load(self) -> int:
        return self.proactive_replenishment_robot_load()

    def proactive_replenishment_robot_commitments_by_pod(self) -> dict[int, str]:
        """Unique proactive pods already committed to queued/active robots.

        Passive pending requests are excluded so the soft cap reflects robot
        usage rather than merely discovered demand.
        """
        return {
            pod_id: source
            for pod_id, source in self.replenishment_commitments_by_pod(include_pending=False).items()
            if source == "proactive"
        }

    def proactive_replenishment_robot_load(self) -> int:
        return sum(
            1 for source in self.proactive_replenishment_robot_commitments_by_pod().values()
            if source == "proactive"
        )

    def replenishment_hard_cap_reached(self) -> bool:
        return self.total_replenishment_load() >= self.replenishment_hard_cap

    def can_admit_replenishment(
        self,
        source: str,
        *,
        idle_bypass: bool = False,
        consume_robot_slot: bool = False,
    ) -> bool:
        """Admission decision for a *new* replenishment commitment.

        Hard cap (11) blocks every new source once total unique load is already
        at the cap. The soft cap (3) only applies when proactive work consumes a
        queued/active robot slot; passive pending requests may wait without
        consuming the soft-cap budget.
        """
        commitments = self.replenishment_commitments_by_pod()
        total = len(commitments)
        if total >= self.replenishment_hard_cap:
            # Task 3 Part E/G: nonsemantic count of which source the cap blocked.
            src = str(source)
            if src == "proactive":
                self._bump_repl_counter("replenishment_cap_blocks_proactive")
            elif src == "rts":
                self._bump_repl_counter("replenishment_cap_blocks_rts")
            else:
                self._bump_repl_counter("replenishment_cap_blocks_post_pick")
            return False
        if source == "proactive" and consume_robot_slot and not idle_bypass:
            proactive = self.proactive_replenishment_robot_load()
            if proactive >= self.replenishment_soft_cap:
                return False
        return True

    def can_start_replenishment_commitment(
        self,
        pod: Pod,
        *,
        source: str,
        idle_bypass: bool = False,
    ) -> bool:
        if pod is None:
            return False
        try:
            pod_id = int(pod.pod_id)
        except (TypeError, ValueError):
            return False
        commitments = self.replenishment_commitments_by_pod()
        if pod_id not in commitments and len(commitments) >= self.replenishment_hard_cap:
            return False
        if source == "proactive":
            proactive_robot_commitments = self.proactive_replenishment_robot_commitments_by_pod()
            if (
                pod_id not in proactive_robot_commitments
                and len(proactive_robot_commitments) >= self.replenishment_soft_cap
                and not idle_bypass
            ):
                return False
        return True

    def enqueue_pending_replenishment_dispatch(
        self,
        pod: Pod,
        skus_to_replenish,
        *,
        source: str = "post_pick",
        guaranteed_on_release: bool = False,
        idle_bypass: bool = False,
    ) -> bool:
        if pod is None:
            return False
        normalized_skus = sorted({int(sku) for sku in skus_to_replenish if sku is not None})
        if not normalized_skus:
            return False
        if getattr(pod, "is_awaiting_replenishment", False):
            return False
        if getattr(pod, "rts_return_in_progress", False):
            return False
        if getattr(pod, "committed_next_owner_robot_id", None):
            return False

        existing = self.get_pending_replenishment_dispatch(pod.pod_id)
        if existing is not None:
            # Merging into an already-admitted request does not create a new
            # commitment and must not re-apply admission caps or change load.
            existing["skus_to_replenish"] = sorted(
                set(existing.get("skus_to_replenish", [])).union(normalized_skus)
            )
            existing["guaranteed_on_release"] = (
                bool(existing.get("guaranteed_on_release", False))
                or guaranteed_on_release
            )
            pod.has_pending_replenishment_dispatch = True
            if existing["guaranteed_on_release"]:
                pod.must_replenish_before_pick = True
            return False

        # Duplicate request for a pod already committed elsewhere (queued/active)
        # must not increase load.
        if int(pod.pod_id) in self.replenishment_commitments_by_pod():
            return False

        if not self.can_admit_replenishment(source, idle_bypass=idle_bypass):
            return False

        self.pending_replenishment_dispatches.append(
            {
                "pod_id": int(pod.pod_id),
                "skus_to_replenish": list(normalized_skus),
                "created_tick": int(self._tick),
                "guaranteed_on_release": bool(guaranteed_on_release),
                "source": str(source),
            }
        )
        pod.has_pending_replenishment_dispatch = True
        if guaranteed_on_release:
            pod.must_replenish_before_pick = True
        return True

    def remove_pending_replenishment_dispatch(self, pod_id: int) -> bool:
        removed = False
        remaining = []
        for request in self.pending_replenishment_dispatches:
            if int(request["pod_id"]) == int(pod_id):
                removed = True
                continue
            remaining.append(request)
        self.pending_replenishment_dispatches = remaining
        pod = self.pod_manager.get_pod_by_id(pod_id)
        if pod is not None:
            pod.has_pending_replenishment_dispatch = False
            pod.must_replenish_before_pick = False
        return removed

    def get_most_depleted_eligible_pod_for_sku(self, sku_id: int):
        pods_with_sku = self.pod_manager.get_pods_by_sku(sku_id) or []
        best_candidate = None
        best_key = None
        for pod in pods_with_sku:
            if pod is None or getattr(pod, "is_awaiting_replenishment", False):
                continue
            if getattr(pod, "rts_return_in_progress", False):
                continue
            if getattr(pod, "committed_next_owner_robot_id", None):
                continue
            plan = self.evaluate_pod_replenishment_eligibility(pod)
            # For a specific globally-low SKU, only consider pods where this SKU
            # is genuinely refillable (present and below its limit).
            if sku_id not in plan["global_low_refillable_skus"]:
                continue
            skus_to_replenish = plan["trigger_skus"] or sorted(pod.skus.keys())
            local_fill = plan["local_fill"] if plan["local_fill"] is not None else 1.0
            sku_details = pod.skus.get(sku_id, {})
            limit_qty = sku_details.get("limit_qty", 0)
            current_qty = sku_details.get("current_qty", 0)
            fill_ratio = current_qty / limit_qty if limit_qty > 0 else 1.0
            candidate_key = (
                local_fill,
                fill_ratio,
                0 if pod.is_idle else 1,
                int(pod.pod_id),
            )
            if best_key is None or candidate_key < best_key:
                best_key = candidate_key
                best_candidate = (pod, skus_to_replenish)
        return best_candidate

    def enqueue_best_replenishment_dispatch_for_sku(self, sku_id: int) -> bool:
        candidate = self.get_most_depleted_eligible_pod_for_sku(sku_id)
        if candidate is None:
            return False
        pod, skus_to_replenish = candidate
        return self.enqueue_pending_replenishment_dispatch(
            pod,
            skus_to_replenish,
            source="proactive",
            guaranteed_on_release=False,
        )

    def iter_proactive_replenishment_candidates(self):
        """Deterministic direct proactive candidates using the approved OR rule."""
        committed = self.replenishment_commitments_by_pod()
        candidates = []
        for pod in self.pod_manager.get_all_pods():
            if pod is None:
                continue
            try:
                pod_id = int(pod.pod_id)
            except (TypeError, ValueError):
                continue
            if pod_id in committed:
                continue
            if getattr(pod, "is_awaiting_replenishment", False):
                continue
            if getattr(pod, "rts_return_in_progress", False):
                continue
            if getattr(pod, "committed_next_owner_robot_id", None):
                continue
            plan = self.evaluate_pod_replenishment_eligibility(pod)
            if not plan["eligible"]:
                continue
            skus_to_replenish = plan["trigger_skus"] or sorted(pod.skus.keys())
            if not skus_to_replenish:
                continue
            local_fill = plan["local_fill"] if plan["local_fill"] is not None else 1.0
            global_fill = 1.0
            global_skus = plan.get("global_low_refillable_skus") or ()
            if global_skus:
                ratios = []
                for sku_id in global_skus:
                    details = pod.skus.get(sku_id, {}) or {}
                    limit_qty = details.get("limit_qty", 0)
                    current_qty = details.get("current_qty", 0)
                    ratios.append((current_qty / limit_qty) if limit_qty else 1.0)
                global_fill = min(ratios) if ratios else 1.0
            candidates.append(
                (
                    (
                        float(local_fill),
                        float(global_fill),
                        0 if getattr(pod, "is_idle", False) else 1,
                        pod_id,
                    ),
                    pod,
                    skus_to_replenish,
                    plan,
                )
            )
        for _key, pod, skus_to_replenish, plan in sorted(candidates, key=lambda item: item[0]):
            yield pod, skus_to_replenish, plan

    def refresh_mandatory_replenishment_pods(self):
        # Task 3 Part E2/E3: release stale/invalid pending cap entries each cycle,
        # before mandatory-flag refresh and any new admission decisions.
        self.prune_stale_pending_replenishment()
        current_tick = int(self._tick)
        for pod in self.pod_manager.get_all_pods():
            if pod is None or getattr(pod, "is_awaiting_replenishment", False):
                continue
            if getattr(pod, "rts_return_in_progress", False):
                continue
            if getattr(pod, "committed_next_owner_robot_id", None):
                continue
            pod.must_replenish_before_pick = False
        for request in self.pending_replenishment_dispatches:
            if not self.should_guarantee_replenishment_request(request, current_tick):
                continue
            pod = self.pod_manager.get_pod_by_id(int(request["pod_id"]))
            if (
                pod is not None
                and not getattr(pod, "is_awaiting_replenishment", False)
                and not getattr(pod, "rts_return_in_progress", False)
                and not getattr(pod, "committed_next_owner_robot_id", None)
            ):
                pod.must_replenish_before_pick = True

    def send_pod_for_replenishment(
        self,
        pod: Pod,
        station: Station,
        skus_to_replenish,
        robot: Robot | None = None,
        source: str = "post_pick",
        idle_bypass: bool = False,
    ) -> bool:
        if pod is None or station is None:
            return False
        if getattr(pod, "is_awaiting_replenishment", False):
            return False
        if getattr(pod, "rts_return_in_progress", False):
            return False
        if getattr(pod, "committed_next_owner_robot_id", None):
            return False
        replenishment_skus = sorted({int(sku) for sku in skus_to_replenish if sku is not None})
        if not replenishment_skus:
            return False
        if not self.can_start_replenishment_commitment(
            pod,
            source=str(source),
            idle_bypass=idle_bypass,
        ):
            return False

        # Proactive replenishment must return the pod to the exact storage it
        # left. Resolve and require that origin up front; if it cannot be pinned
        # (not currently owned by this pod) do not dispatch as proactive — this
        # is a clean pre-commit failure that mutates nothing.
        proactive_origin = None
        if str(source) == "proactive":
            proactive_origin = self.storage_manager.get_owned_storage_for_pod(pod)
            if proactive_origin is None:
                return False

        new_job = RobotJob(pod.coordinate, station_id=station.station_id, pod=pod)
        new_job.add_replenishment_task(pod, replenishment_skus, source=source)
        if proactive_origin is not None:
            new_job.set_proactive_origin(
                proactive_origin,
                getattr(proactive_origin, "storage_id", None),
                (float(getattr(proactive_origin, "pos_x", 0.0)), float(getattr(proactive_origin, "pos_y", 0.0))),
            )
        station.add_pod(pod.pod_id)
        pod.station = station
        self.pod_manager.mark_pod_not_available(pod)
        pod.is_awaiting_replenishment = True
        pod.has_pending_replenishment_dispatch = False
        pod.must_replenish_before_pick = False
        self.remove_pending_replenishment_dispatch(pod.pod_id)

        if robot is not None:
            robot.assign_job_and_set_move_to_station(new_job)
        else:
            self.job_queue.append(new_job)
        return True

    def dispatch_pending_replenishment_requests(self, prioritize_aged_only: bool = False) -> int:
        if not self.pending_replenishment_dispatches:
            return 0

        dispatched = 0
        current_tick = int(self._tick)
        requests = sorted(
            list(self.pending_replenishment_dispatches),
            key=lambda request: (
                0 if self.should_guarantee_replenishment_request(request, current_tick) else 1,
                -(current_tick - int(request.get("created_tick", current_tick))),
                int(request.get("pod_id", 0)),
            ),
        )
        for request in requests:
            guaranteed_request = self.should_guarantee_replenishment_request(request, current_tick)
            if prioritize_aged_only and not guaranteed_request:
                continue

            station = self.station_manager.find_available_replenish_station()
            if station is None:
                break

            pod = self.pod_manager.get_pod_by_id(int(request["pod_id"]))
            if pod is None:
                self.remove_pending_replenishment_dispatch(int(request["pod_id"]))
                continue
            if getattr(pod, "is_awaiting_replenishment", False):
                self.remove_pending_replenishment_dispatch(pod.pod_id)
                continue
            if getattr(pod, "rts_return_in_progress", False):
                continue
            if getattr(pod, "committed_next_owner_robot_id", None):
                continue
            if not pod.is_idle:
                continue

            # A previously admitted request is a stable commitment: it is NOT
            # revoked merely because an eligibility recheck (qj/local fill) now
            # reads differently. Restoration is full-pod, so the trigger list is
            # only diagnostic metadata carried through to the job.
            plan = self.evaluate_pod_replenishment_eligibility(pod)
            skus_to_replenish = plan["trigger_skus"] or sorted(
                {int(sku) for sku in request.get("skus_to_replenish", []) if sku in pod.skus}
            ) or sorted(pod.skus.keys())
            source = request.get("source", "post_pick")

            if self.send_pod_for_replenishment(
                pod,
                station,
                skus_to_replenish,
                source=source,
                idle_bypass=False,
            ):
                dispatched += 1
        return dispatched

    def queue_post_pick_replenishment(self, pod: Pod, critical_skus=None) -> bool:
        if pod is None:
            return False

        critical_set = {
            int(sku)
            for sku in (critical_skus or [])
            if sku is not None
        }
        plan = self.evaluate_pod_replenishment_eligibility(pod)
        # Eligible on the approved OR rule, or forced by a globally-critical SKU
        # discovered at pick time.
        if not plan["eligible"] and not critical_set:
            return False
        skus_to_replenish = set(plan["trigger_skus"]) | critical_set
        if not skus_to_replenish:
            skus_to_replenish = set(pod.skus.keys())

        return self.enqueue_pending_replenishment_dispatch(
            pod,
            sorted(skus_to_replenish),
            source="post_pick",
            guaranteed_on_release=bool(critical_set),
        )

    def _picking_jobs_assignable_now(self):
        """Picking jobs currently eligible for allocation to an idle robot.

        Mirrors the allocator's eligibility filter (excludes RTS-return pods,
        committed-next-owned pods, and committed-next-reserved jobs). Only
        picking jobs (not replenishment jobs) are counted.
        """
        assignable = []
        for job in self.job_queue:
            if getattr(job, "is_replenishment_job", False):
                continue
            pod = getattr(job, "pod", None)
            if pod is None:
                continue
            if getattr(pod, "rts_return_in_progress", False):
                continue
            if getattr(pod, "committed_next_owner_robot_id", None):
                continue
            if getattr(job, "committed_next_reservation_id", None):
                continue
            assignable.append(job)
        return assignable

    def _idle_robots_available_for_replenishment(self):
        return [
            robot for robot in self._iter_robots()
            if (robot.job is None or getattr(robot.job, "is_finished", False))
            and getattr(robot, "current_state", None) == "idle"
            and not robot.is_unavailable_for_work_due_to_charging()
        ]

    def _unmatched_idle_robots_after_picking_allocation(self, idle_robots):
        """Idle robots that cannot receive a currently eligible picking job.

        Use the existing allocator one robot at a time against a shrinking copy
        of eligible picking jobs. This keeps allocator semantics intact while
        avoiding the old warehouse-global "any picking job exists" bypass gate.
        """
        unmatched = []
        remaining_jobs = [
            job
            for job in self._picking_jobs_assignable_now()
            if not getattr(job, "is_replenishment_job", False)
        ]
        for robot in idle_robots:
            if not remaining_jobs:
                unmatched.append(robot)
                continue
            allocation = select_active_job_queue_assignment(
                jobs=remaining_jobs,
                robots=[robot],
                cost_fn=lambda job, candidate_robot: calculateDistance(
                    candidate_robot.pos_x,
                    candidate_robot.pos_y,
                    job.pod_coordinate.x,
                    job.pod_coordinate.y,
                ),
                robot_task_allocator=self.robot_task_allocator,
                regret_k=self.regret_k,
                job_id_fn=lambda job: f"{getattr(job.pod, 'pod_id', job.pod)}:{job.station_id}",
                robot_id_fn=lambda candidate_robot: getattr(candidate_robot, "_id", None),
            )
            if allocation is None:
                unmatched.append(robot)
                continue
            del remaining_jobs[allocation.queue_index]
        return unmatched

    def run_proactive_replenishment_pass(self) -> int:
        """Directly discover eligible pods and enqueue stable pending requests."""
        if not self.proactive_replenishment_enabled:
            return 0
        if self.total_replenishment_load() >= self.replenishment_hard_cap:
            return 0

        admitted = 0
        for pod, skus_to_replenish, _plan in self.iter_proactive_replenishment_candidates():
            if self.total_replenishment_load() >= self.replenishment_hard_cap:
                break
            if self.enqueue_pending_replenishment_dispatch(
                pod,
                skus_to_replenish,
                source="proactive",
                guaranteed_on_release=False,
            ):
                admitted += 1
        return admitted

    def dispatch_proactive_replenishment_to_unmatched_idle_robots(self, idle_robots) -> int:
        """Dispatch at most one proactive job per unmatched idle robot.

        This is the soft-cap bypass seam: the caller supplies robots left idle
        after normal picking allocation has been attempted.
        """
        if not self.proactive_replenishment_enabled:
            return 0
        if not idle_robots:
            return 0
        self.run_proactive_replenishment_pass()
        dispatched = 0
        current_tick = int(self._tick)
        for robot in idle_robots:
            if self.total_replenishment_load() >= self.replenishment_hard_cap:
                break
            if getattr(robot, "current_state", None) != "idle":
                continue
            if getattr(robot, "job", None) is not None and not getattr(robot.job, "is_finished", False):
                continue
            if robot.is_unavailable_for_work_due_to_charging():
                continue
            station = self.station_manager.find_available_replenish_station()
            if station is None:
                break
            requests = sorted(
                [
                    request
                    for request in self.pending_replenishment_dispatches
                    if request.get("source") == "proactive"
                ],
                key=lambda request: (
                    0 if self.should_guarantee_replenishment_request(request, current_tick) else 1,
                    -(current_tick - int(request.get("created_tick", current_tick))),
                    int(request.get("pod_id", 0)),
                ),
            )
            selected_request = None
            selected_pod = None
            selected_skus = None
            for request in requests:
                pod = self.pod_manager.get_pod_by_id(int(request["pod_id"]))
                if pod is None:
                    self.remove_pending_replenishment_dispatch(int(request["pod_id"]))
                    continue
                if getattr(pod, "is_awaiting_replenishment", False):
                    self.remove_pending_replenishment_dispatch(pod.pod_id)
                    continue
                if getattr(pod, "rts_return_in_progress", False):
                    continue
                if getattr(pod, "committed_next_owner_robot_id", None):
                    continue
                if not getattr(pod, "is_idle", False):
                    continue
                plan = self.evaluate_pod_replenishment_eligibility(pod)
                selected_request = request
                selected_pod = pod
                selected_skus = plan["trigger_skus"] or sorted(
                    {int(sku) for sku in request.get("skus_to_replenish", []) if sku in pod.skus}
                ) or sorted(pod.skus.keys())
                break
            if selected_request is None or selected_pod is None or not selected_skus:
                break
            idle_bypass = self.proactive_replenishment_robot_load() >= self.replenishment_soft_cap
            if self.send_pod_for_replenishment(
                selected_pod,
                station,
                selected_skus,
                robot=robot,
                source="proactive",
                idle_bypass=idle_bypass,
            ):
                dispatched += 1
        return dispatched

    def ensure_committed_next_reservation(self, robot: Robot):
        if not self.committed_next_reservations_enabled:
            return None
        registry = self.committed_next_registry
        if registry is None:
            registry = CommittedNextRegistry()
            self.committed_next_registry = registry
        return registry.reserve_for_robot(self, robot)

    def ensure_committed_next_action_proposals(
        self,
        robot: Robot,
        context,
        zone_ids,
        action_contexts=None,
        physical_contexts=None,
        candidate_storage_by_zone=None,
    ):
        if not self.committed_next_reservations_enabled:
            return {}
        registry = self.committed_next_registry
        if registry is None:
            registry = CommittedNextRegistry()
            self.committed_next_registry = registry
        return registry.build_action_proposals(
            self,
            robot,
            context,
            tuple(str(zone_id) for zone_id in zone_ids),
            action_contexts=action_contexts,
            physical_contexts=physical_contexts,
            candidate_storage_by_zone=candidate_storage_by_zone,
        )

    def commit_committed_next_decision(self, robot: Robot, decision):
        if not self.committed_next_reservations_enabled:
            return None
        registry = self.committed_next_registry
        if registry is None:
            registry = CommittedNextRegistry()
            self.committed_next_registry = registry
        return registry.commit_for_decision(self, robot, decision)

    def link_committed_next_decision(self, robot: Robot, decision_event_id) -> None:
        registry = getattr(self, "committed_next_registry", None)
        if registry is not None:
            registry.link_decision_event(robot, decision_event_id)

    def finalize_completed_return(self, job) -> None:
        """Finalize the old pod after a robot physically completes storage return.

        This must run BEFORE any committed-next activation (which replaces
        ``robot.job``) so that replacement never bypasses cleanup of the pod that
        was just returned. The pod is now physically at storage, so it is safe to
        clear ``pod.station``; membership in a station's incoming set is removed
        if it is somehow still present. Idempotent for the ordinary idle path.
        """
        if job is None:
            return
        pod = getattr(job, "pod", None)
        if pod is None:
            return
        # Remove stale station incoming membership (pod is no longer at a station).
        station = getattr(pod, "station", None)
        if station is not None:
            try:
                station.remove_pod(pod.pod_id)
            except Exception:
                pass
        try:
            source_station = self.station_manager.get_station_by_id(job.station_id)
        except Exception:
            source_station = None
        if source_station is not None and source_station is not station:
            try:
                source_station.remove_pod(pod.pod_id)
            except Exception:
                pass
        # Clear pod.station only now that the pod has physically returned.
        pod.remove_pod_station()
        # Mark the returned pod available so replacement of robot.job cannot leave
        # an unavailable, unowned pod behind.
        self.pod_manager.mark_pod_available(pod)

    def activate_committed_next_after_return(self, robot: Robot) -> bool:
        if not self.committed_next_reservations_enabled:
            return False
        registry = getattr(self, "committed_next_registry", None)
        reservation = registry.get_for_robot(robot) if registry is not None else None
        if reservation is None:
            pending_had_reservation = False
            tracker = getattr(getattr(self.rts_rollout_runtime, "tracker", None), "pending_by_robot_id", {})
            pending = tracker.get(str(getattr(robot, "_id", getattr(robot, "id", "")))) if tracker is not None else None
            if pending is not None and getattr(pending, "committed_next_reservation_id", None):
                pending_had_reservation = True
            if registry is not None:
                registry.clear_action_proposals_for_robot(robot)
            if pending_had_reservation:
                self.rts_rollout_runtime.censor_pending_for_robot(
                    robot=robot,
                    status="censored_committed_next_cancelled",
                    reason="committed_next_cancelled_before_activation",
                )
            return False
        activated = registry.activate_for_robot(self, robot)
        if activated is None:
            self.rts_rollout_runtime.censor_pending_for_robot(
                robot=robot,
                status="censored_committed_next_cancelled",
                reason=getattr(reservation, "cancellation_reason", None) or "committed_next_activation_failed",
            )
            return False
        self.rts_rollout_runtime.on_committed_next_activated(robot=robot, reservation=activated)
        return True

    def finalize_committed_next_run_end(self) -> None:
        runtime = getattr(self, "rts_rollout_runtime", None)
        if runtime is not None:
            runtime.censor_all_pending(status="censored_run_end", reason="run_end")
        registry = getattr(self, "committed_next_registry", None)
        if registry is not None:
            registry.cancel_all("run_end")

    def addObject(self, object):
        if object.object_type == "robot":
            object._id = self.total_pod + 1
            self.total_pod += 1
        super().addObject(object)

    def addTrafficPolicyHistory(self, sender, target):
        if target not in self.movement_channel:
            self.movement_channel[target] = []
        self.movement_channel[target].append(sender)

    def getTrafficPolicyHistory(self, target):
        if target not in self.movement_channel:
            return []
        return self.movement_channel[target]

    def tick(self):
        # Get initial state
        result = super().generateResult()
        
        print(f"Current tick: {self._tick}")

        # Reset movement tracking
        self.movement_channel = {}
        
        # Process orders at scheduled intervals
        if int(self._tick) == self.next_process_tick:
            print(f"Processing orders at tick {self._tick}")
            self.find_new_orders()
            self.process_orders()
            if self.update_intersection_using_RL:
                self.intersection_manager.update_allowed_direction_using_q_model(int(self._tick))

        for queued_job in list(self.job_queue):
            if getattr(queued_job, "committed_next_reservation_id", None):
                continue
            self.update_robot_job_for_new_orders(queued_job)

        print(f"Current job queue length: {len(self.job_queue)}")
        self.job_queue_cumulative_sum += len(self.job_queue)
        self.job_queue_sample_count += 1
        self.peak_job_queue = max(self.peak_job_queue, len(self.job_queue))

        idle_robots = [
            o for o in self.get_movable_objects()
            if o.object_type == "robot"
            and (o.job is None or o.job.is_finished)
            and o.current_state == 'idle'
            and not o.is_unavailable_for_work_due_to_charging()
        ]
        assigned_robot = None

        if len(self.job_queue) > 0:
            eligible_jobs = [
                (queue_index, job)
                for queue_index, job in enumerate(self.job_queue)
                if not getattr(getattr(job, "pod", None), "rts_return_in_progress", False)
                and not getattr(getattr(job, "pod", None), "committed_next_owner_robot_id", None)
                and not getattr(job, "committed_next_reservation_id", None)
            ]
            picking_jobs = [
                (queue_index, job)
                for queue_index, job in eligible_jobs
                if not getattr(job, "is_replenishment_job", False)
            ]
            replenishment_jobs = [
                (queue_index, job)
                for queue_index, job in eligible_jobs
                if getattr(job, "is_replenishment_job", False)
            ]
            allocatable_jobs = picking_jobs or replenishment_jobs
            allocation = None
            if allocatable_jobs and idle_robots:
                allocation = select_active_job_queue_assignment(
                    jobs=[job for _queue_index, job in allocatable_jobs],
                    robots=idle_robots,
                    cost_fn=lambda job, robot: calculateDistance(
                        robot.pos_x,
                        robot.pos_y,
                        job.pod_coordinate.x,
                        job.pod_coordinate.y,
                    ),
                    robot_task_allocator=self.robot_task_allocator,
                    regret_k=self.regret_k,
                    job_id_fn=lambda job: f"{getattr(job.pod, 'pod_id', job.pod)}:{job.station_id}",
                    robot_id_fn=lambda robot: getattr(robot, "_id", None),
                )

            if allocation is not None:
                job = allocation.job
                assigned_robot = allocation.robot
                del self.job_queue[allocatable_jobs[allocation.queue_index][0]]
                print(
                    f"Assigning job {job.pod}-{job.station_id} to robot {assigned_robot._id} "
                    f"using {allocation.allocator}"
                )
                assigned_robot.assign_job_and_set_move_to_take_pod(job)
                if not self.fast_train:
                    for triplet in job.orders:
                        upsert_job_task(
                            pod_id=str(job.pod.pod_id),
                            order_id=str(triplet[0]),
                            sku=str(triplet[1]),
                            qty=str(triplet[2]),
                            status="otw",
                            db_path=self.sqlite_db_path,
                        )
            
        remaining_idle_robots = [
            robot for robot in idle_robots
            if robot is not assigned_robot
            and getattr(robot, "current_state", None) == "idle"
            and (getattr(robot, "job", None) is None or getattr(robot.job, "is_finished", False))
        ]
        unmatched_idle_robots = self._unmatched_idle_robots_after_picking_allocation(remaining_idle_robots)
        if unmatched_idle_robots:
            self.dispatch_proactive_replenishment_to_unmatched_idle_robots(unmatched_idle_robots)

        # Update object positions and collect metrics
        total_energy = 0
        total_turning = 0
        total_idle = 0
        for o in self.get_movable_objects():
            if isinstance(o, Robot):
                initial_velocity = o.velocity
                o.move()
                total_energy += o.energy_consumption
                total_turning += o.turning
                total_idle += (o.total_idle * 0.15)
                if o.velocity == 0 and initial_velocity > 0:
                    self.stop_and_go += 1

                # Add newly assigned compatible orders to active pod jobs before
                # deciding whether the pod has finished station processing.
                if o.job is not None and not o.job.is_finished:
                    self.update_robot_job_for_new_orders(o.job)

                # Handle job completion and replenishment
                if self._job_ready_to_finish(o):
                    self.finish_task_in_job(o.job)
                    if not self.fast_train:
                        for triplet in o.job.orders:
                            update_job_task(
                                pod_id=str(o.job.pod.pod_id),
                                order_id=str(triplet[0]),
                                sku=str(triplet[1]),
                                qty=str(triplet[2]),
                                status="finish",
                                finish_time=self._tick,
                                db_path=self.sqlite_db_path,
                            )

                # Reset completed jobs
                if o.current_state == 'idle' and o.job is not None:
                    # self.pod_manager.mark_pod_available(o.job.pod_coordinate)
                    self.pod_manager.mark_pod_available(o.job.pod)
                    o.job = None
                
        # Update global metrics
        self.total_robot_idle = total_idle
        self.total_energy = total_energy
        self.total_turning = total_turning

        # Update process tick and intersection model
        if int(self._tick) == self.next_process_tick:
            self.next_process_tick += 1
            if self.update_intersection_using_RL:
                self.intersection_manager.update_model_after_execution(self._tick)

        # Increment tick
        self._tick += self.tick_to_second

        # Return updated state with station orders
        station_orders = self.get_station_orders_info()
        # with open('result.txt', 'a') as f:
        #     f.write(f"{result}")
        return [result, station_orders]

    def finish_task_in_job(self, job: RobotJob):
        job_station = self.station_manager.get_station_by_id(job.station_id)
        if job_station.is_picker_station():
            try:
                return self.finish_picking_task(job)
            except Exception as e:
                print(f"[ERROR] finish_picking_task for job {job.job_id}")
                print(f"[ERROR] for pod {job.pod} location {job.pod.coordinate}")
                raise e
        elif job_station.is_replenishment_station():
            try:
                return self.finish_replenishment_task(job)
            except Exception as e:
                print(f"[ERROR] finish_replenishment_task for job {job.job_id}")
                print(f"[ERROR] for pod {job.pod} location {job.pod.coordinate}")
                raise e
    
    def _get_order_by_id_flexible(self, order_id):
        order = self.order_manager.get_order_by_id(order_id)
        if order is not None:
            return order
        try:
            return self.order_manager.get_order_by_id(int(order_id))
        except (TypeError, ValueError):
            return None

    def finish_picking_task(self, job: RobotJob):
        # pod: Pod = self.pod_manager.get_pod_by_coordinate(job.pod_coordinate.x, job.pod_coordinate.y)
        pod: Pod = self.pod_manager.get_pod_by_id(job.pod.pod_id)
        pod_info_records = self._fast_pod_info_records if self.fast_train else None
        pod_info_df = None if self.fast_train else pd.read_csv(self.pod_info_csv)
        sku_need_replenished = []
        for order_id, sku, quantity in job.orders:
            order: Order = self._get_order_by_id_flexible(order_id)
            if order is None:
                print(
                    f"[WARN] skipping stale picking task: job={job.my_id} "
                    f"pod={pod.pod_id} order={order_id} sku={sku} qty={quantity}"
                )
                continue
            if not order.has_sku(sku):
                print(
                    f"[WARN] skipping picking task with unknown SKU: job={job.my_id} "
                    f"pod={pod.pod_id} order={order_id} sku={sku} qty={quantity}"
                )
                continue
            order.deliver_quantity(sku, quantity)
            print("order, sku, quantity :" ,order_id, sku, quantity)

            # Authoritative PPS picked-quantity accounting happens at successful
            # delivery (including dynamically added tasks), counting the actual
            # delivered quantity — not at job-creation time.
            self.pps_picked_quantity = getattr(self, "pps_picked_quantity", 0) + quantity

            # Check for SKU Replenishment
            # sku is sku_id (String)

            sku, replenished_status = self.pod_manager.is_sku_need_replenished(sku)

            # SKU Replenished Triggered
            if(replenished_status == True): sku_need_replenished.append(sku)
            # Mark the (order_id, SKU) rows finished in assign_order.csv only when
            # the aggregate delivered quantity has reached the aggregate required
            # quantity for that SKU. A single partial delivery must not flip every
            # row for the (order_id, SKU) to finished.
            sku_details = order.skus.get(sku, {})
            sku_fully_delivered = (
                sku_details.get("quantity_delivered", 0) >= sku_details.get("total_quantity", 0)
            )
            if sku_fully_delivered:
                assign_order_df = self._read_assign_order_csv()
                assign_order_df.loc[((assign_order_df['order_id'] == order.order_id) & (assign_order_df['item_id'] == sku)), 'status'] = 1
                assign_order_df.loc[((assign_order_df['order_id'] == order.order_id) & (assign_order_df['item_id'] == sku)), 'order_finished'] = int(self._tick)
                self._write_assign_order_csv(assign_order_df)
            new_row = {
                "pod_id": pod.pod_id,
                "item_id": sku,
                "qty": quantity,
                "order_id": order_id,
                "processed_time": int(self._tick),
                "task_type": 1
            }

            if self.fast_train:
                pod_info_records.append(new_row)
            else:
                new_row_df = pd.DataFrame([new_row])
                pod_info_df = pd.concat([pod_info_df, new_row_df], ignore_index=True)
            
            if order.is_order_completed():
                self.order_manager.finish_order(order_id, int(self._tick))
                station_id = order.station_id or job.station_id
                if order.station_id is None:
                    order.assign_station(station_id)
                station = self.station_manager.get_station_by_id(station_id)
                station.remove_order(order_id, order)
                self.insert_finished_order_to_csv(order)
                # DB
                # if not isinstance(order.order_id, int):
                    # raise AssertionError(f"WHAT? order {order} order_id {order.order_id} order_id {order_id}")
                if not self.fast_train:
                    upsert_order_history(order_id, order_finish_time=self._tick, db_path=self.sqlite_db_path)
        station = self.station_manager.get_station_by_id(job.station_id)
        station.remove_pod(pod.pod_id)
        
        if not self.fast_train:
            pod_info_df.to_csv(self.pod_info_csv, index=False)
        # Replenishment baseline
        # job.is_finished = True
        job.set_job_finish()
        if sku_need_replenished:
            self.global_critical_skus.update(sku_need_replenished)
        if not self.rts_controls_post_pick_replenishment:
            self.queue_post_pick_replenishment(pod, sku_need_replenished)
        return False
    
    def finish_replenishment_task(self, job: RobotJob):
        # pod: Pod = self.pod_manager.get_pod_by_coordinate(job.pod_coordinate.x, job.pod_coordinate.y)
        pod: Pod = self.pod_manager.get_pod_by_id(job.pod.pod_id)
        # Full-pod replenishment: every visit restores ALL SKU compartments on
        # the pod to their limit, regardless of the (diagnostic) trigger-SKU
        # list. Capture each SKU's old quantity, restore to limit, and apply the
        # exact positive per-SKU delta to the global inventory once (no double
        # counting). Pod mass is updated inside replenish_all_skus().
        restored_quantities = {}
        before_qty = {
            sku_id: details['current_qty']
            for sku_id, details in pod.skus.items()
        }
        pod.replenish_all_skus()
        replenished_count = 0
        for sku_id, old_qty in before_qty.items():
            new_qty = pod.skus[sku_id]['current_qty']
            restored_qty = max(0, new_qty - old_qty)
            if restored_qty > 0:
                restored_quantities[sku_id] = restored_qty
                self.pod_manager.increase_sku_data(sku_id, restored_qty)
                replenished_count += 1
        new_row = {
                "pod_id": pod.pod_id,
                "item_id": -1,
                "qty": -1,
                "order_id": -999,
                "processed_time": int(self._tick),
                "task_type": 2
            }

        if self.fast_train:
            self._fast_pod_info_records.append(new_row)
        else:
            pod_info_df = pd.read_csv(self.pod_info_csv)
            new_row_df = pd.DataFrame([new_row])
            pod_info_df = pd.concat([pod_info_df, new_row_df], ignore_index=True)
            pod_info_df.to_csv(self.pod_info_csv, index= False)
        # job.is_finished = True
        job.set_job_finish()
        station = self.station_manager.get_station_by_id(job.station_id)
        station.remove_pod(pod.pod_id)
        pod.is_awaiting_replenishment = False
        pod.has_pending_replenishment_dispatch = False
        pod.must_replenish_before_pick = False
        if getattr(job, "rts_continuation_active", False):
            job.rts_stage = "post_replenishment_to_storage"
        self.replenishment_trips += 1
        self.replenishment_count += replenished_count
        source = getattr(job, "replenishment_source", None) or "post_pick"
        if source == "rts":
            self.completed_rts_replenishment_actions += 1
        elif source == "proactive":
            self.completed_proactive_replenishment_actions += 1
        else:
            self.completed_post_pick_replenishment_actions += 1
        for sku_id in restored_quantities:
            _, still_below_reorder = self.pod_manager.is_sku_need_replenished(sku_id)
            if still_below_reorder:
                self.global_critical_skus.add(sku_id)
            else:
                self.global_critical_skus.discard(sku_id)
        return False

    def insert_finished_order_to_csv(self, order: Order):
        if self.fast_train:
            self._fast_finished_orders.append(order)
            return
        header = ["order_id", "order_arrival", "process_start_time", "order_complete_time", "station_id"]
        data = [order.order_id, order.order_arrival, order.process_start_time, order.order_complete_time,
                order.station_id]

        self.write_to_csv("order-finished.csv", header, data)

    def find_new_orders(self):
        file_path = self.assign_order_csv
        if os.path.exists(file_path):
            assign_order_df = self._read_assign_order_csv()
        else:
            orders_df = pd.read_csv(self.generated_order_csv)
            assign_order_df = orders_df.copy()
            assign_order_df['assigned_station'] = pd.Series([None] * len(assign_order_df), dtype="object")
            assign_order_df['assigned_pod'] = pd.Series([None] * len(assign_order_df), dtype="object")
            assign_order_df['status'] = -3
            self._write_assign_order_csv(assign_order_df)

        current_second = self.next_process_tick
        previous_second = (self.next_process_tick - 1)

        new_orders = assign_order_df[(assign_order_df['order_arrival']<= current_second) &
                               (assign_order_df['order_arrival'] > previous_second) &
                               (assign_order_df['status'] == -3)]
        grouped_orders = new_orders.groupby('order_id')

        for order_id, group in grouped_orders:
            order_items = group[['item_id', 'item_quantity']].to_dict('records')
            order = Order(order_id=order_id, order_arrival=current_second)

            # Add each item in the group to the order
            for item in order_items:
                order.add_sku(item['item_id'], item['item_quantity'])

            self.order_manager.add_order(order)
            # DB
            if not self.fast_train:
                upsert_order_history(order.order_id, arrival_time=self._tick, db_path=self.sqlite_db_path)

        return new_orders

    def get_movable_objects(self):
        result = []
        for o in self._objects:
            if o.object_type not in self.ignored_types or self._tick == 0:
                result.append(o)

        return result

    def process_orders(self):
        # Joint RL controls both POA and PPS externally — skip everything except
        # robot job init and order start-processing
        if self.joint_rl:
            # Still need to start processing timer for assigned orders
            if os.path.exists(self.assign_order_csv):
                for order in self.order_manager.unfinished_orders:
                    if order.station_id is None:
                        continue
                    if order.process_start_time <= 0:
                        order.start_processing(int(self._tick))
            return

        # Step 2: Trigger preassign logic
        if self.poa_first:
            advanced_table = self.get_advanced_table()

        # Step 3 & 4: POA — dispatch to the selected strategy (no order batching)
        if self.poa_aisyahna:
            # Aisyahna's method has its own timer-based batching
            if self._tick >= 1:
                self.aisyahna_poa()
        else:
            # No threshold batching — run POA whenever there is capacity and demand
            unassigned_count = sum(
                1 for o in self.order_manager.unfinished_orders
                if o.station_id is None and o.order_id not in self.order_manager.preassign_order_ids
            )
            total_empty_bin = self.get_total_empty_bin()
            if (
                unassigned_count > 0
                and sum(total_empty_bin.values()) >= 1
                and self._tick >= 1
            ):
                if self.poa_podmatch:
                    self.assign_order_old()
                if self.poa_first:
                    self.assign_order()
                if self.poa_second:
                    self.xxx()
        # Step 5: Record last order for each station
        if self.poa_first:
            for st in [v for k, v in self.station_manager.stations_by_id.items() if 'picker' in k]:
                self.last_order[st.station_id] = advanced_table.loc[advanced_table['station_id'] == st.station_id, 'order_id'].tolist()
            print(self.last_order)
        # Step 6: Start unfinished orders
        for order in self.order_manager.unfinished_orders:
            if order.station_id is None:
                continue
            if order.process_start_time <= 0:
                order.start_processing(int(self._tick))
        self.refresh_mandatory_replenishment_pods()
        self.run_proactive_replenishment_pass()
        # Step 7: Process PPS logic (skip when RL controls PPS)
        if self.pps_rl:
            self.dispatch_pending_replenishment_requests(prioritize_aged_only=False)
            return  # RL agent handles PPS externally via PPSEnv
        if self.pps_demand or self.pps_pileon:
            for station in filter(lambda s: s.station_type == 'picker' and len(s.incoming_pod) < 11, self.station_manager.stations):
                priority_orders, general_orders = {}, {}
                for order in station.orders:
                    remaining_skus = order.get_remaining_skus()
                    if 0 < len(remaining_skus) <= 2:
                        if self.priority_order:
                            priority_orders[order.order_id] = remaining_skus
                        general_orders[order.order_id] = remaining_skus
                    else:
                        general_orders[order.order_id] = remaining_skus
                print(f"[DEBUG] priority orders {priority_orders}")
                # Handle priority orders first
                if priority_orders:
                    pod_assigned = False
                    for order_id, remaining_skus in priority_orders.items():
                        # raise AssertionError(f"we have priority {priority_orders}")
                        idle_pods = {
                            pod
                            for pod in self.pod_manager.sku_to_pods.get(list(remaining_skus.keys())[0], [])
                            if self.pod_manager.is_idle(pod.pod_id)
                            and not getattr(pod, "is_awaiting_replenishment", False)
                            and not getattr(pod, "must_replenish_before_pick", False)
                        }
                        # for pod_id in [k for k, v in self.pod_manager.pod_idle.items() if v]:
                        for pod in idle_pods:
                            pod_id = pod.pod_id
                            print(f"[DEBUG] pod_id {pod_id} with")
                            # pod = self.pod_manager.get_pod_by_id(pod_id)
                            can_fulfill = any(
                                sku in pod.skus and pod.skus[sku]["current_qty"] >= qty
                                for sku, qty in remaining_skus.items()
                            )
                            print(f"[DEBUG] can fulfill {can_fulfill}")
                            if can_fulfill:
                                sku_to_quantity = {sku: qty for sku, qty in remaining_skus.items()}
                                sku_to_order_map = {sku: [(order_id, qty)] for sku, qty in remaining_skus.items()}
                                job = self.add_picking_task_after_pps(station, pod, sku_to_order_map, sku_to_quantity)
                                if len(job.orders) == 0:
                                    # Transactional PPS produced no task (no
                                    # usable stock): do not queue an empty job.
                                    continue
                                self.job_queue.append(job)
                                for triplet in job.orders:
                                    upsert_job_task(
                                        pod_id=str(job.pod.pod_id),
                                        order_id=str(triplet[0]),
                                        sku=str(triplet[1]),
                                        qty=str(triplet[2]),
                                        assigned_station=station.station_id,
                                        pod_assigned_time=self._tick,
                                        status="queue",
                                        db_path=self.sqlite_db_path,
                                    )
                                # write_record_to("record_record.csv", [f"{self._tick:.2f}", 'job_append', pod, pod.coordinate], ['Time', 'Event', 'Pod ID', 'Location'])
                                pod_assigned = True
                                break
                        if pod_assigned:
                            break
                    if pod_assigned:
                        continue  # skip general orders if priority already assigned

                # Process general orders
                sku_to_quantity, sku_to_order_map = defaultdict(int), defaultdict(list)
                for o_id, remaining_skus in general_orders.items():
                    for sku, qty in remaining_skus.items():
                        sku_to_quantity[sku] += qty
                        sku_to_order_map[sku].append((o_id, qty))

                if not sku_to_quantity:
                    print(f"skipping pod search for station {station.station_id}")
                    continue

                # Pod selection
                if self.pps_demand:
                    backlog_skus = defaultdict(int)
                    for o in filter(lambda o: o.station_id is None and o.order_id not in self.order_manager.preassign_order_ids, self.order_manager.unfinished_orders):
                        for sku, q in o.skus.items():
                            backlog_skus[sku] += q["total_quantity"]
                    pod, score = self.find_best_pod(backlog_skus, list(sku_to_quantity.keys()), mode="demand")
                else:
                    pod, score = self.find_best_pod(sku_to_quantity, list(sku_to_quantity.keys()), mode="pile_on")

                if not pod:
                    continue

                job = self.add_picking_task_after_pps(station, pod, sku_to_order_map, sku_to_quantity)
                if len(job.orders) > 0:
                    self.job_queue.append(job)
                    for triplet in job.orders:
                        upsert_job_task(
                            pod_id=str(job.pod.pod_id),
                            order_id=str(triplet[0]),
                            sku=str(triplet[1]),
                            qty=str(triplet[2]),
                            assigned_station=station.station_id,
                            pod_assigned_time=self._tick,
                            status="queue",
                            db_path=self.sqlite_db_path,
                        )
        self.dispatch_pending_replenishment_requests(prioritize_aged_only=False)

    # def process_orders(self):
    #     robots_location = []
    #     for o in self.get_movable_objects():
    #         if len(self.job_queue) > 0:
    #             job: RobotJob = self.job_queue[0]

    #             if o.object_type == "robot" and (o.job is None or o.job.is_finished) and o.current_state == 'idle':
    #                 robots_location.append([o.pos_x, o.pos_y])

    #     if self.poa_first:
    #         advanced_table = self.get_advanced_table()  # di dalam sini ada proses preassign


    #     # misal kamu mau tau total keseluruhan -> x = sum(self.get_total_empty_bin().values())
    #     total_empty_bin = self.get_total_empty_bin()
    #     if sum(total_empty_bin.values()) >= 1 and self._tick >=1:
    #         if self.poa_podmatch:
    #             self.assign_order_old()
    #         if self.poa_first:
    #             self.assign_order()
    #         if self.poa_second:
    #             self.xxx()

    #     if self.poa_first:
    #         picking_station = [v for k, v in self.station_manager.stations_by_id.items() if 'picker' in k]
    #         for st in picking_station:
    #             self.last_order[st.station_id] = advanced_table.loc[advanced_table['station_id'] == st.station_id, 'order_id'].tolist()
    #         print(self.last_order)
    #     for order in self.order_manager.unfinished_orders:
    #         assign_order_df = pd.read_csv('assign_order.csv')
    #         if order.station_id is None:
    #             continue

    #         # print(f"[DEBUG] order {order.order_id} is not None and keep running the rest")
    #         if order.process_start_time <= 0:
    #             # print(f"[DEBUG] start_process {order.order_id}")
    #             order.start_processing(int(self._tick))
                
    #         assign_order_df.to_csv('assign_order.csv', index=False)

    #     if self.pps_demand:
    #         print("pps_demand")
    #         for station in [st for st in self.station_manager.stations if st.station_type == 'picker']:
    #             print(f"incoming_pod {station.station_id} {station.incoming_pod}")
    #             if len(station.incoming_pod) < 11:
    #                 priority_order = {}
    #                 general_order = {}
    #                 for order in station.orders:
    #                     remaining_skus = order.get_remaining_skus()
    #                     if len(remaining_skus) <= 2:
    #                         # priority_order[order.order_id] = remaining_skus
    #                         general_order[order.order_id] = remaining_skus
    #                     else:
    #                         general_order[order.order_id] = remaining_skus
    #                     # update0617 print(f"order {order.order_id} has remaining skus {remaining_skus}")
    #                 sku_to_quantity = defaultdict(int)
    #                 sku_to_list_order_id_and_quantity = defaultdict(list)
    #                 for o_id, remaining_skus in general_order.items():
    #                     for sku, qty in remaining_skus.items():
    #                         sku_to_quantity[sku] += qty
    #                         sku_to_list_order_id_and_quantity[sku].append((o_id, qty))
    #                 print(f"for station {station.station_id} with sku_to_quantity {sku_to_quantity}")
    #                 print(f"sku in station {station.skus_in_station}")
    #                 if not sku_to_quantity:
    #                     print(f"skipping pod search for station {station.station_id}")
    #                     continue

    #                 ## PPS Demand
    #                 backlog_skus = defaultdict(int)
    #                 unassigned_orders = [order for order in self.order_manager.unfinished_orders if 
    #                                      (order.station_id is None and order.order_id not in self.order_manager.preassign_order_ids)]
    #                 for o in unassigned_orders:
    #                     for sku, q in o.skus.items():
    #                         backlog_skus[sku] += q["total_quantity"]
    #                 highest_demand_on_pod, demand_score = self.find_pod_with_the_highest_demand(backlog_skus, list(sku_to_quantity.keys()))
    #                 print("highest_demand_on_pod", highest_demand_on_pod, "demand_score", demand_score)
    #                 if not highest_demand_on_pod:
    #                     print("assign nothing")
    #                     continue

    #                 job = self.add_picking_task_after_pps(
    #                     station,
    #                     highest_demand_on_pod,
    #                     sku_to_list_order_id_and_quantity,
    #                     sku_to_quantity
    #                 )

    #                 if len(job.orders) > 0:
    #                     self.job_queue.append(job)
    #                     write_record_to("record_record.csv", [f"{self._tick:.2f}", 'job_append', job.pod, job.pod.coordinate], ['Time', 'Event', 'Pod ID', 'Location'])



    #     if self.pps_pileon:
    #         print("pps_pileon")
    #         for station in [st for st in self.station_manager.stations if st.station_type == 'picker']:
    #             print(f"incoming_pod {station.station_id} {station.incoming_pod}")
    #             if len(station.incoming_pod) < 11:
    #                 priority_order = {}
    #                 general_order = {}
    #                 for order in station.orders:
    #                     remaining_skus = order.get_remaining_skus()
    #                     if len(remaining_skus) <= 2:
    #                         priority_order[order.order_id] = remaining_skus
    #                         # general_order[order.order_id] = remaining_skus
    #                     else:
    #                         general_order[order.order_id] = remaining_skus
    #                     # update0617 print(f"order {order.order_id} has remaining skus {remaining_skus}")
    #                 # if priority order, then blabla
    #                 # if not
    #                 sku_to_quantity = defaultdict(int)
    #                 sku_to_list_order_id_and_quantity = defaultdict(list)
    #                 # print("general_order", general_order)
    #                 for o_id, remaining_skus in general_order.items():
    #                     for sku, qty in remaining_skus.items():
    #                         sku_to_quantity[sku] += qty
    #                         sku_to_list_order_id_and_quantity[sku].append((o_id, qty))
    #                 print(f"for station {station.station_id} with sku_to_quantity {sku_to_quantity}")
    #                 print(f"sku in station {station.skus_in_station}")
    #                 if not sku_to_quantity:
    #                     print(f"skipping pod search for station {station.station_id}")
    #                     continue
    #                 ## PPS Pile On
    #                 highest_pile_on_pod, pile_on_score = self.find_pod_with_the_highest_pile_on(sku_to_quantity)
    #                 print("highest_pile_on_pod", highest_pile_on_pod, "pile_on_score", pile_on_score)
                    
    #                 ## try to fix teleport
    #                 # highest_pile_on_pod = self.pod_manager.get_pod_by_id(highest_pile_on_pod.pod_id)
    #                 job = self.add_picking_task_after_pps(
    #                     station,
    #                     highest_pile_on_pod,
    #                     sku_to_list_order_id_and_quantity,
    #                     sku_to_quantity
    #                 )

    #                 if len(job.orders) > 0:
    #                     self.job_queue.append(job)
    #                     for triplet in job.orders:
    #                         upsert_job_task(
    #                             pod_id=str(job.pod.pod_id),
    #                             order_id=str(triplet[0]),
    #                             sku=str(triplet[1]),
    #                             qty=str(triplet[2]),
    #                             assigned_station=station.station_id,
    #                             pod_assigned_time=self._tick,
    #                             status="queue",
    #                         )
    #                     write_record_to("record_record.csv", [f"{self._tick:.2f}", 'job_append', job.pod, job.pod.coordinate], ['Time', 'Event', 'Pod ID', 'Location'])


    #                 # TODO: order.commit_quantity (to tell that that order remaining skus are decreased)
    #                 # TODO: pod.pick_sku(sku, quantity_to_take) (reduct the item inside pod)
    #                 # TODO: self.pod_manager.reduct_sku_data(sku, quantity_to_take) (reduct item in global stock list)
    #                 # TODO: station.add_pod(pod.pod_id)
    #                 # TODO: pod.station = station
    #                 # TODO: update assign_order_df or whatever...
    #                 # TODO: self.pod_manager.mark_pod_not_available(pod.coordinate) (set the pod to not idle)
    #                 # TODO: station.reduce_sku_from_station(sku, quantity_to_take) (reduce the remaining skus in station)
    #                 # TODO: job.add_picking_task(order.order_id, sku, quantity_to_take) (the picking task list inside robotjob)
                    
    #                 # TODO: self.job_queue.append(job)
    def find_best_pod(
        self, 
        sku_to_quantity: dict, 
        relevant_skus: list, 
        mode: str = "pile_on"  # or "demand"
    ):  # type: ignore
        from src.rmfs.decisions.pps.heuristic import find_best_pod
        return find_best_pod(self, sku_to_quantity, relevant_skus, mode)

    def add_picking_task_after_pps(self, station: Station, pod: Pod, sku_to_list_order_id_and_quantity: dict, sku_to_quantity: dict):
        latest_pod_location = get_pod_location(pod.pod_id, db_path=self.sqlite_db_path)
        if latest_pod_location:
            pod.pos_x, pod.pos_y = latest_pod_location
        job = RobotJob(pod.coordinate, station_id=station.station_id, pod=pod)

        # --- Phase 1: inspect and construct the task plan WITHOUT mutating any
        # order, pod, station, or global counter. Discard zero-quantity and
        # missing-order entries as we build.
        plan_entries = []          # list of (order, sku, qty) to apply
        per_sku_take = {}          # sku -> total quantity to remove from pod/global
        for sku in sku_to_list_order_id_and_quantity:
            # sort based on the least quantity for each sku
            sku_to_list_order_id_and_quantity[sku] = sorted(sku_to_list_order_id_and_quantity[sku], key=lambda x: x[1])
            if sku not in pod.skus:
                continue
            available = pod.get_quantity(sku)
            requested = sku_to_quantity.get(sku, 0)
            quantity_to_take = requested if available >= requested else available
            if quantity_to_take <= 0:
                continue

            tmp = quantity_to_take
            allocated = 0
            for o_id, qty in sku_to_list_order_id_and_quantity[sku]:
                if tmp <= 0:
                    break
                order = self._get_order_by_id_flexible(o_id)
                if order is None:
                    print(
                        f"[WARN] skipping PPS task for missing order: "
                        f"station={station.station_id} pod={pod.pod_id} order={o_id} sku={sku}"
                    )
                    continue
                take = min(qty, tmp)
                if take <= 0:
                    continue
                plan_entries.append((order, sku, take))
                allocated += take
                tmp -= take
            if allocated > 0:
                per_sku_take[sku] = allocated

        # --- Phase 2: verify a non-empty, positive-quantity plan was produced.
        total_planned = sum(take for _order, _sku, take in plan_entries)
        if total_planned <= 0:
            # No task produced: return an empty job and leave every order, pod,
            # station, and global counter unchanged.
            return job

        # --- Phase 3: apply all commitments and stock mutations atomically.
        for order, sku, take in plan_entries:
            if order.station_id is None:
                order.assign_station(station.station_id)
            order.commit_quantity(sku, take)
            job.add_picking_task(order.order_id, sku, take)
        for sku, take in per_sku_take.items():
            pod.pick_sku(sku, take)
            self.pod_manager.reduce_sku_data(sku, take)
            station.reduce_sku_from_station(sku, take)

        # --- Phase 4: reserve the pod and return the non-empty job.
        station.add_pod(pod.pod_id)
        pod.station = station
        self.pod_manager.mark_pod_not_available(pod)
        # pps_picked_quantity is now counted authoritatively at delivery time;
        # pps_pod_visits still counts one visit per non-empty PPS assignment.
        self.pps_pod_visits = getattr(self, "pps_pod_visits", 0) + 1
        return job

    def find_pod_with_the_highest_pile_on(self, sku_to_quantity: dict) -> (Pod, int): # type: ignore
        from src.rmfs.decisions.pps.heuristic import find_pod_with_the_highest_pile_on
        return find_pod_with_the_highest_pile_on(self, sku_to_quantity)
    
    def find_pod_with_the_highest_demand(self, sku_to_quantity: dict, station_unfinished_skus: list) -> (Pod, int): # type: ignore
        from src.rmfs.decisions.pps.heuristic import find_pod_with_the_highest_demand
        return find_pod_with_the_highest_demand(self, sku_to_quantity, station_unfinished_skus)

    def write_to_csv(self, filename, header, data):
        if self.fast_train:
            return
        # Use CWD-relative 'output/' so SubprocVecEnv workers each have their own.
        folder_path = os.path.join(os.getcwd(), 'output')
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        filename = os.path.join(folder_path, filename)
        file_exists = os.path.exists(filename)

        with open(filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(header)
            writer.writerow(data)

    def get_station_orders_info(self):
        station_orders = []
        for station in sorted(self.station_manager.stations, key=lambda x: x.station_id):
            if station.is_picker_station():
                order_list = ', '.join(map(str, station.order_ids)) if station.order_ids else "Empty"
                station_orders.append(order_list)
        while len(station_orders) < 3:
            station_orders.append("Empty")
        return station_orders

    def generateResult(self):
        result = super().generateResult()
        station_orders = self.get_station_orders_info()
        return [result, station_orders]

    def get_fulfilment_table(self, mode="FS", excludes=[]):
        # MODE: 
        # FS=fully supplied 
        # OTW= only on incoming pod - pod in stations
        # F3=incoming pod - 3 queue

        # station_ids = [station.station_id for station in self.station_manager.stations]
        # order_ids = [order.order_id for order in self.order_manager.unfinished_orders]

        picking_stations = [station for station in self.station_manager.stations if station.station_type == "picker"]

        # Gather all assigned order IDs across all stations
        # assigned_order_ids = {order_id for station in self.station_manager.stations for order_id in station.order_ids}

        # Filter orders whose order_id is NOT in assigned_order_ids
        # unassigned_orders = [order for order in self.order_manager.unfinished_orders if order.order_id not in assigned_order_ids]
        # unassigned_orders = [order for order in self.order_manager.unfinished_orders if (order.station_id is None)]
        ## Activate this only if you use advanced table
        unassigned_orders = [order for order in self.order_manager.unfinished_orders if (order.station_id is None and order.order_id not in self.order_manager.preassign_order_ids)]
        
        picking_station_ids = [station.station_id for station in picking_stations]
        unassigned_order_ids = [order.order_id for order in unassigned_orders]

        # Initialize fulfillment matrix with zeros
        fulfilment_matrix = pd.DataFrame(0.0, index=unassigned_order_ids, columns=picking_station_ids)

        for station in picking_stations:
            # Build station SKU availability from incoming pods
            station_sku_quantity = {}
            if mode == "FS":
                list_of_pods = station.incoming_pod
            elif mode == "OTW":
                # TODO: incoming pods - is in station
                robots_otw = [o for o in self.get_movable_objects() if 
                          o.object_type == "robot" 
                          and o.job 
                          and o.job.pod.pod_id in station.incoming_pod
                          and not o.is_in_station_path()]
                list_of_pods = [o.job.pod.pod_id for o in robots_otw]
            elif mode == "F3":
                # TODO: incoming pods - self.robot_queue_order
                robots_otw = [o for o in self.get_movable_objects() if 
                          o.object_type == "robot" 
                          and o.job 
                          and o.job.pod.pod_id in station.incoming_pod
                          and o not in excludes]
                list_of_pods = [o.job.pod.pod_id for o in robots_otw]
            for pod_id in list_of_pods:
                pod = self.pod_manager.get_pod_by_id(pod_id)
                for sku, details in pod.skus.items():
                    if details['current_qty'] > 0:
                        station_sku_quantity[sku] = station_sku_quantity.get(sku, 0) + details['current_qty']

            for order in unassigned_orders:
                total_order_qty = sum([x.get('total_quantity') for x in order.skus.values()])
                fulfilled_qty = 0
                for sku, val in order.skus.items():
                    available_qty = station_sku_quantity.get(sku, 0)
                    fulfilled_qty += min(val.get('total_quantity'), available_qty)

                fulfillment_rate = fulfilled_qty / total_order_qty if total_order_qty > 0 else 0.0
                fulfilment_matrix.at[order.order_id, station.station_id] = fulfillment_rate

        return fulfilment_matrix

    def get_total_empty_bin(self):
        bin_dict = {}
        picking_stations = [station for station in self.station_manager.stations if station.station_type == "picker"]
        for station in picking_stations:
            bin_dict[station.station_id] = station.max_orders - len(station.order_ids)
        return bin_dict
    
    def assign_order_with_advanced_table(self, df):
        empty_bin_dict = self.get_total_empty_bin()
        final_selection = {}
        for picker, count in empty_bin_dict.items():
            # for the order_id list in df, if it's not in station
            # self.last_order[picker]
            # self.station_manager.get_station_by_id(picker).order_ids
            try:
                print(f"count: {count}")
                list_of_just_finished_order = [
                    oid for oid in self.last_order[picker] if oid not in df.loc[df['station_id'] == picker, 'order_id'].tolist()
                ]
                print(f"list_of_just_finished_order: {list_of_just_finished_order}")
                list_of_pre_assigned_order = [
                    self.preassign_dict[k] for k in list_of_just_finished_order
                ]
                print(f"preassign_dict: {self.preassign_dict}")
                print(f"list_of_pre_assigned_order: {list_of_pre_assigned_order}")
                if len(list_of_pre_assigned_order) != count:
                    return self.assign_order()
                for n in list_of_pre_assigned_order:
                    final_selection.setdefault(picker, []).append(n)
            except Exception as e:
                print(f"exception with e: {type(e).__name__} {e}")
                return self.assign_order()
            # candidates = df.loc[df['station_id'] == picker, 'pre_assign'].tolist()
            # candidates = [c for c in candidates if c]
            # if not candidates:
            #     return self.assign_order()
                
            # for n in range(count):
            #     final_selection.setdefault(picker, []).append(candidates[0])
            #     del candidates[0]
        print(f"final_selection: {final_selection}")
        self.put_order_to_picking_station(final_selection)

        return final_selection

    def assign_order(self):
        # TODO: for advanced table version, just use the preassign as the assign
        fulfilment_table = self.get_fulfilment_table(mode="OTW")
        empty_bin_dict = self.get_total_empty_bin()

        print("assign_order is triggered")
        print(fulfilment_table)
        print(empty_bin_dict)

        # Step 1: Flatten all candidate values with their source column
        candidates = []
        for picker, count in empty_bin_dict.items():
            top_rows = fulfilment_table[picker].sort_values(ascending=False).head(count * 3)  # get more to allow fallback if conflict
            for index, value in top_rows.items():
                candidates.append({'index': index, 'picker': picker, 'value': value})

        # Step 2: Sort all candidates by value descending
        candidates = sorted(candidates, key=lambda x: x['value'], reverse=True)

        # Step 3: Pick best combination without duplicate indices
        final_selection = {}
        used_indices = set()

        for candidate in candidates:
            picker = candidate['picker']
            index = candidate['index']

            if index in used_indices:
                continue
            if empty_bin_dict[picker] > 0:
                final_selection.setdefault(picker, []).append(index)
                empty_bin_dict[picker] -= 1
                used_indices.add(index)

            # Stop early if all picks are satisfied
            if all(v == 0 for v in empty_bin_dict.values()):
                break
        
        print("Final selection:", final_selection)

        # Put the decision into action
        self.put_order_to_picking_station(final_selection)

        return final_selection
    
    def assign_order_old(self): # buat yang baseline
        fulfilment_table = self.get_fulfilment_table("FS")
        empty_bin_dict = self.get_total_empty_bin()

        print("assign_order is triggered")
        print(fulfilment_table)
        print(empty_bin_dict)

        # Step 1: Flatten all candidate values with their source column
        candidates = []
        for picker, count in empty_bin_dict.items():
            top_rows = fulfilment_table[picker].sort_values(ascending=False).head(count * 3)  # get more to allow fallback if conflict
            for index, value in top_rows.items():
                candidates.append({'index': index, 'picker': picker, 'value': value})

        # Step 2: Sort all candidates by value descending
        candidates = sorted(candidates, key=lambda x: x['value'], reverse=True)

        # Step 3: Pick best combination without duplicate indices
        final_selection = {}
        used_indices = set()

        for candidate in candidates:
            picker = candidate['picker']
            index = candidate['index']

            if index in used_indices:
                continue
            if empty_bin_dict[picker] > 0:
                final_selection.setdefault(picker, []).append(index)
                empty_bin_dict[picker] -= 1
                used_indices.add(index)

            # Stop early if all picks are satisfied
            if all(v == 0 for v in empty_bin_dict.values()):
                break
        
        print("Final selection:", final_selection)

        # Put the decision into action
        self.put_order_to_picking_station(final_selection)

        return final_selection
        
    def put_order_to_picking_station(self, final_selection):
        assign_order_df = pd.read_csv(self.assign_order_csv)
        assign_order_df['assigned_station'] = assign_order_df['assigned_station'].astype("object")
        assign_order_df['assigned_pod'] = assign_order_df['assigned_pod'].astype("object")

        for picker_name, order_ids in final_selection.items():
            for order_id in order_ids:
                order = self.order_manager.get_order_by_id(order_id)
                order.assign_station(picker_name)
                self.station_manager.get_station_by_id(picker_name).add_order(order_id, order)

                assign_order_df.loc[assign_order_df['order_id'] == order.order_id, 'assigned_station'] = picker_name
                assign_order_df.loc[assign_order_df['order_id'] == order.order_id, 'status'] = -1
                # DB
                if not self.fast_train:
                    upsert_order_history(
                        order_id,
                        assigned_station=picker_name,
                        order_assigned_time=self._tick,
                        db_path=self.sqlite_db_path,
                    )
            
        assign_order_df.to_csv(self.assign_order_csv, index=False)

    @staticmethod
    def _calculate_two_coordinates(p1, p2):
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
    
    def distance_robot_to_station(self, robot: Robot, station: Station):
        return self._calculate_two_coordinates((robot.pos_x, robot.pos_y), (station.coordinate.x, station.coordinate.y))

    def sort_pod_order(self, robots: List[Robot], station: Station):
        print("inside sort_pod_order")
        print(f"robot inside: {robots}")
        others = []
        current = None
        one_right = None
        two_right = None
        one_up = None
        two_up = None
        one, two, three = None, None, None
        station_coordinate = station.coordinate
        for r in robots:
            if r.pos_x == station_coordinate.x and r.pos_y == station_coordinate.y:
                current = r
            elif r.pos_x == station_coordinate.x + 1 and r.pos_y == station_coordinate.y:
                one_right = r
            elif math.floor(r.pos_x) == station_coordinate.x + 2 and r.pos_y == station_coordinate.y:
                two_right = r
            elif r.pos_x == station_coordinate.x and math.floor(r.pos_y) == station_coordinate.y + 1:
                one_up = r
            elif r.pos_x == station_coordinate.x and math.floor(r.pos_y) == station_coordinate.y + 2:
                two_up = r
            else:
                others.append(
                    (r, self.distance_robot_to_station(r, station))
                )
        print(f"total others: {others}")
        # TODO: sort others according to the distance
        if current:
            one = current
            if one_up and not one_right:
                two = one_up
                if two_up:
                    three = two_up
                else:
                    three = others[0][0]
                return (one, two, three)
            elif one_right:
                two = one_right
                if two_right:
                    three = two_right
                else:
                    three = others[0][0]
                return (one, two, three)
            else:
                if two_up:
                    two = two_up
                    three = others[0][0]
                else:
                    two = others[0][0]
                    three = others[1][0]
                return (one, two, three)
        else:
            return (None, None, None)

    def get_advanced_table_only(self):
        picking_stations = [station for station in self.station_manager.stations if station.station_type == "picker"]
        if not self.robot_queue_order:
            self.robot_queue_order = {picker.station_id: [] for picker in picking_stations}

        df_dicts = []
        for picker in picking_stations:
            station_id = picker.station_id
            robots_otw = [o for o in self.get_movable_objects() if 
                          o.object_type == "robot" 
                          and o.job 
                          and o.job.pod.pod_id in picker.incoming_pod]
            
            inside_station = []
            currently_picking = None
            for r in robots_otw:
                if r.is_being_process_on_station():
                    currently_picking = r
                
                if r.is_in_station_path():
                    inside_station.append(r)
                    if r.id not in self.robot_queue_order.get(station_id, []):
                        # print(f"adding robot queue {r} to {station_id}")
                        self.robot_queue_order[station_id].append(r.id)
                    print(f"{r} is_in_station_path {station_id}")

            for rid in self.robot_queue_order[station_id]:
                if rid not in [x.id for x in inside_station]:
                    # print(f"removing robot queue {r} from {station_id}")
                    self.robot_queue_order[station_id].remove(rid)
            print(f"current robot inside station {station_id}: {self.robot_queue_order[station_id]}")
            my_robot_queue_order = [None] * len(self.robot_queue_order[station_id])
            for n, rid in enumerate(self.robot_queue_order[station_id]):
                my_robot_queue_order[n] = [o for o in self.get_movable_objects() if o.object_type == "robot" and o.id == rid]
                my_robot_queue_order[n] = my_robot_queue_order[n][0] if my_robot_queue_order[n] else None
            if len(my_robot_queue_order) >= 3:
                first_queue = my_robot_queue_order[0]
                second_queue = my_robot_queue_order[1]
                third_queue = my_robot_queue_order[2]
            elif len(my_robot_queue_order) == 2:
                first_queue = my_robot_queue_order[0]
                second_queue = my_robot_queue_order[1]
                third_queue = None
            elif len(my_robot_queue_order) == 1:
                first_queue = my_robot_queue_order[0]
                second_queue = None
                third_queue = None
            else:
                first_queue = None
                second_queue = None
                third_queue = None
            for order in picker.orders:
                order_id = order.order_id
                unpicked_skus = order.get_unpicked_skus()
                df_dicts.append({
                    "station_id": station_id,
                    "order_id": order_id,
                    "unpicked_skus": self._normalize_sku_qty_dict(unpicked_skus),
                    # "robot_inside_station": self.robot_queue_order[station_id],
                    "pod_1": first_queue,
                    "pod_2": second_queue,
                    "pod_3": third_queue,
                    "occupied_1": first_queue.job.orders if (first_queue and first_queue.job) else None,
                    "occupied_2": second_queue.job.orders if (second_queue and second_queue.job) else None,
                    "occupied_3": third_queue.job.orders if (third_queue and third_queue.job) else None,
                    "next_bin_avail": None,
                    "pre_assign": self.preassign_dict.get(order_id, None)
                })
        if not df_dicts:
            df = pd.DataFrame(columns=[
                "station_id", "order_id", "unpicked_skus", "pod_1", "pod_2", "pod_3",
                "occupied_1", "occupied_2", "occupied_3", "next_bin_avail", "pre_assign"
            ])
        else:
            df = pd.DataFrame(df_dicts)
        df = self.forcast_next_bin_avail(df)
        return df

    def get_advanced_table(self):
        picking_stations = [station for station in self.station_manager.stations if station.station_type == "picker"]
        if not self.robot_queue_order:
            self.robot_queue_order = {picker.station_id: [] for picker in picking_stations}
        # if not self.currently_picking:
        #     self.currently_picking = {picker.station_id: None for picker in picking_stations}
        # print(f"PICKING STATION {picking_stations[0].station_id}")
        # print(f"ORDER IDS {picking_stations[0].order_ids}")
        # for o in picking_stations[0].orders:
        #     print(f"ORDER {o.order_id} has")
        #     print(f"SKUS {o.skus}")
        #     print(f"GET_REMAINING_SKUS {o.get_remaining_skus()}")

        # print(f"PICKING STATION {picking_stations[0].station_id} {picking_stations[0].coordinate}")
        # print(f"INCOMING POD {picking_stations[0].incoming_pod}")
        # print(f"JOB_RUNNING")
        # robots_otw_picking_station = [o for o in self.get_movable_objects() if o.object_type == "robot" and o.job and o.job.pod.pod_id in picking_stations[0].incoming_pod]
        # for r in robots_otw_picking_station:
        #     print(f"sending {r.job.pod.pod_id} to {r.job.station_id} status {r.current_state}")
        # print(f"CURRENTLY PICKING")
        # currently_picking = [r for r in robots_otw_picking_station if r.is_being_process_on_station()]
        # for r in currently_picking:
        #     print(f"sending {r.job.pod.pod_id} to {r.job.station_id} status {r.current_state} {r.pos_x:.2f},{r.pos_y:.2f}")
        # print(f"JOB_QUEUE")
        # for rj in self.job_queue:
        #     print(rj)

        df_dicts = []
        for picker in picking_stations:
            station_id = picker.station_id
            # if self.currently_picking[station_id]:
            #     print(f">>> CURRENT STATUS in {station_id}<<<")
            #     my_robot = [o for o in self.get_movable_objects() if o.object_type == "robot" and o.id == self.currently_picking[station_id]]
            #     my_robot = my_robot[0] if my_robot else None
            #     print(f"{my_robot}")
            #     print(f"ID: {self.currently_picking[station_id]}")
            #     print(f"picking delay {my_robot.job.picking_delay}")
            #     print(f"state {my_robot.current_state}")
            robots_otw = [o for o in self.get_movable_objects() if 
                          o.object_type == "robot" 
                          and o.job 
                          and o.job.pod.pod_id in picker.incoming_pod]
            
            inside_station = []
            currently_picking = None
            for r in robots_otw:
                if r.is_being_process_on_station():
                    currently_picking = r
                    # self.currently_picking[station_id] = r.id
                    # print(f"{r} is_being_process {station_id}")
                    # print(f"ID: {r.id}")
                    # print(f"picking delay {r.job.picking_delay}")
                    # print(f"state {r.current_state}")
                    # print(f"location: {r.pos_x}, {r.pos_y}")
                    # print(f"station location: {picker.coordinate}")
                
                if r.is_in_station_path():
                    inside_station.append(r)
                    if r.id not in self.robot_queue_order.get(station_id, []):
                        # print(f"adding robot queue {r} to {station_id}")
                        self.robot_queue_order[station_id].append(r.id)
                    print(f"{r} is_in_station_path {station_id}")

            for rid in self.robot_queue_order[station_id]:
                if rid not in [x.id for x in inside_station]:
                    # print(f"removing robot queue {r} from {station_id}")
                    self.robot_queue_order[station_id].remove(rid)
            print(f"current robot inside station {station_id}: {self.robot_queue_order[station_id]}")
            my_robot_queue_order = [None] * len(self.robot_queue_order[station_id])
            for n, rid in enumerate(self.robot_queue_order[station_id]):
                my_robot_queue_order[n] = [o for o in self.get_movable_objects() if o.object_type == "robot" and o.id == rid]
                my_robot_queue_order[n] = my_robot_queue_order[n][0] if my_robot_queue_order[n] else None
            if len(my_robot_queue_order) >= 3:
                first_queue = my_robot_queue_order[0]
                second_queue = my_robot_queue_order[1]
                third_queue = my_robot_queue_order[2]
            elif len(my_robot_queue_order) == 2:
                first_queue = my_robot_queue_order[0]
                second_queue = my_robot_queue_order[1]
                third_queue = None
            elif len(my_robot_queue_order) == 1:
                first_queue = my_robot_queue_order[0]
                second_queue = None
                third_queue = None
            else:
                first_queue = None
                second_queue = None
                third_queue = None
            for order in picker.orders:
                order_id = order.order_id
                unpicked_skus = order.get_unpicked_skus()
                df_dicts.append({
                    "station_id": station_id,
                    "order_id": order_id,
                    "unpicked_skus": self._normalize_sku_qty_dict(unpicked_skus),
                    # "robot_inside_station": self.robot_queue_order[station_id],
                    "pod_1": first_queue,
                    "pod_2": second_queue,
                    "pod_3": third_queue,
                    "occupied_1": first_queue.job.orders if (first_queue and first_queue.job) else None,
                    "occupied_2": second_queue.job.orders if (second_queue and second_queue.job) else None,
                    "occupied_3": third_queue.job.orders if (third_queue and third_queue.job) else None,
                    "next_bin_avail": None,
                    "pre_assign": self.preassign_dict.get(order_id, None)
                })
        if not df_dicts:
            df = pd.DataFrame(columns=[
                "station_id", "order_id", "unpicked_skus", "pod_1", "pod_2", "pod_3",
                "occupied_1", "occupied_2", "occupied_3", "next_bin_avail", "pre_assign"
            ])
        else:
            df = pd.DataFrame(df_dicts)
        df = self.forcast_next_bin_avail(df)
        df = self.pre_assign_order(df)
        print(df)
        return df

    def forcast_next_bin_avail(self, df):
        if df.empty:
            return df
        def parse_sku_qty_dict(value):
            if isinstance(value, dict):
                return self._normalize_sku_qty_dict(value)
            if value is None or (isinstance(value, float) and pd.isna(value)):
                return {}
            if isinstance(value, str):
                try:
                    return self._normalize_sku_qty_dict(ast.literal_eval(value))
                except (ValueError, SyntaxError):
                    cleaned = re.sub(
                        r"(?:np\.)?(?:int64|int32|int16|int8)\((-?\d+)\)",
                        r"\1",
                        value,
                    )
                    cleaned = re.sub(
                        r"(?:np\.)?(?:float64|float32)\((-?\d+(?:\.\d+)?)\)",
                        r"\1",
                        cleaned,
                    )
                    try:
                        return self._normalize_sku_qty_dict(ast.literal_eval(cleaned))
                    except (ValueError, SyntaxError):
                        print(f"[WARN] Could not parse unpicked_skus value: {value}")
                        return {}
            return {}

        def is_fulfilled(row):
            required = row['unpicked_skus']
            # Flatten all occupied bins into a list
            all_occupied = []
            for col in ['occupied_1', 'occupied_2', 'occupied_3']:
                val = row.get(col)
                if isinstance(val, list):
                    all_occupied.extend(val)
            
            # Sum quantities by SKU
            available = defaultdict(int)
            for _, sku, qty in all_occupied:
                available[sku] += qty

            # Check if every required SKU has enough quantity
            for sku, req_qty in required.items():
                if available[sku] < req_qty:
                    return False
            return True
        df['unpicked_skus'] = df['unpicked_skus'].apply(parse_sku_qty_dict)
        df['next_bin_avail'] = df.apply(is_fulfilled, axis=1)
        return df
    
    def pre_assign_order(self, df):
        """
        For each row where 'next_bin_avail' is True and 'pre_assign' is None,
        call choose_order with (station_id, pod_1, pod_2, pod_3),
        and store the result in 'pre_assign'.
        """
        if df.empty:
            return df
        print("PREASSIGN IS CALLED !!!!!!!")
        mask = (df['next_bin_avail'] == True) & (df['pre_assign'].isna())  # noqa: E712
        print(mask)
        df.loc[mask, 'pre_assign'] = df[mask].apply(
            lambda row: self.choose_order(row['station_id'], row['order_id'], row['pod_1'], row['pod_2'], row['pod_3']),
            axis=1
        )
        return df

    # Example implementation
    def choose_order(self, station_id: str, order_id: int, pod_1: Pod, pod_2: Pod, pod_3: Pod):
        print("CHOOSE ORDER IS CALLED !!!!!")
        df = self.get_fulfilment_table(mode="F3", excludes=[pod_1, pod_2, pod_3])
        print("\n\nfulfillment table during pre-assignment")
        print(df)
        print("\n\n")
        if df.empty:
            print("[PRE-ASSIGN FULFULLMENT TABLE IS INVALID!]")
            print(f"for station {station_id}")
            print(f"for order_id {order_id}")
            unassigned_orders = [order for order in self.order_manager.unfinished_orders if (order.station_id is None and order.order_id not in self.order_manager.preassign_order_ids)]
            unassigned_order_ids = [order.order_id for order in unassigned_orders]
            print(f"unassigned_order_ids {unassigned_order_ids}")
            print(f"pod queue {pod_1.pod_id}, {pod_2.pod_id}, {pod_3.pod_id}")
        df.sort_values(by=station_id, ascending=False, inplace=True)
        val = df.index[0]
        self.order_manager.preassign_order_ids.append(val)
        self.preassign_dict[order_id] = int(val)
        print(f" VALUE {val} {df.iloc[0]}")
        return df.index[0]

    # ------------------------------------------------------------------ #
    #  Aisyahna's Similarity-Based Order Batching + POA                    #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _jaccard_similarity(skus_a: set, skus_b: set) -> float:
        """Sim(A,B) = (shared / |A|) * (shared / |B|)"""
        if not skus_a or not skus_b:
            return 0.0
        shared = len(skus_a & skus_b)
        return (shared / len(skus_a)) * (shared / len(skus_b))

    def aisyahna_poa(self):
        """Full similarity-based order batching and POA (Steps 2-9)."""

        # -- Step 2: Check if batching timer should fire --
        if self._tick - self._aisyahna_last_batch_tick < self.aisyahna_batch_interval:
            return  # not time yet
        self._aisyahna_last_batch_tick = self._tick

        # -- Identify available stations and free bins --
        picking_stations = [
            s for s in self.station_manager.stations if s.station_type == 'picker'
        ]
        free_bins = {}  # station_id -> int
        for st in picking_stations:
            fb = st.max_orders - len(st.order_ids)
            if fb > 0:
                free_bins[st.station_id] = fb

        if not free_bins:
            return  # no station has a free bin

        total_assignable = sum(free_bins.values())

        # -- Step 3: Take candidate orders from the pool --
        unassigned_orders = [
            o for o in self.order_manager.unfinished_orders
            if o.station_id is None
               and o.order_id not in self.order_manager.preassign_order_ids
        ]
        if not unassigned_orders:
            return

        candidates = unassigned_orders[:total_assignable]

        # -- Step 4: Calculate pairwise similarity --
        # Build SKU sets per order
        order_skus = {}
        for o in candidates:
            order_skus[o.order_id] = set(o.skus.keys())

        n = len(candidates)
        ids = [o.order_id for o in candidates]

        # Similarity matrix (dict of dict for flexible access)
        sim = {a: {} for a in ids}
        for i in range(n):
            for j in range(i, n):
                a, b = ids[i], ids[j]
                if i == j:
                    sim[a][b] = 1.0
                else:
                    s = self._jaccard_similarity(order_skus[a], order_skus[b])
                    sim[a][b] = s
                    sim[b][a] = s

        # -- Step 5: Group orders into batches --
        num_batches = len(free_bins)

        if num_batches >= n:
            # Fewer (or equal) candidates than stations — each order is its own batch
            batches = [[oid] for oid in ids]
        else:
            # Total similarity score per order (sum of similarities to all others)
            total_sim = {
                oid: sum(sim[oid][other] for other in ids if other != oid)
                for oid in ids
            }
            # Pick top-N seeds (highest total similarity)
            sorted_by_sim = sorted(total_sim.items(), key=lambda x: x[1], reverse=True)
            seeds = [oid for oid, _ in sorted_by_sim[:num_batches]]

            # Initialize batches with seeds
            batches = [[seed] for seed in seeds]

            # Assign remaining orders to the closest seed
            remaining = [oid for oid in ids if oid not in seeds]
            for oid in remaining:
                best_idx = 0
                best_score = -1.0
                for bi, batch in enumerate(batches):
                    seed = batch[0]  # first element is the seed
                    score = sim[oid][seed]
                    if score > best_score:
                        best_score = score
                        best_idx = bi
                batches[best_idx].append(oid)

        # -- Step 6: Compare each batch to each station --
        station_ids = list(free_bins.keys())

        # Collect current SKU sets at each station (from active orders)
        station_skus = {}
        for sid in station_ids:
            st = self.station_manager.get_station_by_id(sid)
            skus_at_station = set()
            for order in st.orders:
                skus_at_station.update(order.skus.keys())
            station_skus[sid] = skus_at_station

        # Collect collective SKU set per batch
        batch_skus = []
        for batch in batches:
            collective = set()
            for oid in batch:
                collective.update(order_skus[oid])
            batch_skus.append(collective)

        # Greedy assignment: for each batch, pick the best unassigned station
        assignment = {}  # station_id -> list of order_ids
        assigned_stations = set()
        assigned_batches = set()

        # Build score matrix: batch_idx -> station_id -> similarity
        score_matrix = []
        for bi, bskus in enumerate(batch_skus):
            scores = {}
            for sid in station_ids:
                scores[sid] = self._jaccard_similarity(bskus, station_skus[sid])
            score_matrix.append(scores)

        # Greedy: pick the highest-scoring (batch, station) pair repeatedly
        for _ in range(min(len(batches), len(station_ids))):
            best_score = -1.0
            best_bi = -1
            best_sid = None
            for bi in range(len(batches)):
                if bi in assigned_batches:
                    continue
                for sid in station_ids:
                    if sid in assigned_stations:
                        continue
                    if score_matrix[bi][sid] > best_score:
                        best_score = score_matrix[bi][sid]
                        best_bi = bi
                        best_sid = sid
            if best_bi == -1:
                break
            assignment[best_sid] = batches[best_bi]
            assigned_stations.add(best_sid)
            assigned_batches.add(best_bi)

        # Any remaining batches that didn't get a station — their orders stay in pool
        # Any remaining stations that didn't get a batch — they stay idle this cycle

        # -- Step 7: Trim each batch to fit station's free bins --
        final_selection = {}  # station_id -> list of order_ids to assign
        for sid, order_ids in assignment.items():
            capacity = free_bins[sid]
            if len(order_ids) <= capacity:
                final_selection[sid] = order_ids
            else:
                final_selection[sid] = order_ids[:capacity]
                # rest stay in pool automatically (they're just not assigned)

        if not final_selection:
            return

        # -- Step 8: Execute the assignment --
        print(f"[AISYAHNA POA] tick={self._tick:.1f} assigning: {final_selection}")
        self.put_order_to_picking_station(final_selection)

    # ------------------------------------------------------------------ #

    def xxx(self):
        picking_stations = [s for s in self.station_manager.stations if s.station_type == 'picker']
        # Step 1: Get current empty bin per station
        empty_bins = {
            station.station_id: station.max_orders - len(station.order_ids)
            for station in picking_stations
            if (station.max_orders - len(station.order_ids)) > 0
        }

        if not empty_bins:
            return
        
        order_ids = []
        current_picker = list(empty_bins.keys())[0]
        total_order_ids = empty_bins[current_picker]
        fulfilment_fs = self.get_fulfilment_table(mode="FS")
        advanced_df = self.get_advanced_table_only()
        while self.preassign_per_station[current_picker] and empty_bins[current_picker] > 0:
            order_ids.append(self.preassign_per_station[current_picker].popleft())
            empty_bins[current_picker] -= 1
        if empty_bins[current_picker] == 0:
            # assign everything in order_ids
            self.yyy(current_picker, order_ids)
            return
        
        # exclude the preassigned
        exclude_indices = set()
        for q in self.preassign_per_station.values():
            exclude_indices.update(q)
        fulfilment_fs = fulfilment_fs[~fulfilment_fs.index.isin(exclude_indices)]

        order_candidates = fulfilment_fs[current_picker].sort_values(
            ascending=False
        )
        # print("### ORDER CANDIDATES")
        # print(order_candidates)

        next_bin_counts = (
            advanced_df[advanced_df['next_bin_avail'] == True]  # noqa: E712
            .groupby('station_id')
            .size()
            .to_dict()
        )
        # print("### NEXT BIN COUNTS")
        # print(next_bin_counts)
        next_bin_counts = {k: v for k, v in next_bin_counts.items() if k != current_picker}
        # print(f"after filter {next_bin_counts}")
        if not next_bin_counts:
            order_ids.extend(
                order_candidates.index[:empty_bins[current_picker]]
            )
            # print("### ORDER TO BE ASSIGNED")
            # print(order_ids)
            # assign everything in order_ids
            self.yyy(current_picker, order_ids)
            return
        else:
            print(f"next_bin_counts {next_bin_counts}")
            # raise AssertionError(f"there is next_bin_counts current picker {current_picker} next_bin_counts {next_bin_counts}")
            fulfilment_f3 = self.get_fulfilment_table(mode="F3")
            diverted_indices = set()
            for idx, val in order_candidates.items():
                best_picker = fulfilment_f3.loc[idx].idxmax()
                best_value = fulfilment_f3.loc[idx].max()
                # if best_picker != current_picker and best_value > val:
                #     raise AssertionError(f"current picker {current_picker} score {val} best_picker {best_picker} score {best_value}")
                if (
                        best_value > val and 
                        best_picker != current_picker and
                        best_picker in next_bin_counts and
                        next_bin_counts[best_picker] > 0
                    ):
                    # preassign
                    print("\n\n\n YESSSSSSS WE HAVE PREASSIGN!!!!! \n\n\n")
                    print("original")
                    print(order_candidates)
                    # with open('preassign_record.txt', 'a') as f:
                    #     f.write(f"[tick {self._tick}]current {current_picker} order {idx} score {val} bestpicker {best_picker} score {best_value}\n")
                    if not self.fast_train:
                        insert_pre_assign(
                            self._tick,
                            current_picker,
                            idx,
                            val,
                            best_picker,
                            best_value,
                            db_path=self.sqlite_db_path,
                        )
                    self.preassign_per_station[best_picker].append(idx)
                    next_bin_counts[best_picker] -= 1
                    diverted_indices.add(idx)
                    # raise AssertionError
                else:
                    order_ids.append(idx)
                
                if len(order_ids) >= total_order_ids:
                    # process
                    self.yyy(current_picker, order_ids)
                    return

            # Fallback: if future-bin preassignment consumed too many candidate
            # orders, fill the current picker with remaining unassigned orders
            # instead of crashing the simulation.
            for idx, _val in order_candidates.items():
                if len(order_ids) >= total_order_ids:
                    break
                if idx in diverted_indices or idx in order_ids:
                    continue
                order_ids.append(idx)

            if order_ids:
                self.yyy(current_picker, order_ids[:total_order_ids])
            return

    def yyy(self, station_id, order_ids):
        self.put_order_to_picking_station({station_id: order_ids})
        return
    
    def update_robot_job_for_new_orders(self, job: RobotJob):
        if not self.dynamic_job_update_enabled:
            return 0
        if job is None or job.is_finished:
            return 0

        try:
            station: Station = self.station_manager.get_station_by_id(job.station_id)
        except KeyError:
            return 0
        if station is None or not station.is_picker_station():
            return 0

        pod: Pod = job.pod
        if pod is None or not any(
            details.get("current_qty", 0) > 0
            for details in pod.skus.values()
        ):
            return 0

        added_tasks = 0
        existing_order_skus = {
            (order_id, sku)
            for order_id, sku, qty in job.orders
            if qty > 0
        }
        orders: list[Order] = station.get_orders_in_station() or []
        for order in orders:
            remaining_skus = order.get_remaining_skus()
            for sku, qty in remaining_skus.items():
                if (order.order_id, sku) in existing_order_skus:
                    continue
                if sku not in pod.skus:
                    continue

                available_qty = pod.get_quantity(sku)
                quantity_to_take = min(available_qty, qty)
                if quantity_to_take <= 0:
                    continue

                order.commit_quantity(sku, quantity_to_take)
                job.add_picking_task(order.order_id, sku, quantity_to_take)
                pod.pick_sku(sku, quantity_to_take)
                self.pod_manager.reduce_sku_data(sku, quantity_to_take)
                station.reduce_sku_from_station(sku, quantity_to_take)
                existing_order_skus.add((order.order_id, sku))
                added_tasks += 1

        return added_tasks
