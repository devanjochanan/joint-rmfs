"""NetLogo Python bridge – implementation module.

This module contains the full simulation bridge implementation that was
originally in the root ``netlogo.py`` file.  It is imported by the root
compatibility shim so that ``import netlogo`` continues to work for
``simulation.nlogo`` and local scripts such as ``profile_netlogo.py``.

All engine/** and model/** imports remain as-is (relative to the
repository root which must be on sys.path).
"""

import csv
import pickle
import os
import traceback
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import List
import random
import secrets
import warnings

import networkx as nx
import pandas as pd
from pandas import DataFrame
import numpy as np

from pandas import DataFrame
from sklearn.cluster import KMeans

from src.rmfs.runtime_io import RunContext
from src.rmfs.runtime_io.detail_db import configure_detail_db, is_detail_db_configured
from src.rmfs.runtime_io.logging import debug_print
from src.rmfs.runtime_io.run_profiles import resolve_run_profile
from src.rmfs.runtime_io.timing import timed
from src.rmfs.runtime_io.scenario_bundle import (
    activate_scenario_inputs as _activate_scenario_inputs,
    list_available_scenarios as _list_available_scenarios,
)
from src.rmfs.runtime_io.layout_randomization import slot_index_to_pod_id
from src.rmfs.rl.rts.runtime_invariants import check_runtime_invariants
from src.rmfs.decisions.pps import (
    DEFAULT_PPS_MODEL_PATH,
    pps_model_candidates,
    configure_pps_rl_strategy,
    runtime_set_pps_mode,
    get_pps_mode,
    load_pps_rl_model,
    build_pps_rl_sku_index,
    PPS_RL_NUM_STATIONS,
    PPS_RL_TOP_K_SKUS,
    PPS_RL_MAX_PODS,
    PPS_RL_NUM_TRAFFIC_ZONES,
    PPS_RL_MAX_ZONE_ROBOT_COUNT,
    PPS_RL_TRAFFIC_ZONES,
    PPS_RL_POD_FEATURE_DIM,
    PPS_RL_MODEL_PATH,
)
from engine.netlogo_coordinate import NetLogoCoordinate
from engine.object import Object
from model.intersection import Intersection
from model.inventory import Inventory
from model.order import Order
from src.rmfs.order_generation import config_orders, generate_orders_from_raw_bootstrap, PodGenerator
from model.pod import Pod
from model.pod_manager import PodManager
from model.robot import Robot
from model.station import Station
from model.layout import Layout
# DB
from model.tools.pod_location import (
    clear_pod_locations,
    configure_default_db_path as configure_default_pod_location_db_path,
    initialize_pod_location_table,
    upsert_pod_location,
)
from model.tools.pod_travel import (
    clear_pod_travel,
    configure_default_db_path as configure_default_pod_travel_db_path,
    initialize_pod_travel_table,
)
from model.tools.job_task import clear_job_task_table, initialize_job_task_table, upsert_job_task
from model.tools.order_history import clear_order_history, initialize_order_history_table

warnings.simplefilter(action='ignore', category=FutureWarning)

__all__ = [
    # Constants
    "ACTIVATE_NEAREST",
    "PPS_RL_MODEL_PATH",
    # Classes
    "DirectedGraph",
    # Module-level state
    "intersections",
    "stations",
    # Helper functions
    "initRobots",
    "draw_layout",
    "draw_layout_from_generated_file",
    "jaccard_similarity",
    "compute_jaccard_similarity",
    "cluster_backlog_orders",
    "assign_cluster_labels",
    "assign_backlog_orders",
    "draw_storage_from_generated_file",
    "construct_station_path",
    "add_all_direction_paths",
    "assign_skus_to_pods",
    "assign_skus_to_pods_from_file",
    "activate_scenario_inputs",
    "list_available_scenarios",
    # Public API (called by simulation.nlogo / profile_netlogo.py)
    "setup",
    "tick",
    "console_tick",
    "SimulationTermination",
    "SimulationStepResult",
    "HeadlessSimulationSession",
    "setup_in_memory",
    "tick_in_memory",
    "finalize_headless_run",
    "setup_py",
    "get_run_context",
    "configure_run_context",
    "reset_run_context",
    "set_pps_mode",
    "set_sim_seed",
    "get_sim_seed",
    "set_order_cycle_time",
    "get_order_cycle_time",
]

ACTIVATE_NEAREST = True
_RUN_CONTEXT = RunContext.default()


class SimulationTermination(str, Enum):
    RUNNING = "running"
    MAXIMUM_HORIZON = "maximum_horizon"
    NORMAL_COMPLETION = "normal_completion"
    NO_ACTIVE_WORK = "no_active_work"
    CONGESTION = "congestion"
    MANUAL_CANCELLATION = "manual_cancellation"
    WORKER_EXCEPTION = "worker_exception"


@dataclass(frozen=True)
class SimulationStepResult:
    status: SimulationTermination
    payload: list | None = None
    steps_executed: int = 0
    warehouse_time: float | None = None
    netlogo_step: int | None = None
    terminal_reason: str | None = None

    @property
    def terminal(self) -> bool:
        return self.status is not SimulationTermination.RUNNING


def activate_scenario_inputs(scenario_name=None, target_root=None, dry_run=False):
    return _activate_scenario_inputs(
        scenario_name=scenario_name,
        target_root=target_root,
        dry_run=dry_run,
    )


def list_available_scenarios():
    return _list_available_scenarios()


def get_run_context():
    return _RUN_CONTEXT


def configure_run_context(context=None, runtime_root=None):
    global _RUN_CONTEXT
    if context is not None and runtime_root is not None:
        raise ValueError("Pass either context or runtime_root, not both.")
    if context is None:
        context = RunContext.isolated(runtime_root) if runtime_root is not None else RunContext.default()
    context.ensure_runtime_dirs()
    _RUN_CONTEXT = context
    return _RUN_CONTEXT


def reset_run_context():
    return configure_run_context(RunContext.default())


def _path(name):
    return getattr(get_run_context(), name)


def _str_path(name):
    return str(_path(name))


def _maybe_activate_configured_scenario():
    scenario_name = os.environ.get("RMFS_SCENARIO_NAME", "").strip()
    if not scenario_name:
        return None
    metadata_path = get_run_context().runtime_root / "active_scenario.json"
    metadata = _activate_scenario_inputs(
        scenario_name=scenario_name,
        target_root=str(get_run_context().input_root),
        metadata_path=str(metadata_path),
    )
    if metadata is not None:
        debug_print(
            "[SCENARIO] "
            f"{metadata['scenario_name']} "
            f"items={metadata['items_rows']} "
            f"pods={metadata['unique_pods']} "
            f"input_root={get_run_context().input_root}"
        )
    return metadata


_SIM_SEED = (
    int(os.environ["RMFS_SIM_SEED"])
    if os.environ.get("RMFS_SIM_SEED", "").strip()
    else None
)
_SIM_SEED_EXPLICIT = _SIM_SEED is not None
_ORDER_CYCLE_TIME = int(os.environ.get("RMFS_ORDER_CYCLE_TIME", "500"))


def set_sim_seed(seed):
    """Set a reproducible seed, or use 0/None to request a fresh setup seed."""
    global _SIM_SEED, _SIM_SEED_EXPLICIT

    resolved = 0 if seed is None else int(seed)
    if resolved == 0:
        _SIM_SEED = None
        _SIM_SEED_EXPLICIT = False
        os.environ.pop("RMFS_SIM_SEED", None)
        print("[SIM_SEED] Automatic random seed enabled for the next setup.")
        return None
    if resolved < 0:
        raise ValueError("Simulation seed must be zero or a positive integer.")

    _SIM_SEED = resolved
    _SIM_SEED_EXPLICIT = True
    os.environ["RMFS_SIM_SEED"] = str(_SIM_SEED)
    random.seed(_SIM_SEED)
    np.random.seed(_SIM_SEED)
    print(f"[SIM_SEED] Current simulation seed: {_SIM_SEED}")
    return _SIM_SEED


def get_sim_seed():
    return _SIM_SEED


def _prepare_setup_seed():
    """Choose a fresh seed for unseeded runs and apply it to all RNGs."""
    global _SIM_SEED

    if not _SIM_SEED_EXPLICIT:
        _SIM_SEED = secrets.randbelow(2_147_483_646) + 1
    os.environ["RMFS_SIM_SEED"] = str(_SIM_SEED)
    random.seed(_SIM_SEED)
    np.random.seed(_SIM_SEED)
    print(f"[SIM_SEED] Setup seed: {_SIM_SEED}")
    return _SIM_SEED


def set_order_cycle_time(value):
    """Set the shared order arrival rate in orders per simulated hour."""
    global _ORDER_CYCLE_TIME

    resolved = int(value)
    if resolved <= 0:
        raise ValueError("order_cycle_time must be a positive orders-per-hour value.")
    _ORDER_CYCLE_TIME = resolved
    os.environ["RMFS_ORDER_CYCLE_TIME"] = str(resolved)
    print(f"[ORDER_CYCLE] Current order rate: {resolved} orders/hour")
    return _ORDER_CYCLE_TIME


def get_order_cycle_time():
    env_value = os.environ.get("RMFS_ORDER_CYCLE_TIME", "").strip()
    return int(env_value) if env_value else _ORDER_CYCLE_TIME


def _env_bool(name, default=False):
    value = os.environ.get(name)
    if value is None or str(value).strip() == "":
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name, default=None):
    value = os.environ.get(name)
    if value is None or str(value).strip() == "":
        return default
    return int(value)


def _current_run_profile():
    profile = os.environ.get("RMFS_RUN_PROFILE", "gui")
    return resolve_run_profile(
        profile,
        run_horizon_ticks=_env_int("RMFS_RUN_HORIZON_TICKS"),
        bootstrap_n_orders=_env_int("RMFS_BOOTSTRAP_N_ORDERS"),
        demand_horizon_ticks=_env_int("RMFS_DEMAND_HORIZON_TICKS"),
        demand_buffer_ticks=_env_int("RMFS_DEMAND_BUFFER_TICKS"),
        order_generation_mode=os.environ.get("RMFS_ORDER_GENERATION_MODE"),
        full_raw_order_replay=_env_bool("RMFS_FULL_RAW_ORDER_REPLAY", False),
        detail_db=_env_bool("RMFS_DETAIL_DB", True) if os.environ.get("RMFS_DETAIL_DB") is not None else None,
        pod_location_mode=os.environ.get("RMFS_POD_LOCATION_MODE"),
        pod_location_seed=_env_int("RMFS_POD_LOCATION_SEED"),
        seed=_SIM_SEED,
    )


def _apply_runtime_config(universe):
    allocator = os.environ.get("RMFS_ROBOT_TASK_ALLOCATOR")
    if allocator:
        universe.robot_task_allocator = allocator
    regret_k = _env_int("RMFS_REGRET_K")
    if regret_k is not None:
        universe.regret_k = regret_k
    scope = os.environ.get("RMFS_TASK_ALLOCATOR_SCOPE")
    if scope:
        universe.task_allocator_scope = scope
    universe.committed_next_reservations_enabled = _env_bool(
        "RMFS_COMMITTED_NEXT_RESERVATIONS",
        getattr(universe, "committed_next_reservations_enabled", False),
    )


def set_pps_mode(mode):
    """Switch PPS mode: 'ppo', 'random', 'heuristic'/'rika', or 'demand'."""
    return runtime_set_pps_mode(
        mode,
        state_file_path=_str_path("state_file"),
        items_csv=_str_path("items_csv"),
        skus_data_csv=_str_path("skus_data_csv"),
    )


def _pps_rl_station_demands(universe, use_committed=True):
    demands = {}
    for station in universe.station_manager.picking_stations:
        station_demand = defaultdict(int)
        for order in station.orders:
            order_skus = order.get_remaining_skus() if use_committed else order.get_unpicked_skus()
            for sku, qty in order_skus.items():
                station_demand[sku] += qty
        demands[station.station_id] = dict(station_demand)
    return demands


def _pps_rl_incoming_pod_commits(universe):
    commits = defaultdict(lambda: defaultdict(int))
    for job in universe.job_queue:
        if job is None or job.is_finished:
            continue
        for _order_id, sku, qty in job.orders:
            commits[job.station_id][sku] += qty

    for obj in universe.get_movable_objects():
        if obj.object_type != "robot":
            continue
        job = obj.job
        if job is None or job.is_finished:
            continue
        for _order_id, sku, qty in job.orders:
            commits[job.station_id][sku] += qty
    return commits


def _pps_rl_future_station_demands(universe):
    current = _pps_rl_station_demands(universe, use_committed=False)
    incoming = _pps_rl_incoming_pod_commits(universe)
    future = {}
    for station_id, cur_demand in current.items():
        inc_demand = incoming.get(station_id, {})
        remaining = {}
        for sku, qty in cur_demand.items():
            qty_left = qty - inc_demand.get(sku, 0)
            if qty_left > 0:
                remaining[sku] = qty_left
        future[station_id] = remaining
    return future


def _pps_rl_candidate_pods(universe):
    station_demands = _pps_rl_station_demands(universe)
    demand_skus = set()
    for demand in station_demands.values():
        demand_skus.update(demand.keys())

    if not demand_skus:
        return []

    candidates = []
    for pod in universe.pod_manager.pods:
        if not universe.pod_manager.is_idle(pod.pod_id):
            continue
        pod_skus = {
            sku for sku, details in pod.skus.items()
            if details["current_qty"] > 0
        }
        if pod_skus & demand_skus:
            candidates.append(pod)
    return candidates[:PPS_RL_MAX_PODS]


def _pps_rl_decision_needed(universe):
    for station in universe.station_manager.picking_stations:
        for order in station.orders:
            if order.get_remaining_skus():
                return len(_pps_rl_candidate_pods(universe)) > 0
    return False


def _pps_rl_traffic_zone_index(x, y):
    for idx, ((min_x, max_x), (min_y, max_y)) in enumerate(PPS_RL_TRAFFIC_ZONES):
        if min_x <= x <= max_x and min_y <= y <= max_y:
            return idx
    return None


def _pps_rl_zone_robot_counts(universe):
    counts = np.zeros(PPS_RL_NUM_TRAFFIC_ZONES, dtype=np.float32)
    for obj in universe.get_movable_objects():
        if getattr(obj, "object_type", None) != "robot":
            continue
        zone_idx = _pps_rl_traffic_zone_index(obj.pos_x, obj.pos_y)
        if zone_idx is not None:
            counts[zone_idx] += 1.0
    return counts


def _build_pps_rl_observation(universe):
    if not hasattr(universe, "pps_rl_sku_index"):
        universe.pps_rl_sku_index = _build_pps_rl_sku_index(universe)

    sku_index = universe.pps_rl_sku_index
    candidates = _pps_rl_candidate_pods(universe)
    station_demands = _pps_rl_station_demands(universe)
    future_demands = _pps_rl_future_station_demands(universe)
    stations = sorted(universe.station_manager.picking_stations, key=lambda s: s.station_id)
    station_ids = [station.station_id for station in stations]
    station_pos = [(station.pos_x, station.pos_y) for station in stations]

    pod_features = np.zeros((PPS_RL_MAX_PODS, PPS_RL_POD_FEATURE_DIM), dtype=np.float32)
    station_features = np.zeros((PPS_RL_NUM_STATIONS, PPS_RL_TOP_K_SKUS), dtype=np.float32)
    zone_robot_counts = _pps_rl_zone_robot_counts(universe)

    for j, station_id in enumerate(station_ids[:PPS_RL_NUM_STATIONS]):
        for sku, qty in future_demands.get(station_id, {}).items():
            idx = sku_index.get(sku)
            if idx is not None:
                station_features[j, idx] = min(qty / 100.0, 1.0)

    max_dist = 49.0 + 31.0
    for i, pod in enumerate(candidates):
        for sku, details in pod.skus.items():
            idx = sku_index.get(sku)
            if idx is not None:
                pod_features[i, idx] = min(details["current_qty"] / 100.0, 1.0)

        for j, (station_x, station_y) in enumerate(station_pos[:PPS_RL_NUM_STATIONS]):
            dist = abs(pod.pos_x - station_x) + abs(pod.pos_y - station_y)
            pod_features[i, PPS_RL_TOP_K_SKUS + j] = 1.0 - min(dist / max_dist, 1.0)

        for j, station_id in enumerate(station_ids[:PPS_RL_NUM_STATIONS]):
            demand = station_demands.get(station_id, {})
            if not demand:
                continue

            matched = 0
            total_demand = sum(demand.values())
            for sku, req_qty in demand.items():
                if sku in pod.skus:
                    matched += min(pod.skus[sku]["current_qty"], req_qty)
            pod_features[i, PPS_RL_TOP_K_SKUS + PPS_RL_NUM_STATIONS + j] = (
                min(matched / max(total_demand, 1), 1.0)
            )

        zone_idx = _pps_rl_traffic_zone_index(pod.pos_x, pod.pos_y)
        if zone_idx is not None:
            zone_offset = (
                PPS_RL_TOP_K_SKUS
                + PPS_RL_NUM_STATIONS
                + PPS_RL_NUM_STATIONS
            )
            pod_features[i, zone_offset + zone_idx] = 1.0

    return {
        "pod_features": pod_features,
        "station_features": station_features,
        "num_candidates": np.array([len(candidates)], dtype=np.int32),
        "zone_robot_counts": zone_robot_counts,
    }


def _execute_pps_rl_actions(universe, actions):
    candidates = _pps_rl_candidate_pods(universe)
    station_ids = sorted(
        station.station_id
        for station in universe.station_manager.picking_stations
    )
    flat_actions = np.asarray(actions).reshape(-1)
    assignments = 0

    for i in range(min(len(candidates), len(flat_actions))):
        action = int(flat_actions[i])
        if action == 0 or action < 1 or action > PPS_RL_NUM_STATIONS:
            continue

        pod = candidates[i]
        station = universe.station_manager.get_station_by_id(station_ids[action - 1])

        if not universe.pod_manager.is_idle(pod.pod_id):
            continue
        if len(station.incoming_pod) >= station.max_robots:
            continue
        if not station.orders:
            continue

        sku_to_quantity = defaultdict(int)
        sku_to_order_map = defaultdict(list)
        for order in station.orders:
            for sku, qty in order.get_remaining_skus().items():
                sku_to_quantity[sku] += qty
                sku_to_order_map[sku].append((order.order_id, qty))

        if not sku_to_quantity:
            continue
        has_match = any(
            sku in pod.skus and pod.skus[sku]["current_qty"] > 0
            for sku in sku_to_quantity
        )
        if not has_match:
            continue

        job = universe.add_picking_task_after_pps(
            station,
            pod,
            sku_to_order_map,
            sku_to_quantity,
        )
        if len(job.orders) == 0:
            continue

        universe.job_queue.append(job)
        assignments += 1
        for order_id, sku, qty in job.orders:
            upsert_job_task(
                pod_id=str(job.pod.pod_id),
                order_id=str(order_id),
                sku=str(sku),
                qty=str(qty),
                assigned_station=station.station_id,
                pod_assigned_time=universe._tick,
                status="queue",
            )

    return assignments


def _apply_pps_rl_policy(universe):
    if not getattr(universe, "pps_rl", False):
        return 0
    if not _pps_rl_decision_needed(universe):
        return 0

    if getattr(universe, "pps_rl_random", False) or get_pps_mode() == "random":
        actions = np.random.randint(
            0,
            PPS_RL_NUM_STATIONS + 1,
            size=PPS_RL_MAX_PODS,
            dtype=np.int64,
        )
        return _execute_pps_rl_actions(universe, actions)

    model = load_pps_rl_model()
    if model is None:
        return 0

    observation = _build_pps_rl_observation(universe)
    actions, _state = model.predict(observation, deterministic=True)
    return _execute_pps_rl_actions(universe, actions)


def _get_throughput(universe):
    """Completed orders so far, matching the PPS training throughput metric."""
    total_orders = len(universe.order_manager.orders)
    unfinished_orders = len(universe.order_manager.unfinished_orders)
    return max(total_orders - unfinished_orders, 0)


def _get_avg_order_completion_time(universe):
    completed_times = []
    for order in universe.order_manager.orders:
        if order.order_complete_time >= 0 and order.process_start_time >= 0:
            completed_times.append(order.order_complete_time - order.process_start_time)
    if not completed_times:
        return 0
    return sum(completed_times) / len(completed_times)


def _get_pod_visits(universe):
    return getattr(universe, "pps_pod_visits", getattr(universe, "pps_rl_pod_visits", 0))


def _get_picked_quantity(universe):
    return getattr(universe, "pps_picked_quantity", getattr(universe, "pps_rl_picked_quantity", 0))


def _get_pile_on_rate(universe):
    pod_visits = _get_pod_visits(universe)
    if pod_visits <= 0:
        return 0
    return _get_picked_quantity(universe) / pod_visits


def _print_gui_tick_status(warehouse):
    cadence = _env_int("RMFS_GUI_TICK_PRINT_CADENCE", 1) or 1
    step = _netlogo_step(warehouse)
    if step is not None and step % max(1, cadence) != 0:
        return
    robots = [
        obj for obj in getattr(warehouse, "_objects", []) or []
        if getattr(obj, "object_type", None) == "robot"
    ]
    robot_states = defaultdict(int)
    charging_count = 0
    for robot in robots:
        robot_states[str(getattr(robot, "current_state", "unknown"))] += 1
        if (
            bool(getattr(robot, "is_charging", False))
            or bool(getattr(robot, "is_charging_pending", False))
            or getattr(robot, "_claimed_charger", None) is not None
        ):
            charging_count += 1
    orders_loaded = len(getattr(warehouse.order_manager, "orders", []) or [])
    orders_unfinished = len(getattr(warehouse.order_manager, "unfinished_orders", []) or [])
    orders_completed = max(orders_loaded - orders_unfinished, 0)
    state_text = ",".join(f"{name}:{count}" for name, count in sorted(robot_states.items()))
    print(
        "[RMFS_TICK] "
        f"step={step} "
        f"warehouse_seconds={float(getattr(warehouse, '_tick', 0.0)):.2f} "
        f"job_queue={len(getattr(warehouse, 'job_queue', []) or [])} "
        f"robots={state_text} "
        f"charging_active={charging_count} "
        f"charging_enabled={bool(getattr(warehouse, 'charging_enabled', False))} "
        f"rts_controls_replenishment={bool(getattr(warehouse, 'rts_controls_post_pick_replenishment', False))} "
        f"orders_completed={orders_completed} "
        f"orders_loaded={orders_loaded} "
        f"orders_unfinished={orders_unfinished} "
        f"pod_visits={_get_pod_visits(warehouse)} "
        f"picked_quantity={_get_picked_quantity(warehouse)}"
    )



class DirectedGraph:
    key = ''

    def __init__(self):
        """Initialize an instance with a directed graph."""
        self.graph = nx.DiGraph()
        self._undirected_graph_cache = None

    @staticmethod
    def node_valid(node):
        """Check if a node is valid based on custom logic.

        Args:
            node (str): The node in format 'x,y'.

        Returns:
            bool: True if the node is valid, False otherwise.
        """
        x, y = map(int, node.split(","))
        return x >= 2 and y >= 0

    def add_node(self, node):
        """Add a node to the graph if it's valid.

        Args:
            node (str): The node to add.
        """
        if self.node_valid(node):
            self.graph.add_node(node)
            self._undirected_graph_cache = None

    def add_edge(self, start, end, weight):
        """Add an edge between two nodes with a weight if both nodes are valid.

        Args:
            start (str): The start node.
            end (str): The end node.
            weight (float): The weight of the edge.
        """
        if self.node_valid(start) and self.node_valid(end):
            self.graph.add_edge(start, end, weight=weight)
            self._undirected_graph_cache = None

    @staticmethod
    def get_heading(p1: NetLogoCoordinate, p2: NetLogoCoordinate):
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

    def _undirected_fallback_path(self, G, start, end):
        """Return an emergency route when directed routing cannot connect nodes."""
        try:
            if G is self.graph:
                if self._undirected_graph_cache is None:
                    self._undirected_graph_cache = self.graph.to_undirected()
                fallback_graph = self._undirected_graph_cache
            else:
                fallback_graph = G.to_undirected()
            return nx.shortest_path(fallback_graph, source=start, target=end, weight='weight')
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    def dijkstra_modified(self, start, end, penalties, zone_boundary, avoid=None):
        """Find the shortest path between two nodes using Dijkstra's algorithm, avoiding specified nodes.

        Args:
            start (str): The start node.
            end (str): The end node.
            avoid (list, optional): Nodes to avoid in the path.

        Returns:
            list or None: The path from start to end if one exists, otherwise None.
        """
        # Avoid copying the graph in the hot path. Copy only when avoid penalties are needed.
        G = self.graph.copy() if avoid else self.graph

        # Increase the weight of the edges leading to and from the nodes to avoid
        if avoid:
            for node in avoid:
                for neighbor in list(G.neighbors(node)) + list(G.predecessors(node)):
                    # Increase the weight significantly to discourage using these paths
                    if G.has_edge(neighbor, node):
                        G[neighbor][node]['weight'] += 10000
                    if G.has_edge(node, neighbor):
                        G[node][neighbor]['weight'] += 10000

        # Increase the weight of edges in every zone based on the penalty
        for index, zone in enumerate(zone_boundary):
            for row in range(zone[1][0], zone[0][0]):
                for col in range(zone[0][1], zone[1][1]):
                    coordinate_str = f"{row},{col}"
                    for neighbor in list(G.neighbors(coordinate_str)) + list(G.predecessors(coordinate_str)):
                        if G.has_edge(neighbor, coordinate_str):
                            G[neighbor][coordinate_str]['weight'] = penalties[index]
                        if G.has_edge(coordinate_str, neighbor):
                            G[coordinate_str][neighbor]['weight'] = penalties[index]

        try:
            # Use Dijkstra's algorithm to find the shortest path
            path = nx.shortest_path(G, source=start, target=end, weight='weight', method='bellman-ford')
            return path
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            fallback_path = self._undirected_fallback_path(G, start, end)
            if fallback_path is not None:
                debug_print(f"[WARN] directed route unavailable from {start} to {end}; using undirected fallback.")
            return fallback_path

    def dijkstra(self, start, end, avoid=None):
        """Find the shortest path between two nodes using Dijkstra's algorithm, avoiding specified nodes.

        Args:
            start (str): The start node.
            end (str): The end node.
            avoid (list, optional): Nodes to avoid in the path.

        Returns:
            list or None: The path from start to end if one exists, otherwise None.
        """
        # Avoid copying the graph in the hot path. Copy only when avoid penalties are needed.
        G = self.graph.copy() if avoid else self.graph

        # Increase the weight of the edges leading to and from the nodes to avoid
        if avoid:
            for node in avoid:
                for neighbor in list(G.neighbors(node)) + list(G.predecessors(node)):
                    # Increase the weight significantly to discourage using these paths
                    if G.has_edge(neighbor, node):
                        G[neighbor][node]['weight'] += 1000
                    if G.has_edge(node, neighbor):
                        G[node][neighbor]['weight'] += 1000

        try:
            # Use Dijkstra's algorithm to find the shortest path
            path = nx.shortest_path(G, source=start, target=end, weight='weight', method='bellman-ford')
            return path
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            fallback_path = self._undirected_fallback_path(G, start, end)
            if fallback_path is not None:
                debug_print(f"[WARN] directed route unavailable from {start} to {end}; using undirected fallback.")
            return fallback_path


intersections: List[Intersection] = []

stations = [
    [2, 33],
    [2, 27],
    [2, 21],
    [2, 15],
    [2, 9],
    [2, 3],
]


# def initStation(universe: Inventory):
#     # Iterate over each station defined in the 'stations' list
#     # Assuming 'stations' is a list of tuples/lists where each item contains the x and y coordinates of a station
#     for s in stations:
#         # Create a new Station object
#         station = Station(1, "picker")

#         # Set the x and y positions from the station data
#         station.pos_x = s[0]
#         station.pos_y = s[1]

#         # Set the coordinates for the station using a helper function or class
#         # NetLogoCoordinate may be a function or class designed to handle coordinate transformations or representations
#         station.coordinate = NetLogoCoordinate(s[0], s[1])

#         # Add the station object to the universe's list of objects
#         # This could be for general object management within the universe
#         universe.addObject(station)

#         # Specifically add the station object to the universe's list of stations
#         # This could be for easy access to stations or station-specific management
#         universe.station_manager.add_station(station)


def initRobots(universe: Inventory):

    num_robot = _env_int("RMFS_NUM_ROBOTS", 20)
    if num_robot < 1:
        raise ValueError(f"RMFS_NUM_ROBOTS must be >= 1; got {num_robot}")

    layout_frame = pd.read_csv(_str_path("generated_pod_csv"), header=None)
    row_count, col_count = layout_frame.shape
    x_min = min(5, max(0, col_count - 1))
    x_max = max(x_min, col_count - 6)
    y_min = 0
    y_max = max(y_min, row_count - 1)

    occupied_coordinates = {
        (int(obj.pos_x), int(obj.pos_y))
        for obj in universe._objects
        if getattr(obj, "object_type", None) in {"pod", "picker", "replenishment", "station"}
    }
    blocked_coordinates = occupied_coordinates | {
        (int(x), int(y)) for x, y in getattr(universe, "charger_cells", set())
    }
    candidate_coordinates = []
    if universe.graph is not None:
        for node in universe.graph.graph.nodes:
            x, y = map(int, str(node).split(","))
            if (
                x_min <= x <= x_max
                and y_min <= y <= y_max
                and (x, y) not in blocked_coordinates
            ):
                candidate_coordinates.append((x, y))
    if len(candidate_coordinates) < num_robot:
        raise ValueError(
            f"Not enough valid graph nodes to place {num_robot} robots; only found {len(candidate_coordinates)}"
        )

    random.shuffle(candidate_coordinates)
    robots = [
        {
            'velocity': 0,
            'heading': 0,
            'x': x,
            'y': y,
        }
        for x, y in candidate_coordinates[:num_robot]
    ]

    # Iterate through each robot in the list to initialize and add to the universe
    for r in robots:
        # Create a new Robot instance
        robot = Robot(universe)

        # Set the robot's attributes based on the dictionary values
        robot.velocity = r['velocity']
        robot.heading = r['heading']
        robot.pos_x = r['x']
        robot.pos_y = r['y']

        # Optionally, set the robot's coordinates using a specific coordinate system
        robot.coordinate = NetLogoCoordinate(robot.pos_x, robot.pos_y)

        # Add the robot to the universe, which likely involves adding it to some internal list or map
        universe.addObject(robot)


def draw_layout(universe):
    # Check if generated_pod.csv exists in the current directory
    if _path("generated_pod_csv").exists():
        print("Generated pod already exist, delete generated_pod.csv if you want to change")
        draw_layout_from_generated_file(universe)
    else:
        layout = Layout()
        # This one to generate new configuration
        layout.generate(output_path=_str_path("generated_pod_csv"))
        draw_layout_from_generated_file(universe)


def draw_layout_from_generated_file(universe: Inventory):
    draw_storage_from_generated_file(universe)

    # Build the shared shuffled historical-order stream with cycle-rate arrivals.
    assign_skus_to_pods(universe.pod_manager)
    run_profile = _current_run_profile()
    config_orders(
        seed=_SIM_SEED,
        source_path=_str_path("raw_order_csv"),
        target_dir=str(get_run_context().runtime_root),
        items_csv_path=_str_path("items_csv"),
        n_orders=run_profile.bootstrap_n_orders,
        run_horizon_ticks=run_profile.run_horizon_ticks,
        demand_horizon_ticks=run_profile.demand_horizon_ticks,
        demand_buffer_ticks=run_profile.demand_buffer_ticks,
        order_cycle_time=get_order_cycle_time(),
        order_generation_mode=run_profile.order_generation_mode,
        full_raw_order_replay=run_profile.full_raw_order_replay,
        profile=run_profile.profile,
    )
    initRobots(universe)

    pod = list(universe.pod_manager.coordinate_to_pods.values())[0]
    destinations = [
        [pod.pos_x, pod.pos_y, 0]
    ]


def jaccard_similarity(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))

    return intersection / union


def compute_jaccard_similarity(data):
    similarity_dict = {}
    grouped = data.groupby('order_id')['item_id'].apply(set)
    for order_dum, items in grouped.items():
        similarities = []
        for other_order_dum, other_items in grouped.items():
            if order_dum == other_order_dum:
                similarities.append(1.0)  # similarity with itself is 1
            else:
                similarity = jaccard_similarity(items, other_items)
                similarities.append(similarity)
        similarity_dict[order_dum] = similarities
    return grouped, similarity_dict


def cluster_backlog_orders(jaccard_similarities, total_station, station_capacity_df):
    jaccard_similarities_list = [similarities for similarities in jaccard_similarities.values()]
    # print(jaccard_similarities_list)
    cluster_labels = [-1] * len(jaccard_similarities_list)
    station_remaining_capacity = station_capacity_df['capacity_left'].tolist()

    # K-Means clustering
    kmeans = KMeans(n_clusters=total_station)
    kmeans.fit(jaccard_similarities_list)

    cluster_labels1 = kmeans.labels_

    cluster_distances = []

    # calculate distances for each order
    for i, label in enumerate(cluster_labels1):
        centroid = kmeans.cluster_centers_[label]
        distance = np.linalg.norm(jaccard_similarities_list[i] - centroid)
        cluster_distances.append((i, label, distance))

    cluster_distances.sort(key=lambda x: x[2])

    # assign each backlog order to a cluster
    for order_idx, label, distance in cluster_distances:
        station_id = station_capacity_df.iloc[label]['id_station']
        if station_remaining_capacity[label] > 0:
            cluster_labels[order_idx] = station_id
            station_remaining_capacity[label] -= 1
        else:
            cluster_labels[order_idx] = None

    print("cluster label:")
    print(cluster_labels)

    return cluster_labels


def assign_cluster_labels(universe: Inventory, data_backlog_order_df, full_order, cluster_labels, station_capacity_df):
    order_dum_to_cluster = dict(zip(full_order.index, cluster_labels))
    temp = float('inf')
    new_order = None

    orders_df = pd.read_csv(_str_path("generated_order_csv"))

    file_path = _str_path("assign_order_csv")
    if os.path.exists(file_path):
        assign_order_df = pd.read_csv(file_path)
        # pass
    else:
        assign_order_df = orders_df.copy()
        assign_order_df['assigned_station'] = pd.Series([None] * len(assign_order_df), dtype="object")
        assign_order_df['assigned_pod'] = pd.Series([None] * len(assign_order_df), dtype="object")
        assign_order_df['status'] = -3
        assign_order_df['order_processed'] = None
        assign_order_df['order_finished'] = None
        assign_order_df.to_csv(_str_path("assign_order_csv"), index=False)

    unique_orders = set()
    order_sku_map = {}
    new_order = None
    for index, row in data_backlog_order_df.iterrows():
        order_dum = row['order_id']
        station_id = order_dum_to_cluster[order_dum]

        if station_id is not None and order_dum not in unique_orders:
            unique_orders.add(order_dum)
            new_order = Order(order_dum, 0)
            # print("order: ", new_order.order_id)
            # print("station: ", station_id)

            assign_order_df.loc[assign_order_df['order_id'] == new_order.order_id, 'assigned_station'] = station_id
            assign_order_df.loc[assign_order_df['order_id'] == new_order.order_id, 'status'] = -1
            assign_order_df.loc[assign_order_df['order_id'] == new_order.order_id, 'order_processed'] = int(
                universe.tick_to_second)
            assign_order_df.to_csv(_str_path("assign_order_csv"), index=False)
            new_order.assign_station(station_id)
            station = universe.station_manager.get_station_by_id(station_id)
            universe.order_manager.add_order(new_order)
            order_sku_map[order_dum] = 0

        if order_dum in unique_orders:
            order = universe.order_manager.get_order_by_id(order_dum)
            order.add_sku(row['item_id'], row['item_quantity'])
            order_sku_map[order_dum] += 1
        if order_dum in order_sku_map:
            order = universe.order_manager.get_order_by_id(order_dum)
            expected_sku_count = data_backlog_order_df[data_backlog_order_df['order_id'] == order_dum].shape[0]
            if order_sku_map[order_dum] == expected_sku_count:
                station.add_order(order_dum, order)

    return station_capacity_df


def assign_backlog_orders(universe: Inventory):
    # open file order
    order_path = _str_path("generated_order_csv")
    data_order_df = pd.read_csv(order_path)

    # filter order_id < 0
    unassigned_backlog_order = data_order_df.loc[(data_order_df['order_id'] < 0)].sort_values(by=['order_id']).reset_index(
        drop=True)

    columns = ['id_station', 'capacity_left']
    station_id_cap_df = pd.DataFrame(columns=columns)

    for station in universe.station_manager.stations:
        id = station.station_id
        cap = station.max_orders - len(station.order_ids)

        new_row = pd.DataFrame({'id_station': [id], 'capacity_left': [cap]})
        # station_id_cap_df = station_id_cap_df.append({'id_station': id, 'capacity_left': cap}, ignore_index=True)
        station_id_cap_df = pd.concat([station_id_cap_df, new_row], ignore_index=True)
    is_picker = station_id_cap_df['id_station'].str.startswith('picker')

    station_id_cap_df = station_id_cap_df[is_picker]
    station_id_cap_df.reset_index(drop=True, inplace=True)

    if len(unassigned_backlog_order) > 0:
        total_station = len(station_id_cap_df)

        full_order, jaccard_similarities = compute_jaccard_similarity(unassigned_backlog_order)

        cluster_labels = cluster_backlog_orders(jaccard_similarities, total_station, station_id_cap_df)

        station_id_cap_df = assign_cluster_labels(universe, unassigned_backlog_order, full_order, cluster_labels,
                                                  station_id_cap_df)


def draw_storage_from_generated_file(universe: Inventory):
    # ── PATCH C: load charging config, apply policy, register chargers ────
    # Runs before initRobots(), so Robot class policy reaches the fleet.
    import json as _json
    from model.robot import Robot as _Robot
    from src.rmfs.decisions.charging.config import canonical_charging_config_path
    _cfg_path = os.environ.get("RMFS_CHARGING_CONFIG", "") or str(canonical_charging_config_path())
    _cfg = {}
    if os.path.exists(_cfg_path):
        try:
            with open(_cfg_path, "r", encoding="utf-8") as _f:
                _cfg = _json.load(_f)
        except Exception as _e:
            print(f"[charging] could not read {_cfg_path}: {_e}")
    # Apply policy to the Robot class BEFORE the fleet is created.
    for _k, _a in (("battery_low_pct", "BATTERY_LOW_PCT"),
                   ("battery_charged_pct", "BATTERY_CHARGED_PCT"),
                   ("battery_interrupt_pct", "BATTERY_INTERRUPT_PCT"),
                   ("initial_battery_frac", "INITIAL_BATTERY_FRAC")):
        if _k in _cfg:
            setattr(_Robot, _a, float(_cfg[_k]))
    if "corrected_energy_model" in _cfg:
        _Robot.CORRECTED_ENERGY_MODEL = bool(_cfg["corrected_energy_model"])
    # Register charger overlay positions ([row, col] in config -> (x=col, y=row)).
    for _pos in _cfg.get("charger_positions", []):
        universe.charger_cells.add((int(_pos[1]), int(_pos[0])))
    for _pos in _cfg.get("active_charger_positions", []):
        universe.active_charger_cells.add((int(_pos[1]), int(_pos[0])))
    universe.disable_active_charging = bool(_cfg.get("disable_active_charging", False))
    # Charging is ON by default (realistic battery model). It is active whenever
    # chargers exist (config positions OR grid value-2 cells). Set
    # RMFS_CHARGING_ENABLED=0 to fall back to the old no-battery behavior.
    _env = os.environ.get("RMFS_CHARGING_ENABLED", "").strip().lower()
    _explicit_off = _env in {"0", "false", "no", "off"}
    universe.charging_enabled = (not _explicit_off) and bool(universe.charger_cells)
    if universe.charging_enabled:
        print(f"[charging] ON: {len(universe.charger_cells)} charger cells "
              f"(active={len(universe.active_charger_cells)}), policy "
              f"{_Robot.BATTERY_LOW_PCT}/{_Robot.BATTERY_CHARGED_PCT}/"
              f"{_Robot.BATTERY_INTERRUPT_PCT} from {os.path.basename(_cfg_path)}")
    else:
        print("[charging] OFF (RMFS_CHARGING_ENABLED=0, or no chargers found)")

    station_picker_counter = 1
    station_replenish_counter = 1
    pods_horizontal_length = 5
    pods_vertical_length = 2
    pod_counter = 0
    pod_slot_counter = 0
    pod_location_mode = os.environ.get("RMFS_POD_LOCATION_MODE", "fixed").strip().lower()
    randomized_pod_ids = {}
    if pod_location_mode == "randomize_slots":
        seed_text = os.environ.get("RMFS_POD_LOCATION_SEED", "").strip()
        randomization_seed = int(seed_text) if seed_text else (_SIM_SEED if _SIM_SEED is not None else 0)
        randomized_pod_ids = slot_index_to_pod_id(
            _str_path("generated_pod_csv"),
            seed=randomization_seed,
            mode=pod_location_mode,
        )
        debug_print(f"[POD_LOCATION] randomize_slots enabled with seed {randomization_seed}")
    graph = DirectedGraph()
    graph_pod = DirectedGraph()
    graph_pod.key = 'pod'
    universe.graph = graph
    universe.graph_pod = graph_pod
    data = pd.read_csv(_str_path("generated_pod_csv"), header=None)
    total_rows = len(data)
    total_cols = 0
    for y, row in data.iterrows():
        # Invert Y only to draw
        for x, value in row.items():
            obj = Object()
            obj.object_type = 'way-direction'
            obj_key = f"{x},{y}"
            obj.pos_x = x
            obj.pos_y = y

            obj_left_coordinate = f"{x - 1},{y}"
            obj_right_coordinate = f"{x + 1},{y}"
            obj_above_coordinate = f"{x},{y - 1}"
            obj_below_coordinate = f"{x},{y + 1}"

            obj_left_value = data.iloc[y, x - 1] if x > 0 else None
            obj_right_value = data.iloc[y, x + 1] if x < len(row) - 1 else None
            obj_above_value = data.iloc[y - 1, x] if y > 0 else None
            obj_below_value = data.iloc[y + 1, x] if y < total_rows - 1 else None

            weight = 1
            turning_weight = 5
            intersection_weight = 4
            if x <= 7:
                weight = 3

            if value == 0 or value == 1 or value == 2:
                add_all_direction_paths(graph, obj_key, weight)

                if value == 0:
                    obj.shape = 'empty-space'
                    if ACTIVATE_NEAREST:
                        universe.storage_manager.createStorage(x, y)
                elif value == 1:
                    pod_id = randomized_pod_ids.get(pod_slot_counter, pod_counter)
                    obj = Pod(pod_id)
                    if ACTIVATE_NEAREST:
                        storage = universe.storage_manager.createStorage(x, y)
                    pod_counter += 1
                    pod_slot_counter += 1
                    # obj.coordinate = NetLogoCoordinate(x, y)
                    obj.pos_x = x
                    obj.pos_y = y
                    upsert_pod_location(obj.pod_id, obj.pos_x, obj.pos_y, db_path=_str_path("sqlite_db"))
                    
                    if ACTIVATE_NEAREST:
                        universe.storage_manager.addPodToStorage(obj, storage)
                    graph_pod.add_node(obj_key)
                    universe.pod_manager.add_pod(obj)
                elif value == 2:
                    obj.shape = 'square 2'
                    universe.charger_cells.add((x, y))   # PATCH C: grid-encoded charger

                if obj_left_value != 1:
                    graph_pod.add_edge(obj_key, obj_left_coordinate, weight=100)
                if obj_right_value != 1:
                    graph_pod.add_edge(obj_key, obj_right_coordinate, weight=100)
                if obj_above_value != 1:
                    graph_pod.add_edge(obj_key, obj_above_coordinate, weight=100)
                if obj_below_value != 1:
                    graph_pod.add_edge(obj_key, obj_below_coordinate, weight=100)
            elif value == 3:
                obj.shape = 'empty-space'

                intersection = Intersection(NetLogoCoordinate(x, y))
                approaching_path_coordinates = []

                if obj_right_value in [4, 6, 7]:
                    right_x = x + 1
                    while data.iloc[y, right_x] in [4, 6, 7]:
                        approaching_path_coordinates.append((right_x, y))
                        right_x += 1

                    if data.iloc[y, right_x] == 3:
                        intersection.add_connected_intersection_id(right_x, y)
                if obj_left_value in [5, 6, 7]:
                    left_x = x - 1
                    while data.iloc[y, left_x] in [5, 6, 7]:
                        approaching_path_coordinates.append((left_x, y))
                        left_x -= 1

                    if data.iloc[y, left_x] == 3:
                        intersection.add_connected_intersection_id(left_x, y)
                if obj_below_value == 6:
                    below_y = y + 1
                    while data.iloc[below_y, x] == 6:
                        approaching_path_coordinates.append((x, below_y))
                        below_y += 1

                    if data.iloc[below_y, x] == 3:
                        intersection.add_connected_intersection_id(x, below_y)
                if obj_above_value == 7:
                    above_y = y - 1
                    while data.iloc[above_y, x] == 7:
                        approaching_path_coordinates.append((x, above_y))
                        above_y -= 1

                    if data.iloc[above_y, x] == 3:
                        intersection.add_connected_intersection_id(x, above_y)

                for each_approaching_coordinate in approaching_path_coordinates:
                    intersection.approaching_path_coordinates.append(each_approaching_coordinate)

                if obj.pos_x == 15:
                    intersection.use_reinforcement_learning = True
                    if obj.pos_y == 0:
                        intersection.set_RL_model_name("BOTTOM")
                    elif obj.pos_y == 30:
                        intersection.set_RL_model_name("TOP")
                    else:
                        intersection.set_RL_model_name("MIDDLE")

                universe.intersection_manager.add_intersection(intersection)

                if obj_left_value == 4 or obj_right_value == 4:
                    graph.add_edge(obj_key, obj_left_coordinate, weight=intersection_weight)
                    graph_pod.add_edge(obj_key, obj_left_coordinate, weight=intersection_weight)
                elif obj_left_value == 5 or obj_right_value == 5:
                    graph.add_edge(obj_key, obj_right_coordinate, weight=intersection_weight)
                    graph_pod.add_edge(obj_key, obj_right_coordinate, weight=intersection_weight)

                if obj_above_value == 6 or obj_above_value == 6:
                    graph.add_edge(obj_key, obj_above_coordinate, weight=intersection_weight)
                    graph_pod.add_edge(obj_key, obj_above_coordinate, weight=intersection_weight)
                elif obj_below_value == 7 or obj_below_value == 7:
                    graph.add_edge(obj_key, obj_below_coordinate, weight=intersection_weight)
                    graph_pod.add_edge(obj_key, obj_below_coordinate, weight=intersection_weight)

                if obj_left_value == 6 or obj_left_value == 7:
                    graph.add_edge(obj_key, obj_left_coordinate, weight=intersection_weight)
                    graph_pod.add_edge(obj_key, obj_left_coordinate, weight=intersection_weight)
                elif obj_right_value == 6 or obj_right_value == 7:
                    graph.add_edge(obj_key, obj_right_coordinate, weight=intersection_weight)
                    graph_pod.add_edge(obj_key, obj_right_coordinate, weight=intersection_weight)
            elif value == 4:
                obj.shape = 'arrow-left'
                graph.add_edge(obj_key, obj_left_coordinate, weight=weight)
                graph_pod.add_edge(obj_key, obj_left_coordinate, weight=weight)

                graph.add_edge(obj_key, obj_above_coordinate, weight=turning_weight)
                graph_pod.add_edge(obj_key, obj_above_coordinate, weight=100)
                graph.add_edge(obj_key, obj_below_coordinate, weight=turning_weight)
                graph_pod.add_edge(obj_key, obj_below_coordinate, weight=100)
            elif value == 5:
                obj.shape = 'arrow-right'
                graph.add_edge(obj_key, obj_right_coordinate, weight=weight)
                graph_pod.add_edge(obj_key, obj_right_coordinate, weight=weight)

                graph.add_edge(obj_key, obj_above_coordinate, weight=turning_weight)
                graph_pod.add_edge(obj_key, obj_above_coordinate, weight=100)
                graph.add_edge(obj_key, obj_below_coordinate, weight=turning_weight)
                graph_pod.add_edge(obj_key, obj_below_coordinate, weight=100)
            elif value == 6:
                obj.shape = 'arrow-up'
                graph.add_edge(obj_key, obj_above_coordinate, weight=weight)
                graph_pod.add_edge(obj_key, obj_above_coordinate, weight=weight)

                graph.add_edge(obj_key, obj_left_coordinate, weight=turning_weight)
                graph_pod.add_edge(obj_key, obj_left_coordinate, weight=100)
                graph.add_edge(obj_key, obj_right_coordinate, weight=turning_weight)
                graph_pod.add_edge(obj_key, obj_right_coordinate, weight=100)
            elif value == 7:
                obj.shape = 'arrow-down'
                graph.add_edge(obj_key, obj_below_coordinate, weight=weight)
                graph_pod.add_edge(obj_key, obj_below_coordinate, weight=weight)

                graph.add_edge(obj_key, obj_left_coordinate, weight=weight)
                graph_pod.add_edge(obj_key, obj_left_coordinate, weight=100)
                graph.add_edge(obj_key, obj_right_coordinate, weight=weight)
                graph_pod.add_edge(obj_key, obj_right_coordinate, weight=100)
            elif value == 11 or value == 21:
                obj.shape = 'person-red'
            elif value == 12 or value == 23:
                graph_pod.add_edge(obj_key, obj_right_coordinate, weight=weight)
                obj.shape = 'rail'
            elif value == 13 or value == 22:
                graph_pod.add_edge(obj_key, obj_left_coordinate, weight=weight)
                obj.shape = 'rail'
            elif value == 14 or value == 24:
                if obj_left_value == 11:
                    obj = Station(station_picker_counter, "picker")
                    station_picker_counter += 1
                    obj.pos_x = x
                    obj.pos_y = y
                    obj.coordinate = NetLogoCoordinate(x, y)
                    obj.short_path = construct_station_path(data, x, y, station_type='picking')
                    obj.long_path = construct_station_path(data, x, y, station_type='picking', short_path=False)
                    universe.station_manager.add_station(obj)
                elif obj_right_value == 21:
                    obj = Station(station_replenish_counter, "replenishment")
                    station_replenish_counter += 1
                    obj.pos_x = x
                    obj.pos_y = y
                    obj.coordinate = NetLogoCoordinate(x, y)
                    obj.short_path = construct_station_path(data, x, y, station_type='replenishment')
                    obj.long_path = construct_station_path(data, x, y, station_type='replenishment', short_path=False)
                    universe.station_manager.add_station(obj)

                obj.shape = 'rail-triangle'
                if value == 14:
                    obj.heading = 270
                elif value == 24:
                    obj.heading = 90
                graph_pod.add_edge(obj_key, obj_above_coordinate, weight=weight)
            elif value == 16:
                obj.shape = 'rail-corner'
                obj.heading = 270
                graph_pod.add_edge(obj_key, obj_right_coordinate, weight=weight)
            elif value == 17:
                obj.shape = 'rail-corner'
                graph_pod.add_edge(obj_key, obj_above_coordinate, weight=weight)
            elif value == 18:
                obj.shape = 'rail-corner'
                obj.heading = 180
                graph_pod.add_edge(obj_key, obj_left_coordinate, weight=weight)
            elif value == 19:
                obj.shape = 'rail-corner'
                obj.heading = 90
                graph_pod.add_edge(obj_key, obj_above_coordinate, weight=weight)
            elif value == 26:
                obj.shape = 'rail-corner'
                obj.heading = 180
                graph_pod.add_edge(obj_key, obj_left_coordinate, weight=weight)
            elif value == 27:
                obj.shape = 'rail-corner'
                obj.heading = 90
                graph_pod.add_edge(obj_key, obj_below_coordinate, weight=weight)
            elif value == 28:
                obj.shape = 'rail-corner'
                obj.heading = 270
                graph_pod.add_edge(obj_key, obj_right_coordinate, weight=weight)
            elif value == 29:
                obj.shape = 'rail-corner'
                obj.heading = 0
                graph_pod.add_edge(obj_key, obj_above_coordinate, weight=weight)
            elif value == 99:
                obj.shape = 'empty-space'
            else:
                continue

            if obj_left_coordinate == 13:
                graph_pod.add_edge(obj_key, obj_left_coordinate, weight=weight)

            obj.pos_x = x
            obj.pos_y = y
            total_cols += 1
            universe.addObject(obj)

    universe.set_warehouse_size([total_rows, total_cols])


def construct_station_path(data: DataFrame, start_x, start_y, station_type: str, short_path=True):
    station_path: List[NetLogoCoordinate] = [NetLogoCoordinate(start_x, start_y)]
    row_count, col_count = data.shape

    if station_type not in ['picking', 'replenishment']:
        raise ValueError("station_type must be either 'picking' or 'replenishment'")

    x_increment = 1 if station_type == 'picking' else -1
    if not short_path:
        station_path.insert(0, NetLogoCoordinate(start_x + 1 * x_increment, start_y))
        station_path.insert(0, NetLogoCoordinate(start_x + 2 * x_increment, start_y))
        station_path.insert(0, NetLogoCoordinate(start_x + 2 * x_increment, start_y + 1))
        station_path.insert(0, NetLogoCoordinate(start_x + 1 * x_increment, start_y + 1))

    # go to bottom
    y, x = start_y + 1, start_x
    while 0 <= y < row_count and 0 <= x < col_count and data.iloc[y, x] in (14, 17, 24, 27):
        station_path.insert(0, NetLogoCoordinate(x, y))

        if data.iloc[y, x] in (17, 27):
            x += x_increment
            while 0 <= y < row_count and 0 <= x < col_count and data.iloc[y, x] in (13, 23):
                station_path.insert(0, NetLogoCoordinate(x, y))
                x += x_increment

        y += 1

    return station_path


def add_all_direction_paths(graph, obj_key, weight):
    x, y = map(int, obj_key.split(','))
    directions = {
        'left': (x - 1, y),
        'right': (x + 1, y),
        'up': (x, y + 1),
        'down': (x, y - 1)
    }

    for dir_key, (nx, ny) in directions.items():
        neighbor_key = f"{nx},{ny}"
        graph.add_edge(obj_key, neighbor_key, weight=weight)


def assign_skus_to_pods(pod_manager):
    # Check if pods.csv exists in the current directory
    if _path("pods_csv").exists():
        assign_skus_to_pods_from_file(pod_manager)
    else:
        # Fungsi generate pods.csv
        # PodGenerator(pod_manager).generate()
        PodGenerator(pod_types=[0], pod_num=[420], total_sku=500,
                    #   items_class_conf={"A": 0.07, "B": 0.28, "C": 0.65}, 
                      items_class_conf={"A": 0.1, "B": 0.3, "C": 0.6},
                      items_pods_inventory_levels={"A": 0.4, "B": 0.5, "C": 0.6}, #intial inventory , how much of each class's total inventory should be place in pods
                      items_warehouse_inventory_levels={"A": 0.3, "B": 0.4, "C": 0.5}, #replenishment threshold
                      items_pods_class_conf={"A": 0.7, "B": 0.1, "C": 0.2}, 
                    #   items_warehouse_inventory_levels={"A": 0.4, "B": 0.5, "C": 0.6}, #original
                    #   items_pods_class_conf={"A": 0.6, "B": 0.3, "C": 0.1}, #original 
                    #   items_pods_class_conf={"A": 0.7, "B": 0.2, "C": 0.1}, #data 1 - 8 used this config
                    #   items_pods_class_conf={"A": 0.4, "B": 0.4, "C": 0.2}, # data 10 
                    #   items_pods_class_conf={"A": 0.5, "B": 0.3, "C": 0.2}, # data 11 
                    #   items_pods_class_conf={"A": 0.7, "B": 0.2, "C": 0.1}, # data 12 
               
                      pod_manager=pod_manager,
                      dev_mode=False).generate()
        assign_skus_to_pods_from_file(pod_manager)


def assign_skus_to_pods_from_file(pod_manager: PodManager):

    # ── Phase 1: read all rows and aggregate duplicates per (pod_id, item) ──
    raw_rows = []
    with open(_str_path("pods_csv"), mode='r', newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:
            raw_rows.append(row)

    # Group by (pod_id, item) and aggregate additive fields.
    # Validate that non-additive metadata are identical within each group.
    from collections import OrderedDict
    aggregated = OrderedDict()  # key: (pod_id, sku) -> aggregated record
    for row in raw_rows:
        pod_id = int(row['pod_id'])
        sku = int(row['item'])
        qty = int(row['qty'])
        max_qty = int(row['max_qty'])
        weight = float(row['item_weight'])
        threshold = row['item_pod_inventory_level']
        global_threshold = row['item_warehouse_inventory_level']
        key = (pod_id, sku)

        if key not in aggregated:
            aggregated[key] = {
                'pod_id': pod_id,
                'sku': sku,
                'qty': qty,
                'max_qty': max_qty,
                'weight': weight,
                'threshold': threshold,
                'global_threshold': global_threshold,
            }
        else:
            entry = aggregated[key]
            # Accumulate additive quantities
            entry['qty'] += qty
            entry['max_qty'] += max_qty
            # Validate non-additive metadata consistency
            for field, new_val in [
                ('weight', weight),
                ('threshold', threshold),
                ('global_threshold', global_threshold),
            ]:
                if str(entry[field]) != str(new_val):
                    raise ValueError(
                        f"Conflicting non-additive metadata for pod_id={pod_id}, "
                        f"item={sku}: field='{field}', "
                        f"values=[{entry[field]!r}, {new_val!r}]. "
                        f"Cannot safely aggregate duplicate pod-SKU rows."
                    )

    # ── Phase 2: apply aggregated entries to Pod and PodManager ──
    for (pod_id, sku), entry in aggregated.items():
        pod: Pod = pod_manager.get_pod_by_id(pod_id)
        pod.add_sku(
            sku,
            limit_qty=entry['max_qty'],
            current_qty=entry['qty'],
            threshold=entry['threshold'],
            weight=entry['weight'],
        )
        pod_manager.add_sku_to_pod(sku, pod)
        pod_manager.add_sku_data(
            sku,
            entry['qty'],
            entry['max_qty'],
            entry['global_threshold'],
        )

    # ── Phase 3: post-load inventory invariant check ──
    for sku_id, global_data in pod_manager.skus_data.items():
        pod_current_sum = 0
        pod_max_sum = 0
        for pod in pod_manager.sku_to_pods.get(sku_id, []):
            if sku_id in pod.skus:
                pod_current_sum += pod.skus[sku_id]['current_qty']
                pod_max_sum += pod.skus[sku_id]['limit_qty']
        expected_current = int(global_data['current_global_qty'])
        expected_max = int(global_data['max_global_qty'])
        if pod_current_sum != expected_current:
            raise RuntimeError(
                f"[INVENTORY INVARIANT] SKU {sku_id}: "
                f"global current_qty={expected_current} != "
                f"sum-of-pod current_qty={pod_current_sum}"
            )
        if pod_max_sum != expected_max:
            raise RuntimeError(
                f"[INVENTORY INVARIANT] SKU {sku_id}: "
                f"global max_qty={expected_max} != "
                f"sum-of-pod max_qty={pod_max_sum}"
            )

    csv_file = _str_path("skus_data_csv")
    if os.path.exists(csv_file):
        os.remove(csv_file)
    skus_data = pod_manager.get_all_skus_data()

    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['item_id', 'current_global_qty', 'max_global_qty', 'global_inv_level'])
        for key, value in skus_data.items():
            writer.writerow([key, value['current_global_qty'], value['max_global_qty'], value['global_inv_level']])

    pod_info = pd.DataFrame(columns=["pod_id", "item_id", "qty", "order_id", "processed_time", "task_type"])
    pod_info.to_csv(_str_path("pod_info_csv"), index=False)

    print(f"Data has been saved to {csv_file}")
    df = pd.read_csv(csv_file)
    df_sorted = df.sort_values(by='item_id')
    sorted_csv_file = _str_path("sorted_skus_data_csv")
    df_sorted.to_csv(sorted_csv_file, index=False)


def _initialize_universe():
    ctx = get_run_context()
    ctx.ensure_runtime_dirs()
    _maybe_activate_configured_scenario()
    _prepare_setup_seed()
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    db_path = _str_path("sqlite_db")
    if not is_detail_db_configured():
        detail_env = os.environ.get("RMFS_DETAIL_DB")
        if detail_env is None:
            detail_enabled = (
                os.environ.get("RMFS_FAST_TRAIN", "0").strip().lower()
                not in {"1", "true", "yes", "on"}
            )
        else:
            detail_enabled = detail_env.strip().lower() in {"1", "true", "yes", "on"}
        configure_detail_db(enabled=detail_enabled, db_path=db_path)
    configure_default_pod_location_db_path(db_path)
    configure_default_pod_travel_db_path(db_path)

    initialize_job_task_table(timestamp, db_path=db_path)
    initialize_order_history_table(timestamp, db_path=db_path)
    initialize_pod_location_table(timestamp, db_path=db_path)
    initialize_pod_travel_table(timestamp, db_path=db_path)

    clear_job_task_table(db_path=db_path)
    clear_order_history(db_path=db_path)
    clear_pod_locations(db_path=db_path)
    clear_pod_travel(db_path=db_path)
    for path_attr in ("assign_order_csv", "pod_info_csv"):
        path = _str_path(path_attr)
        if os.path.exists(path):
            os.remove(path)

    # The order stream is regenerated on every setup so bootstrap seeds take
    # effect immediately and no stale synthetic files are reused.
    for generated_attr in (
        "generated_order_csv",
        "generated_database_order_csv",
        "generated_order_meta_json",
    ):
        generated_path = _str_path(generated_attr)
        if os.path.exists(generated_path):
            os.remove(generated_path)

    warehouse = Inventory(runtime_paths=ctx.inventory_paths(), sqlite_db_path=db_path)
    _apply_runtime_config(warehouse)
    draw_layout(warehouse)
    warehouse.tick_to_second = 0.15
    configure_pps_rl_strategy(warehouse, items_csv=_str_path("items_csv"), skus_data_csv=_str_path("skus_data_csv"))
    return warehouse, warehouse.generateResult()[0]


def _reattach_universe(warehouse):
    for obj in warehouse._objects:
        obj.setUniverse(warehouse)
    configure_pps_rl_strategy(warehouse, items_csv=_str_path("items_csv"), skus_data_csv=_str_path("skus_data_csv"))


def _tick_payload(warehouse, next_result):
    return [
        next_result[0],
        warehouse.total_energy,
        len(warehouse.job_queue),
        warehouse.stop_and_go,
        warehouse.total_turning,
        next_result[1],
        _get_throughput(warehouse),
        _get_avg_order_completion_time(warehouse),
        _get_pod_visits(warehouse),
        _get_pile_on_rate(warehouse),
        _get_picked_quantity(warehouse),
    ]


def _netlogo_step(warehouse):
    tick_to_second = getattr(warehouse, "tick_to_second", None)
    if tick_to_second in (None, 0):
        return None
    return int(round(float(getattr(warehouse, "_tick", 0.0)) / float(tick_to_second)))


def _run_semantic_tick(warehouse):
    _reattach_universe(warehouse)
    next_result = warehouse.tick()
    _apply_pps_rl_policy(warehouse)
    return _tick_payload(warehouse, next_result)


def _persist_universe(warehouse):
    with timed("pickle_dump"):
        with open(_str_path("state_file"), "wb") as config_dictionary_file:
            pickle.dump(warehouse, config_dictionary_file)


def _load_universe():
    with timed("pickle_load"):
        with open(_str_path("state_file"), "rb") as file:
            return pickle.load(file)


def _horizon_limit():
    limit = _env_int("RMFS_RUN_HORIZON_TICKS")
    return limit if limit is not None and limit > 0 else None


def setup_in_memory(*, persist_initial_state: bool = False):
    warehouse, setup_payload = _initialize_universe()
    if persist_initial_state:
        _persist_universe(warehouse)
    return warehouse, setup_payload


def tick_in_memory(warehouse) -> SimulationStepResult:
    payload = _run_semantic_tick(warehouse)
    return SimulationStepResult(
        status=SimulationTermination.RUNNING,
        payload=payload,
        steps_executed=1,
        warehouse_time=float(getattr(warehouse, "_tick", 0.0)),
        netlogo_step=_netlogo_step(warehouse),
    )


def _censor_status_for_termination(reason: SimulationTermination) -> str:
    return {
        SimulationTermination.MAXIMUM_HORIZON: "censored_maximum_horizon",
        SimulationTermination.CONGESTION: "censored_congestion",
        SimulationTermination.NO_ACTIVE_WORK: "censored_no_active_work",
        SimulationTermination.MANUAL_CANCELLATION: "censored_manual_cancellation",
        SimulationTermination.WORKER_EXCEPTION: "censored_worker_exception",
    }.get(reason, "censored_run_end")


def finalize_headless_run(
    warehouse,
    *,
    reason: SimulationTermination | str,
    success: bool,
    persist_final_state: bool = True,
) -> dict:
    if warehouse is None:
        return {"finalized": False, "reason": str(reason), "success": bool(success)}
    if getattr(warehouse, "_rmfs_finalized", False):
        return {"finalized": False, "already_finalized": True, "reason": str(reason), "success": bool(success)}
    warehouse._rmfs_finalized = True
    reason_enum = reason if isinstance(reason, SimulationTermination) else SimulationTermination(str(reason))
    diagnostics = {
        "finalized": True,
        "reason": reason_enum.value,
        "success": bool(success),
        "warehouse_time": float(getattr(warehouse, "_tick", 0.0)),
        "netlogo_step": _netlogo_step(warehouse),
        "cancelled_committed_next_reservations": 0,
        "released_charger_claims": 0,
    }
    runtime = getattr(warehouse, "rts_rollout_runtime", None)
    if runtime is not None:
        runtime.censor_all_pending(
            status=_censor_status_for_termination(reason_enum),
            reason=reason_enum.value,
            warehouse=warehouse,
        )
    registry = getattr(warehouse, "committed_next_registry", None)
    if registry is not None:
        cancelled = registry.cancel_all(f"run_end_{reason_enum.value}")
        diagnostics["cancelled_committed_next_reservations"] = len(cancelled)
    for robot in getattr(warehouse, "_objects", []) or []:
        if getattr(robot, "object_type", None) != "robot":
            continue
        claimed = getattr(robot, "_claimed_charger", None)
        if claimed is not None:
            robot._release_charger()
            diagnostics["released_charger_claims"] += 1
    if runtime is not None:
        runtime.close(censor_status=_censor_status_for_termination(reason_enum), reason=reason_enum.value)
    invariants = check_runtime_invariants(warehouse)
    diagnostics["runtime_invariants"] = invariants
    fail_on_invariants = _env_bool("RMFS_RTS_FAIL_ON_INVARIANTS", False) or _env_bool("RMFS_DEBUG_TRACE", False)
    if fail_on_invariants and invariants.get("hard_violation_count", 0):
        raise RuntimeError(f"RTS runtime invariant violations: {invariants}")
    if persist_final_state:
        _persist_universe(warehouse)
    return diagnostics


class HeadlessSimulationSession:
    """Resident worker path; GUI compatibility continues through setup()/tick()."""

    def __init__(self, *, persist_final_state: bool = False):
        self.persist_final_state = bool(persist_final_state)
        self.warehouse = None
        self.setup_payload = None
        self.steps_completed = 0
        self.finalized = False

    def setup(self):
        self.warehouse, self.setup_payload = setup_in_memory(persist_initial_state=False)
        return self.setup_payload

    def step(self) -> SimulationStepResult:
        if self.warehouse is None:
            raise RuntimeError("HeadlessSimulationSession.setup() must be called before step()")
        result = tick_in_memory(self.warehouse)
        self.steps_completed += result.steps_executed
        return result

    def finalize(self, *, reason: SimulationTermination, success: bool) -> dict:
        self.finalized = True
        return finalize_headless_run(
            self.warehouse,
            reason=reason,
            success=success,
            persist_final_state=self.persist_final_state,
        )


def setup():
    try:
        warehouse, setup_payload = setup_in_memory(persist_initial_state=True)
        return setup_payload
    except Exception:
        traceback.print_exc()
        return "An error occurred. See the details above."


def tick():
    try:
        warehouse = _load_universe()
        limit = _horizon_limit()
        if limit is not None and getattr(warehouse, "_tick", 0) >= limit:
            return SimulationStepResult(
                status=SimulationTermination.MAXIMUM_HORIZON,
                steps_executed=0,
                warehouse_time=float(getattr(warehouse, "_tick", 0.0)),
                netlogo_step=_netlogo_step(warehouse),
                terminal_reason=SimulationTermination.MAXIMUM_HORIZON.value,
            )
        payload = _run_semantic_tick(warehouse)
        _print_gui_tick_status(warehouse)
        _persist_universe(warehouse)
        return payload
    except Exception:
        traceback.print_exc()
        return "An error occurred. See the details above."
    
def console_tick():
    try:
        warehouse = _load_universe()
        last_payload = None
        limit = _horizon_limit() or 100000
        while getattr(warehouse, "_tick", 0) < limit:
            last_payload = _run_semantic_tick(warehouse)
        _persist_universe(warehouse)
        return SimulationStepResult(
            status=SimulationTermination.MAXIMUM_HORIZON,
            payload=last_payload,
            steps_executed=0,
            warehouse_time=float(getattr(warehouse, "_tick", 0.0)),
            netlogo_step=_netlogo_step(warehouse),
            terminal_reason=SimulationTermination.MAXIMUM_HORIZON.value,
        )
    except Exception:
        traceback.print_exc()
        return "An error occurred. See the details above."


def setup_py():
    def install_package(package_name):
        from pip._internal import main as pipmain
        pipmain(['install', package_name])

    # List of packages to install
    packages = ["networkx", "matplotlib"]

    # Install each package
    for package in packages:
        install_package(package)
