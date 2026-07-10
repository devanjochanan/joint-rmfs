"""Canonical KPI schema definitions for the RMFS sensitivity campaign.

This module defines the **full_kpi_v3** output contract with two classes
of fields:

* **Primitive fields** — recorded directly by the simulator.
* **Derived fields** — calculated once from primitives during finalization.

Every derived KPI can be recomputed from clearly defined primitive
counters, eliminating inconsistent duplicates.

The v2 schema (``SENSITIVITY_KPI_FIELDS``) is preserved in
``local_executor.py`` for backward compatibility.  This module defines
only the v3 contract; the finalization function that populates the
fields lives in ``local_executor.py``.
"""

from __future__ import annotations

# ── schema version ────────────────────────────────────────────────

FULL_KPI_V3_SCHEMA_VERSION = "full_kpi_v3"

# ── field type annotations (informational) ────────────────────────

FULL_KPI_V3_PRIMITIVE_FIELDS: tuple[str, ...] = ()
FULL_KPI_V3_DERIVED_FIELDS: tuple[str, ...] = ()

# ================================================================
# 1. Order-flow KPIs
# ================================================================

_ORDER_FLOW_PRIMITIVE = (
    "orders_available_in_source",   # input metadata — replaces orders_generated
    "orders_released",
    "orders_started",
    "orders_completed",
    "orders_unfinished_end",
    "order_lines_released",
    "order_lines_picked",
    "order_lines_completed",
    "order_lines_unfinished_end",
    "order_lines_blocked_stockout",
    "sum_completed_order_cycle_time_s",
    "completed_order_cycle_time_count",
    "max_completed_order_cycle_time_s",
    "oldest_unfinished_order_age_s",
)

_ORDER_FLOW_DERIVED = (
    "throughput_orders_per_hour",
    "order_completion_ratio",
    "order_line_completion_ratio",
    "mean_completed_order_cycle_time_s",
)

# ================================================================
# 2. Picking and PPS KPIs
# ================================================================

_PICKING_PPS_PRIMITIVE = (
    "pps_decision_rounds",
    "pps_candidates_evaluated",
    "pps_actions_zero",
    "pps_actions_invalid",
    "pps_rejected_station_full",
    "pps_rejected_station_no_orders",
    "pps_rejected_sku_mismatch",
    "pps_rejected_pod_ineligible",
    "pps_rejected_pod_reserved",
    "pps_assignments_accepted",
    "picking_jobs_created",
    "picking_jobs_completed",
    "picking_pod_visits",
    "picked_items_count",
)

_PICKING_PPS_DERIVED = (
    "pps_assignment_acceptance_rate",
    "pps_assignment_to_job_conversion_rate",
    "pile_on_items_per_picking_visit",
)

# ================================================================
# 3. Robot-job lifecycle KPIs (per task type)
# ================================================================

_JOB_TASK_TYPES = ("picking", "replenishment", "return_to_storage")

_JOB_LIFECYCLE_SUFFIXES = (
    "_jobs_created",
    "_jobs_assigned",
    "_jobs_started",
    "_jobs_completed",
    "_jobs_cancelled",
    "_jobs_requeued",
    "_jobs_blocked",
    "_jobs_pending_end",
    "_max_job_age_s",
)

_JOB_LIFECYCLE_PRIMITIVE = tuple(
    f"{task_type}{suffix}"
    for task_type in _JOB_TASK_TYPES
    for suffix in _JOB_LIFECYCLE_SUFFIXES
) + (
    "job_queue_time_integral",
    "job_queue_max",
    "job_queue_observation_time_s",
)

_JOB_LIFECYCLE_DERIVED = (
    "mean_job_queue_length",
)

# ================================================================
# 4. Robot-cycle and movement KPIs
# ================================================================

_ROBOT_MOVEMENT_PRIMITIVE = (
    "robot_cycles_completed",
    "sum_robot_cycle_time_s",
    "max_robot_cycle_time_s",
    "loaded_robot_distance",
    "empty_robot_distance",
    "stop_and_go_count",
    "turning_count",
)

_ROBOT_MOVEMENT_DERIVED = (
    "mean_robot_cycle_time_s",
    "total_robot_distance",
    "robot_distance_per_completed_order",
    "loaded_distance_share",
    "empty_distance_share",
)

# ================================================================
# 5. Robot-state utilization KPIs (mutually exclusive robot-seconds)
# ================================================================

_ROBOT_UTILIZATION_PRIMITIVE = (
    "robot_seconds_available",
    "robot_seconds_working",
    "robot_seconds_idle",
    "robot_seconds_going_to_charge",
    "robot_seconds_waiting_for_charger",
    "robot_seconds_physically_charging",
    "robot_seconds_dead",
)

_ROBOT_UTILIZATION_DERIVED = (
    "fleet_available_fraction",
    "fleet_working_fraction",
    "fleet_idle_fraction",
    "fleet_charging_fraction",
    "fleet_dead_fraction",
)

# ================================================================
# 6. Charging KPIs
# ================================================================

_CHARGING_PRIMITIVE = (
    "charging_requests",
    "charging_sessions_started",
    "charging_sessions_completed",
    "charging_sessions_interrupted",
    "drive_by_charging_events",
    "charger_route_failures",
    "charger_unavailable_events",
    "charger_claims_created",
    "charger_claims_released",
    "stale_charger_claims_detected",
    "robots_died_from_battery",
    "first_low_soc_time_s",
    "first_robot_death_time_s",
    "initial_battery_energy_j",
    "final_battery_energy_j",
    "minimum_final_robot_soc",
    "mean_final_robot_soc",
    "charging_energy_added_kj",
)

# No derived charging fields — time-in-charging is already in robot-state.

# ================================================================
# 7. Replenishment KPIs
# ================================================================

_REPLENISHMENT_PRIMITIVE = (
    "proactive_replenishment_requests",
    "post_pick_replenishment_requests",
    "rts_replenish_store_selected",
    "rts_replenish_store_masked",
    "replenishment_jobs_created",
    "replenishment_jobs_started",
    "replenishment_jobs_completed",
    "replenishment_trips_completed",
    "replenishment_units_restored",
    "replenishment_pending_end",
    "replenishment_queued_end",
    "replenishment_active_end",
    "replenishment_cap_blocks_proactive",
    "replenishment_cap_blocks_post_pick",
    "replenishment_cap_blocks_rts",
    "replenishment_station_busy_seconds",
    "replenishment_station_idle_seconds",
    "pods_awaiting_replenishment_end",
    "max_replenishment_request_age_s",
)

_REPLENISHMENT_DERIVED = (
    "replenishment_units_per_trip",
    "replenishment_to_picking_visit_ratio",
    "replenishment_station_utilization",
)

# ================================================================
# 8. Committed-next and ownership KPIs
# ================================================================

_COMMITTED_NEXT_PRIMITIVE = (
    "committed_next_reservations_created",
    "committed_next_reservations_activated",
    "committed_next_reservations_cancelled_charging",
    "committed_next_reservations_cancelled_other",
    "committed_next_reservations_stale_end",
    "committed_next_jobs_restored",
    "committed_next_jobs_lost",
    "committed_next_duplicate_restorations",
    "max_committed_next_reservation_age_s",
    "stale_rts_ownership_markers_end",
    "stale_replenishment_markers_end",
    "max_ownership_marker_age_s",
)

# ================================================================
# 9. Station-performance KPIs
# ================================================================
# Per-station detail is stored in a structured sidecar field
# ("station_detail"), not as flat columns. Only system-level
# derived metrics appear in the flat KPI namespace.

_STATION_SIDECAR_FIELD = (
    "station_detail",   # JSON dict: station_id -> {assignments, pod_visits, ...}
)

_STATION_DERIVED = (
    "max_station_assignment_share",
    "station_assignment_imbalance_ratio",
    "mean_station_utilization",
    "max_station_utilization",
    "min_station_utilization",
)

# ================================================================
# 10. Energy KPIs
# ================================================================

_ENERGY_PRIMITIVE = (
    "movement_energy_kj",
    "fixed_load_energy_kj",
    # charging_energy_added_kj already in §6
    "lifting_energy_kj",
    "rotation_energy_kj",
)

_ENERGY_DERIVED = (
    "total_consumed_energy_kj",
    "energy_per_completed_order_kj",
    "energy_per_completed_order_line_kj",
    "mean_consumed_energy_per_robot_kj",
    "battery_energy_balance_residual_kj",
)

# ================================================================
# 11. Operational-health KPIs
# ================================================================

_OPERATIONAL_HEALTH_PRIMITIVE = (
    "longest_no_order_completion_interval_s",
    "longest_no_picking_job_created_interval_s",
    "longest_no_robot_job_completed_interval_s",
    "first_starvation_time_s",
    "first_busy_unproductive_time_s",
    "first_congestion_time_s",
    "starvation_detected",
    "busy_unproductive_detected",
    "congestion_detected",
    "simulation_termination_reason",
    "simulation_completed_full_horizon",
)

# ================================================================
# 12. Run metadata — NOT in KPI namespace
# ================================================================
# These belong in the result identity / provenance section and are
# recorded in run_spec.json and the run-outcome identity block.
# Listed here for documentation only.

RUN_METADATA_FIELDS = (
    "campaign_id",
    "condition_key",
    "host_id",
    "replication",
    "seed",
    "simulated_horizon_seconds",
    "simulated_ticks",
    "robot_count",
    "picking_station_count",
    "replenishment_station_count",
    "order_rate_per_hour",
    "policy_configuration",
    "pps_mode",
    "rts_policy_mode",
    "charging_enabled",
    "committed_next_reservations_enabled",
    "source_commit",
    "source_tree_hash",
    "rts_checkpoint_id",
    "rts_checkpoint_sha256",
    "pps_model_sha256",
    "charging_config_sha256",
    "kpi_schema_version",
    "warning_count",
    "retry_count",
    "run_status",
)

# ================================================================
# Assembled field tuples
# ================================================================

FULL_KPI_V3_PRIMITIVE_FIELDS = (
    *_ORDER_FLOW_PRIMITIVE,
    *_PICKING_PPS_PRIMITIVE,
    *_JOB_LIFECYCLE_PRIMITIVE,
    *_ROBOT_MOVEMENT_PRIMITIVE,
    *_ROBOT_UTILIZATION_PRIMITIVE,
    *_CHARGING_PRIMITIVE,
    *_REPLENISHMENT_PRIMITIVE,
    *_COMMITTED_NEXT_PRIMITIVE,
    *_ENERGY_PRIMITIVE,
    *_OPERATIONAL_HEALTH_PRIMITIVE,
)

FULL_KPI_V3_DERIVED_FIELDS = (
    *_ORDER_FLOW_DERIVED,
    *_PICKING_PPS_DERIVED,
    *_JOB_LIFECYCLE_DERIVED,
    *_ROBOT_MOVEMENT_DERIVED,
    *_ROBOT_UTILIZATION_DERIVED,
    *_REPLENISHMENT_DERIVED,
    *_STATION_DERIVED,
    *_ENERGY_DERIVED,
)

# Sidecar structured fields (not flat KPI columns)
FULL_KPI_V3_SIDECAR_FIELDS = (
    *_STATION_SIDECAR_FIELD,
)

# All KPI fields in the flat namespace (primitive + derived)
FULL_KPI_V3_FIELDS = (
    *FULL_KPI_V3_PRIMITIVE_FIELDS,
    *FULL_KPI_V3_DERIVED_FIELDS,
)

# Complete field set including sidecars
FULL_KPI_V3_ALL_FIELDS = (
    *FULL_KPI_V3_FIELDS,
    *FULL_KPI_V3_SIDECAR_FIELDS,
)

# ================================================================
# V2 → V3 deprecation / rename mapping
# ================================================================
# Used during finalization to map old warehouse attribute names to
# the canonical v3 field names.

V2_TO_V3_FIELD_MAP: dict[str, str | None] = {
    "orders_generated": None,                          # split: orders_available_in_source + orders_released
    "order_throughput": "throughput_orders_per_hour",
    "replenishment_count": "replenishment_units_restored",
    "replenishment_trips": "replenishment_trips_completed",
    "pod_visit_picking": "picking_pod_visits",
    "pod_visit_replenishment": "replenishment_trips_completed",
    "robot_idle_time": "robot_seconds_idle",
    "total_energy_kj": "total_consumed_energy_kj",
    "average_energy": "mean_consumed_energy_per_robot_kj",
    "energy_efficiency": None,                          # derive from energy_per_completed_order_kj
    "energy_efficiency_fixed": None,                    # removed; fixed_load is a component
    "pile_on": "pile_on_items_per_picking_visit",
    "total_robot_distance": None,                       # derived: loaded + empty
    "robot_distance_per_order": "robot_distance_per_completed_order",
    "average_order_cycle_time": "mean_completed_order_cycle_time_s",
    "maximum_order_cycle_time": "max_completed_order_cycle_time_s",
    "robot_cycle_time": "mean_robot_cycle_time_s",
    "average_job_queue": "mean_job_queue_length",
    "peak_job_queue": "job_queue_max",
    "max_station_assignment_share": "max_station_assignment_share",
    "station_imbalance_ratio": "station_assignment_imbalance_ratio",
}

# ================================================================
# 12. Interval snapshot field list
# ================================================================

KPI_SNAPSHOT_CADENCE_SECONDS_DEFAULT = [1000.0, 2500.0, 5000.0, 10000.0]

INTERVAL_SNAPSHOT_FIELDS = (
    "snapshot_simulated_seconds",
    # Order flow
    "orders_released",
    "orders_completed",
    "order_lines_completed",
    # PPS / picking
    "pps_assignments_accepted",
    "picking_jobs_created",
    "picking_jobs_completed",
    "picking_pod_visits",
    # Job queue
    "job_queue_max",
    # Replenishment
    "replenishment_jobs_completed",
    "replenishment_trips_completed",
    "replenishment_pending_end",
    "replenishment_queued_end",
    "replenishment_active_end",
    # Robot state
    "robot_seconds_available",
    "robot_seconds_working",
    "robot_seconds_idle",
    "robot_seconds_waiting_for_charger",
    "robot_seconds_physically_charging",
    "robot_seconds_dead",
    # Charger claims
    "charger_claims_created",
    "stale_charger_claims_detected",
    # Station
    "max_station_assignment_share",
    # Distance
    "loaded_robot_distance",
    "empty_robot_distance",
    # Energy
    "movement_energy_kj",
    "fixed_load_energy_kj",
    "charging_energy_added_kj",
)


# ================================================================
# Derivation helpers
# ================================================================

def _safe_div(numerator: float | None, denominator: float | None) -> float | None:
    """Return numerator / denominator, or None if invalid."""
    if numerator is None or denominator is None:
        return None
    try:
        n = float(numerator)
        d = float(denominator)
    except (TypeError, ValueError):
        return None
    if d <= 0.0:
        return None
    return n / d


def derive_v3_fields(primitives: dict[str, object]) -> dict[str, object]:
    """Compute all derived v3 KPI fields from a primitives dict.

    Missing primitives produce ``None`` for the corresponding derived field.
    """
    def _get(key: str) -> float | None:
        val = primitives.get(key)
        if val is None:
            return None
        try:
            return float(val)
        except (TypeError, ValueError):
            return None

    derived: dict[str, object] = {}

    # ── 1. Order flow ─────────────────────────────────────────────
    sim_seconds = _get("simulated_seconds")
    orders_completed = _get("orders_completed")
    orders_released = _get("orders_released")
    order_lines_completed = _get("order_lines_completed")
    order_lines_released = _get("order_lines_released")

    derived["throughput_orders_per_hour"] = (
        _safe_div(orders_completed, sim_seconds / 3600.0)
        if sim_seconds is not None and sim_seconds > 0 else None
    )
    derived["order_completion_ratio"] = _safe_div(orders_completed, orders_released)
    derived["order_line_completion_ratio"] = _safe_div(order_lines_completed, order_lines_released)
    derived["mean_completed_order_cycle_time_s"] = _safe_div(
        _get("sum_completed_order_cycle_time_s"),
        _get("completed_order_cycle_time_count"),
    )

    # ── 2. Picking / PPS ─────────────────────────────────────────
    pps_candidates = _get("pps_candidates_evaluated")
    pps_accepted = _get("pps_assignments_accepted")
    picking_created = _get("picking_jobs_created")
    picking_visits = _get("picking_pod_visits")
    picked_items = _get("picked_items_count")

    derived["pps_assignment_acceptance_rate"] = _safe_div(pps_accepted, pps_candidates)
    derived["pps_assignment_to_job_conversion_rate"] = _safe_div(picking_created, pps_accepted)
    derived["pile_on_items_per_picking_visit"] = _safe_div(picked_items, picking_visits)

    # ── 3. Job lifecycle ──────────────────────────────────────────
    derived["mean_job_queue_length"] = _safe_div(
        _get("job_queue_time_integral"),
        _get("job_queue_observation_time_s"),
    )

    # ── 4. Robot movement ─────────────────────────────────────────
    derived["mean_robot_cycle_time_s"] = _safe_div(
        _get("sum_robot_cycle_time_s"),
        _get("robot_cycles_completed"),
    )
    loaded = _get("loaded_robot_distance")
    empty = _get("empty_robot_distance")
    total_dist = None
    if loaded is not None and empty is not None:
        total_dist = loaded + empty
    derived["total_robot_distance"] = total_dist
    derived["robot_distance_per_completed_order"] = _safe_div(total_dist, orders_completed)
    derived["loaded_distance_share"] = _safe_div(loaded, total_dist)
    derived["empty_distance_share"] = _safe_div(empty, total_dist)

    # ── 5. Robot-state utilization ────────────────────────────────
    available = _get("robot_seconds_available")
    working = _get("robot_seconds_working")
    idle = _get("robot_seconds_idle")
    going_charge = _get("robot_seconds_going_to_charge")
    waiting_charge = _get("robot_seconds_waiting_for_charger")
    phys_charge = _get("robot_seconds_physically_charging")
    dead = _get("robot_seconds_dead")

    total_robot_seconds = None
    if all(v is not None for v in (available, working, idle, going_charge, waiting_charge, phys_charge, dead)):
        total_robot_seconds = available + working + idle + going_charge + waiting_charge + phys_charge + dead

    derived["fleet_available_fraction"] = _safe_div(available, total_robot_seconds)
    derived["fleet_working_fraction"] = _safe_div(working, total_robot_seconds)
    derived["fleet_idle_fraction"] = _safe_div(idle, total_robot_seconds)
    charging_total = None
    if all(v is not None for v in (going_charge, waiting_charge, phys_charge)):
        charging_total = going_charge + waiting_charge + phys_charge
    derived["fleet_charging_fraction"] = _safe_div(charging_total, total_robot_seconds)
    derived["fleet_dead_fraction"] = _safe_div(dead, total_robot_seconds)

    # ── 7. Replenishment ──────────────────────────────────────────
    derived["replenishment_units_per_trip"] = _safe_div(
        _get("replenishment_units_restored"),
        _get("replenishment_trips_completed"),
    )
    derived["replenishment_to_picking_visit_ratio"] = _safe_div(
        _get("replenishment_trips_completed"),
        picking_visits,
    )
    rep_busy = _get("replenishment_station_busy_seconds")
    rep_idle = _get("replenishment_station_idle_seconds")
    rep_total = None
    if rep_busy is not None and rep_idle is not None:
        rep_total = rep_busy + rep_idle
    derived["replenishment_station_utilization"] = _safe_div(rep_busy, rep_total)

    # ── 9. Station ────────────────────────────────────────────────
    # max_station_assignment_share, imbalance, utilizations are
    # computed from station_detail sidecar during finalization, not
    # here — they need per-station data.
    for field in _STATION_DERIVED:
        if field not in derived:
            derived[field] = primitives.get(field)

    # ── 10. Energy ────────────────────────────────────────────────
    movement_e = _get("movement_energy_kj")
    fixed_e = _get("fixed_load_energy_kj")
    lifting_e = _get("lifting_energy_kj")
    rotation_e = _get("rotation_energy_kj")
    components = [v for v in (movement_e, fixed_e, lifting_e, rotation_e) if v is not None]
    total_energy = sum(components) if components else None
    derived["total_consumed_energy_kj"] = total_energy
    derived["energy_per_completed_order_kj"] = _safe_div(total_energy, orders_completed)
    derived["energy_per_completed_order_line_kj"] = _safe_div(total_energy, order_lines_completed)
    # mean_consumed_energy_per_robot_kj needs robot_count from metadata
    robot_count = _get("robot_count")
    derived["mean_consumed_energy_per_robot_kj"] = _safe_div(total_energy, robot_count)
    # residual = initial_battery - final_battery + charging_added - total_consumed
    initial_batt = _get("initial_battery_energy_j")
    final_batt = _get("final_battery_energy_j")
    charge_added = _get("charging_energy_added_kj")
    if all(v is not None for v in (initial_batt, final_batt, total_energy, charge_added)):
        derived["battery_energy_balance_residual_kj"] = (
            (initial_batt - final_batt) / 1000.0 + charge_added - total_energy
        )
    else:
        derived["battery_energy_balance_residual_kj"] = None

    return derived


def empty_v3_kpi_payload() -> dict[str, object]:
    """Return a dict with all v3 KPI fields initialized to ``None``."""
    payload: dict[str, object] = {}
    for field in FULL_KPI_V3_ALL_FIELDS:
        payload[field] = None
    return payload
