"""Run-local order generation policy helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass

from .run_profiles import resolve_run_profile


TRUE_VALUES = {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class OrderGenerationPolicy:
    profile: str
    order_generation_mode: str
    full_raw_order_replay: bool
    bootstrap_n_orders: int | None
    run_horizon_ticks: int | None
    demand_horizon_ticks: int | None
    demand_buffer_ticks: int


def _env_int(name: str) -> int | None:
    value = os.environ.get(name)
    if value is None or str(value).strip() == "":
        return None
    return int(value)


def env_full_raw_replay() -> bool:
    return os.environ.get("RMFS_FULL_RAW_ORDER_REPLAY", "").strip().lower() in TRUE_VALUES


def resolve_order_generation_policy(
    *,
    profile: str | None = None,
    n_orders: int | None = None,
    run_horizon_ticks: int | None = None,
    demand_horizon_ticks: int | None = None,
    demand_buffer_ticks: int | None = None,
    order_generation_mode: str | None = None,
    full_raw_order_replay: bool | None = None,
) -> OrderGenerationPolicy:
    profile_name = profile or os.environ.get("RMFS_RUN_PROFILE", "gui")
    explicit_n_orders = n_orders if n_orders is not None else _env_int("RMFS_BOOTSTRAP_N_ORDERS")
    full_replay = env_full_raw_replay() if full_raw_order_replay is None else bool(full_raw_order_replay)
    resolved_profile = resolve_run_profile(
        profile_name,
        run_horizon_ticks=run_horizon_ticks if run_horizon_ticks is not None else _env_int("RMFS_RUN_HORIZON_TICKS"),
        bootstrap_n_orders=explicit_n_orders,
        demand_horizon_ticks=demand_horizon_ticks if demand_horizon_ticks is not None else _env_int("RMFS_DEMAND_HORIZON_TICKS"),
        demand_buffer_ticks=demand_buffer_ticks if demand_buffer_ticks is not None else _env_int("RMFS_DEMAND_BUFFER_TICKS"),
        order_generation_mode=order_generation_mode or os.environ.get("RMFS_ORDER_GENERATION_MODE"),
        full_raw_order_replay=full_replay,
    )
    return OrderGenerationPolicy(
        profile=resolved_profile.profile,
        order_generation_mode=resolved_profile.order_generation_mode,
        full_raw_order_replay=resolved_profile.full_raw_order_replay,
        bootstrap_n_orders=resolved_profile.bootstrap_n_orders,
        run_horizon_ticks=resolved_profile.run_horizon_ticks,
        demand_horizon_ticks=resolved_profile.demand_horizon_ticks,
        demand_buffer_ticks=resolved_profile.demand_buffer_ticks,
    )
