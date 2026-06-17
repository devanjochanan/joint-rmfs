"""Human-facing run profile defaults for RMFS."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal


RunProfileName = Literal["smoke", "training", "ablation", "debug", "gui"]


@dataclass(frozen=True)
class RunProfile:
    profile: str
    run_horizon_ticks: int | None
    bootstrap_n_orders: int | None
    demand_horizon_ticks: int | None
    demand_buffer_ticks: int
    order_generation_mode: str
    full_raw_order_replay: bool
    detail_db: bool
    debug_trace: bool
    keep_runtime_artifacts: bool
    pod_location_mode: str
    pod_location_seed: int | None = None

    def env(self) -> dict[str, str]:
        env = {
            "RMFS_RUN_PROFILE": self.profile,
            "RMFS_ORDER_GENERATION_MODE": self.order_generation_mode,
            "RMFS_FULL_RAW_ORDER_REPLAY": "1" if self.full_raw_order_replay else "0",
            "RMFS_DETAIL_DB": "1" if self.detail_db else "0",
            "RMFS_POD_LOCATION_MODE": self.pod_location_mode,
        }
        if self.run_horizon_ticks is not None:
            env["RMFS_RUN_HORIZON_TICKS"] = str(int(self.run_horizon_ticks))
        if self.bootstrap_n_orders is not None:
            env["RMFS_BOOTSTRAP_N_ORDERS"] = str(int(self.bootstrap_n_orders))
        if self.demand_horizon_ticks is not None:
            env["RMFS_DEMAND_HORIZON_TICKS"] = str(int(self.demand_horizon_ticks))
        env["RMFS_DEMAND_BUFFER_TICKS"] = str(int(self.demand_buffer_ticks))
        if self.pod_location_seed is not None:
            env["RMFS_POD_LOCATION_SEED"] = str(int(self.pod_location_seed))
        return env


_PROFILES = {
    "smoke": RunProfile(
        profile="smoke",
        run_horizon_ticks=100,
        bootstrap_n_orders=100,
        demand_horizon_ticks=1_100,
        demand_buffer_ticks=1_000,
        order_generation_mode="controlled_count",
        full_raw_order_replay=False,
        detail_db=False,
        debug_trace=False,
        keep_runtime_artifacts=False,
        pod_location_mode="randomize_slots",
    ),
    "training": RunProfile(
        profile="training",
        run_horizon_ticks=5_000,
        bootstrap_n_orders=1_000,
        demand_horizon_ticks=6_000,
        demand_buffer_ticks=1_000,
        order_generation_mode="controlled_count",
        full_raw_order_replay=False,
        detail_db=False,
        debug_trace=False,
        keep_runtime_artifacts=False,
        pod_location_mode="randomize_slots",
    ),
    "ablation": RunProfile(
        profile="ablation",
        run_horizon_ticks=100_000,
        bootstrap_n_orders=10_000,
        demand_horizon_ticks=105_000,
        demand_buffer_ticks=5_000,
        order_generation_mode="controlled_count",
        full_raw_order_replay=False,
        detail_db=False,
        debug_trace=False,
        keep_runtime_artifacts=False,
        pod_location_mode="randomize_slots",
    ),
    "debug": RunProfile(
        profile="debug",
        run_horizon_ticks=100,
        bootstrap_n_orders=100,
        demand_horizon_ticks=1_100,
        demand_buffer_ticks=1_000,
        order_generation_mode="controlled_count",
        full_raw_order_replay=False,
        detail_db=True,
        debug_trace=True,
        keep_runtime_artifacts=True,
        pod_location_mode="randomize_slots",
    ),
    "gui": RunProfile(
        profile="gui",
        run_horizon_ticks=None,
        bootstrap_n_orders=None,
        demand_horizon_ticks=None,
        demand_buffer_ticks=0,
        order_generation_mode="legacy_compat",
        full_raw_order_replay=False,
        detail_db=True,
        debug_trace=False,
        keep_runtime_artifacts=True,
        pod_location_mode="fixed",
    ),
}


def available_profiles() -> tuple[str, ...]:
    return tuple(_PROFILES)


def resolve_run_profile(
    profile: str = "smoke",
    *,
    run_horizon_ticks: int | None = None,
    bootstrap_n_orders: int | None = None,
    demand_horizon_ticks: int | None = None,
    demand_buffer_ticks: int | None = None,
    order_generation_mode: str | None = None,
    full_raw_order_replay: bool | None = None,
    detail_db: bool | None = None,
    debug_trace: bool | None = None,
    keep_runtime_artifacts: bool | None = None,
    pod_location_mode: str | None = None,
    pod_location_seed: int | None = None,
    seed: int | None = None,
) -> RunProfile:
    name = str(profile or "smoke").strip().lower()
    if name not in _PROFILES:
        raise ValueError(f"Unknown RMFS run profile: {profile!r}. Expected one of {available_profiles()}.")
    resolved = _PROFILES[name]
    buffer_ticks = resolved.demand_buffer_ticks if demand_buffer_ticks is None else int(demand_buffer_ticks)
    horizon = resolved.run_horizon_ticks if run_horizon_ticks is None else int(run_horizon_ticks)
    demand_horizon = demand_horizon_ticks
    if demand_horizon is None:
        demand_horizon = None if horizon is None else int(horizon) + int(buffer_ticks)
    pod_seed = pod_location_seed if pod_location_seed is not None else seed
    return replace(
        resolved,
        run_horizon_ticks=horizon,
        bootstrap_n_orders=resolved.bootstrap_n_orders if bootstrap_n_orders is None else int(bootstrap_n_orders),
        demand_horizon_ticks=demand_horizon,
        demand_buffer_ticks=buffer_ticks,
        order_generation_mode=resolved.order_generation_mode if order_generation_mode is None else order_generation_mode,
        full_raw_order_replay=resolved.full_raw_order_replay if full_raw_order_replay is None else bool(full_raw_order_replay),
        detail_db=resolved.detail_db if detail_db is None else bool(detail_db),
        debug_trace=resolved.debug_trace if debug_trace is None else bool(debug_trace),
        keep_runtime_artifacts=resolved.keep_runtime_artifacts if keep_runtime_artifacts is None else bool(keep_runtime_artifacts),
        pod_location_mode=resolved.pod_location_mode if pod_location_mode is None else pod_location_mode,
        pod_location_seed=pod_seed,
    )
