"""Run specification for local isolated RMFS workers."""

from dataclasses import asdict, dataclass
from pathlib import Path

from src.rmfs.runtime_io.run_profiles import TICK_TO_SECOND


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    ticks: int
    runtime_root: Path
    repo_root: Path
    input_root: Path | None = None
    branch: str | None = None
    commit: str | None = None
    python_executable: str | None = None
    timestamp: str | None = None
    debug_trace: bool = False
    trace_cadence: int = 1000
    trace_first_n: int = 0
    rts_policy_mode: str = "current"
    rts_rollout_enabled: bool = False
    rts_zone_ids: list[str] | None = None
    rts_reward_reference_path: str | None = None
    rts_seed_base: int | None = None
    rts_random_seed: int | None = None
    rts_max_events: int | None = None
    rts_policy_checkpoint_dir: str | None = None
    rts_policy_checkpoint_id: str | None = None
    rts_policy_action_mode: str = "sample"
    rts_policy_device: str = "cpu"
    rts_feature_ablation: str = "full"
    rts_feature_ablation_hash: str | None = None
    rts_state_capture_mode: str = "auto"
    rts_charging_mode: str = "inherit"
    robot_count: int = 20
    expected_picking_station_count: int | None = None
    expected_replenishment_station_count: int | None = None
    pps_mode: str = "heuristic"
    pps_model_path: str | None = None
    charging_enabled: bool | None = None
    charging_config_path: str | None = None
    keep_runtime_artifacts: bool = False
    detail_db: bool = False
    timing: bool = False
    worker_status_cadence: int = 100
    run_profile: str = "smoke"
    run_horizon_ticks: int | None = None
    bootstrap_n_orders: int | None = None
    demand_horizon_ticks: int | None = None
    demand_buffer_ticks: int | None = None
    order_generation_mode: str = "controlled_count"
    full_raw_order_replay: bool = False
    order_rate_per_hour: int | None = None
    pod_location_mode: str = "randomize_slots"
    pod_location_seed: int | None = None
    experiment_id: str | None = None
    scenario_id: str | None = None
    artifact_label: str | None = None
    batch_id: int | None = None
    worker_id: int | None = None
    robot_task_allocator: str = "regret_k"
    regret_k: int | None = 2
    task_allocator_scope: str = "active_job_queue"
    committed_next_reservations_enabled: bool = False
    rts_torch_threads: int | None = None
    rts_torch_interop_threads: int | None = None

    @property
    def netlogo_steps_requested(self) -> int:
        return self.ticks

    @property
    def tick_to_second(self) -> float:
        return TICK_TO_SECOND

    @property
    def simulated_horizon_seconds(self) -> float:
        return float(self.netlogo_steps_requested) * self.tick_to_second

    @property
    def demand_horizon_steps(self) -> int | None:
        return self.demand_horizon_ticks

    @property
    def demand_horizon_simulated_seconds(self) -> float | None:
        if self.demand_horizon_ticks is None:
            return None
        return float(self.demand_horizon_ticks) * self.tick_to_second

    @property
    def demand_buffer_steps(self) -> int | None:
        return self.demand_buffer_ticks

    @property
    def demand_buffer_simulated_seconds(self) -> float | None:
        if self.demand_buffer_ticks is None:
            return None
        return float(self.demand_buffer_ticks) * self.tick_to_second

    def validate_runtime_semantics(self) -> None:
        if int(self.ticks) <= 0:
            raise ValueError("RunSpec.ticks must be positive backend / NetLogo steps")
        if int(self.robot_count) < 1:
            raise ValueError("RunSpec.robot_count must be >= 1")
        if self.order_rate_per_hour is not None and int(self.order_rate_per_hour) <= 0:
            raise ValueError("RunSpec.order_rate_per_hour must be positive when supplied")
        if not str(self.order_generation_mode or "").strip():
            raise ValueError("RunSpec.order_generation_mode must be nonblank")
        if str(self.run_profile).strip().lower() != "gui":
            if self.order_generation_mode == "legacy_compat":
                raise ValueError("headless RunSpec rejects legacy_compat order generation")
            if self.order_generation_mode == "controlled_count" and self.bootstrap_n_orders is None:
                raise ValueError("headless controlled_count RunSpec requires bootstrap_n_orders")
            if self.order_generation_mode == "shuffled_historical_cycle" and self.order_rate_per_hour is None:
                raise ValueError("headless shuffled_historical_cycle RunSpec requires order_rate_per_hour")

    def to_json_dict(self):
        self.validate_runtime_semantics()
        data = asdict(self)
        data["runtime_root"] = str(self.runtime_root)
        data["repo_root"] = str(self.repo_root)
        data["input_root"] = str(self.input_root) if self.input_root is not None else None
        data["netlogo_steps_requested"] = self.netlogo_steps_requested
        data["simulated_horizon_seconds"] = self.simulated_horizon_seconds
        data["tick_to_second"] = self.tick_to_second
        data["demand_horizon_steps"] = self.demand_horizon_steps
        data["demand_horizon_simulated_seconds"] = self.demand_horizon_simulated_seconds
        data["demand_buffer_steps"] = self.demand_buffer_steps
        data["demand_buffer_simulated_seconds"] = self.demand_buffer_simulated_seconds
        return data

    @classmethod
    def from_json_dict(cls, data):
        payload = dict(data)
        for derived in (
            "netlogo_steps_requested",
            "simulated_horizon_seconds",
            "tick_to_second",
            "demand_horizon_steps",
            "demand_horizon_simulated_seconds",
            "demand_buffer_steps",
            "demand_buffer_simulated_seconds",
        ):
            payload.pop(derived, None)
        payload["runtime_root"] = Path(payload["runtime_root"])
        payload["repo_root"] = Path(payload["repo_root"])
        payload["input_root"] = Path(payload["input_root"]) if payload.get("input_root") else None
        spec = cls(**payload)
        spec.validate_runtime_semantics()
        return spec
