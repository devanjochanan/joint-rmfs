"""Validate local executor contracts without running workers."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.rmfs.orchestration.local_executor import expected_worker_files, worker_environment_overrides
from src.rmfs.orchestration.run_spec import RunSpec
from src.rmfs.runtime_io.run_profiles import resolve_run_profile


def main() -> int:
    if "warehouse.db" in expected_worker_files(detail_db=False):
        raise SystemExit("warehouse.db must not be expected when detail_db=False")
    if "warehouse.db" not in expected_worker_files(detail_db=True):
        raise SystemExit("warehouse.db must be expected when detail_db=True")

    smoke = resolve_run_profile("smoke")
    ablation = resolve_run_profile("ablation")
    if smoke.detail_db:
        raise SystemExit("smoke profile should disable detail DB")
    if ablation.run_horizon_ticks != 100_000:
        raise SystemExit("ablation profile should default to 100000 ticks")
    if ablation.pod_location_mode != "randomize_slots":
        raise SystemExit("ablation profile should randomize pod slots")

    spec = RunSpec(
        run_id="contract",
        ticks=3,
        runtime_root=REPO_ROOT / "data" / "runtime" / "tmp" / "contract",
        repo_root=REPO_ROOT,
        detail_db=False,
        run_profile="smoke",
        run_horizon_ticks=3,
        bootstrap_n_orders=100,
        demand_horizon_ticks=1003,
        demand_buffer_ticks=1000,
        order_generation_mode="controlled_count",
        robot_task_allocator="fifo",
        regret_k=7,
        task_allocator_scope="active_job_queue",
        committed_next_reservations_enabled=True,
        pod_location_mode="randomize_slots",
        pod_location_seed=123,
        rts_random_seed=123,
    )
    env = worker_environment_overrides(spec)
    expected = {
        "RMFS_ROBOT_TASK_ALLOCATOR": "fifo",
        "RMFS_REGRET_K": "7",
        "RMFS_COMMITTED_NEXT_RESERVATIONS": "1",
        "RMFS_DETAIL_DB": "0",
        "RMFS_POD_LOCATION_MODE": "randomize_slots",
        "RMFS_POD_LOCATION_SEED": "123",
        "RMFS_BOOTSTRAP_N_ORDERS": "100",
    }
    for key, value in expected.items():
        if env.get(key) != value:
            raise SystemExit(f"{key} expected {value!r}, got {env.get(key)!r}")

    print("local executor contract smoke OK")
    print(f"smoke_horizon_ticks: {smoke.run_horizon_ticks}")
    print(f"ablation_horizon_ticks: {ablation.run_horizon_ticks}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
