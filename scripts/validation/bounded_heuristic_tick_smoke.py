"""Run a tiny bounded heuristic setup/tick smoke in an isolated runtime."""

from __future__ import annotations

import hashlib
import os
import shutil
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


ROOT_SENSITIVE_FILES = (
    "netlogo.state",
    "warehouse.db",
    "assign_order.csv",
    "pod_info.csv",
    "skus_data.csv",
    "sorted_skus_data.csv",
    "generated_order.csv",
    "generated_database_order.csv",
    "generated_backlog.csv",
    "generated_order_meta.json",
)


def digest(path: Path) -> tuple[bool, int, str | None]:
    if not path.exists():
        return False, 0, None
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            hasher.update(chunk)
    return True, path.stat().st_size, hasher.hexdigest()


def snapshot_root() -> dict[str, tuple[bool, int, str | None]]:
    return {name: digest(REPO_ROOT / name) for name in ROOT_SENSITIVE_FILES}


def main() -> int:
    root_before = snapshot_root()
    runtime_root = Path(tempfile.mkdtemp(prefix="rmfs_bounded_heuristic_"))
    env_keys = {
        "RMFS_RUN_PROFILE": "smoke",
        "RMFS_RUN_HORIZON_TICKS": "3",
        "RMFS_BOOTSTRAP_N_ORDERS": "100",
        "RMFS_DEMAND_HORIZON_TICKS": "1003",
        "RMFS_DEMAND_BUFFER_TICKS": "1000",
        "RMFS_ORDER_GENERATION_MODE": "controlled_count",
        "RMFS_FULL_RAW_ORDER_REPLAY": "0",
        "RMFS_DETAIL_DB": "0",
        "RMFS_FAST_TRAIN": "1",
        "RMFS_SIM_SEED": "123",
        "RMFS_POD_LOCATION_MODE": "randomize_slots",
        "RMFS_POD_LOCATION_SEED": "123",
        "PPS_MODE": "heuristic",
    }
    previous_env = {key: os.environ.get(key) for key in env_keys}
    os.environ.update(env_keys)

    try:
        from src.rmfs.runtime_io import RunContext
        from src.rmfs.runtime_io.detail_db import configure_detail_db
        import netlogo

        ctx = RunContext.isolated(runtime_root, repo_root=REPO_ROOT)
        ctx.ensure_runtime_dirs()
        configure_detail_db(enabled=False, db_path=ctx.sqlite_db)
        netlogo.configure_run_context(ctx)
        netlogo.set_sim_seed(123)
        netlogo.set_pps_mode("heuristic")

        setup_result = netlogo.setup()
        if isinstance(setup_result, str) and setup_result.startswith("An error occurred"):
            raise SystemExit("netlogo.setup failed during bounded smoke")

        last_result = None
        for _ in range(3):
            last_result = netlogo.tick()
            if isinstance(last_result, str) and last_result.startswith("An error occurred"):
                raise SystemExit("netlogo.tick failed during bounded smoke")

        required_runtime = (
            ctx.state_file,
            ctx.assign_order_csv,
            ctx.pod_info_csv,
            ctx.skus_data_csv,
            ctx.sorted_skus_data_csv,
            ctx.generated_order_csv,
            ctx.generated_database_order_csv,
            ctx.generated_order_meta_json,
        )
        missing = [str(path) for path in required_runtime if not path.exists()]
        if missing:
            raise SystemExit(f"missing runtime files: {missing}")
        if ctx.sqlite_db.exists():
            raise SystemExit("warehouse.db should not be created when detail_db=False")
        if not isinstance(last_result, list) or len(last_result) < 11:
            raise SystemExit("tick return shape changed unexpectedly")

        root_after = snapshot_root()
        if root_before != root_after:
            changed = [name for name in ROOT_SENSITIVE_FILES if root_before[name] != root_after[name]]
            raise SystemExit(f"root runtime files changed: {changed}")

        print("bounded heuristic tick smoke OK")
        print(f"runtime_root: {runtime_root}")
        print(f"tick_return_length: {len(last_result)}")
        return 0
    finally:
        try:
            import netlogo

            netlogo.reset_run_context()
        except Exception:
            pass
        for key, value in previous_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        shutil.rmtree(runtime_root, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
