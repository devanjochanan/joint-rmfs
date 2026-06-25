"""Validate bounded, deterministic run-local order generation policy."""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.rmfs.order_generation import generate_orders_from_raw_bootstrap
from src.rmfs.runtime_io.run_profiles import resolve_run_profile


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    base = REPO_ROOT / "data" / "input" / "base"
    protected = [base / "items.csv", base / "pods.csv", base / "generated_pod.csv", base / "raw_order.csv"]
    before = {path: sha256(path) for path in protected}
    smoke_profile = resolve_run_profile("smoke", run_horizon_ticks=100, bootstrap_n_orders=25, order_generation_mode="controlled_count", seed=123)
    ablation_profile = resolve_run_profile("ablation")
    if ablation_profile.run_horizon_ticks != 100_000:
        raise SystemExit("ablation profile should default to 100000 ticks")

    root_tmp = Path(tempfile.mkdtemp(prefix="rmfs_order_generation_policy_"))
    try:
        outputs = []
        for index in range(2):
            target = root_tmp / f"same_seed_{index}"
            generate_orders_from_raw_bootstrap(
                seed=123,
                n_orders=smoke_profile.bootstrap_n_orders,
                source_path=base / "raw_order.csv",
                target_dir=target,
                items_csv_path=base / "items.csv",
                run_horizon_ticks=smoke_profile.run_horizon_ticks,
                demand_horizon_ticks=smoke_profile.demand_horizon_ticks,
                demand_buffer_ticks=smoke_profile.demand_buffer_ticks,
                order_generation_mode=smoke_profile.order_generation_mode,
                full_raw_order_replay=smoke_profile.full_raw_order_replay,
                profile=smoke_profile.profile,
            )
            outputs.append(target)

        first_order = outputs[0] / "generated_order.csv"
        second_order = outputs[1] / "generated_order.csv"
        if sha256(first_order) != sha256(second_order):
            raise SystemExit("same seed/policy produced different order streams")

        meta = json.loads((outputs[0] / "generated_order_meta.json").read_text(encoding="utf-8"))
        if meta["generated_unique_orders"] != 25:
            raise SystemExit(f"expected 25 generated orders, got {meta['generated_unique_orders']}")
        if meta["full_raw_order_replay"]:
            raise SystemExit("smoke policy should not use full raw replay")
        if meta["generated_max_arrival"] > int(smoke_profile.demand_horizon_ticks):
            raise SystemExit("generated order arrivals exceeded demand horizon")

        after = {path: sha256(path) for path in protected}
        if before != after:
            raise SystemExit("canonical input files changed during order generation smoke")

        print("order generation policy smoke OK")
        print(f"generated_unique_orders: {meta['generated_unique_orders']}")
        print(f"demand_horizon_ticks: {meta['demand_horizon_ticks']}")
        print(f"ablation_horizon_ticks: {ablation_profile.run_horizon_ticks}")
        return 0
    finally:
        shutil.rmtree(root_tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
