"""Run one RMFS backend episode without opening the NetLogo GUI.

This uses the same Python backend functions as the NetLogo interface, but keeps
the Inventory object in memory instead of loading/saving netlogo.state every
tick. Fast training I/O is enabled by default for quicker result checks.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import pickle
import sys
import time
from contextlib import ExitStack
from pathlib import Path



_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
os.chdir(_REPO_ROOT)
from src.rmfs.decisions.pps import DEFAULT_PPS_MODEL_PATH
from src.rmfs.runtime_io.run_profiles import available_profiles, resolve_run_profile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one RMFS Python-backend episode headlessly."
    )
    parser.add_argument(
        "--mode",
        "--pps-mode",
        dest="mode",
        choices=("ppo", "random", "rika", "heuristic", "demand"),
        default="rika",
        help="PPS mode to use.",
    )
    parser.add_argument(
        "--profile",
        choices=available_profiles(),
        default="smoke",
        help="Run profile for horizon, demand, detail DB, and pod-location defaults.",
    )
    parser.add_argument(
        "--max-ticks",
        type=float,
        default=3000.0,
        help="Stop when the backend simulation clock reaches this value.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help=f"Optional PPO .zip model path. Defaults to {DEFAULT_PPS_MODEL_PATH}.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional simulation seed to apply before backend setup.",
    )
    parser.add_argument(
        "--bootstrap-n-orders",
        type=int,
        default=None,
        help="Override profile bootstrap order count.",
    )
    parser.add_argument(
        "--demand-horizon-ticks",
        type=int,
        default=None,
        help="Override generated demand horizon.",
    )
    parser.add_argument(
        "--demand-buffer-ticks",
        type=int,
        default=None,
        help="Override generated demand buffer beyond run horizon.",
    )
    parser.add_argument(
        "--pod-location-mode",
        choices=("fixed", "randomize_slots"),
        default=None,
        help="Override profile pod-location mode.",
    )
    parser.add_argument(
        "--pod-location-seed",
        type=int,
        default=None,
        help="Override pod-location randomization seed.",
    )
    parser.add_argument(
        "--full-raw-order-replay",
        action="store_true",
        default=False,
        help="Opt in to replaying all unique raw orders.",
    )
    parser.add_argument(
        "--normal-io",
        action="store_true",
        help="Disable RMFS_FAST_TRAIN and run with normal CSV/database I/O.",
    )
    parser.add_argument(
        "--detail-db",
        action="store_true",
        help="Enable detail SQLite DB writes. Default stays off in fast training I/O.",
    )
    parser.add_argument(
        "--show-log",
        action="store_true",
        help="Show backend debug prints while the episode is running.",
    )
    parser.add_argument(
        "--progress-seconds",
        type=float,
        default=10.0,
        help="Print a progress line every N real seconds. Use 0 to disable.",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default=None,
        help="Optional scenario bundle name under joint-rmfs/data/input/scenarios.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.normal_io:
        os.environ.pop("RMFS_FAST_TRAIN", None)
    else:
        os.environ["RMFS_FAST_TRAIN"] = "1"

    if args.model_path:
        os.environ["PPS_RL_MODEL_PATH"] = args.model_path
    if args.seed is not None:
        os.environ["RMFS_SIM_SEED"] = str(args.seed)
    if args.scenario:
        os.environ["RMFS_SCENARIO_NAME"] = args.scenario
    profile_cfg = resolve_run_profile(
        args.profile,
        run_horizon_ticks=int(args.max_ticks),
        bootstrap_n_orders=args.bootstrap_n_orders,
        demand_horizon_ticks=args.demand_horizon_ticks,
        demand_buffer_ticks=args.demand_buffer_ticks,
        full_raw_order_replay=args.full_raw_order_replay,
        detail_db=args.detail_db if args.detail_db else None,
        pod_location_mode=args.pod_location_mode,
        pod_location_seed=args.pod_location_seed,
        seed=args.seed,
    )
    os.environ.update(profile_cfg.env())
    if args.detail_db:
        os.environ["RMFS_DETAIL_DB"] = "1"
    elif args.normal_io:
        os.environ.setdefault("RMFS_DETAIL_DB", "1")
    else:
        os.environ["RMFS_DETAIL_DB"] = "0"

    import netlogo
    from src.rmfs.app.netlogo_api import (
        _apply_pps_rl_policy,
        _get_throughput,
        _get_avg_order_completion_time,
        _get_pod_visits,
        _get_picked_quantity,
        _get_pile_on_rate,
    )
    from src.rmfs.decisions.pps import configure_pps_rl_strategy

    progress_out = sys.stdout

    def maybe_silence_logs() -> ExitStack:
        stack = ExitStack()
        if not args.show_log:
            devnull = stack.enter_context(open(os.devnull, "w"))
            stack.enter_context(contextlib.redirect_stdout(devnull))
            stack.enter_context(contextlib.redirect_stderr(devnull))
        return stack

    def progress(message: str) -> None:
        print(message, file=progress_out, flush=True)

    setup_start = time.perf_counter()
    with maybe_silence_logs():
        if args.seed is not None:
            netlogo.set_sim_seed(args.seed)
        netlogo.set_pps_mode(args.mode)
        setup_result = netlogo.setup()

    if (
        isinstance(setup_result, str)
        and setup_result.startswith("An error occurred")
    ):
        raise SystemExit(
            "Backend setup failed. If assign_order.csv or pod_info.csv is open "
            "in Excel, NetLogo, or another Python process, close it and run again. "
            "Rerun with --show-log if you need the full traceback."
        )

    state_file = netlogo.get_run_context().state_file
    if not state_file.exists():
        raise SystemExit(f"Backend setup did not create {state_file}.")

    scenario_meta = netlogo.get_run_context().runtime_root / "active_scenario.json"
    if scenario_meta.exists():
        try:
            meta = json.loads(scenario_meta.read_text(encoding="utf-8"))
            print(
                "[INPUT] "
                f"scenario={meta.get('scenario_name', '')} "
                f"items_rows={meta.get('items_rows', '')} "
                f"pods_rows={meta.get('pods_rows', '')} "
                f"unique_pods={meta.get('unique_pods', '')} "
                f"unique_pod_items={meta.get('unique_pod_items', '')}"
            )
        except Exception:
            pass

    with open(state_file, "rb") as file:
        universe = pickle.load(file)

    for obj in universe._objects:
        obj.setUniverse(universe)

    with maybe_silence_logs():
        configure_pps_rl_strategy(universe)

    run_start = time.perf_counter()
    last_progress = run_start
    backend_steps = 0

    with maybe_silence_logs():
        while universe._tick < args.max_ticks:
            universe.tick()
            _apply_pps_rl_policy(universe)
            backend_steps += 1
            now = time.perf_counter()
            if args.progress_seconds > 0 and now - last_progress >= args.progress_seconds:
                last_progress = now
                progress(
                    "progress: "
                    f"tick={universe._tick:.2f}/{args.max_ticks:g}, "
                    f"steps={backend_steps}, "
                    f"throughput={_get_throughput(universe)}, "
                    f"elapsed={now - run_start:.1f}s"
                )

    run_elapsed = time.perf_counter() - run_start
    total_elapsed = time.perf_counter() - setup_start

    print(f"Mode: {args.mode}")
    if args.scenario:
        print(f"Scenario: {args.scenario}")
    print(f"Seed: {args.seed if args.seed is not None else ''}")
    print(f"Run profile: {profile_cfg.profile}")
    print(f"Bootstrap orders: {profile_cfg.bootstrap_n_orders}")
    print(f"Demand horizon ticks: {profile_cfg.demand_horizon_ticks}")
    print(f"Pod location mode: {profile_cfg.pod_location_mode}")
    print(f"Fast training I/O: {'off' if args.normal_io else 'on'}")
    print(f"Detail DB: {'on' if args.detail_db else 'default'}")
    print(f"Backend steps: {backend_steps}")
    print(f"Simulation tick: {universe._tick:.2f}")
    print(f"Setup + run seconds: {total_elapsed:.2f}")
    print(f"Run seconds: {run_elapsed:.2f}")
    print(f"Throughput: {_get_throughput(universe)}")
    print(f"Avg order completion time: {_get_avg_order_completion_time(universe)}")
    print(f"Pod visits: {_get_pod_visits(universe)}")
    print(f"Pile-on rate: {_get_pile_on_rate(universe)}")
    print(f"Picked quantity: {_get_picked_quantity(universe)}")
    print(f"Total energy: {universe.total_energy}")
    print(f"Stop-and-go: {universe.stop_and_go}")
    print(f"Total turning: {universe.total_turning}")


if __name__ == "__main__":
    main()
