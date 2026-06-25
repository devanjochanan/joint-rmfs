"""Human-facing RMFS operator CLI.

This wrapper keeps operators focused on profile, scenario, and seed. It
delegates simulation work to existing bounded scripts and runners.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON = sys.executable

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.rmfs.runtime_io.run_profiles import available_profiles, resolve_run_profile  # noqa: E402


PROFILE_FIELDS = (
    "profile",
    "run_horizon_ticks",
    "bootstrap_n_orders",
    "demand_horizon_ticks",
    "demand_buffer_ticks",
    "order_generation_mode",
    "full_raw_order_replay",
    "detail_db",
    "debug_trace",
    "keep_runtime_artifacts",
    "pod_location_mode",
    "pod_location_seed",
)


def load_config(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SystemExit(f"Config file not found: {config_path}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid JSON config {config_path}: {exc}") from exc
    if not isinstance(data, dict):
        raise SystemExit(f"Config must be a JSON object: {config_path}")
    return data


def get_value(args: argparse.Namespace, config: dict[str, Any], name: str, default=None):
    value = getattr(args, name, None)
    if value is not None:
        return value
    return config.get(name, default)


def resolve_from_args(
    args: argparse.Namespace,
    *,
    default_profile: str | None = None,
) -> tuple[Any, dict[str, Any]]:
    config = load_config(getattr(args, "config", None))
    profile_name = (
        getattr(args, "profile_name", None)
        or default_profile
        or config.get("profile")
        or "smoke"
    )
    profile = resolve_run_profile(
        profile_name,
        run_horizon_ticks=get_value(args, config, "ticks"),
        bootstrap_n_orders=get_value(args, config, "bootstrap_n_orders"),
        demand_horizon_ticks=get_value(args, config, "demand_horizon_ticks"),
        demand_buffer_ticks=get_value(args, config, "demand_buffer_ticks"),
        full_raw_order_replay=get_value(args, config, "full_raw_order_replay"),
        detail_db=get_value(args, config, "detail_db"),
        debug_trace=get_value(args, config, "debug_trace"),
        keep_runtime_artifacts=get_value(args, config, "keep_runtime_artifacts"),
        pod_location_mode=get_value(args, config, "pod_location_mode"),
        pod_location_seed=get_value(args, config, "pod_location_seed"),
        seed=get_value(args, config, "seed"),
    )
    operator = {
        "scenario": get_value(args, config, "scenario", "base"),
        "seed": get_value(args, config, "seed"),
        "target": get_value(args, config, "target", "pps"),
        "expected_runtime_root": str(
            REPO_ROOT
            / "data"
            / "runtime"
            / "tmp"
            / f"operator_{profile.profile}_{get_value(args, config, 'seed', 'unseeded')}"
        ),
        "expected_output_root": str(REPO_ROOT / "data" / "output"),
    }
    return profile, operator


def profile_dict(profile) -> dict[str, Any]:
    data = asdict(profile)
    return {field: data.get(field) for field in PROFILE_FIELDS}


def print_profile(profile, *, title: str = "Resolved RMFS profile") -> None:
    print(title)
    print("-" * len(title))
    data = profile_dict(profile)
    for field in PROFILE_FIELDS:
        print(f"{field}: {json.dumps(data[field]) if isinstance(data[field], bool) else data[field]}")


def print_dry_run(profile, operator: dict[str, Any], *, command_name: str, runner: str) -> None:
    print("DRY RUN: no simulation was executed.")
    print_profile(profile, title=f"Resolved {command_name} run")
    print(f"scenario: {operator['scenario']}")
    print(f"seed: {operator['seed']}")
    print(f"expected_runtime_root: {operator['expected_runtime_root']}")
    print(f"expected_output_root: {operator['expected_output_root']}")
    print(f"lower_level_runner: {runner}")


def add_common_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", help="Flat JSON config file. CLI args override config values.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--ticks", type=int, default=None)
    parser.add_argument("--bootstrap-n-orders", dest="bootstrap_n_orders", type=int, default=None)
    parser.add_argument("--demand-horizon-ticks", dest="demand_horizon_ticks", type=int, default=None)
    parser.add_argument("--demand-buffer-ticks", dest="demand_buffer_ticks", type=int, default=None)
    parser.add_argument("--detail-db", dest="detail_db", action="store_true", default=None)
    parser.add_argument("--no-detail-db", dest="detail_db", action="store_false")
    parser.add_argument("--debug-trace", dest="debug_trace", action="store_true", default=None)
    parser.add_argument("--no-debug-trace", dest="debug_trace", action="store_false")
    parser.add_argument("--keep-runtime-artifacts", dest="keep_runtime_artifacts", action="store_true", default=None)
    parser.add_argument("--pod-location-mode", dest="pod_location_mode", choices=("fixed", "randomize_slots"), default=None)
    parser.add_argument("--pod-location-seed", dest="pod_location_seed", type=int, default=None)
    parser.add_argument("--full-raw-order-replay", dest="full_raw_order_replay", action="store_true", default=None)


def command_profiles(_args: argparse.Namespace) -> int:
    for name in available_profiles():
        profile = resolve_run_profile(name)
        print_profile(profile, title=f"profile: {name}")
        print()
    return 0


def command_profile(args: argparse.Namespace) -> int:
    profile, _operator = resolve_from_args(args)
    if args.json:
        print(json.dumps(profile_dict(profile), indent=2, sort_keys=True))
    else:
        print_profile(profile)
    return 0


def check_specs(fast: bool, skip_compileall: bool) -> list[tuple[str, list[str], Path | None]]:
    all_checks = []
    if not skip_compileall and not fast:
        all_checks.append(("compileall", [PYTHON, "-m", "compileall", "model", "src", "scripts"], None))
    script_checks = [
        ("run_context_smoke", "scripts/validation/run_context_smoke.py"),
        ("layout_randomization_smoke", "scripts/validation/layout_randomization_smoke.py"),
        ("order_generation_policy_smoke", "scripts/validation/order_generation_policy_smoke.py"),
        ("order_generation_shuffled_historical_smoke", "scripts/validation/order_generation_shuffled_historical_smoke.py"),
        ("bounded_heuristic_tick_smoke", "scripts/validation/bounded_heuristic_tick_smoke.py"),
        ("scenario_list", "scripts/data/activate_scenario.py"),
        ("scenario_dry_run", "scripts/data/activate_scenario.py"),
        ("cleanup_dry_run", "scripts/runtime/cleanup_runtime_artifacts.py"),
    ]
    if fast:
        script_checks = [
            ("run_context_smoke", "scripts/validation/run_context_smoke.py"),
            ("layout_randomization_smoke", "scripts/validation/layout_randomization_smoke.py"),
            ("order_generation_policy_smoke", "scripts/validation/order_generation_policy_smoke.py"),
            ("order_generation_shuffled_historical_smoke", "scripts/validation/order_generation_shuffled_historical_smoke.py"),
            ("cleanup_dry_run", "scripts/runtime/cleanup_runtime_artifacts.py"),
        ]
    for name, rel_path in script_checks:
        path = REPO_ROOT / rel_path
        command = [PYTHON, rel_path]
        if name == "scenario_list":
            command.append("--list")
        elif name == "scenario_dry_run":
            command.extend(["--scenario", "cindy_s1", "--dry-run"])
        elif name == "cleanup_dry_run":
            command.append("--dry-run")
        all_checks.append((name, command, path))
    return all_checks


def command_validate(args: argparse.Namespace) -> int:
    results = []
    for name, command, path in check_specs(args.fast, args.skip_compileall):
        if path is not None and not path.exists():
            result = {"name": name, "command": command, "status": "FAIL", "result": "missing script"}
            results.append(result)
            if not args.json:
                print(f"FAIL {name}: missing {path.relative_to(REPO_ROOT)}")
            continue
        completed = subprocess.run(command, cwd=REPO_ROOT, text=True, capture_output=True)
        status = "PASS" if completed.returncode == 0 else "FAIL"
        output = (completed.stdout or completed.stderr).strip().splitlines()
        short = output[-1] if output else f"exit {completed.returncode}"
        result = {"name": name, "command": command, "status": status, "result": short}
        results.append(result)
        if not args.json:
            print(f"{status} {name}: {' '.join(command)}")
            print(f"  {short}")
    failures = [row for row in results if row["status"] != "PASS"]
    if args.json:
        print(json.dumps({"checks": results, "failures": len(failures)}, indent=2))
    else:
        print(f"Validation summary: {len(results) - len(failures)} passed, {len(failures)} failed")
        if args.fast:
            print("FAST mode: reduced bounded check set.")
    return 1 if failures else 0


def command_smoke(args: argparse.Namespace) -> int:
    profile, operator = resolve_from_args(args, default_profile="smoke")
    if args.dry_run:
        print_dry_run(
            profile,
            operator,
            command_name="smoke",
            runner="scripts/validation/bounded_heuristic_tick_smoke.py",
        )
        return 0
    seed = operator["seed"]
    if seed is not None and int(seed) != 123:
        print("Note: Phase 6 smoke delegates to bounded_heuristic_tick_smoke.py, which uses seed 123.")
    return subprocess.call([PYTHON, "scripts/validation/bounded_heuristic_tick_smoke.py"], cwd=REPO_ROOT)


def command_ablation(args: argparse.Namespace) -> int:
    profile, operator = resolve_from_args(args, default_profile="ablation")
    dry_run = args.dry_run or not args.yes
    runner = "src.rmfs.orchestration.local_executor controller (future dedicated ablation runner)"
    if dry_run:
        print_dry_run(profile, operator, command_name="ablation", runner=runner)
        return 0
    print("ablation execution runner is not implemented in Phase 6; use dry-run or implement dedicated experiment runner later.")
    return 2


def training_command(profile, operator: dict[str, Any]) -> list[str]:
    command = [
        PYTHON,
        "scripts/training/train_pps_rl.py",
        "--profile",
        profile.profile,
    ]
    if operator["seed"] is not None:
        command.extend(["--seed", str(operator["seed"])])
    if profile.run_horizon_ticks is not None:
        command.extend(["--max-ticks", str(profile.run_horizon_ticks)])
    if profile.bootstrap_n_orders is not None:
        command.extend(["--bootstrap-n-orders", str(profile.bootstrap_n_orders)])
    if profile.demand_horizon_ticks is not None:
        command.extend(["--demand-horizon-ticks", str(profile.demand_horizon_ticks)])
    if profile.detail_db:
        command.append("--detail-db")
    return command


def command_training(args: argparse.Namespace) -> int:
    profile, operator = resolve_from_args(args, default_profile="training")
    operator["target"] = args.target
    if args.help_runner:
        return subprocess.call([PYTHON, "scripts/training/train_pps_rl.py", "--help"], cwd=REPO_ROOT)
    command = training_command(profile, operator)
    print_dry_run(profile, operator, command_name="training", runner="scripts/training/train_pps_rl.py")
    print(f"target: {args.target}")
    print(f"delegated_command: {' '.join(command)}")
    if args.yes and not args.dry_run:
        print("PPO training execution is intentionally not launched by the Phase 6 wrapper.")
        return 2
    return 0


def command_debug(args: argparse.Namespace) -> int:
    profile, operator = resolve_from_args(args, default_profile="debug")
    print_dry_run(
        profile,
        operator,
        command_name="debug",
        runner="scripts/validation/bounded_heuristic_tick_smoke.py with debug profile env",
    )
    return 0


def command_cleanup(args: argparse.Namespace) -> int:
    command = [PYTHON, "scripts/runtime/cleanup_runtime_artifacts.py"]
    if args.apply:
        command.append("--apply")
    else:
        command.append("--dry-run")
    if args.keep_failed:
        command.append("--keep-failed")
    if args.keep_latest:
        command.append("--keep-latest")
    if args.max_age_days is not None:
        command.extend(["--max-age-days", str(args.max_age_days)])
    return subprocess.call(command, cwd=REPO_ROOT)


def command_run(args: argparse.Namespace) -> int:
    profile, operator = resolve_from_args(args)
    print_dry_run(profile, operator, command_name="configured", runner="profile-selected lower-level runner")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RMFS operator CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("profiles", help="List available run profiles").set_defaults(func=command_profiles)

    profile_parser = subparsers.add_parser("profile", help="Resolve one run profile")
    profile_parser.add_argument("profile_name", nargs="?", choices=available_profiles())
    add_common_options(profile_parser)
    profile_parser.add_argument("--json", action="store_true", default=False)
    profile_parser.set_defaults(func=command_profile)

    validate_parser = subparsers.add_parser("validate", help="Run safe bounded validation checks")
    validate_parser.add_argument("--fast", action="store_true", default=False)
    validate_parser.add_argument("--skip-compileall", action="store_true", default=False)
    validate_parser.add_argument("--json", action="store_true", default=False)
    validate_parser.set_defaults(func=command_validate)

    smoke_parser = subparsers.add_parser("smoke", help="Run the bounded heuristic smoke")
    add_common_options(smoke_parser)
    smoke_parser.add_argument("--dry-run", action="store_true", default=False)
    smoke_parser.set_defaults(func=command_smoke)

    ablation_parser = subparsers.add_parser("ablation", help="Prepare an ablation run")
    add_common_options(ablation_parser)
    ablation_parser.add_argument("--scenario", default=None)
    ablation_parser.add_argument("--dry-run", action="store_true", default=False)
    ablation_parser.add_argument("--yes", action="store_true", default=False)
    ablation_parser.set_defaults(func=command_ablation)

    training_parser = subparsers.add_parser("training", help="Prepare a training run")
    add_common_options(training_parser)
    training_parser.add_argument("--target", choices=("pps",), default="pps")
    training_parser.add_argument("--dry-run", action="store_true", default=False)
    training_parser.add_argument("--yes", action="store_true", default=False)
    training_parser.add_argument("--help-runner", action="store_true", default=False)
    training_parser.set_defaults(func=command_training)

    debug_parser = subparsers.add_parser("debug", help="Inspect debug run defaults")
    add_common_options(debug_parser)
    debug_parser.add_argument("--scenario", default=None)
    debug_parser.add_argument("--dry-run", action="store_true", default=False)
    debug_parser.set_defaults(func=command_debug)

    cleanup_parser = subparsers.add_parser("cleanup", help="Clean runtime artifacts, dry-run first")
    cleanup_parser.add_argument("--dry-run", action="store_true", default=True)
    cleanup_parser.add_argument("--apply", action="store_true", default=False)
    cleanup_parser.add_argument("--keep-failed", action="store_true", default=False)
    cleanup_parser.add_argument("--keep-latest", action="store_true", default=False)
    cleanup_parser.add_argument("--max-age-days", type=float, default=None)
    cleanup_parser.set_defaults(func=command_cleanup)

    run_parser = subparsers.add_parser("run", help="Resolve a config-file based run")
    add_common_options(run_parser)
    run_parser.add_argument("--scenario", default=None)
    run_parser.add_argument("--dry-run", action="store_true", default=True)
    run_parser.set_defaults(func=command_run)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except ValueError as exc:
        parser.error(str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
