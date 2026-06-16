"""List or activate RMFS scenario input bundles."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.rmfs.runtime_io.scenario_bundle import (  # noqa: E402
    activate_scenario_inputs,
    list_available_scenarios,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="List or activate RMFS scenario bundles.")
    parser.add_argument("--list", action="store_true", help="List available scenarios.")
    parser.add_argument("--scenario", help="Scenario name or alias to activate.")
    parser.add_argument(
        "--target-root",
        default=str(REPO_ROOT),
        help="Target root for items.csv and pods.csv. Defaults to the repository root.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and report what would be written without changing inputs.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.list:
        for name in list_available_scenarios():
            print(name)
        return 0
    if not args.scenario:
        raise SystemExit("Pass --list or --scenario <name>.")

    metadata = activate_scenario_inputs(
        scenario_name=args.scenario,
        target_root=args.target_root,
        dry_run=args.dry_run,
    )
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
