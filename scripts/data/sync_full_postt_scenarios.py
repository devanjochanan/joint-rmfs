"""Sync shared full-postT scenario inputs into joint-rmfs scenario bundles."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.rmfs.runtime_io.scenario_bundle import normalize_scenario_name  # noqa: E402


DEFAULT_SOURCE_ROOT = (
    REPO_ROOT.parent
    / "_full_postt_parallel_runs"
    / "four_scenario_1000_shared_latest"
)
DEFAULT_BUNDLE_ROOT = REPO_ROOT / "data" / "input" / "scenarios"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sync items.csv, pods.csv, generated_pod.csv, and cutoff raw orders into joint-rmfs scenario bundles."
    )
    parser.add_argument(
        "--source-root",
        default=str(DEFAULT_SOURCE_ROOT),
        help="Source root containing full-postT scenario folders.",
    )
    parser.add_argument(
        "--bundle-root",
        default=str(DEFAULT_BUNDLE_ROOT),
        help="Destination scenario bundle root under joint-rmfs.",
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=("scenario4_sij", "cindy_s3", "my_scenario"),
        help="Scenario names to sync.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report planned copies without writing files.",
    )
    return parser.parse_args()


def scenario_file_map(source_root: Path, scenario_name: str) -> dict[str, Path]:
    scenario_dir = source_root / scenario_name / "netlogo-rmfs"
    candidates = {
        "items.csv": scenario_dir / "data" / "output" / "items.csv",
        "pods.csv": scenario_dir / "data" / "output" / "pods.csv",
        "generated_pod.csv": scenario_dir / "data" / "output" / "generated_pod.csv",
        "raw_order.csv": scenario_dir / "data" / "input" / "cutoff_test_orders.csv",
    }
    missing = [name for name, path in candidates.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Scenario '{scenario_name}' is missing required source files: {', '.join(missing)}"
        )
    return candidates


def sync_one_scenario(
    *,
    source_root: Path,
    bundle_root: Path,
    scenario_name: str,
    dry_run: bool,
) -> dict[str, object]:
    canonical = normalize_scenario_name(scenario_name)
    if canonical is None:
        raise ValueError(f"Invalid scenario name: {scenario_name!r}")

    source_files = scenario_file_map(source_root, canonical)
    destination_dir = bundle_root / canonical
    operations = {
        name: {
            "source": str(source_path),
            "target": str(destination_dir / name),
        }
        for name, source_path in source_files.items()
    }

    if not dry_run:
        destination_dir.mkdir(parents=True, exist_ok=True)
        for name, source_path in source_files.items():
            shutil.copy2(source_path, destination_dir / name)

    return {
        "scenario": canonical,
        "destination_dir": str(destination_dir),
        "dry_run": bool(dry_run),
        "files": operations,
    }


def main() -> int:
    args = parse_args()
    source_root = Path(args.source_root).resolve()
    bundle_root = Path(args.bundle_root).resolve()

    payload = {
        "source_root": str(source_root),
        "bundle_root": str(bundle_root),
        "results": [],
    }
    for scenario_name in args.scenarios:
        payload["results"].append(
            sync_one_scenario(
                source_root=source_root,
                bundle_root=bundle_root,
                scenario_name=scenario_name,
                dry_run=args.dry_run,
            )
        )

    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
