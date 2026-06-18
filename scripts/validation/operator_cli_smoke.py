"""Smoke test scripts/run/rmfs.py without long simulations."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON = sys.executable
RMFS = REPO_ROOT / "scripts" / "run" / "rmfs.py"


def run_command(args: list[str]) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        [PYTHON, str(RMFS), *args],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise AssertionError(
            f"command failed: {' '.join(args)}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    return completed


def snapshot_runtime_tmp() -> set[str]:
    root = REPO_ROOT / "data" / "runtime" / "tmp"
    if not root.exists():
        return set()
    return {path.name for path in root.iterdir() if path.name != ".gitkeep"}


def main() -> int:
    for config in (
        "configs/smoke.example.json",
        "configs/ablation.example.json",
        "configs/training_pps.example.json",
        "configs/debug.example.json",
    ):
        json.loads((REPO_ROOT / config).read_text(encoding="utf-8"))

    before_tmp = snapshot_runtime_tmp()

    checks = [
        ["profiles"],
        ["profile", "smoke"],
        ["profile", "ablation", "--seed", "123"],
        ["ablation", "--scenario", "base", "--seed", "123", "--dry-run"],
        ["training", "--target", "pps", "--seed", "123", "--dry-run"],
        ["cleanup", "--dry-run"],
        ["profile", "--config", "configs/ablation.example.json"],
    ]
    outputs = []
    for command in checks:
        outputs.append((command, run_command(command).stdout))

    json_output = run_command(["profile", "ablation", "--seed", "123", "--json"]).stdout
    parsed = json.loads(json_output)
    if parsed.get("profile") != "ablation":
        raise AssertionError("JSON profile output did not resolve ablation")
    if parsed.get("run_horizon_ticks") != 100000:
        raise AssertionError("JSON ablation profile did not include 100000 horizon")

    ablation_text = "\n".join(output for _command, output in outputs if _command[:1] == ["ablation"])
    if "100000" not in ablation_text:
        raise AssertionError("ablation dry-run output should include 100000")
    if "DRY RUN: no simulation was executed." not in ablation_text:
        raise AssertionError("ablation command should be dry-run")

    after_tmp = snapshot_runtime_tmp()
    if before_tmp != after_tmp:
        raise AssertionError("operator dry-run commands created runtime tmp artifacts")

    print("operator cli smoke ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
