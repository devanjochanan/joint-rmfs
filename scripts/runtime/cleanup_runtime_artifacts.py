"""Dry-run-first cleanup for RMFS runtime scratch artifacts."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_ROOT = REPO_ROOT / "data" / "runtime"
ALLOWED_TARGETS = (
    RUNTIME_ROOT / "tmp",
    RUNTIME_ROOT / "debug",
    RUNTIME_ROOT / "latest",
)
NEVER_TOUCH = (
    REPO_ROOT / "data" / "input",
    REPO_ROOT / "data" / "models",
    REPO_ROOT / "data" / "output",
    REPO_ROOT / "data" / "reference",
)


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _validate_runtime_path(path: Path) -> None:
    resolved = path.resolve()
    if not any(_is_relative_to(resolved, target) for target in ALLOWED_TARGETS):
        raise ValueError(f"Refusing to target path outside allowed runtime scratch roots: {path}")
    if any(_is_relative_to(resolved, protected) for protected in NEVER_TOUCH):
        raise ValueError(f"Refusing to target protected path: {path}")


def _read_status(path: Path) -> str | None:
    for name in ("worker_summary.json", "worker_status.json", "controller_summary.json"):
        candidate = path / name
        if not candidate.exists():
            continue
        try:
            data = json.loads(candidate.read_text(encoding="utf-8"))
        except Exception:
            continue
        status = data.get("status")
        if isinstance(status, str):
            return status.lower()
    return None


def _is_failed(path: Path) -> bool:
    status = _read_status(path)
    return status not in {None, "success"}


def _iter_cleanup_candidates(keep_latest: bool, keep_failed: bool, max_age_days: float | None):
    now = time.time()
    for target in ALLOWED_TARGETS:
        target.mkdir(parents=True, exist_ok=True)
        entries = [
            path for path in target.iterdir()
            if path.name != ".gitkeep" and not path.name.startswith(".")
        ]
        latest_entry = max(entries, key=lambda p: p.stat().st_mtime, default=None)
        for path in entries:
            _validate_runtime_path(path)
            reason = []
            if keep_latest and latest_entry is not None and path == latest_entry:
                reason.append("kept latest")
            if keep_failed and path.is_dir() and _is_failed(path):
                reason.append("kept failed")
            if max_age_days is not None:
                age_days = (now - path.stat().st_mtime) / 86400.0
                if age_days < max_age_days:
                    reason.append(f"younger than {max_age_days:g} days")
            if reason:
                yield {"path": path, "action": "keep", "reason": ", ".join(reason)}
            else:
                yield {"path": path, "action": "delete", "reason": "eligible runtime scratch artifact"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean RMFS runtime scratch artifacts.")
    parser.add_argument("--dry-run", action="store_true", default=True, help="Print the deletion plan without deleting. This is the default.")
    parser.add_argument("--apply", action="store_true", help="Actually delete eligible artifacts.")
    parser.add_argument("--keep-failed", action="store_true", help="Preserve run folders whose status is not success.")
    parser.add_argument("--keep-latest", action="store_true", help="Preserve the newest entry in each scratch root.")
    parser.add_argument("--max-age-days", type=float, default=None, help="Only delete artifacts at least N days old.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    apply = bool(args.apply)
    plan = list(_iter_cleanup_candidates(args.keep_latest, args.keep_failed, args.max_age_days))
    deletions = [entry for entry in plan if entry["action"] == "delete"]

    mode = "APPLY" if apply else "DRY-RUN"
    print(f"{mode}: runtime cleanup plan")
    print(f"Allowed targets: {', '.join(str(p) for p in ALLOWED_TARGETS)}")
    for entry in plan:
        print(f"{entry['action'].upper():6} {entry['path']}  # {entry['reason']}")
    print(f"Eligible deletions: {len(deletions)}")

    if apply:
        for entry in deletions:
            path = entry["path"]
            _validate_runtime_path(path)
            if path.is_dir():
                shutil.rmtree(path)
            elif path.exists():
                path.unlink()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
