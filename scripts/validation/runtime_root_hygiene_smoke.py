#!/usr/bin/env python3
"""Validate isolated RunContext routing for root-sensitive runtime artifacts."""

from __future__ import annotations

import hashlib
from pathlib import Path
import shutil
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.rmfs.runtime_io.context import RunContext


ARTIFACT_ROOT = REPO_ROOT / "data" / "runtime" / "runtime_root_hygiene_smoke"
ROOT_ARTIFACTS = ("warehouse.db", "netlogo.state", "assign_order.csv", "pod_info.csv")


def file_digest(path: Path) -> dict[str, object]:
    if not path.exists():
        return {"exists": False, "sha256": None, "size": 0}
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            digest.update(chunk)
    return {"exists": True, "sha256": digest.hexdigest(), "size": path.stat().st_size}


def snapshot_root() -> dict[str, dict[str, object]]:
    return {name: file_digest(REPO_ROOT / name) for name in ROOT_ARTIFACTS}


def assert_under_runtime_root(path: Path, runtime_root: Path) -> None:
    assert path.resolve().is_relative_to(runtime_root.resolve()), f"{path} is not under {runtime_root}"


def main() -> int:
    root_before = snapshot_root()
    shutil.rmtree(ARTIFACT_ROOT, ignore_errors=True)
    ctx = RunContext.isolated(ARTIFACT_ROOT, repo_root=REPO_ROOT)
    ctx.ensure_runtime_dirs()

    routed_paths = {
        "warehouse.db": ctx.sqlite_db,
        "netlogo.state": ctx.state_file,
        "assign_order.csv": ctx.assign_order_csv,
        "pod_info.csv": ctx.pod_info_csv,
    }
    for path in routed_paths.values():
        assert_under_runtime_root(path, ctx.runtime_root)
        path.write_text("runtime-root-hygiene-smoke\n", encoding="utf-8")
        assert path.exists()

    root_after = snapshot_root()
    assert root_before == root_after, "isolated runtime routing changed root artifacts"

    for name, path in routed_paths.items():
        assert path.exists(), f"expected isolated artifact missing: {name}"

    shutil.rmtree(ARTIFACT_ROOT, ignore_errors=True)
    print("runtime root hygiene smoke ok")
    print("validated isolated routing for warehouse.db, netlogo.state, assign_order.csv, and pod_info.csv")
    print("cleaned task-specific runtime output after validation")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
