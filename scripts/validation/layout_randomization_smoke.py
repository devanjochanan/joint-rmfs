"""Smoke test pod-location-only randomization invariants."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.rmfs.runtime_io import RunContext  # noqa: E402
from src.rmfs.runtime_io.layout_randomization import (  # noqa: E402
    randomize_pod_locations,
    read_pod_storage_slots,
)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def load_mapping(path: Path) -> pd.DataFrame:
    return pd.read_csv(path).sort_values("slot_index", kind="stable").reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate pod-location randomization invariants.")
    parser.add_argument("--debug", action="store_true", help="Keep temp outputs under data/runtime/tmp.")
    args = parser.parse_args()

    ctx = RunContext.default(repo_root=REPO_ROOT)
    generated_pod = ctx.generated_pod_csv
    pods_csv = ctx.pods_csv
    items_csv = ctx.items_csv
    workdir = REPO_ROOT / "data" / "runtime" / "tmp" / "layout_randomization_smoke"
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    before_pods = digest(pods_csv)
    before_items = digest(items_csv)
    before_layout = digest(generated_pod)
    source_rows = sum(1 for _ in generated_pod.open("rb"))
    slots = read_pod_storage_slots(generated_pod)

    same_a = randomize_pod_locations(generated_pod, workdir / "seed_123_a.csv", seed=123)
    same_b = randomize_pod_locations(generated_pod, workdir / "seed_123_b.csv", seed=123)
    diff = randomize_pod_locations(generated_pod, workdir / "seed_456.csv", seed=456)

    assert digest(same_a) == digest(same_b), "same seed produced different mappings"
    if len(slots) > 1:
        assert digest(same_a) != digest(diff), "different seed did not change mapping"

    mapping = load_mapping(same_a)
    assert len(mapping) == len(slots), "mapping row count differs from pod storage slot count"
    assert set(mapping["pod_id"]) == set(range(len(slots))), "pod ID set changed"
    assert set(zip(mapping["row"], mapping["col"])) == set(slots), "storage slot set changed"
    assert sum(1 for _ in generated_pod.open("rb")) == source_rows, "generated_pod row count changed"
    assert digest(generated_pod) == before_layout, "generated_pod.csv was mutated"
    assert digest(pods_csv) == before_pods, "pods.csv was mutated"
    assert digest(items_csv) == before_items, "items.csv was mutated"

    print("[PASS] same seed is deterministic.")
    print("[PASS] different seed changes mapping when enough slots exist.")
    print("[PASS] pod ID set and storage slot set are unchanged.")
    print("[PASS] generated_pod/items/pods checksums are preserved.")

    if not args.debug:
        shutil.rmtree(workdir)
    else:
        print(f"[DEBUG] kept temp outputs at {workdir}")
    print("[ALL PASSED] layout randomization smoke test.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
