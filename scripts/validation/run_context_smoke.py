"""Smoke test: RunContext default and isolated constructors.

Verifies that all expected path attributes exist, that isolated mode routes
generated outputs to runtime_root and canonical inputs to input_root,
and that the new raw_order_csv and generated_order_meta_json fields are present.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.rmfs.runtime_io.context import RunContext


def main():
    # ---- Default context ----
    ctx = RunContext.default(repo_root=REPO_ROOT)
    assert ctx.repo_root == REPO_ROOT
    assert ctx.input_root == REPO_ROOT
    assert ctx.runtime_root == REPO_ROOT
    assert ctx.raw_order_csv == REPO_ROOT / "raw_order.csv"
    assert ctx.items_csv == REPO_ROOT / "items.csv"
    assert ctx.pods_csv == REPO_ROOT / "pods.csv"
    assert ctx.generated_order_csv == REPO_ROOT / "generated_order.csv"
    assert ctx.generated_database_order_csv == REPO_ROOT / "generated_database_order.csv"
    assert ctx.generated_order_meta_json == REPO_ROOT / "generated_order_meta.json"
    assert ctx.generated_pod_csv == REPO_ROOT / "generated_pod.csv"
    print("[PASS] Default context: all paths resolve to repo root.")

    # ---- Isolated context ----
    rt = REPO_ROOT / "_test_worker_0"
    ctx_iso = RunContext.isolated(runtime_root=rt, repo_root=REPO_ROOT)
    rt_resolved = rt.resolve()

    # Runtime outputs should be in the worker's runtime_root
    assert ctx_iso.runtime_root == rt_resolved
    assert ctx_iso.generated_order_csv == rt_resolved / "generated_order.csv"
    assert ctx_iso.generated_database_order_csv == rt_resolved / "generated_database_order.csv"
    assert ctx_iso.generated_order_meta_json == rt_resolved / "generated_order_meta.json"
    assert ctx_iso.generated_backlog_csv == rt_resolved / "generated_backlog.csv"
    assert ctx_iso.state_file == rt_resolved / "netlogo.state"
    assert ctx_iso.sqlite_db == rt_resolved / "warehouse.db"
    assert ctx_iso.assign_order_csv == rt_resolved / "assign_order.csv"
    assert ctx_iso.pod_info_csv == rt_resolved / "pod_info.csv"

    # Canonical inputs should be in input_root (repo root by default)
    rr = REPO_ROOT.resolve()
    assert ctx_iso.raw_order_csv == rr / "raw_order.csv"
    assert ctx_iso.items_csv == rr / "items.csv"
    assert ctx_iso.pods_csv == rr / "pods.csv"
    assert ctx_iso.generated_pod_csv == rr / "generated_pod.csv"
    print("[PASS] Isolated context: generated outputs → runtime_root, inputs → input_root.")

    # ---- inventory_paths helper ----
    inv = ctx_iso.inventory_paths()
    assert "assign_order_csv" in inv
    assert "pod_info_csv" in inv
    assert "generated_order_csv" in inv
    print("[PASS] inventory_paths() contains expected keys.")

    print("\n[ALL PASSED] RunContext smoke test.")


if __name__ == "__main__":
    main()
