"""Smoke test canonical and isolated RunContext path routing."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.rmfs.runtime_io.context import RunContext  # noqa: E402


def assert_under(path: Path, parent: Path) -> None:
    path.resolve().relative_to(parent.resolve())


def main() -> int:
    canonical_input = REPO_ROOT / "data" / "input" / "base"
    dictionaries = REPO_ROOT / "data" / "input" / "dictionaries"
    runtime_latest = REPO_ROOT / "data" / "runtime" / "latest"
    output_root = REPO_ROOT / "data" / "output"
    model_root = REPO_ROOT / "data" / "models"

    ctx = RunContext.default(repo_root=REPO_ROOT)
    assert ctx.input_root == canonical_input.resolve()
    assert ctx.dictionaries_root == dictionaries
    assert ctx.runtime_root == runtime_latest.resolve()
    assert ctx.output_root == output_root
    assert ctx.model_root == model_root
    assert ctx.items_csv == canonical_input / "items.csv"
    assert ctx.pods_csv == canonical_input / "pods.csv"
    assert ctx.generated_pod_csv == canonical_input / "generated_pod.csv"
    assert ctx.raw_order_csv == canonical_input / "raw_order.csv"
    assert ctx.items_dictionary_csv == dictionaries / "items_dictionary.csv"
    assert ctx.pods_dictionary_csv == dictionaries / "pods_dictionary.csv"
    assert ctx.items_slots_configuration_csv == dictionaries / "items_slots_configuration.csv"
    print("[PASS] default context resolves canonical input/dictionary paths.")

    for required in (ctx.items_csv, ctx.pods_csv, ctx.generated_pod_csv, ctx.raw_order_csv):
        assert required.exists(), f"missing canonical input: {required}"
    print("[PASS] canonical baseline inputs exist.")

    runtime = REPO_ROOT / "data" / "runtime" / "tmp" / "run_context_smoke"
    iso = RunContext.isolated(runtime, repo_root=REPO_ROOT)
    assert iso.runtime_root == runtime.resolve()
    for runtime_file in (
        iso.state_file,
        iso.sqlite_db,
        iso.assign_order_csv,
        iso.pod_info_csv,
        iso.skus_data_csv,
        iso.sorted_skus_data_csv,
        iso.generated_order_csv,
        iso.generated_database_order_csv,
        iso.generated_backlog_csv,
        iso.generated_order_meta_json,
    ):
        assert_under(runtime_file, runtime)
        assert runtime_file.parent != REPO_ROOT
    print("[PASS] isolated context routes mutable runtime files under runtime root.")

    assert iso.items_csv == ctx.items_csv
    assert iso.pods_csv == ctx.pods_csv
    assert iso.generated_pod_csv == ctx.generated_pod_csv
    assert iso.raw_order_csv == ctx.raw_order_csv
    print("[PASS] isolated context keeps canonical inputs outside runtime root.")

    inventory_paths = iso.inventory_paths()
    assert inventory_paths["assign_order_csv"] == str(iso.assign_order_csv)
    assert inventory_paths["pod_info_csv"] == str(iso.pod_info_csv)
    assert inventory_paths["generated_order_csv"] == str(iso.generated_order_csv)
    assert iso.sqlite_db_path == iso.sqlite_db
    assert iso.state_path == iso.state_file
    print("[PASS] inventory_paths and compatibility aliases are context-aware.")

    print("[ALL PASSED] RunContext smoke test.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
