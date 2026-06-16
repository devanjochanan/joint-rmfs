import json
import os
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
BUNDLED_SCENARIO_ROOT = REPO_ROOT / "data" / "scenarios"
LEGACY_SHARED_SCENARIO_ROOT = (
    REPO_ROOT.parent / "_full_postt_parallel_runs" / "four_scenario_1000_shared_latest"
)

SCENARIO_ALIASES = {
    "cindy_s1": "cindy_s1",
    "cindy1": "cindy_s1",
    "scenario1": "cindy_s1",
    "cindy_s3": "cindy_s3",
    "cindy3": "cindy_s3",
    "scenario3": "cindy_s3",
    "my_scenario": "my_scenario",
    "myscenario": "my_scenario",
    "my-scenario": "my_scenario",
    "scenario4_sij": "scenario4_sij",
    "scenario4": "scenario4_sij",
    "sij": "scenario4_sij",
}


def _normalize_name(name):
    if name is None:
        return None
    return str(name).strip().lower().replace("-", "_")


def _normalize_identifier(value):
    text = str(value).replace("\ufeff", "").strip()
    if not text or text.lower() in {"nan", "none"}:
        return ""
    return text[:-2] if text.endswith(".0") else text


def _shared_scenario_root():
    env_root = os.getenv("RMFS_SHARED_SCENARIO_ROOT")
    if env_root:
        return Path(env_root)
    if BUNDLED_SCENARIO_ROOT.exists():
        return BUNDLED_SCENARIO_ROOT
    return LEGACY_SHARED_SCENARIO_ROOT


def _canonical_scenario_name(name):
    normalized = _normalize_name(name)
    if normalized is None or normalized == "":
        return None
    canonical = SCENARIO_ALIASES.get(normalized)
    if canonical is None:
        available = ", ".join(list_available_scenarios())
        raise ValueError(
            f"Unknown RMFS scenario '{name}'. Available scenarios: {available}"
        )
    return canonical


def _scenario_output_dir(scenario_name):
    root = _shared_scenario_root()
    bundled_output_dir = root / scenario_name
    if (bundled_output_dir / "items.csv").exists() and (bundled_output_dir / "pods.csv").exists():
        return bundled_output_dir
    return root / scenario_name / "netlogo-rmfs" / "data" / "output"


def list_available_scenarios():
    available = []
    seen = set()
    for alias_target in SCENARIO_ALIASES.values():
        if alias_target in seen:
            continue
        output_dir = _scenario_output_dir(alias_target)
        if (output_dir / "items.csv").exists() and (output_dir / "pods.csv").exists():
            available.append(alias_target)
            seen.add(alias_target)
    return tuple(available)


def _read_csv_auto(path):
    frame = pd.read_csv(path, sep=None, engine="python")
    frame.columns = [str(col).replace("\ufeff", "").strip() for col in frame.columns]
    return frame


def _normalize_items_frame(frame):
    if "item_id" not in frame.columns or "item_code" not in frame.columns:
        raise ValueError(
            "Scenario items.csv must contain at least item_id and item_code columns."
        )

    normalized = frame.copy()
    normalized = normalized.dropna(subset=["item_id", "item_code"]).copy()
    normalized["item_id"] = pd.to_numeric(normalized["item_id"], errors="coerce")
    normalized = normalized.dropna(subset=["item_id"]).copy()
    normalized["item_id"] = normalized["item_id"].astype(int)
    normalized["item_code"] = normalized["item_code"].map(_normalize_identifier)
    normalized = normalized[normalized["item_code"] != ""].copy()
    for column in (
        "item_order_frequency",
        "item_initial_quantity_inventory",
        "number_of_item_in_a_box",
        "max_fit",
        "item_initial_quantity_inventory/max_fit",
    ):
        if column in normalized.columns:
            numeric = pd.to_numeric(normalized[column], errors="coerce")
            if numeric.notna().all():
                normalized[column] = numeric.astype(int)
    normalized = normalized.sort_values("item_id", kind="stable").drop_duplicates(
        subset=["item_id"],
        keep="first",
    )
    return normalized.reset_index(drop=True)


def _normalize_pods_frame(frame):
    required = {"pod_id", "slot_id", "item", "qty", "max_qty"}
    missing = required.difference(frame.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"Scenario pods.csv is missing required columns: {missing_text}")

    normalized = frame.copy()
    normalized = normalized.dropna(subset=["pod_id", "slot_id", "item"]).copy()
    for column in ("pod_id", "slot_id", "item", "qty", "max_qty"):
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    normalized = normalized.dropna(subset=["pod_id", "slot_id", "item"]).copy()
    normalized["pod_id"] = normalized["pod_id"].astype(int)
    normalized["slot_id"] = normalized["slot_id"].astype(int)
    normalized["item"] = normalized["item"].astype(int)
    normalized["qty"] = normalized["qty"].fillna(0).astype(int)
    normalized["max_qty"] = normalized["max_qty"].fillna(0).astype(int)
    return normalized.reset_index(drop=True)


def activate_scenario_inputs(scenario_name=None, target_root=None):
    canonical_name = _canonical_scenario_name(
        scenario_name if scenario_name is not None else os.getenv("RMFS_SCENARIO_NAME")
    )
    if canonical_name is None:
        return None

    output_dir = _scenario_output_dir(canonical_name)
    items_source = output_dir / "items.csv"
    pods_source = output_dir / "pods.csv"
    if not items_source.exists() or not pods_source.exists():
        raise FileNotFoundError(
            f"Scenario bundle '{canonical_name}' is incomplete under {output_dir}."
        )

    destination_root = Path(target_root) if target_root is not None else REPO_ROOT
    destination_root.mkdir(parents=True, exist_ok=True)
    items_target = destination_root / "items.csv"
    pods_target = destination_root / "pods.csv"

    items_frame = _normalize_items_frame(_read_csv_auto(items_source))
    pods_frame = _normalize_pods_frame(_read_csv_auto(pods_source))

    items_frame.to_csv(items_target, index=False)
    pods_frame.to_csv(pods_target, index=False)

    metadata = {
        "scenario_name": canonical_name,
        "scenario_bundle_root": str(_shared_scenario_root()),
        "shared_scenario_root": str(_shared_scenario_root()),
        "items_source": str(items_source),
        "pods_source": str(pods_source),
        "items_target": str(items_target),
        "pods_target": str(pods_target),
        "items_rows": int(len(items_frame)),
        "pods_rows": int(len(pods_frame)),
        "unique_item_ids": int(items_frame["item_id"].nunique()),
        "unique_item_codes": int(items_frame["item_code"].nunique()),
        "unique_pods": int(pods_frame["pod_id"].nunique()),
        "unique_pod_items": int(pods_frame["item"].nunique()),
    }

    runtime_dir = destination_root / "data" / "runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = runtime_dir / "active_scenario.json"
    with open(metadata_path, "w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, indent=2)

    return metadata
