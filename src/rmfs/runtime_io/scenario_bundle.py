"""Scenario bundle helpers for RMFS input CSVs."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SCENARIO_ROOT = REPO_ROOT / "data" / "input" / "scenarios"
DEFAULT_METADATA_PATH = REPO_ROOT / "data" / "runtime" / "active_scenario.json"

SCENARIO_ALIASES = {
    "cindy_s1": "cindy_s1",
    "cindy1": "cindy_s1",
    "scenario1": "cindy_s1",
    "cindy_s2": "cindy_s2",
    "cindy2": "cindy_s2",
    "scenario2": "cindy_s2",
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


def _scenario_root(root: str | os.PathLike[str] | None = None) -> Path:
    return Path(
        root
        or os.environ.get("RMFS_SCENARIO_ROOT", "")
        or DEFAULT_SCENARIO_ROOT
    ).resolve()


def normalize_scenario_name(name: str | None) -> str | None:
    """Normalize a scenario name or alias to the canonical bundle directory."""
    if name is None:
        return None
    normalized = str(name).strip().lower().replace("-", "_")
    if not normalized:
        return None
    return SCENARIO_ALIASES.get(normalized, normalized)


def list_available_scenarios(
    scenario_root: str | os.PathLike[str] | None = None,
) -> tuple[str, ...]:
    """Return scenario directories containing both items.csv and pods.csv."""
    root = _scenario_root(scenario_root)
    if not root.exists():
        return ()
    scenarios = []
    for path in sorted(root.iterdir()):
        if not path.is_dir():
            continue
        if (path / "items.csv").exists() and (path / "pods.csv").exists():
            scenarios.append(path.name)
    return tuple(scenarios)


def _scenario_dir(
    scenario_name: str | None,
    scenario_root: str | os.PathLike[str] | None = None,
) -> tuple[str, Path]:
    canonical = normalize_scenario_name(
        scenario_name if scenario_name is not None else os.environ.get("RMFS_SCENARIO_NAME")
    )
    if canonical is None:
        raise ValueError("Scenario name is required.")

    root = _scenario_root(scenario_root)
    path = root / canonical
    if not path.exists():
        available = ", ".join(list_available_scenarios(root)) or "(none)"
        raise ValueError(f"Unknown RMFS scenario '{scenario_name}'. Available scenarios: {available}")
    return canonical, path


def _read_csv_auto(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, sep=None, engine="python")
    frame.columns = [str(col).replace("\ufeff", "").strip() for col in frame.columns]
    return frame


def _normalize_identifier(value: Any) -> str:
    text = str(value).replace("\ufeff", "").strip()
    if not text or text.lower() in {"nan", "none"}:
        return ""
    return text[:-2] if text.endswith(".0") else text


def normalize_items_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize scenario items.csv contents."""
    required = {"item_id", "item_code"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(
            "Scenario items.csv is missing required columns: "
            + ", ".join(sorted(missing))
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
            values = pd.to_numeric(normalized[column], errors="coerce")
            if values.notna().all():
                normalized[column] = values.astype(int)

    normalized = normalized.sort_values("item_id", kind="stable").drop_duplicates(
        subset=["item_id"],
        keep="first",
    )
    return normalized.reset_index(drop=True)


def normalize_pods_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize scenario pods.csv contents."""
    required = {"pod_id", "slot_id", "item", "qty", "max_qty"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(
            "Scenario pods.csv is missing required columns: "
            + ", ".join(sorted(missing))
        )

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


def read_scenario_inputs(
    scenario_name: str | None,
    scenario_root: str | os.PathLike[str] | None = None,
) -> tuple[str, pd.DataFrame, pd.DataFrame]:
    """Read and validate a scenario's item and pod input frames."""
    canonical, path = _scenario_dir(scenario_name, scenario_root)
    items_path = path / "items.csv"
    pods_path = path / "pods.csv"
    if not items_path.exists() or not pods_path.exists():
        raise FileNotFoundError(f"Scenario bundle '{canonical}' is missing items.csv or pods.csv.")
    return (
        canonical,
        normalize_items_frame(_read_csv_auto(items_path)),
        normalize_pods_frame(_read_csv_auto(pods_path)),
    )


def activate_scenario_inputs(
    scenario_name: str | None = None,
    target_root: str | os.PathLike[str] | None = None,
    scenario_root: str | os.PathLike[str] | None = None,
    dry_run: bool = False,
) -> dict[str, Any] | None:
    """Copy normalized scenario items.csv and pods.csv into a target root."""
    selected_name = scenario_name if scenario_name is not None else os.environ.get("RMFS_SCENARIO_NAME")
    if selected_name is None or str(selected_name).strip() == "":
        return None

    canonical, scenario_path = _scenario_dir(selected_name, scenario_root)
    items_source = scenario_path / "items.csv"
    pods_source = scenario_path / "pods.csv"
    items_frame = normalize_items_frame(_read_csv_auto(items_source))
    pods_frame = normalize_pods_frame(_read_csv_auto(pods_source))

    destination_root = Path(target_root).resolve() if target_root is not None else REPO_ROOT
    items_target = destination_root / "items.csv"
    pods_target = destination_root / "pods.csv"
    metadata_path = destination_root / "data" / "runtime" / "active_scenario.json"
    metadata = {
        "scenario_name": canonical,
        "scenario_bundle_root": str(_scenario_root(scenario_root)),
        "items_source": str(items_source),
        "pods_source": str(pods_source),
        "items_target": str(items_target),
        "pods_target": str(pods_target),
        "metadata_target": str(metadata_path),
        "items_rows": int(len(items_frame)),
        "pods_rows": int(len(pods_frame)),
        "unique_item_ids": int(items_frame["item_id"].nunique()),
        "unique_item_codes": int(items_frame["item_code"].nunique()),
        "unique_pods": int(pods_frame["pod_id"].nunique()),
        "unique_pod_items": int(pods_frame["item"].nunique()),
        "dry_run": bool(dry_run),
    }

    if dry_run:
        return metadata

    destination_root.mkdir(parents=True, exist_ok=True)
    items_frame.to_csv(items_target, index=False)
    pods_frame.to_csv(pods_target, index=False)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def copy_scenario_bundle(
    source_dir: str | os.PathLike[str],
    scenario_name: str,
    scenario_root: str | os.PathLike[str] | None = None,
) -> Path:
    """Copy a bundle directory into data/input/scenarios after validation."""
    source = Path(source_dir)
    canonical = normalize_scenario_name(scenario_name)
    if canonical is None:
        raise ValueError("Scenario name is required.")
    read_scenario_inputs(canonical, source.parent)
    destination = _scenario_root(scenario_root) / canonical
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source / "items.csv", destination / "items.csv")
    shutil.copy2(source / "pods.csv", destination / "pods.csv")
    return destination
