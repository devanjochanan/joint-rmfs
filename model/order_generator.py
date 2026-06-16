import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


current_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.dirname(current_directory)

RAW_ORDER_ID_CANDIDATES = ["order_id", "Order ID"]
RAW_ITEM_CODE_CANDIDATES = ["item_code", "Item Code"]
RAW_QUANTITY_CANDIDATES = ["item_quantity", "quantity", "qty", "Item Quantity"]
RAW_CREATED_AT_CANDIDATES = ["order_date", "created_at", "order_time", "Order Date"]


def _normalize_item_code(value):
    text = str(value).replace("\ufeff", "").strip()
    if not text or text.lower() in {"nan", "none"}:
        return ""
    return text[:-2] if text.endswith(".0") else text


def _find_column(columns, candidates):
    normalized = {str(col).replace("\ufeff", "").strip(): col for col in columns}
    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]
    raise KeyError(
        f"Could not find any of the expected columns {candidates}. "
        f"Available columns: {list(columns)}"
    )


def _env_int(name, default):
    value = os.getenv(name)
    if value is None or str(value).strip() == "":
        return default
    return int(value)


def _env_optional_int(name):
    value = os.getenv(name)
    if value is None or str(value).strip() == "":
        return None
    return int(value)


def _bootstrap_source_path(source_path=None):
    if source_path is not None:
        return Path(source_path)
    env_path = os.getenv("RMFS_BOOTSTRAP_ORDER_PATH")
    if env_path:
        return Path(env_path)
    return Path(parent_directory) / "raw_order.csv"


def _load_item_lookup(items_csv_path=None):
    items_path = Path(items_csv_path) if items_csv_path else Path(parent_directory) / "items.csv"
    items = pd.read_csv(items_path)
    items["item_code"] = items["item_code"].map(_normalize_item_code)
    return items[["item_id", "item_code"]].drop_duplicates("item_code")


def _load_raw_orders(source_path, items_csv_path=None):
    # Auto-detect delimiter so the bootstrap input can be either comma- or
    # semicolon-delimited without requiring a separate conversion step.
    raw_orders = pd.read_csv(source_path, sep=None, engine="python")

    order_col = _find_column(raw_orders.columns, RAW_ORDER_ID_CANDIDATES)
    item_code_col = _find_column(raw_orders.columns, RAW_ITEM_CODE_CANDIDATES)
    qty_col = _find_column(raw_orders.columns, RAW_QUANTITY_CANDIDATES)
    created_col = _find_column(raw_orders.columns, RAW_CREATED_AT_CANDIDATES)

    raw_orders = raw_orders[[order_col, item_code_col, qty_col, created_col]].copy()
    raw_orders.columns = ["source_order_id", "item_code", "item_quantity", "created_at"]
    raw_orders["source_order_id"] = raw_orders["source_order_id"].astype(str).str.strip()
    raw_orders["item_code"] = raw_orders["item_code"].map(_normalize_item_code)
    raw_orders["item_quantity"] = pd.to_numeric(raw_orders["item_quantity"], errors="coerce")
    raw_orders["created_at"] = pd.to_datetime(
        raw_orders["created_at"],
        errors="coerce",
        dayfirst=True,
    )

    raw_orders = raw_orders.dropna(
        subset=["source_order_id", "item_code", "item_quantity", "created_at"]
    ).copy()
    raw_orders = raw_orders[raw_orders["item_code"] != ""].copy()
    raw_orders["item_quantity"] = np.ceil(raw_orders["item_quantity"]).astype(int)
    raw_orders = raw_orders[raw_orders["item_quantity"] > 0].copy()

    if raw_orders.empty:
        raise ValueError(f"No valid order rows remained after parsing {source_path}.")

    item_lookup = _load_item_lookup(items_csv_path=items_csv_path)
    raw_orders = raw_orders.merge(item_lookup, on="item_code", how="left")

    missing_item_mask = raw_orders["item_id"].isna()
    missing_sku_count = int(missing_item_mask.sum())
    missing_codes = []
    if missing_item_mask.any():
        missing_codes = sorted(
            raw_orders.loc[missing_item_mask, "item_code"].astype(str).unique().tolist()
        )
        raw_orders = raw_orders.loc[~missing_item_mask].copy()

    if raw_orders.empty:
        raise ValueError(
            "No valid order rows remained after filtering to item codes present in items.csv."
        )

    raw_orders["item_id"] = raw_orders["item_id"].astype(int)
    raw_orders = raw_orders.sort_values(
        ["created_at", "source_order_id", "item_id"],
        kind="stable",
    ).reset_index(drop=True)

    return raw_orders, missing_item_mask, missing_sku_count, missing_codes


def _build_empirical_order_table(raw_orders):
    order_table = (
        raw_orders.groupby("source_order_id", sort=False)
        .agg(created_at=("created_at", "min"))
        .reset_index()
        .sort_values(["created_at", "source_order_id"], kind="stable")
        .reset_index(drop=True)
    )
    base_time = order_table["created_at"].min()
    order_table["arrival_seconds"] = (
        order_table["created_at"] - base_time
    ).dt.total_seconds().round().astype(int)
    return order_table


def _bootstrap_arrivals(order_table, sample_size, rng, arrival_mode, order_start_arrival_time):
    start_time = int(order_start_arrival_time)
    if sample_size <= 0:
        return np.asarray([], dtype=np.int64)

    empirical_arrivals = order_table["arrival_seconds"].to_numpy(dtype=np.int64)
    if empirical_arrivals.size == 0:
        return np.full(sample_size, start_time, dtype=np.int64)

    if arrival_mode == "sample_original_times":
        sampled_arrivals = rng.choice(empirical_arrivals, size=sample_size, replace=True)
        sampled_arrivals = np.sort(sampled_arrivals.astype(np.int64))
        return sampled_arrivals + start_time

    if arrival_mode != "empirical_interarrival":
        raise ValueError(
            "Unsupported bootstrap arrival mode: "
            f"{arrival_mode}. Expected 'empirical_interarrival' or 'sample_original_times'."
        )

    horizon = int(empirical_arrivals.max())
    if empirical_arrivals.size == 1 or horizon <= 0:
        return np.full(sample_size, start_time, dtype=np.int64)

    empirical_gaps = np.diff(empirical_arrivals)
    if empirical_gaps.size == 0:
        return np.full(sample_size, start_time, dtype=np.int64)

    sampled_gaps = rng.choice(empirical_gaps, size=max(sample_size - 1, 0), replace=True)
    sampled_arrivals = np.concatenate(
        [np.asarray([0], dtype=np.float64), np.cumsum(sampled_gaps, dtype=np.float64)]
    )

    if sampled_arrivals[-1] > 0:
        sampled_arrivals = (sampled_arrivals / sampled_arrivals[-1]) * horizon

    sampled_arrivals = np.rint(sampled_arrivals).astype(np.int64)
    sampled_arrivals = np.maximum.accumulate(sampled_arrivals)
    return sampled_arrivals + start_time


def _build_generated_orders(sampled_order_ids, raw_orders, sampled_arrivals):
    order_lines = []
    sequence_id = 0

    for generated_order_id, (source_order_id, order_arrival) in enumerate(
        zip(sampled_order_ids, sampled_arrivals)
    ):
        source_lines = raw_orders.loc[
            raw_orders["source_order_id"] == source_order_id,
            ["item_id", "item_quantity"],
        ]
        for line in source_lines.itertuples(index=False):
            order_lines.append(
                {
                    "sequence_id": int(sequence_id),
                    "order_id": int(generated_order_id),
                    "order_type": 1,
                    "item_id": int(line.item_id),
                    "item_quantity": int(line.item_quantity),
                    "order_arrival": int(order_arrival),
                    "source_order_id": str(source_order_id),
                }
            )
            sequence_id += 1

    generated_order = pd.DataFrame(order_lines)
    if generated_order.empty:
        raise ValueError("Bootstrap generation produced no order lines.")

    return generated_order


def _build_generated_database_order(generated_order):
    database_order = pd.DataFrame(
        {
            "order_id": generated_order["sequence_id"].astype(int),
            "order_dum": generated_order["order_id"].astype(int),
            "order_type": generated_order["order_type"].astype(int),
            "item": generated_order["item_id"].astype(int),
            "qty": generated_order["item_quantity"].astype(int),
            "facing": -1,
            "due_date": 99999,
            "station": -1,
            "pod_id": -1,
            "status": -3,
            "finish_time": -1,
            "date": 1,
            "time_gen": generated_order["order_arrival"].astype(int),
            "source_order_id": generated_order["source_order_id"].astype(str),
        }
    )
    return database_order


def generate_orders_from_raw_bootstrap(
    seed=None,
    n_orders=None,
    arrival_mode="empirical_interarrival",
    order_start_arrival_time=0,
    source_path=None,
    target_dir=None,
    items_csv_path=None,
):
    # Unified seed: prefer explicit seed arg, then RMFS_SIM_SEED, then
    # RMFS_BOOTSTRAP_SEED for backward compatibility, then default 42.
    if seed is not None:
        resolved_seed = int(seed)
    else:
        sim_seed = os.getenv("RMFS_SIM_SEED", "").strip()
        if sim_seed:
            resolved_seed = int(sim_seed)
        else:
            resolved_seed = _env_int("RMFS_BOOTSTRAP_SEED", 42)
    resolved_n_orders = (
        _env_optional_int("RMFS_BOOTSTRAP_N_ORDERS") if n_orders is None else int(n_orders)
    )
    resolved_arrival_mode = os.getenv("RMFS_BOOTSTRAP_ARRIVAL_MODE", arrival_mode)
    resolved_start_arrival = (
        _env_int("RMFS_BOOTSTRAP_START_ARRIVAL", 0)
        if order_start_arrival_time is None
        else int(order_start_arrival_time)
    )
    resolved_source_path = _bootstrap_source_path(source_path)

    rng = np.random.default_rng(resolved_seed)

    raw_orders, _, missing_sku_count, missing_codes = _load_raw_orders(
        resolved_source_path, items_csv_path=items_csv_path,
    )
    order_table = _build_empirical_order_table(raw_orders)
    source_order_ids = order_table["source_order_id"].to_numpy(dtype=object)
    source_unique_orders = int(source_order_ids.size)

    if source_unique_orders <= 0:
        raise ValueError("Bootstrap source contains no valid unique orders.")

    if resolved_n_orders is None:
        resolved_n_orders = source_unique_orders
    if resolved_n_orders <= 0:
        raise ValueError("Bootstrap n_orders must be positive.")

    sampled_order_ids = rng.choice(
        source_order_ids,
        size=int(resolved_n_orders),
        replace=True,
    )
    sampled_arrivals = _bootstrap_arrivals(
        order_table=order_table,
        sample_size=int(resolved_n_orders),
        rng=rng,
        arrival_mode=resolved_arrival_mode,
        order_start_arrival_time=resolved_start_arrival,
    )
    generated_order = _build_generated_orders(
        sampled_order_ids=sampled_order_ids,
        raw_orders=raw_orders,
        sampled_arrivals=sampled_arrivals,
    )
    generated_database_order = _build_generated_database_order(generated_order)

    output_dir = Path(target_dir) if target_dir else Path(parent_directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_order_path = output_dir / "generated_order.csv"
    generated_database_order_path = output_dir / "generated_database_order.csv"
    generated_meta_path = output_dir / "generated_order_meta.json"

    generated_order.to_csv(generated_order_path, index=False)
    generated_database_order.to_csv(generated_database_order_path, index=False)

    metadata = {
        "generator": "bootstrap_raw_order",
        "source_path": str(resolved_source_path),
        "seed": int(resolved_seed),
        "n_orders": int(resolved_n_orders),
        "arrival_mode": resolved_arrival_mode,
        "order_start_arrival_time": int(resolved_start_arrival),
        "source_unique_orders": int(source_unique_orders),
        "sampled_unique_source_orders": int(pd.Series(sampled_order_ids).nunique()),
        "generated_unique_orders": int(generated_order["order_id"].nunique()),
        "generated_order_lines": int(len(generated_order)),
        "generated_max_arrival": int(generated_order["order_arrival"].max()),
        "missing_sku_lines_dropped": int(missing_sku_count),
        "missing_sku_examples": missing_codes[:10],
    }
    with open(generated_meta_path, "w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, indent=2, ensure_ascii=False)

    print(
        "Generated bootstrap order stream "
        f"with {metadata['generated_unique_orders']:,} orders and "
        f"{metadata['generated_order_lines']:,} lines "
        f"(seed={metadata['seed']}, arrival_mode={metadata['arrival_mode']})."
    )
    if missing_sku_count > 0:
        preview = ", ".join(metadata["missing_sku_examples"])
        print(
            f"Dropped {missing_sku_count:,} raw-order lines whose item_code was not found in items.csv. "
            f"Sample missing codes: {preview}"
        )

    return generated_order


def config_orders(
    initial_order=None,
    total_requested_item=None,
    items_orders_class_configuration=None,
    quantity_range=None,
    order_cycle_time=None,
    order_period_time=None,
    order_start_arrival_time=None,
    date=None,
    sim_ver=None,
    dev_mode=False,
    seed=None,
    n_orders=None,
    arrival_mode="empirical_interarrival",
    source_path=None,
    target_dir=None,
    items_csv_path=None,
):
    # Legacy synthetic parameters are intentionally ignored. joint-rmfs now
    # always prepares its order stream from bootstrap resampling of raw_order.csv.
    return generate_orders_from_raw_bootstrap(
        seed=seed,
        n_orders=n_orders,
        arrival_mode=arrival_mode,
        order_start_arrival_time=order_start_arrival_time,
        source_path=source_path,
        target_dir=target_dir,
        items_csv_path=items_csv_path,
    )
