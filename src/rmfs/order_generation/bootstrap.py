"""Bootstrap order generation from raw order data.

Moved from model/order_generator.py to become part of the order generation
owner module at src/rmfs/order_generation/.
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.rmfs.order_generation.policy import resolve_order_generation_policy


current_directory = os.path.dirname(os.path.abspath(__file__))
# Parent of src/rmfs/order_generation/ is src/rmfs/, grandparent is src/, great-grandparent is repo root
_repo_root = Path(__file__).resolve().parents[3]
CANONICAL_INPUT_BASE = _repo_root / "data" / "input" / "base"
CANONICAL_RUNTIME_TMP = _repo_root / "data" / "runtime" / "tmp"

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
    canonical = CANONICAL_INPUT_BASE / "raw_order.csv"
    return canonical if canonical.exists() else _repo_root / "raw_order.csv"


def _load_item_lookup(items_csv_path=None):
    if items_csv_path:
        items_path = Path(items_csv_path)
    else:
        canonical = CANONICAL_INPUT_BASE / "items.csv"
        items_path = canonical if canonical.exists() else _repo_root / "items.csv"
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


def _bootstrap_arrivals(
    order_table,
    sample_size,
    rng,
    arrival_mode,
    order_start_arrival_time,
    target_horizon_ticks=None,
):
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

    horizon = int(target_horizon_ticks) if target_horizon_ticks is not None else int(empirical_arrivals.max())
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


def _cycle_rate_arrivals(
    sample_size,
    rng,
    order_cycle_time,
    order_start_arrival_time,
):
    """Generate random arrivals at an exact configured orders-per-hour rate."""
    start_time = int(order_start_arrival_time)
    orders_per_hour = int(order_cycle_time)
    if orders_per_hour <= 0:
        raise ValueError("order_cycle_time must be a positive orders-per-hour value.")
    if sample_size <= 0:
        return np.asarray([], dtype=np.int64)

    cycle_arrivals = []
    for cycle_index, offset in enumerate(range(0, sample_size, orders_per_hour)):
        cycle_count = min(orders_per_hour, sample_size - offset)
        mean_gap_seconds = 3600.0 / orders_per_hour
        random_gaps = rng.exponential(mean_gap_seconds, size=cycle_count)
        cumulative = np.cumsum(random_gaps, dtype=np.float64)

        # Keep exactly order_cycle_time orders in each full hour while
        # retaining random exponential spacing inside that hour.
        target_span = (3600.0 * cycle_count / orders_per_hour) - 1.0
        if cumulative[-1] > 0 and target_span > 0:
            cumulative = (cumulative / cumulative[-1]) * target_span

        arrivals = np.rint(cumulative).astype(np.int64)
        arrivals += start_time + cycle_index * 3600
        cycle_arrivals.append(arrivals)

    return np.concatenate(cycle_arrivals)


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
    run_horizon_ticks=None,
    demand_horizon_ticks=None,
    demand_buffer_ticks=None,
    order_cycle_time=None,
    shuffle_full_order_sequence=False,
    order_generation_mode=None,
    full_raw_order_replay=None,
    profile=None,
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
    policy = resolve_order_generation_policy(
        profile=profile,
        n_orders=n_orders,
        run_horizon_ticks=run_horizon_ticks,
        demand_horizon_ticks=demand_horizon_ticks,
        demand_buffer_ticks=demand_buffer_ticks,
        order_generation_mode=order_generation_mode,
        full_raw_order_replay=full_raw_order_replay,
    )
    resolved_n_orders = policy.bootstrap_n_orders
    resolved_full_raw_replay = bool(policy.full_raw_order_replay)
    resolved_mode = policy.order_generation_mode
    if resolved_mode == "full_raw_replay":
        resolved_full_raw_replay = True
    resolved_arrival_mode = os.getenv("RMFS_BOOTSTRAP_ARRIVAL_MODE", arrival_mode)
    resolved_start_arrival = (
        _env_int("RMFS_BOOTSTRAP_START_ARRIVAL", 0)
        if order_start_arrival_time is None
        else int(order_start_arrival_time)
    )
    resolved_source_path = _bootstrap_source_path(source_path)
    resolved_order_cycle_time = (
        int(order_cycle_time)
        if order_cycle_time is not None
        else _env_optional_int("RMFS_ORDER_CYCLE_TIME")
    )
    resolved_shuffle_full_sequence = bool(shuffle_full_order_sequence)

    rng = np.random.default_rng(resolved_seed)

    raw_orders, _, missing_sku_count, missing_codes = _load_raw_orders(
        resolved_source_path, items_csv_path=items_csv_path,
    )
    order_table = _build_empirical_order_table(raw_orders)
    source_order_ids = order_table["source_order_id"].to_numpy(dtype=object)
    source_unique_orders = int(source_order_ids.size)

    if source_unique_orders <= 0:
        raise ValueError("Bootstrap source contains no valid unique orders.")

    if resolved_shuffle_full_sequence:
        if resolved_order_cycle_time is None:
            raise ValueError(
                "order_cycle_time is required when shuffle_full_order_sequence=True."
            )
        resolved_n_orders = source_unique_orders
        resolved_full_raw_replay = False
        resolved_mode = "shuffled_full_cycle"
    elif resolved_n_orders is None and not resolved_full_raw_replay:
        resolved_full_raw_replay = True
        resolved_mode = "legacy_compat"
        print(
            "[ORDER_GENERATION] Legacy GUI/manual fallback is replaying the full raw order source. "
            "Set RMFS_BOOTSTRAP_N_ORDERS or RMFS_FULL_RAW_ORDER_REPLAY=1 explicitly for headless runs."
        )

    if resolved_full_raw_replay:
        resolved_n_orders = source_unique_orders
    if int(resolved_n_orders) <= 0:
        raise ValueError("Bootstrap n_orders must be positive.")

    if resolved_shuffle_full_sequence:
        sampled_order_ids = rng.permutation(source_order_ids)
        sampled_arrivals = _cycle_rate_arrivals(
            sample_size=source_unique_orders,
            rng=rng,
            order_cycle_time=resolved_order_cycle_time,
            order_start_arrival_time=resolved_start_arrival,
        )
        resolved_arrival_mode = "cycle_exponential"
    elif resolved_full_raw_replay:
        sampled_order_ids = source_order_ids
        sampled_arrivals = order_table["arrival_seconds"].to_numpy(dtype=np.int64) + int(resolved_start_arrival)
    else:
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
            target_horizon_ticks=policy.demand_horizon_ticks,
        )
    generated_order = _build_generated_orders(
        sampled_order_ids=sampled_order_ids,
        raw_orders=raw_orders,
        sampled_arrivals=sampled_arrivals,
    )
    generated_database_order = _build_generated_database_order(generated_order)

    output_dir = Path(target_dir) if target_dir else CANONICAL_RUNTIME_TMP
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_order_path = output_dir / "generated_order.csv"
    generated_database_order_path = output_dir / "generated_database_order.csv"
    generated_meta_path = output_dir / "generated_order_meta.json"

    generated_order.to_csv(generated_order_path, index=False)
    generated_database_order.to_csv(generated_database_order_path, index=False)

    metadata = {
        "generator": "bootstrap_raw_order",
        "order_generation_mode": resolved_mode,
        "profile": policy.profile,
        "source_path": str(resolved_source_path),
        "items_csv_path": str(Path(items_csv_path)) if items_csv_path else None,
        "target_dir": str(output_dir),
        "seed": int(resolved_seed),
        "n_orders": int(resolved_n_orders),
        "bootstrap_n_orders": int(resolved_n_orders),
        "run_horizon_ticks": policy.run_horizon_ticks,
        "demand_horizon_ticks": policy.demand_horizon_ticks,
        "demand_buffer_ticks": policy.demand_buffer_ticks,
        "full_raw_order_replay": bool(resolved_full_raw_replay),
        "shuffle_full_order_sequence": bool(resolved_shuffle_full_sequence),
        "order_cycle_time": resolved_order_cycle_time,
        "order_cycle_time_unit": "orders_per_hour" if resolved_order_cycle_time is not None else None,
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
    run_horizon_ticks=None,
    demand_horizon_ticks=None,
    demand_buffer_ticks=None,
    order_generation_mode=None,
    full_raw_order_replay=None,
    shuffle_full_order_sequence=False,
    profile=None,
):
    # SKU composition always comes from complete historical orders in raw_order.csv.
    # PPS training may additionally shuffle the full sequence and use order_cycle_time
    # as an orders-per-hour arrival-rate control.
    return generate_orders_from_raw_bootstrap(
        seed=seed,
        n_orders=n_orders,
        arrival_mode=arrival_mode,
        order_start_arrival_time=order_start_arrival_time,
        source_path=source_path,
        target_dir=target_dir,
        items_csv_path=items_csv_path,
        run_horizon_ticks=run_horizon_ticks,
        demand_horizon_ticks=demand_horizon_ticks,
        demand_buffer_ticks=demand_buffer_ticks,
        order_cycle_time=order_cycle_time,
        shuffle_full_order_sequence=shuffle_full_order_sequence,
        order_generation_mode=order_generation_mode,
        full_raw_order_replay=full_raw_order_replay,
        profile=profile,
    )
