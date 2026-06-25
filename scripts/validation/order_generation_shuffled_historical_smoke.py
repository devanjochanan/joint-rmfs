"""Validation smoke test for shuffled_historical_cycle order generation mode."""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.rmfs.order_generation import generate_orders_from_raw_bootstrap


def create_fixtures(temp_dir: Path) -> tuple[Path, Path]:
    items_path = temp_dir / "items.csv"
    raw_order_path = temp_dir / "raw_order.csv"

    # Create dummy items.csv
    items_df = pd.DataFrame([
        {"item_id": 1, "item_code": "AAA"},
        {"item_id": 2, "item_code": "BBB"},
        {"item_id": 3, "item_code": "CCC"},
    ])
    items_df.to_csv(items_path, index=False)

    # Create dummy raw_order.csv
    orders_df = pd.DataFrame([
        {"order_id": "ORD100", "item_code": "AAA", "item_quantity": 2, "created_at": "2026-06-01 10:00:00"},
        {"order_id": "ORD100", "item_code": "BBB", "item_quantity": 1, "created_at": "2026-06-01 10:00:00"},
        {"order_id": "ORD200", "item_code": "BBB", "item_quantity": 3, "created_at": "2026-06-01 10:05:00"},
        {"order_id": "ORD300", "item_code": "CCC", "item_quantity": 5, "created_at": "2026-06-01 10:10:00"},
    ])
    orders_df.to_csv(raw_order_path, index=False)

    return items_path, raw_order_path


def main() -> int:
    temp_dir = Path(tempfile.mkdtemp(prefix="rmfs_shuffled_hist_smoke_"))
    try:
        items_path, raw_order_path = create_fixtures(temp_dir)

        # 1. Run with seed 123
        out_123_a = temp_dir / "out_123_a"
        generate_orders_from_raw_bootstrap(
            seed=123,
            order_generation_mode="shuffled_historical_cycle",
            order_cycle_time=100,
            source_path=raw_order_path,
            target_dir=out_123_a,
            items_csv_path=items_path,
            order_start_arrival_time=0,
        )

        # Load generated orders and metadata
        orders_a = pd.read_csv(out_123_a / "generated_order.csv")
        with open(out_123_a / "generated_order_meta.json", "r", encoding="utf-8") as f:
            meta_a = json.load(f)

        # Assertions for shuffled_historical_cycle mode properties:
        # A. Mode metadata correctly recorded
        assert meta_a["order_generation_mode"] == "shuffled_historical_cycle", "Mode should be shuffled_historical_cycle"
        # B. Counts are correct
        assert meta_a["source_unique_orders"] == 3, "Source unique orders should be 3"
        assert meta_a["generated_unique_orders"] == 3, "Generated unique orders should be 3"
        assert meta_a["n_orders"] is None, "n_orders should be None in metadata to avoid misleading claims"
        assert meta_a["bootstrap_n_orders"] is None, "bootstrap_n_orders should be None in metadata"
        assert meta_a["order_cycle_time"] == 100, "Configured order rate should be recorded"
        assert meta_a["arrival_mode"] == "cycle_exponential", "Arrival mode should be cycle_exponential"
        assert meta_a["shuffle_full_order_sequence"] is True, "shuffle_full_order_sequence should be True"

        # C. Verify basket composition and quantities are preserved exactly
        def check_order_baskets(orders_df):
            # Check ORD100 basket
            ord100 = orders_df[orders_df["source_order_id"] == "ORD100"]
            assert len(ord100) == 2, "ORD100 should have 2 lines"
            assert set(ord100["item_id"]) == {1, 2}, "ORD100 should contain item 1 and 2"
            assert ord100[ord100["item_id"] == 1]["item_quantity"].values[0] == 2
            assert ord100[ord100["item_id"] == 2]["item_quantity"].values[0] == 1

            # Check ORD200 basket
            ord200 = orders_df[orders_df["source_order_id"] == "ORD200"]
            assert len(ord200) == 1, "ORD200 should have 1 line"
            assert ord200["item_id"].values[0] == 2
            assert ord200["item_quantity"].values[0] == 3

            # Check ORD300 basket
            ord300 = orders_df[orders_df["source_order_id"] == "ORD300"]
            assert len(ord300) == 1, "ORD300 should have 1 line"
            assert ord300["item_id"].values[0] == 3
            assert ord300["item_quantity"].values[0] == 5

        check_order_baskets(orders_a)

        # 2. Verify determinism: repeating with the same seed produces identical output
        out_123_b = temp_dir / "out_123_b"
        generate_orders_from_raw_bootstrap(
            seed=123,
            order_generation_mode="shuffled_historical_cycle",
            order_cycle_time=100,
            source_path=raw_order_path,
            target_dir=out_123_b,
            items_csv_path=items_path,
            order_start_arrival_time=0,
        )
        orders_b = pd.read_csv(out_123_b / "generated_order.csv")
        pd.testing.assert_frame_equal(orders_a, orders_b)

        # 3. Verify permutation: different seed produces different sequence but same orders
        out_456 = temp_dir / "out_456"
        generate_orders_from_raw_bootstrap(
            seed=456,
            order_generation_mode="shuffled_historical_cycle",
            order_cycle_time=100,
            source_path=raw_order_path,
            target_dir=out_456,
            items_csv_path=items_path,
            order_start_arrival_time=0,
        )
        orders_diff = pd.read_csv(out_456 / "generated_order.csv")
        check_order_baskets(orders_diff)

        # Extract sequence of source orders
        seq_a = orders_a.drop_duplicates("order_id")["source_order_id"].tolist()
        seq_diff = orders_diff.drop_duplicates("order_id")["source_order_id"].tolist()
        # Verify sequence is a different permutation
        assert seq_a != seq_diff, f"Expected different permutation with different seed, but got: {seq_a} vs {seq_diff}"
        assert set(seq_a) == set(seq_diff), "Order membership must be preserved"

        print("order generation shuffled historical smoke OK")
        return 0
    except Exception as e:
        print(f"FAIL: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
