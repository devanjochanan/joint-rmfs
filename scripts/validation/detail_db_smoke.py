"""Smoke checks for optional detail DB policy."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.rmfs.runtime_io.detail_db import configure_detail_db
from model.tools.job_task import initialize_job_task_table, upsert_job_task
from model.tools.order_history import initialize_order_history_table, upsert_order_history
from model.tools.pod_location import (
    get_pod_location,
    initialize_pod_location_table,
    upsert_pod_location,
)
from model.tools.pod_travel import initialize_pod_travel_table, upsert_pod_travel
from model.tools.pre_assign import initialize_pre_assign_table, insert_pre_assign


def main() -> int:
    repo_root = _REPO_ROOT
    root_db = repo_root / "warehouse.db"
    root_db_existed = root_db.exists()
    root_db_mtime = root_db.stat().st_mtime_ns if root_db_existed else None

    with tempfile.TemporaryDirectory(prefix="rmfs_detail_db_smoke_") as tmp:
        temp_db = Path(tmp) / "warehouse.db"

        configure_detail_db(enabled=False, db_path=temp_db)
        initialize_pod_location_table("smoke_disabled", db_path=temp_db)
        upsert_pod_location("pod-a", 3, 4, db_path=temp_db)
        assert get_pod_location("pod-a", db_path=temp_db) == (3, 4)

        initialize_job_task_table("smoke_disabled", db_path=temp_db)
        upsert_job_task("pod-a", "order-a", "sku-a", 1, status="queue", db_path=temp_db)
        initialize_order_history_table("smoke_disabled", db_path=temp_db)
        upsert_order_history("order-a", arrival_time=1.0, db_path=temp_db)
        initialize_pod_travel_table("smoke_disabled", db_path=temp_db)
        upsert_pod_travel("job-a", 1, "pod-a", "taking_pod", start_time=1.0, db_path=temp_db)
        initialize_pre_assign_table("smoke_disabled", db_path=temp_db)
        insert_pre_assign(1.0, "picker-1", "order-a", 0.5, "picker-2", 0.7, db_path=temp_db)
        assert not temp_db.exists(), "disabled detail DB should not create warehouse.db"

        configure_detail_db(enabled=True, db_path=temp_db)
        initialize_pod_location_table("smoke_enabled", db_path=temp_db)
        upsert_pod_location("pod-b", 5, 6, db_path=temp_db)
        assert get_pod_location("pod-b", db_path=temp_db) == (5, 6)
        assert temp_db.exists(), "enabled detail DB should create warehouse.db in temp dir"

    if root_db_existed:
        assert root_db.exists()
        assert root_db.stat().st_mtime_ns == root_db_mtime
    else:
        assert not root_db.exists(), "detail DB smoke must not create root warehouse.db"

    print("detail db smoke ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
