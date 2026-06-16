"""Smoke checks for lightweight runtime timing instrumentation."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.rmfs.runtime_io import RunContext
from src.rmfs.runtime_io.timing import (
    configure_timing,
    increment_counter,
    timed,
    write_timing_summary,
)


def main() -> int:
    repo_root = _REPO_ROOT
    ctx = RunContext.default(repo_root=repo_root)

    with tempfile.TemporaryDirectory(prefix="rmfs_timing_smoke_") as tmp:
        tmp_path = Path(tmp)
        summary_path = tmp_path / "timing_summary.json"
        configure_timing(enabled=True, output_path=summary_path)

        with timed("csv_read"):
            with ctx.generated_pod_csv.open(newline="", encoding="utf-8") as fh:
                rows = list(csv.reader(fh))

        out_csv = tmp_path / "copy.csv"
        with timed("csv_write"):
            with out_csv.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.writer(fh)
                writer.writerows(rows[:5])

        increment_counter("worker_status_write", 2)
        written = write_timing_summary()
        assert written == summary_path
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        assert payload["enabled"] is True
        assert payload["sections"]["csv_read"]["count"] == 1
        assert payload["sections"]["csv_write"]["count"] == 1
        assert payload["counters"]["worker_status_write"] == 2

    print("runtime io performance smoke ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
