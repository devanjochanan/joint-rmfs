import os
from pathlib import Path

import pandas as pd

from model.inventory import Inventory


def _inventory(tmp_path: Path) -> Inventory:
    inv = Inventory.__new__(Inventory)
    inv.runtime_paths = {
        "assign_order_csv": str(tmp_path / "assign_order.csv"),
        "generated_order_csv": str(tmp_path / "generated_order.csv"),
    }
    return inv


def test_assign_order_atomic_write_retries_transient_permission_error(tmp_path, monkeypatch):
    inv = _inventory(tmp_path)
    real_replace = os.replace
    calls = {"count": 0}

    def flaky_replace(src, dst):
        calls["count"] += 1
        if calls["count"] == 1:
            raise PermissionError("temporarily locked")
        return real_replace(src, dst)

    monkeypatch.setattr("model.inventory.os.replace", flaky_replace)

    inv._write_assign_order_csv(pd.DataFrame([{"order_id": 1, "status": -3}]))

    assert calls["count"] == 2
    assert pd.read_csv(inv.assign_order_csv).to_dict("records") == [{"order_id": 1, "status": -3}]


def test_assign_order_read_parser_failures_are_not_silently_accepted(tmp_path, monkeypatch):
    inv = _inventory(tmp_path)
    Path(inv.assign_order_csv).write_text("not,a,stable,csv\n\"unterminated", encoding="utf-8")

    def always_bad(_path):
        raise pd.errors.ParserError("partial write")

    monkeypatch.setattr("model.inventory.pd.read_csv", always_bad)

    try:
        inv._read_assign_order_csv()
    except RuntimeError as exc:
        assert "assign_order.csv read failed" in str(exc)
        assert "ParserError" in str(exc)
    else:
        raise AssertionError("partial assign_order.csv parse was silently accepted")
