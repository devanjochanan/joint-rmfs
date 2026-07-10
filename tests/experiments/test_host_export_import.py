import importlib.util
import json
from pathlib import Path

from src.rmfs.orchestration.host_ledger import HostLedger


ROOT = Path(__file__).resolve().parents[2]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / "experiments" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_importer_accepts_hash_checked_warning_complete_result(tmp_path):
    importer = _load("import_host_exports")
    import zipfile, hashlib
    condition = {"condition_key": "x", "run_id": "run-x"}
    ledger = HostLedger.create("host", "campaign", "manifest", [condition])
    ledger.start_condition("x")
    ledger.mark_completed("x", {"result_path": "runs/run-x/worker_summary.json"}, warning_count=1)
    spec = {"campaign_id": "campaign", "policy_configuration": "all_off", "robot_count": 20,
            "order_rate_per_hour": 500, "replication": 1, "campaign_seed": 42}
    summary = {"status": "success", "finalization": {"finalized": True}, "kpi": {"orders_completed": 3}}
    members = {
        "host_ledger.json": json.dumps(ledger.__dict__, default=str).encode(),
        "runs/run-x/run_spec.json": json.dumps(spec).encode(),
        "runs/run-x/worker_summary.json": json.dumps(summary).encode(),
    }
    members["sha256_manifest.json"] = json.dumps({"files": {k: hashlib.sha256(v).hexdigest() for k, v in members.items()}}).encode()
    archive = tmp_path / "host_export_h.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        for name, content in members.items():
            zf.writestr(name, content)
    report = importer.import_exports([archive], tmp_path / "imported")
    assert report["accepted"] == 1
    assert (tmp_path / "imported" / "imported_outcomes.csv").exists()
