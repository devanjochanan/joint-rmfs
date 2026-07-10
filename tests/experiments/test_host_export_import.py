import csv
import hashlib
import importlib.util
import io
import json
import zipfile
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
    condition = {"condition_key": "x", "run_id": "run-x"}
    ledger = HostLedger.create("host", "campaign", "manifest", [condition])
    ledger.start_condition("x")
    ledger.mark_completed("x", {"result_path": "run_outcomes.csv#run-x", "row_sha256": "row"}, warning_count=1)
    row = {
        "campaign_id": "campaign", "run_id": "run-x", "paired_group_id": "pair",
        "policy_configuration": "all_off", "robot_count": "20", "order_rate": "500",
        "replication": "1", "seed": "42", "source_tree_hash": "source", "layout_hash": "layout",
        "charging_placement_source": "generated_reference", "charging_config_sha256": "cfg",
        "effective_charger_coordinate_hash": "coords", "rts_checkpoint_sha256": "rts",
        "pps_model_sha256": "pps", "kpi_schema_version": "full_kpi_v3",
        "simulation_semantics_id": "semantics", "status": "completed_with_warnings", "orders_completed": "3",
    }
    buffer = io.StringIO(); writer = csv.DictWriter(buffer, fieldnames=list(row)); writer.writeheader(); writer.writerow(row)
    members = {
        "host_ledger.json": json.dumps(ledger.__dict__, default=str).encode(),
        "run_outcomes.csv": buffer.getvalue().encode(),
        "failed_conditions.csv": b"run_id,row_sha256\n",
        "host_assignment.json": json.dumps({"assigned_conditions": [condition]}).encode(),
        "export_metadata.json": json.dumps({"campaign_id": "campaign"}).encode(),
    }
    members["sha256_manifest.json"] = json.dumps({"files": {k: hashlib.sha256(v).hexdigest() for k, v in members.items()}}).encode()
    archive = tmp_path / "host_export_h.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        for name, content in members.items():
            zf.writestr(name, content)
    report = importer.import_exports([archive], tmp_path / "imported")
    assert report["accepted"] == 1
    assert (tmp_path / "imported" / "imported_outcomes.csv").exists()
