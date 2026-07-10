"""End-to-end synthetic regression for the compact autonomous-host contract."""

from __future__ import annotations

import importlib.util
import json
import zipfile
from pathlib import Path

from scripts.experiments import distributed_sensitivity_campaign as campaign
from tests.experiments.test_distributed_sensitivity_campaign import fake_assets


ROOT = Path(__file__).resolve().parents[2]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / "experiments" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _success_summary(spec):
    kpi = {
        "orders_released": 1,
        "orders_completed": 1,
        "order_lines_completed": 1,
        "loaded_robot_distance": 2.0,
        "empty_robot_distance": 1.0,
        "movement_energy_kj": 0.2,
        "fixed_load_energy_kj": 0.1,
        "simulation_termination_reason": "completed",
        "simulation_completed_full_horizon": True,
    }
    return {
        "status": "success", "finalization": {"finalized": True, "reason": "completed"},
        "kpi_schema_version": "full_kpi_v3", "kpi_complete": True, "kpi": kpi,
        "effective_charger_coordinate_hash": "effective-coordinates",
    }


def test_host_lifecycle_persists_rows_before_workspace_deletion_and_exports_kpi_only(tmp_path, monkeypatch):
    machines = [campaign.default_machines()[0]]
    plan = campaign.build_campaign_plan(
        campaign_id=campaign.generate_campaign_id(machines, fake_assets(), 1.35),
        machines=machines,
        assets=fake_assets(),
    )
    off = next(run for run in plan["runs"] if run["policy_configuration"] == "all_off")
    on = next(run for run in plan["runs"] if run["policy_configuration"] == "all_on_rl")
    failed = dict(off)
    failed["condition_key"] = off["condition_key"] + "|synthetic-failure"
    failed["run_id"] = off["run_id"] + "-failed"
    runs = [{**off, "machine_id": machines[0].machine_id}, {**on, "machine_id": machines[0].machine_id}, {**failed, "machine_id": machines[0].machine_id}]
    plan = {**plan, "machines": [machines[0].__dict__], "runs": runs}
    manifest_path = tmp_path / "manifest.json"
    campaign.write_json(manifest_path, plan)

    def fake_run_specs(specs, **_kwargs):
        results = []
        for spec in specs:
            spec.runtime_root.mkdir(parents=True, exist_ok=True)
            if spec.run_id == failed["run_id"]:
                summary, code = {"status": "failure", "error_type": "synthetic", "error_message": "expected failure"}, 1
            else:
                summary, code = _success_summary(spec), 0
            (spec.runtime_root / "worker_summary.json").write_text(json.dumps(summary), encoding="utf-8")
            results.append({"spec": spec, "return_code": code, "worker_summary": summary, "kpi_payload": summary.get("kpi")})
        return results

    monkeypatch.setattr(campaign, "run_specs", fake_run_specs)
    monkeypatch.setattr(campaign, "validate_local_assets", lambda *_args, **_kwargs: None)
    host_root = tmp_path / "host"
    assert campaign.execute_host(
        manifest_path=manifest_path, machine_id=machines[0].machine_id, stages=[1, 2, 3, 4],
        resume=True, progress=False, max_retries=0, host_data_root=host_root,
    ) == 1

    ledger = json.loads((host_root / "host_ledger.json").read_text(encoding="utf-8"))
    states = ledger["condition_states"]
    assert states[off["condition_key"]]["status"] == "completed_strict"
    assert states[on["condition_key"]]["status"] == "completed_strict"
    assert states[failed["condition_key"]]["status"] == "failed_final"
    assert not (host_root / "runs" / off["run_id"]).exists()
    assert not (host_root / "runs" / on["run_id"]).exists()
    assert not (host_root / "runs" / failed["run_id"]).exists()
    assert (host_root / "run_outcomes.csv").exists()
    assert (host_root / "failed_conditions.csv").exists()

    exporter = _load("host_export")
    archive = exporter.export_host(manifest_path, machines[0].machine_id, host_root, tmp_path / "exports")
    with zipfile.ZipFile(archive) as zf:
        assert not any(name.startswith("runs/") for name in zf.namelist())
        assert {"run_outcomes.csv", "failed_conditions.csv", "host_assignment.json", "host_ledger.json"}.issubset(zf.namelist())
    importer = _load("import_host_exports")
    report = importer.import_exports([archive], tmp_path / "imported")
    assert report["accepted"] == 2
    assert any(item.get("terminal_failure") for item in report["failures"])
