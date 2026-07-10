import csv
import json
import zipfile
from pathlib import Path

import pytest

from scripts.experiments import append_distributed_sensitivity_campaign as append
from scripts.experiments import distributed_sensitivity_campaign as base
from tests.experiments.test_distributed_sensitivity_campaign import fake_assets

# The replications-21-to-40 append/extension workflow is SUPERSEDED by the
# finalized 720-run design, which builds 40 replications directly into stage 4.
# The append runner + its shard/resume-patch exchange are deprecated (Section 10);
# these tests are retained for historical reference only.
pytestmark = pytest.mark.skip(
    reason="append/extension (rep 21-40) workflow superseded by the 720-run 40-replication design"
)


def _parent_manifest(tmp_path: Path):
    assets = fake_assets()
    machines = base.default_machines()
    manifest = base.build_campaign_plan(
        campaign_id=base.generate_campaign_id(machines, assets, base.DEFAULT_RL_OVERHEAD_MULTIPLIER),
        machines=machines,
        assets=assets,
    )
    manifest = {
        **manifest,
        "campaign_id": append.ORIGINAL_CAMPAIGN_ID,
        "allocation_patch_id": append.ORIGINAL_ALLOCATION_PATCH_ID,
        "campaign_root_relative": f"data/runtime/distributed_sensitivity/{append.ORIGINAL_CAMPAIGN_ID}",
    }
    parent_path = tmp_path / "manifest.json"
    base.write_json(parent_path, manifest)
    return manifest, parent_path


def test_extension_manifest_has_replications_21_to_40_and_1200_tuple_design(tmp_path, monkeypatch):
    parent, parent_path = _parent_manifest(tmp_path)
    monkeypatch.setattr(append, "original_manifest_path", lambda repo_root=append.REPO_ROOT: parent_path)
    machines = append.append_machines()
    allocation_patch_id = append.append_allocation_patch_id(machines)

    extension = append.build_extension_manifest(parent, machines, allocation_patch_id)

    assert extension["campaign_id"].startswith("sensitivity_full_kpi_rep21_40_")
    assert extension["parent_campaign_id"] == append.ORIGINAL_CAMPAIGN_ID
    assert extension["extension_replication_first"] == 21
    assert extension["extension_replication_last"] == 40
    assert extension["seed_first"] == 62
    assert extension["seed_last"] == 81
    assert extension["assertions"]["condition_count"] == 600
    assert extension["assertions"]["unique_run_ids"] == 600
    assert extension["assertions"]["central_extension_count"] == 40
    assert extension["assertions"]["noncentral_extension_count"] == 560
    assert {run["replication"] for run in extension["runs"]} == set(range(21, 41))
    assert {run["seed"] for run in extension["runs"]} == set(range(62, 82))

    combined_tuples = {
        (run["policy_configuration"], run["robot_count"], run["order_rate"], run["replication"])
        for run in parent["runs"] + extension["runs"]
    }
    assert len(combined_tuples) == 1200


def test_append_scheduler_skips_strict_original_and_assigns_each_remaining_once(tmp_path, monkeypatch):
    parent, parent_path = _parent_manifest(tmp_path)
    monkeypatch.setattr(append, "original_manifest_path", lambda repo_root=append.REPO_ROOT: parent_path)
    machines = append.append_machines()
    allocation_patch_id = append.append_allocation_patch_id(machines)
    extension = append.build_extension_manifest(parent, machines, allocation_patch_id)
    strict_run = next(run for run in parent["runs"] if run["stage_first_requested"] == 1)
    inventory = {
        "strict_records": {strict_run["run_id"]: {"condition": strict_run, "copy": {}}},
        "duration_records": [],
        "counts": {"strict_completed": 1},
    }
    duration_model = append.DurationModel([], {machine.machine_id: machine for machine in machines})

    items, report = append.schedule_items(parent, extension, inventory, machines, duration_model)
    append.validate_design(parent, extension, inventory, items)

    assert strict_run["run_id"] not in {item["run_id"] for item in items}
    assert len([item for item in items if item["source_campaign_id"] == append.ORIGINAL_CAMPAIGN_ID]) == 599
    assert len([item for item in items if item["source_campaign_id"] == extension["campaign_id"]]) == 600
    assert len({(item["source_campaign_id"], item["run_id"]) for item in items}) == len(items)
    assert "dewa_macbook" not in {item["assigned_machine_id"] for item in items}
    assert "dewa_macbook" not in report["central_allocation_counts"]



def _csv_text(rows, fieldnames):
    from io import StringIO

    buf = StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue()


def test_merged_zip_inventory_requeues_unfinished_and_excludes_macbook(tmp_path, monkeypatch):
    parent, parent_path = _parent_manifest(tmp_path)
    monkeypatch.setattr(append, "original_manifest_path", lambda repo_root=append.REPO_ROOT: parent_path)
    machines = append.append_machines()
    allocation_patch_id = append.append_allocation_patch_id(machines)
    extension = append.build_extension_manifest(parent, machines, allocation_patch_id)
    original_strict = parent["runs"][0]
    original_failed = parent["runs"][1]
    extension_mac_strict = extension["runs"][0]
    extension_missing = extension["runs"][1]
    fieldnames = [
        "campaign_id", "run_id", "condition_key", "policy_configuration", "robot_count", "order_rate",
        "replication", "seed", "stage_first_requested", "merge_status", "selected_snapshot",
        "selected_machine_id", "observed_copy_count", "strict_copy_count", "summary_status",
        "netlogo_steps_completed", "netlogo_steps_requested", "error_type", "error_message",
    ]
    rows = []
    for manifest in (parent, extension):
        for run in manifest["runs"]:
            status = "strict_completed"
            machine = "codex_local"
            if run["run_id"] == original_failed["run_id"]:
                status = "failed"
            if run["run_id"] == extension_missing["run_id"]:
                status = "missing"
                machine = ""
            if run["run_id"] == extension_mac_strict["run_id"]:
                machine = "dewa_macbook"
            rows.append({
                "campaign_id": manifest["campaign_id"],
                "run_id": run["run_id"],
                "condition_key": run["condition_key"],
                "policy_configuration": run["policy_configuration"],
                "robot_count": run["robot_count"],
                "order_rate": run["order_rate"],
                "replication": run["replication"],
                "seed": run["seed"],
                "stage_first_requested": run["stage_first_requested"],
                "merge_status": status,
                "selected_snapshot": "synthetic.zip" if status == "strict_completed" else "",
                "selected_machine_id": machine,
                "observed_copy_count": "1",
                "strict_copy_count": "1" if status == "strict_completed" else "0",
                "summary_status": "success" if status == "strict_completed" else status,
                "netlogo_steps_completed": "580000" if status == "strict_completed" else "0",
                "netlogo_steps_requested": "580000",
                "error_type": "RuntimeError" if status == "failed" else "",
                "error_message": "boom" if status == "failed" else "",
            })
    zip_path = tmp_path / "merged.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("manifests/canonical_original_manifest.json", json.dumps(parent))
        zf.writestr("manifests/extension_manifest.json", json.dumps(extension))
        zf.writestr("datasets/condition_status_1200.csv", _csv_text(rows, fieldnames))
        zf.writestr("datasets/strict_completion_provenance.csv", _csv_text([
            {"campaign_id": extension["campaign_id"], "run_id": extension_mac_strict["run_id"], "machine_id": "dewa_macbook"}
        ], ["campaign_id", "run_id", "machine_id"]))
        zf.writestr(
            f"merged_run_artifacts/{parent['campaign_id']}/codex_local/runs/{original_strict['run_id']}/worker_summary.json",
            json.dumps({"worker_wall_time_elapsed": 100.0}),
        )

    inventory = append.completion_inventory_from_merged_zip(parent, extension, zip_path)
    assert original_strict["run_id"] in inventory["strict_records_by_campaign"][parent["campaign_id"]]
    assert extension_mac_strict["run_id"] in inventory["strict_records_by_campaign"][extension["campaign_id"]]

    duration_model = append.DurationModel(inventory["duration_records"], {machine.machine_id: machine for machine in machines})
    items, _report = append.schedule_items(parent, extension, inventory, machines, duration_model)
    append.validate_design(parent, extension, inventory, items)

    planned = {(item["source_campaign_id"], item["run_id"]) for item in items}
    assert (parent["campaign_id"], original_strict["run_id"]) not in planned
    assert (extension["campaign_id"], extension_mac_strict["run_id"]) not in planned
    assert (parent["campaign_id"], original_failed["run_id"]) in planned
    assert (extension["campaign_id"], extension_missing["run_id"]) in planned
    assert all(item["assigned_machine_id"] != "dewa_macbook" for item in items)
