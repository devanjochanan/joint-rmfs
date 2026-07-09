from pathlib import Path

from scripts.experiments import append_distributed_sensitivity_campaign as append
from scripts.experiments import distributed_sensitivity_campaign as base
from tests.experiments.test_distributed_sensitivity_campaign import fake_assets


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
    assert report["central_allocation_counts"]["citi_gojira"] == {"all_off": 20, "all_on_rl": 2}
    assert report["central_allocation_counts"]["dewa_macbook"]["all_on_rl"] == 6

    mac_queue = sorted([item for item in items if item["assigned_machine_id"] == "dewa_macbook"], key=append.item_sort_key)
    assert [item["execution_priority"] for item in mac_queue[:8]] == [0, 0, 0, 0, 0, 1, 1, 1]
