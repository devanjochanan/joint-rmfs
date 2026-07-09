import csv
from collections import Counter
from pathlib import Path

from scripts.experiments.distributed_sensitivity_campaign import (
    ALLOCATION_PATCH_LABEL,
    CANONICAL_RTS_CHECKPOINT_ID,
    DEFAULT_CHARGING_CONFIG_PATH,
    SIMULATION_SEMANTICS_ID,
    AssetBundle,
    Machine,
    build_campaign_plan,
    build_run_spec_from_condition,
    default_machines,
    execute_machine,
    generate_allocation_patch_id,
    generate_campaign_id,
    relative_to_repo,
    rebuild_run_outcomes_csv,
    run_complete_for_campaign,
    seed_for_replication,
    write_json,
)


def fake_assets() -> AssetBundle:
    return AssetBundle(
        pps_model_relative_path="data/models/pps/pps_rl_best.zip",
        pps_model_sha256="p" * 64,
        pps_observation_schema={"pod_features": {"shape": [60, 511]}},
        rts_checkpoint_relative_dir="data/models/rts/batch_000014/checkpoint",
        rts_checkpoint_id=CANONICAL_RTS_CHECKPOINT_ID,
        rts_model_sha256="a" * 64,
        rts_metadata_sha256="b" * 64,
        rts_feature_schema_sha256="c" * 64,
        rts_feature_schema_id="rts_feature_schema_fake",
        rts_training_artifact="batch_000014",
        rts_training_latest_relative_path="",
        rts_lineage={
            "initialization_method": "vrsla_behavior_cloning",
            "teacher_policy": "vrsla_event_driven",
        },
        rts_lineage_source_relative_dir="data/models/rts/batch_000014/checkpoint",
        charging_config_relative_path=relative_to_repo(DEFAULT_CHARGING_CONFIG_PATH),
        charging_config_sha256="d" * 64,
    )


def build_manifest(machines=None):
    machines = machines or default_machines()
    assets = fake_assets()
    campaign_id = generate_campaign_id(machines, assets, 1.35)
    return build_campaign_plan(campaign_id=campaign_id, machines=machines, assets=assets)


def test_campaign_plan_counts_allocations_and_seed_design():
    manifest = build_manifest()

    assert manifest["simulation_semantics_id"] == SIMULATION_SEMANTICS_ID
    assert manifest["allocation_patch_id"].startswith(f"{ALLOCATION_PATCH_LABEL}_")
    assert manifest["assertions"]["machine_count"] == 7
    assert manifest["assertions"]["stage_new_runs"] == {"1": 30, "2": 19, "3": 19, "4": 532}
    assert manifest["assertions"]["total_unique_fresh_runs"] == 600
    assert len(manifest["runs"]) == 600
    assert manifest["old_capacity_study_completions_used"] == 0
    assert manifest["assertions"]["old_capacity_study_roots_contribute_completions"] == 0
    assert manifest["assertions"]["macbook_stage4_count"] == 0

    for replication, expected_seed in ((1, 42), (20, 61)):
        assert seed_for_replication(replication) == expected_seed
        assert {run["seed"] for run in manifest["runs"] if run["replication"] == replication} == {expected_seed}


def test_campaign_plan_has_no_duplicate_identities_or_shard_overlap():
    manifest = build_manifest()

    run_ids = [run["run_id"] for run in manifest["runs"]]
    condition_keys = [run["condition_key"] for run in manifest["runs"]]
    assert len(run_ids) == len(set(run_ids)) == 600
    assert len(condition_keys) == len(set(condition_keys)) == 600
    assert all(count == 1 for count in Counter(run_ids).values())
    assert all(count == 1 for count in Counter(condition_keys).values())
    assert sum(manifest["assertions"]["total_fresh_runs_by_machine"].values()) == 600

    all_on_rl = [run for run in manifest["runs"] if run["policy_configuration"] == "all_on_rl"]
    assert {run["identity"]["rts_checkpoint_sha256"] for run in all_on_rl} == {"a" * 64}
    assert {run["identity"]["pps_model_sha256"] for run in all_on_rl} == {"p" * 64}


def test_campaign_id_ignores_allocation_and_local_machine_details():
    assets = fake_assets()
    machines = default_machines()
    campaign_id = generate_campaign_id(machines, assets, 1.35)
    manifest = build_campaign_plan(campaign_id=campaign_id, machines=machines, assets=assets)

    changed = [
        Machine(
            machine_id=m.machine_id,
            os="windows" if m.os != "windows" else "linux",
            repository=f"/different/{m.machine_id}",
            python=f"/python/{m.machine_id}",
            max_workers=(m.max_workers + 1 if m.machine_id == "alisha_pc" else m.max_workers),
            effective_steps_per_second=(m.effective_steps_per_second + 17.0 if m.machine_id == "alisha_pc" else m.effective_steps_per_second),
            eligible_stages=m.eligible_stages,
            anydesk_id=m.anydesk_id,
        )
        for m in machines
    ]
    changed_campaign_id = generate_campaign_id(changed, assets, 9.99)
    changed_manifest = build_campaign_plan(campaign_id=changed_campaign_id, machines=changed, assets=assets)

    assert changed_campaign_id == campaign_id
    assert changed_manifest["allocation_patch_id"] != manifest["allocation_patch_id"]
    assert {
        run["condition_key"]: run["run_id"]
        for run in changed_manifest["runs"]
    } == {
        run["condition_key"]: run["run_id"]
        for run in manifest["runs"]
    }


def test_allocation_patch_id_changes_without_scientific_identity_change():
    assets = fake_assets()
    machines = default_machines()
    modified = [
        Machine(
            machine_id=m.machine_id,
            os=m.os,
            repository=m.repository,
            python=m.python,
            max_workers=(m.max_workers + 1 if m.machine_id == "alisha_pc" else m.max_workers),
            effective_steps_per_second=m.effective_steps_per_second,
            eligible_stages=m.eligible_stages,
            anydesk_id=m.anydesk_id,
        )
        for m in machines
    ]
    assert generate_campaign_id(machines, assets, 1.35) == generate_campaign_id(modified, assets, 1.35)
    assert generate_allocation_patch_id(machines, 1.35) != generate_allocation_patch_id(modified, 1.35)


def test_sensitivity_run_specs_are_no_state_and_all_off_checkpoint_not_applicable():
    manifest = build_manifest()
    machine_by_id = {m.machine_id: m for m in default_machines()}
    all_off = next(run for run in manifest["runs"] if run["policy_configuration"] == "all_off")
    all_on = next(run for run in manifest["runs"] if run["policy_configuration"] == "all_on_rl")

    all_off_spec = build_run_spec_from_condition(all_off, manifest=manifest, machine=machine_by_id[all_off["machine_id"]], repo_root=Path.cwd())
    all_on_spec = build_run_spec_from_condition(all_on, manifest=manifest, machine=machine_by_id[all_on["machine_id"]], repo_root=Path.cwd())

    assert all_off_spec.persist_final_state is False
    assert all_on_spec.persist_final_state is False
    assert all_off_spec.rts_policy_checkpoint_id == "not_applicable"
    assert all_on_spec.rts_policy_checkpoint_id == CANONICAL_RTS_CHECKPOINT_ID


def test_continuous_execution_combines_stages_before_calling_run_specs(tmp_path, monkeypatch):
    manifest = build_manifest()
    machine = default_machines()[0]
    selected = [
        {**run, "machine_id": machine.machine_id}
        for run in manifest["runs"]
        if run["stage_first_requested"] == 1
    ][:6]
    selected.extend(
        {**run, "machine_id": machine.machine_id}
        for run in manifest["runs"]
        if run["stage_first_requested"] == 2
    )
    manifest = {
        **manifest,
        "campaign_root_relative": tmp_path.relative_to(Path.cwd()).as_posix() if tmp_path.is_relative_to(Path.cwd()) else "data/runtime/distributed_sensitivity/test_combined_queue",
        "machines": [machine.__dict__],
        "runs": selected,
        "shards": {machine.machine_id: "shards/test_shard.json"},
    }
    root = Path.cwd() / manifest["campaign_root_relative"]
    shard_path = root / "shards" / "test_shard.json"
    write_json(root / "manifest.json", manifest)
    write_json(shard_path, {"campaign_id": manifest["campaign_id"], "machine_id": machine.machine_id, "runs": selected})

    launched = {}

    def fake_run_specs(specs, **kwargs):
        launched["stage_order"] = [spec.stage_first_requested for spec in specs]
        return []

    monkeypatch.setattr("scripts.experiments.distributed_sensitivity_campaign.validate_local_assets", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("scripts.experiments.distributed_sensitivity_campaign.run_specs", fake_run_specs)
    monkeypatch.setattr("scripts.experiments.distributed_sensitivity_campaign.rebuild_run_outcomes_csv", lambda **_kwargs: root / machine.machine_id / "run_outcomes.csv")

    execute_machine(
        manifest_path=root / "manifest.json",
        machine_id=machine.machine_id,
        stages=[1, 2],
        resume=False,
        progress=False,
    )

    assert launched["stage_order"][:8] == [1, 1, 1, 1, 1, 1, 2, 2]


def test_run_outcomes_csv_rebuilds_from_retained_worker_summary(tmp_path):
    manifest = {**build_manifest(), "campaign_root_relative": "campaign"}
    machine = default_machines()[0]
    condition = next(run for run in manifest["runs"] if run["machine_id"] == machine.machine_id)
    run_root = tmp_path / "campaign" / machine.machine_id / "runs" / condition["run_id"]
    write_json(
        run_root / "worker_summary.json",
        {
            "status": "success",
            "repo_commit": "commit123",
            "finalization": {"reason": "maximum_horizon"},
            "campaign_id": manifest["campaign_id"],
            "allocation_patch_id": manifest["allocation_patch_id"],
            "simulation_semantics_id": manifest["simulation_semantics_id"],
            "run_id": condition["run_id"],
            "kpi": {
                "kpi_complete": True,
                "rts_checkpoint_id": condition["identity"].get("rts_checkpoint_id", ""),
                "rts_checkpoint_sha256": condition["identity"].get("rts_checkpoint_sha256", ""),
                "pps_model_sha256": condition["identity"].get("pps_model_sha256", ""),
            },
        },
    )

    path = rebuild_run_outcomes_csv(manifest=manifest, machine=machine, repo_root=tmp_path)

    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 1
    row = rows[0]
    assert row["campaign_id"] == manifest["campaign_id"]
    assert row["allocation_patch_id"] == manifest["allocation_patch_id"]
    assert row["simulation_semantics_id"] == SIMULATION_SEMANTICS_ID
    assert row["run_id"] == condition["run_id"]
    assert row["condition_key"] == condition["condition_key"]
    assert row["source_machine_id"] == machine.machine_id
    assert row["status"] == "success"
    assert row["termination_reason"] == "maximum_horizon"
    assert row["repo_commit"] == "commit123"


def test_resume_identity_accepts_cleaned_success_summary(tmp_path):
    manifest = {**build_manifest(), "campaign_root_relative": "campaign"}
    machine = default_machines()[0]
    condition = next(run for run in manifest["runs"] if run["machine_id"] == machine.machine_id)
    spec = build_run_spec_from_condition(condition, manifest=manifest, machine=machine, repo_root=tmp_path)
    write_json(spec.runtime_root / "run_spec.json", spec.to_json_dict())
    write_json(
        spec.runtime_root / "worker_summary.json",
        {
            "run_id": condition["run_id"],
            "campaign_id": manifest["campaign_id"],
            "simulation_semantics_id": SIMULATION_SEMANTICS_ID,
            "kpi_schema_version": manifest["kpi_schema_version"],
            "policy_configuration": condition["policy_configuration"],
            "replication": condition["replication"],
            "seed": condition["seed"],
            "requested_robot_count": condition["robot_count"],
            "rts_checkpoint_sha256": condition["identity"]["rts_checkpoint_sha256"],
            "pps_model_sha256": condition["identity"]["pps_model_sha256"],
            "status": "success",
            "finalization": {"finalized": True},
            "kpi_complete": True,
        },
    )

    assert run_complete_for_campaign(condition, spec, manifest) is True
