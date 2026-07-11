import csv
import json
from collections import Counter
from pathlib import Path

import pytest

from scripts.experiments.distributed_sensitivity_campaign import (
    ALLOCATION_PATCH_LABEL,
    POLICY_CONFIGURATIONS,
    CANONICAL_RTS_CHECKPOINT_ID,
    DEFAULT_CHARGING_CONFIG_PATH,
    SIMULATION_SEMANTICS_ID,
    AssetBundle,
    Machine,
    build_input_identity,
    build_campaign_plan,
    build_run_spec_from_condition,
    build_scientific_identity,
    condition_sort_key,
    default_machines,
    ensure_clean_tracked_scientific_files,
    execute_machine,
    generate_allocation_patch_id,
    generate_campaign_id,
    generate_campaign_id_from_identity,
    main,
    relative_to_repo,
    rebuild_run_outcomes_csv,
    run_complete_for_campaign,
    seed_for_replication,
    treatment_execution_contracts,
    validate_source_identity,
    write_campaign_files,
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
    assert manifest["assertions"]["stage_new_runs"] == {"1": 18, "2": 39, "3": 39, "4": 624}
    assert manifest["assertions"]["total_unique_fresh_runs"] == 720
    assert len(manifest["runs"]) == 720
    assert manifest["assertions"]["old_capacity_study_roots_contribute_completions"] == 0

    for replication, expected_seed in ((1, 42), (40, 81)):
        assert seed_for_replication(replication) == expected_seed
        assert {run["seed"] for run in manifest["runs"] if run["replication"] == replication} == {expected_seed}


def test_campaign_plan_has_no_duplicate_identities_or_shard_overlap():
    manifest = build_manifest()

    run_ids = [run["run_id"] for run in manifest["runs"]]
    condition_keys = [run["condition_key"] for run in manifest["runs"]]
    assert len(run_ids) == len(set(run_ids)) == 720
    assert len(condition_keys) == len(set(condition_keys)) == 720
    assert all(count == 1 for count in Counter(run_ids).values())
    assert all(count == 1 for count in Counter(condition_keys).values())
    assert sum(manifest["assertions"]["total_fresh_runs_by_machine"].values()) == 720

    assert {run["policy_configuration"] for run in manifest["runs"]} == set(POLICY_CONFIGURATIONS)


def test_new_campaign_files_write_host_assignments_not_shards(tmp_path, monkeypatch):
    manifest = {**build_manifest(), "campaign_root_relative": "unused"}
    root = tmp_path / "campaign"
    monkeypatch.setattr(
        "scripts.experiments.distributed_sensitivity_campaign.campaign_root",
        lambda _repo_root, _manifest: root,
    )
    write_campaign_files(manifest)
    written = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert "shards" not in written
    assert set(written["host_assignments"]) == {row["machine_id"] for row in manifest["machines"]}
    assert not (root / "shards").exists()
    for launcher in (root / "launchers").iterdir():
        content = launcher.read_text(encoding="utf-8")
        assert "--launch-host --progress" in content
        assert "--resume" not in content
        assert "--stage" not in content


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
            allowed_treatments=m.allowed_treatments,
            reference_effective_steps_per_second=m.reference_effective_steps_per_second,
            rl_effective_steps_per_second=m.rl_effective_steps_per_second,
            max_rl_workers=m.max_rl_workers,
            rl_rate_source=m.rl_rate_source,
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
            allowed_treatments=m.allowed_treatments,
            reference_effective_steps_per_second=m.reference_effective_steps_per_second,
            rl_effective_steps_per_second=m.rl_effective_steps_per_second,
            max_rl_workers=m.max_rl_workers,
            rl_rate_source=m.rl_rate_source,
        )
        for m in machines
    ]
    assert generate_campaign_id(machines, assets, 1.35) == generate_campaign_id(modified, assets, 1.35)
    assert generate_allocation_patch_id(machines, 1.35) != generate_allocation_patch_id(modified, 1.35)


def test_primary_run_specs_are_no_state_and_use_explicit_rts_contracts():
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
    selected = []
    for stage, count in ((1, 6), (2, 2), (3, 2), (4, 2)):
        selected.extend([
            {**run, "machine_id": machine.machine_id}
            for run in manifest["runs"]
            if run["stage_first_requested"] == stage
        ][:count])
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

    launched = []

    def fake_run_specs(specs, **kwargs):
        launched.append([spec.stage_first_requested for spec in specs])
        completed = []
        for spec in specs:
            write_json(spec.runtime_root / "run_spec.json", spec.to_json_dict())
            write_json(
                spec.runtime_root / "worker_summary.json",
                {
                    "run_id": spec.run_id,
                    "campaign_id": spec.campaign_id,
                    "simulation_semantics_id": spec.simulation_semantics_id,
                    "kpi_schema_version": spec.kpi_schema_version,
                    "policy_configuration": spec.policy_configuration,
                    "replication": spec.replication,
                    "seed": spec.campaign_seed,
                    "requested_robot_count": spec.robot_count,
                    "rts_checkpoint_sha256": spec.rts_checkpoint_sha256,
                    "pps_model_sha256": spec.pps_model_sha256,
                    "status": "success",
                    "finalization": {"finalized": True},
                    "kpi_complete": True,
                },
            )
            completed.append({"spec": spec, "return_code": 0, "peak_runtime_directory_bytes": 0})
        return completed

    monkeypatch.setattr("scripts.experiments.distributed_sensitivity_campaign.ensure_clean_tracked_scientific_files", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("scripts.experiments.distributed_sensitivity_campaign.validate_local_assets", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("scripts.experiments.distributed_sensitivity_campaign.run_specs", fake_run_specs)
    monkeypatch.setattr("scripts.experiments.distributed_sensitivity_campaign.rebuild_run_outcomes_csv", lambda **_kwargs: root / machine.machine_id / "run_outcomes.csv")
    monkeypatch.setattr("scripts.experiments.distributed_sensitivity_campaign.run_complete_for_campaign", lambda *_args, **_kwargs: True)

    assert execute_machine(
        manifest_path=root / "manifest.json",
        machine_id=machine.machine_id,
        stages=[1, 2, 3, 4],
        resume=False,
        progress=False,
    ) == 0
    assert launched == [[1, 1, 1, 1, 1, 1, 2, 2, 3, 3, 4, 4]]


def test_gojira_first_wave_backfills_idle_critical_slots_with_stage4():
    manifest = build_manifest()
    gojira = next(machine for machine in default_machines() if machine.machine_id == "citi_gojira")
    queue = sorted(
        [run for run in manifest["runs"] if run["machine_id"] == gojira.machine_id],
        key=condition_sort_key,
    )
    first_wave = queue[:gojira.max_workers]
    assert sum(run["stage_first_requested"] in {1, 2, 3} for run in queue) == 20
    assert len(first_wave) == 24
    assert [run["stage_first_requested"] for run in first_wave[:20]] == sorted(
        run["stage_first_requested"] for run in first_wave[:20]
    )
    assert all(run["stage_first_requested"] == 4 for run in first_wave[20:])


def test_execute_machine_returns_nonzero_for_invalid_selected_run(tmp_path, monkeypatch):
    root_rel = f"data/runtime/distributed_sensitivity/test_invalid_{tmp_path.name}"
    manifest = {**build_manifest(), "campaign_root_relative": root_rel}
    machine = default_machines()[0]
    condition = next(run for run in manifest["runs"] if run["machine_id"] == machine.machine_id)
    manifest = {
        **manifest,
        "machines": [machine.__dict__],
        "runs": [condition],
        "shards": {machine.machine_id: "shards/test_shard.json"},
    }
    root = Path.cwd() / root_rel
    write_json(root / "manifest.json", manifest)
    write_json(root / "shards" / "test_shard.json", {"runs": [condition]})

    def fake_run_specs(specs, **_kwargs):
        spec = specs[0]
        write_json(spec.runtime_root / "run_spec.json", spec.to_json_dict())
        write_json(
            spec.runtime_root / "worker_summary.json",
            {
                "run_id": spec.run_id,
                "status": "failure",
                "error_type": "RuntimeError",
                "error_message": "boom",
            },
        )
        return [{"spec": spec, "return_code": 1, "peak_runtime_directory_bytes": 0}]

    monkeypatch.setattr("scripts.experiments.distributed_sensitivity_campaign.ensure_clean_tracked_scientific_files", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("scripts.experiments.distributed_sensitivity_campaign.validate_local_assets", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("scripts.experiments.distributed_sensitivity_campaign.run_specs", fake_run_specs)

    assert execute_machine(
        manifest_path=root / "manifest.json",
        machine_id=machine.machine_id,
        stages=[condition["stage_first_requested"]],
        resume=False,
        progress=False,
    ) == 1


def test_execute_machine_returns_zero_only_after_strict_completion(tmp_path, monkeypatch):
    root_rel = f"data/runtime/distributed_sensitivity/test_success_{tmp_path.name}"
    manifest = {**build_manifest(), "campaign_root_relative": root_rel}
    machine = default_machines()[0]
    condition = next(run for run in manifest["runs"] if run["machine_id"] == machine.machine_id)
    manifest = {
        **manifest,
        "machines": [machine.__dict__],
        "runs": [condition],
        "shards": {machine.machine_id: "shards/test_shard.json"},
    }
    root = Path.cwd() / root_rel
    write_json(root / "manifest.json", manifest)
    write_json(root / "shards" / "test_shard.json", {"runs": [condition]})

    def fake_run_specs(specs, **_kwargs):
        spec = specs[0]
        write_json(spec.runtime_root / "run_spec.json", spec.to_json_dict())
        write_json(
            spec.runtime_root / "worker_summary.json",
            {
                **spec.to_json_dict(),
                "run_id": spec.run_id,
                "campaign_id": spec.campaign_id,
                "simulation_semantics_id": spec.simulation_semantics_id,
                "kpi_schema_version": spec.kpi_schema_version,
                "policy_configuration": spec.policy_configuration,
                "replication": spec.replication,
                "seed": spec.campaign_seed,
                "requested_robot_count": spec.robot_count,
                "rts_checkpoint_sha256": spec.rts_checkpoint_sha256,
                "pps_model_sha256": spec.pps_model_sha256,
                "status": "success",
                "finalization": {"finalized": True},
                "kpi_complete": True,
            },
        )
        return [{"spec": spec, "return_code": 1, "peak_runtime_directory_bytes": 0}]

    monkeypatch.setattr("scripts.experiments.distributed_sensitivity_campaign.ensure_clean_tracked_scientific_files", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("scripts.experiments.distributed_sensitivity_campaign.validate_local_assets", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("scripts.experiments.distributed_sensitivity_campaign.run_specs", fake_run_specs)

    assert execute_machine(
        manifest_path=root / "manifest.json",
        machine_id=machine.machine_id,
        stages=[condition["stage_first_requested"]],
        resume=False,
        progress=False,
    ) == 0


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
            "source_tree_hash": condition["identity"]["source_tree_hash"],
            "status": "success",
            "finalization": {"finalized": True},
            "kpi_complete": True,
        },
    )

    assert run_complete_for_campaign(condition, spec, manifest) is True


def test_clean_scientific_inputs_warn_for_dirty_tracked_input(monkeypatch, capsys):
    monkeypatch.setattr(
        "scripts.experiments.distributed_sensitivity_campaign.dirty_tracked_files",
        lambda _repo_root: ["data/input/base/raw_order.csv"],
    )

    ensure_clean_tracked_scientific_files(Path.cwd())
    assert "data/input/base/raw_order.csv" in capsys.readouterr().out


def test_clean_scientific_inputs_ignore_runtime_outputs(monkeypatch):
    monkeypatch.setattr(
        "scripts.experiments.distributed_sensitivity_campaign.dirty_tracked_files",
        lambda _repo_root: ["data/runtime/distributed_sensitivity/manifest.json"],
    )

    ensure_clean_tracked_scientific_files(Path.cwd())


def test_portable_snapshot_accepts_matching_source_hash(tmp_path):
    (tmp_path / "worker.py").write_text("VALUE = 1\n", encoding="utf-8")
    from src.rmfs.orchestration.source_identity import SourceIdentity
    identity = SourceIdentity.compute(tmp_path).source_tree_hash
    assert validate_source_identity({"source_tree_hash": identity}, tmp_path).source_tree_hash == identity


def test_portable_snapshot_warns_for_source_hash_mismatch(tmp_path, capsys):
    (tmp_path / "worker.py").write_text("VALUE = 1\n", encoding="utf-8")
    assert validate_source_identity({"source_tree_hash": "0" * 64}, tmp_path).source_tree_hash
    assert "source-tree hash differs" in capsys.readouterr().out


def test_launch_host_materializes_and_executes_without_a_manifest(monkeypatch, tmp_path):
    manifest_path = tmp_path / "manifest.json"
    write_json(manifest_path, {"campaign_id": "local", "runs": []})
    called = {}
    monkeypatch.setattr(
        "scripts.experiments.distributed_sensitivity_campaign.materialize_local_campaign_plan",
        lambda _args: (manifest_path, {"campaign_id": "local"}),
    )
    monkeypatch.setattr(
        "scripts.experiments.distributed_sensitivity_campaign.execute_machine",
        lambda **kwargs: called.update(kwargs) or 0,
    )
    assert main(["--launch-host", "--machine-id", "win_lukman", "--progress"]) == 0
    assert called["manifest_path"] == manifest_path
    assert called["stages"] == [1, 2, 3, 4]
    assert called["resume"] is False


def test_input_identity_uses_git_blob_for_layout_not_raw_sha(monkeypatch):
    input_meta = {
        "file_digests": {
            "generated_pod.csv": {"sha256": "raw-layout"},
            "items.csv": {"sha256": "raw-items"},
            "pods.csv": {"sha256": "raw-pods"},
            "raw_order.csv": {"sha256": "raw-orders"},
        },
        "layout": {"layout_sha256": "raw-layout"},
    }
    identities = {
        "data/input/base/generated_pod.csv": "blob-layout",
        "data/input/base/items.csv": "blob-items",
        "data/input/base/pods.csv": "blob-pods",
        "data/input/base/raw_order.csv": "blob-orders",
    }
    monkeypatch.setattr(
        "scripts.experiments.distributed_sensitivity_campaign.git_blob_identity",
        lambda rel_path: identities[rel_path],
    )

    identity = build_input_identity(input_meta)

    assert identity["layout_identity"] == "blob-layout"
    assert identity["tracked_file_identities"]["generated_pod.csv"] == "blob-layout"
    assert "raw-layout" not in identity.values()


def test_treatment_contract_changes_campaign_id_but_allocation_does_not():
    assets = fake_assets()
    manifest = build_manifest()
    identity = dict(manifest["scientific_identity"])
    changed = {
        **identity,
        "treatment_execution_contracts": {
            **identity["treatment_execution_contracts"],
            "all_on_rl": {
                **identity["treatment_execution_contracts"]["all_on_rl"],
                "rts_policy_action_mode": "sample",
            },
        },
    }

    assert generate_campaign_id_from_identity(changed) != manifest["campaign_id"]
    assert generate_campaign_id(default_machines(), assets, 99.0) == manifest["campaign_id"]
    assert treatment_execution_contracts(assets)["all_off"]["persist_final_state"] is False



def _write_complete_summary(spec, manifest, condition, **overrides):
    summary = {
        "run_id": condition["run_id"],
        "campaign_id": spec.campaign_id,
        "allocation_patch_id": spec.allocation_patch_id,
        "machine_id": spec.machine_id,
        "repo_commit": "different-commit",
        "simulation_semantics_id": spec.simulation_semantics_id,
        "kpi_schema_version": manifest["kpi_schema_version"],
        "policy_configuration": condition["policy_configuration"],
        "replication": condition["replication"],
        "seed": condition["seed"],
        "requested_robot_count": condition["robot_count"],
        "rts_checkpoint_sha256": "different-rts-hash",
        "pps_model_sha256": "different-pps-hash",
        "rts_checkpoint_id": "different-checkpoint-id",
        "source_tree_hash": condition["identity"]["source_tree_hash"],
        "status": "success",
        "finalization": {"finalized": True},
        "kpi_complete": True,
    }
    summary.update(overrides)
    write_json(spec.runtime_root / "worker_summary.json", summary)


def test_relaxed_completion_ignores_provenance_and_asset_metadata(tmp_path):
    manifest = {**build_manifest(), "campaign_root_relative": "campaign"}
    machine = default_machines()[0]
    condition = next(run for run in manifest["runs"] if run["machine_id"] == machine.machine_id)
    spec = build_run_spec_from_condition(condition, manifest=manifest, machine=machine, repo_root=tmp_path)
    spec_payload = spec.to_json_dict()
    spec_payload.update({
        "allocation_patch_id": "different-allocation",
        "machine_id": "different-machine",
        "repo_commit": "different-commit",
        "rts_checkpoint_id": "different-checkpoint-id",
        "rts_checkpoint_sha256": "different-rts-hash",
        "pps_model_sha256": "different-pps-hash",
    })
    write_json(spec.runtime_root / "run_spec.json", spec_payload)
    _write_complete_summary(spec, manifest, condition)

    assert run_complete_for_campaign(condition, spec, manifest) is False
    assert run_complete_for_campaign(condition, spec, manifest, relaxed=True) is True


def test_relaxed_completion_still_rejects_bad_condition_and_incomplete_kpi(tmp_path):
    manifest = {**build_manifest(), "campaign_root_relative": "campaign"}
    machine = default_machines()[0]
    condition = next(run for run in manifest["runs"] if run["machine_id"] == machine.machine_id)
    spec = build_run_spec_from_condition(condition, manifest=manifest, machine=machine, repo_root=tmp_path)
    write_json(spec.runtime_root / "run_spec.json", spec.to_json_dict())
    _write_complete_summary(spec, manifest, condition, seed=condition["seed"] + 1)
    assert run_complete_for_campaign(condition, spec, manifest, relaxed=True) is False

    _write_complete_summary(spec, manifest, condition, seed=condition["seed"], kpi_complete=False)
    assert run_complete_for_campaign(condition, spec, manifest, relaxed=True) is False

    _write_complete_summary(spec, manifest, condition, kpi_complete=True, finalization={"finalized": False})
    assert run_complete_for_campaign(condition, spec, manifest, relaxed=True) is False

    _write_complete_summary(spec, manifest, condition, finalization={"finalized": True}, status="failure")
    assert run_complete_for_campaign(condition, spec, manifest, relaxed=True) is False


def test_completion_inventory_works_without_worker_summary_json(tmp_path, monkeypatch):
    from scripts.experiments import append_distributed_sensitivity_campaign as append
    manifest = {**build_manifest(), "campaign_root_relative": "campaign"}
    machine = default_machines()[0]
    condition = next(run for run in manifest["runs"] if run["machine_id"] == machine.machine_id)

    outcomes_dir = tmp_path / "campaign" / machine.machine_id
    outcomes_dir.mkdir(parents=True, exist_ok=True)
    outcomes_path = outcomes_dir / "run_outcomes.csv"

    with outcomes_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(append.base.RUN_OUTCOME_FIELDS))
        writer.writeheader()
        row = {field: "" for field in append.base.RUN_OUTCOME_FIELDS}
        row.update({
            "campaign_id": manifest["campaign_id"],
            "allocation_patch_id": manifest["allocation_patch_id"],
            "simulation_semantics_id": manifest["simulation_semantics_id"],
            "run_id": condition["run_id"],
            "condition_key": condition["condition_key"],
            "policy_configuration": condition["policy_configuration"],
            "robot_count": str(condition["robot_count"]),
            "order_rate": str(condition["order_rate"]),
            "replication": str(condition["replication"]),
            "seed": str(condition["seed"]),
            "status": "success",
            "kpi_schema_version": manifest["kpi_schema_version"],
            "simulated_seconds": "580.0",
        })
        writer.writerow(row)

    inventory = append.completion_inventory(manifest, tmp_path, tmp_path / "snapshots")
    assert inventory["counts"]["strict_completed"] == 1
    assert condition["run_id"] in inventory["strict_records"]


def test_rebuild_plan_outcomes_preserves_rows(tmp_path):
    from scripts.experiments import append_distributed_sensitivity_campaign as append
    manifest = {**build_manifest(), "campaign_root_relative": "campaign"}
    machine = default_machines()[0]
    condition = next(run for run in manifest["runs"] if run["machine_id"] == machine.machine_id)

    outcomes_dir = tmp_path / "campaign" / machine.machine_id
    outcomes_dir.mkdir(parents=True, exist_ok=True)
    outcomes_path = outcomes_dir / "run_outcomes.csv"

    with outcomes_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(append.base.RUN_OUTCOME_FIELDS))
        writer.writeheader()
        row = {field: "" for field in append.base.RUN_OUTCOME_FIELDS}
        row.update({
            "campaign_id": manifest["campaign_id"],
            "allocation_patch_id": manifest["allocation_patch_id"],
            "simulation_semantics_id": manifest["simulation_semantics_id"],
            "run_id": condition["run_id"],
            "condition_key": condition["condition_key"],
            "policy_configuration": condition["policy_configuration"],
            "robot_count": str(condition["robot_count"]),
            "order_rate": str(condition["order_rate"]),
            "replication": str(condition["replication"]),
            "seed": str(condition["seed"]),
            "status": "success",
            "kpi_schema_version": manifest["kpi_schema_version"],
        })
        writer.writerow(row)

    items = [{
        "run_id": condition["run_id"],
        "source_campaign_id": manifest["campaign_id"],
        "allocation_patch_id": manifest["allocation_patch_id"],
    }]
    monkeypatch_repo_root = append.REPO_ROOT
    try:
        append.REPO_ROOT = tmp_path
        path = append.rebuild_plan_outcomes(manifest, machine, items)
        with path.open(newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) == 1
        assert rows[0]["run_id"] == condition["run_id"]
    finally:
        append.REPO_ROOT = monkeypatch_repo_root

