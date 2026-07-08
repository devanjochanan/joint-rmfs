from collections import Counter

from scripts.experiments.distributed_sensitivity_campaign import (
    AssetBundle,
    build_campaign_plan,
    default_machines,
    seed_for_replication,
)


def fake_assets() -> AssetBundle:
    return AssetBundle(
        pps_model_relative_path="data/models/pps/pps_rl_best.zip",
        pps_model_sha256="p" * 64,
        pps_observation_schema={"pod_features": {"shape": [60, 511]}},
        rts_checkpoint_relative_dir="data/runtime/rts_training/fake/batch_000016/checkpoint",
        rts_checkpoint_id="batch_000016",
        rts_model_sha256="a" * 64,
        rts_metadata_sha256="b" * 64,
        rts_feature_schema_sha256="c" * 64,
        rts_feature_schema_id="rts_feature_schema_fake",
        rts_training_artifact="fake_vrsla_ppo",
        rts_training_latest_relative_path="data/runtime/rts_training/fake/latest.json",
        rts_lineage={
            "initialization_method": "vrsla_behavior_cloning",
            "teacher_policy": "vrsla_event_driven",
        },
        rts_lineage_source_relative_dir="data/runtime/rts_training/fake_teacher/batch_000010/checkpoint",
        charging_config_relative_path="data/input/charging/canonical_chargers.json",
        charging_config_sha256="d" * 64,
    )


def test_campaign_plan_counts_allocations_and_seed_design():
    machines = default_machines()
    manifest = build_campaign_plan(
        campaign_id="sensitivity_full_kpi_v2_test",
        machines=machines,
        assets=fake_assets(),
    )

    assert manifest["assertions"]["machine_count"] == 4
    assert manifest["assertions"]["stage_new_runs"] == {"1": 30, "2": 38, "3": 532, "4": 900}
    assert manifest["assertions"]["stage_allocations"] == {
        "1": {"win_lukman": 9, "win_admin": 8, "citi_angiebow": 7, "codex_local": 6},
        "2": {"win_lukman": 11, "win_admin": 10, "citi_angiebow": 9, "codex_local": 8},
        "3": {"win_lukman": 150, "win_admin": 143, "citi_angiebow": 131, "codex_local": 108},
        "4": {"win_lukman": 253, "win_admin": 242, "citi_angiebow": 222, "codex_local": 183},
    }
    assert manifest["assertions"]["total_fresh_runs_by_machine"] == {
        "win_lukman": 423,
        "win_admin": 403,
        "citi_angiebow": 369,
        "codex_local": 305,
    }
    assert len(manifest["runs"]) == 1500
    assert manifest["assertions"]["total_unique_fresh_runs"] == 1500
    assert manifest["old_capacity_study_completions_used"] == 0
    assert manifest["assertions"]["old_capacity_study_roots_contribute_completions"] == 0

    for replication, expected_seed in ((1, 42), (20, 61), (50, 91)):
        assert seed_for_replication(replication) == expected_seed
        assert {run["seed"] for run in manifest["runs"] if run["replication"] == replication} == {expected_seed}


def test_campaign_plan_has_no_duplicate_identities_or_shard_overlap():
    manifest = build_campaign_plan(
        campaign_id="sensitivity_full_kpi_v2_test",
        machines=default_machines(),
        assets=fake_assets(),
    )

    run_ids = [run["run_id"] for run in manifest["runs"]]
    condition_keys = [run["condition_key"] for run in manifest["runs"]]
    assert len(run_ids) == len(set(run_ids)) == 1500
    assert len(condition_keys) == len(set(condition_keys)) == 1500
    assert all(count == 1 for count in Counter(run_ids).values())
    assert all(count == 1 for count in Counter(condition_keys).values())

    all_on_rl = [run for run in manifest["runs"] if run["policy_configuration"] == "all_on_rl"]
    assert {run["identity"]["rts_checkpoint_sha256"] for run in all_on_rl} == {"a" * 64}
    assert {run["identity"]["pps_model_sha256"] for run in all_on_rl} == {"p" * 64}
    stage_1_rl_machines = {
        run["machine_id"]
        for run in manifest["runs"]
        if run["stage_first_requested"] == 1 and run["policy_configuration"] == "all_on_rl"
    }
    assert stage_1_rl_machines == {"win_lukman", "win_admin", "citi_angiebow", "codex_local"}
