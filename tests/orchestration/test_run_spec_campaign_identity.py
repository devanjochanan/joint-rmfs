from pathlib import Path

from src.rmfs.orchestration.run_spec import RunSpec
from src.rmfs.rl.rts.runtime_config import RTSRuntimeConfig


def test_run_spec_round_trips_campaign_identity(tmp_path: Path):
    spec = RunSpec(
        run_id="all_on_rl__r20__arr500__rep001__abc",
        ticks=580000,
        runtime_root=tmp_path / "run",
        repo_root=tmp_path,
        input_root=tmp_path / "input",
        run_profile="training",
        order_generation_mode="shuffled_historical_cycle",
        order_rate_per_hour=500,
        campaign_id="sensitivity_full_kpi_v2_abc",
        allocation_patch_id="allocation_patch_0001_abc",
        simulation_semantics_id="sensitivity_simulation_semantics.v2",
        machine_id="codex_local",
        stage_first_requested=1,
        kpi_schema_version="sensitivity_full_kpi.v2",
        policy_configuration="all_on_rl",
        replication=1,
        campaign_seed=42,
        rts_checkpoint_sha256="a" * 64,
        pps_model_sha256="p" * 64,
    )

    payload = spec.to_json_dict()
    assert payload["campaign_id"] == "sensitivity_full_kpi_v2_abc"
    assert payload["machine_id"] == "codex_local"
    assert payload["stage_first_requested"] == 1
    assert payload["campaign_seed"] == 42
    assert payload["allocation_patch_id"] == "allocation_patch_0001_abc"
    assert payload["simulation_semantics_id"] == "sensitivity_simulation_semantics.v2"
    assert payload["rts_checkpoint_sha256"] == "a" * 64
    assert payload["persist_final_state"] is False

    restored = RunSpec.from_json_dict(payload)
    assert restored == spec


def test_vrsla_post_pick_flag_round_trips_worker_and_runtime_configs(tmp_path: Path):
    spec = RunSpec(
        run_id="vrsla_rep001",
        ticks=10,
        runtime_root=tmp_path / "run",
        repo_root=tmp_path,
        run_profile="gui",
        rts_policy_mode="vrsla_teacher",
        rts_rollout_enabled=True,
        rts_state_capture_mode="full",
        rts_vrsla_always_post_pick_replenish=True,
    )

    restored_spec = RunSpec.from_json_dict(spec.to_json_dict())
    assert restored_spec.rts_vrsla_always_post_pick_replenish is True

    runtime_config = RTSRuntimeConfig.from_dict(
        {
            "policy_mode": "vrsla_teacher",
            "rollout_enabled": True,
            "state_capture_mode": "full",
            "committed_next_reservations_enabled": True,
            "vrsla_always_post_pick_replenish": True,
        }
    )
    assert runtime_config.vrsla_always_post_pick_replenish is True
    assert runtime_config.to_json_dict()["vrsla_always_post_pick_replenish"] is True
