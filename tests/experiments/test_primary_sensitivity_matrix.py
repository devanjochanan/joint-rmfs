from scripts.experiments import distributed_sensitivity_campaign as campaign
from tests.experiments.test_distributed_sensitivity_campaign import fake_assets


def _plan():
    machines = campaign.default_machines()
    assets = fake_assets()
    plan = campaign.build_campaign_plan(
        campaign_id=campaign.generate_campaign_id(machines, assets, 1.35),
        machines=machines, assets=assets,
    )
    return plan, assets


def test_primary_matrix_is_130_run_main_plus_interactions_full_kpi_v3():
    plan, assets = _plan()
    assert plan["kpi_schema_version"] == "full_kpi_v3"
    assert "charging_config_relative_path" not in plan["assets"]
    assert "charging_config_sha256" not in plan["assets"]
    assert len(plan["runs"]) == 130
    a = plan["assertions"]
    assert a["total_unique_fresh_runs"] == 130
    assert a["stage_new_runs"] == {"1": 18, "2": 19, "3": 19, "4": 74}
    assert a["main_run_count"] == 88
    assert a["mixed_run_count"] == 42
    assert a["complete_pair_count"] == 44
    assert a["pair_seed_mismatches"] == 0 and a["pair_layout_mismatches"] == 0
    assert {run["machine_id"] for run in plan["runs"]}.isdisjoint({"dewa_macbook"})


def test_treatment_contracts_are_bundled_and_charging_enabled():
    plan, assets = _plan()
    off = campaign.treatment_execution_contract("all_off", assets)
    on = campaign.treatment_execution_contract("all_on_rl", assets)
    # Charging enabled in BOTH treatments, different packages.
    assert off["charging_enabled"] is True and on["charging_enabled"] is True
    assert off["charging_placement_source"] == "generated_reference"
    assert on["charging_placement_source"] == "generated_salsa_adaptive"
    # PPS: heuristic for all_off, constrained PPO for all_on_rl.
    assert off["pps_mode"] == "heuristic" and on["pps_mode"] == "ppo_constrained"
    # RTS + committed-next bundled correctly.
    assert off["rts_policy_mode"] == "current" and on["rts_policy_mode"] == "rts_rl_explicit"
    assert off["committed_next_reservations_enabled"] is False
    assert on["committed_next_reservations_enabled"] is True
    mixed = campaign.treatment_execution_contract("mix_sij_cur_ppoc_salsa", assets)
    assert mixed["rts_policy_mode"] == "current"
    assert mixed["pps_mode"] == "ppo_constrained"
    assert mixed["charging_placement_source"] == "generated_salsa_adaptive"


def test_pairing_and_charging_identity_consistency():
    plan, _ = _plan()
    for run in plan["runs"]:
        assert run["charging"]["charging_enabled"] is True
        assert run["charging"]["charging_placement_source"] in {"generated_reference", "generated_salsa_adaptive"}
        assert run["identity"]["charging_placement_source"] == run["charging"]["charging_placement_source"]
        assert run["scenario_id"] in {"cindy_s3", "scenario4_sij"}
        assert "paired_group_id" in run
    # Every main pair id has exactly one all_off and one all_on_rl member with matching seed.
    pairs = {}
    for run in plan["runs"]:
        if run["design_group"] != "main":
            continue
        pairs.setdefault(run["paired_group_id"], {}).setdefault(run["policy_configuration"], []).append(run)
    assert len(pairs) == 44
    for members in pairs.values():
        assert len(members.get("all_off", [])) == 1
        assert len(members.get("all_on_rl", [])) == 1
        assert members["all_off"][0]["seed"] == members["all_on_rl"][0]["seed"]
