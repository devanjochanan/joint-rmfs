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


def test_primary_matrix_is_bundled_two_treatment_720_full_kpi_v3():
    plan, assets = _plan()
    assert plan["kpi_schema_version"] == "full_kpi_v3"
    # Bundled two-treatment comparison: 2 x (15,20,25) x (400,500,600) x 40 = 720.
    assert len(plan["runs"]) == 720
    assert {run["policy_configuration"] for run in plan["runs"]} == {"all_off", "all_on_rl"}
    a = plan["assertions"]
    assert a["total_unique_fresh_runs"] == 720
    assert a["stage_new_runs"] == {"1": 18, "2": 39, "3": 39, "4": 624}
    assert a["complete_pair_count"] == 360
    assert a["pair_seed_mismatches"] == 0 and a["pair_layout_mismatches"] == 0


def test_treatment_contracts_are_bundled_and_charging_enabled():
    plan, assets = _plan()
    off = campaign.treatment_execution_contract("all_off", assets)
    on = campaign.treatment_execution_contract("all_on_rl", assets)
    # Charging enabled in BOTH treatments, different packages.
    assert off["charging_enabled"] is True and on["charging_enabled"] is True
    assert off["charging_placement_source"] == "generated_reference"
    assert on["charging_placement_source"] == "generated_salsa_adaptive"
    # PPS: heuristic for all_off, PPO for all_on_rl.
    assert off["pps_mode"] == "heuristic" and on["pps_mode"] == "ppo"
    # RTS + committed-next bundled correctly.
    assert off["rts_policy_mode"] == "current" and on["rts_policy_mode"] == "rts_rl_explicit"
    assert off["committed_next_reservations_enabled"] is False
    assert on["committed_next_reservations_enabled"] is True


def test_pairing_and_charging_identity_consistency():
    plan, _ = _plan()
    for run in plan["runs"]:
        assert run["charging"]["charging_enabled"] is True
        assert run["charging"]["charging_placement_source"] in {"generated_reference", "generated_salsa_adaptive"}
        assert run["identity"]["charging_placement_source"] == run["charging"]["charging_placement_source"]
        assert "paired_group_id" in run
    # Every pair id has exactly one all_off and one all_on_rl member with matching seed.
    pairs = {}
    for run in plan["runs"]:
        pairs.setdefault(run["paired_group_id"], {}).setdefault(run["policy_configuration"], []).append(run)
    assert len(pairs) == 360
    for members in pairs.values():
        assert len(members.get("all_off", [])) == 1
        assert len(members.get("all_on_rl", [])) == 1
        assert members["all_off"][0]["seed"] == members["all_on_rl"][0]["seed"]
