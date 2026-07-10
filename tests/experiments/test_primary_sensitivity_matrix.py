from scripts.experiments import distributed_sensitivity_campaign as campaign
from tests.experiments.test_distributed_sensitivity_campaign import fake_assets


def test_primary_matrix_is_four_treatment_full_kpi_v3():
    machines = campaign.default_machines()
    assets = fake_assets()
    plan = campaign.build_campaign_plan(
        campaign_id=campaign.generate_campaign_id(machines, assets, 1.35), machines=machines, assets=assets,
    )
    assert plan["kpi_schema_version"] == "full_kpi_v3"
    assert len(plan["runs"]) == 1200
    assert {run["policy_configuration"] for run in plan["runs"]} == set(campaign.POLICY_CONFIGURATIONS)
    for run in plan["runs"]:
        contract = campaign.treatment_execution_contract(run["policy_configuration"], assets)
        assert contract["pps_mode"] == "heuristic"
        assert run["charging"]["charging_placement_source"] in {"reference_off", "salsa_adaptive_on"}
        assert run["identity"]["charging_placement_source"] == run["charging"]["charging_placement_source"]
