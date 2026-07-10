from collections import defaultdict
from dataclasses import replace

import pytest

from src.rmfs.experiments.sensitivity_allocation import (
    REFERENCE_TREATMENT,
    RL_TREATMENT,
    MachineAllocationProfile,
    allocate_stage,
    allocate_waves,
    allocation_config_rows,
    default_machine_profiles,
)


def _rows():
    """Synthetic 720-condition campaign with the real stage/treatment counts."""
    rows = []
    for pair in range(1, 361):
        seed = 41 + ((pair - 1) % 40 + 1)
        if pair <= 9:
            stages = {REFERENCE_TREATMENT: 1, RL_TREATMENT: 1}
        elif pair <= 48:
            stages = {REFERENCE_TREATMENT: 2, RL_TREATMENT: 3}
        else:
            stages = {REFERENCE_TREATMENT: 4, RL_TREATMENT: 4}
        for treatment in (REFERENCE_TREATMENT, RL_TREATMENT):
            rows.append({
                "condition_key": f"pair={pair}|{treatment}", "run_id": f"run-{pair}-{treatment}",
                "paired_group_id": f"pair={pair}", "policy_configuration": treatment,
                "stage_first_requested": stages[treatment], "robot_count": (15, 20, 25)[pair % 3],
                "order_rate": (400, 500, 600)[pair % 3], "replication": (pair - 1) % 40 + 1,
                "seed": seed, "cost": 580_000.0 * (1.2 if treatment == RL_TREATMENT else 1.0),
            })
    return rows


def _cost(row):
    return float(row["cost"])


def _plan():
    return allocate_waves(_rows(), profiles=default_machine_profiles(), estimate_steps=_cost)


def _by_stage(plan, stage):
    return [row for row in plan["runs"] if row["stage_first_requested"] == stage]


def test_capability_table_has_explicit_treatment_and_stage_eligibility():
    profiles = {profile.machine_id: profile for profile in default_machine_profiles()}
    for machine in ("alisha_pc", "dewa_macbook"):
        assert profiles[machine].allowed_treatments == (REFERENCE_TREATMENT,)
        assert profiles[machine].max_rl_workers == 0
        assert profiles[machine].eligible_stages == (1, 2, 3, 4)
    assert all(profile.max_rl_workers <= profile.max_workers for profile in profiles.values())
    assert allocation_config_rows(profiles.values())


def test_mac_and_alisha_reject_rl_but_mac_accepts_stage4_reference():
    plan = _plan()
    for row in plan["runs"]:
        if row["machine_id"] in {"alisha_pc", "dewa_macbook"}:
            assert row["policy_configuration"] == REFERENCE_TREATMENT
    assert any(row["machine_id"] == "dewa_macbook" and row["stage_first_requested"] == 4 for row in plan["runs"])


def test_stage1_places_constrained_rl_before_reference_work():
    stage1 = _by_stage(_plan(), 1)
    assert len(stage1) == 18
    assert [row["policy_configuration"] for row in stage1[:9]] == [RL_TREATMENT] * 9
    assert {row["machine_id"] for row in stage1 if row["policy_configuration"] == REFERENCE_TREATMENT}.issuperset({"alisha_pc", "dewa_macbook"})


def test_stage2_uses_all_seven_hosts_and_stage3_uses_exact_rl_hosts():
    plan = _plan()
    assert {row["machine_id"] for row in _by_stage(plan, 2)} == {profile.machine_id for profile in default_machine_profiles()}
    assert {row["machine_id"] for row in _by_stage(plan, 3)} == {
        "win_lukman", "win_admin", "citi_angiebow", "codex_local", "citi_gojira",
    }


def test_stage4_uses_all_hosts_for_reference_and_only_rl_hosts_for_rl():
    stage4 = _by_stage(_plan(), 4)
    reference_hosts = {row["machine_id"] for row in stage4 if row["policy_configuration"] == REFERENCE_TREATMENT}
    rl_hosts = {row["machine_id"] for row in stage4 if row["policy_configuration"] == RL_TREATMENT}
    assert reference_hosts == {profile.machine_id for profile in default_machine_profiles()}
    assert rl_hosts <= {"win_lukman", "win_admin", "citi_angiebow", "codex_local", "citi_gojira"}


def test_wave_sizes_counts_and_pair_identity_are_preserved():
    source = _rows()
    plan = _plan()
    assert len(plan["critical_wave"]) == 96
    assert len(plan["best_effort_wave"]) == 624
    assert {stage: len(_by_stage(plan, stage)) for stage in (1, 2, 3, 4)} == {1: 18, 2: 39, 3: 39, 4: 624}
    before = {(row["condition_key"], row["seed"], row["paired_group_id"]) for row in source}
    after = {(row["condition_key"], row["seed"], row["paired_group_id"]) for row in plan["runs"]}
    assert before == after
    assert len({row["paired_group_id"] for row in plan["runs"]}) == 360


def test_faster_rl_hosts_receive_no_less_stage3_rl_work_than_slower_hosts():
    stage3 = _by_stage(_plan(), 3)
    work = defaultdict(float)
    for row in stage3:
        work[row["machine_id"]] += row["estimated_backend_steps"]
    ordered = ["citi_gojira", "win_lukman", "win_admin", "citi_angiebow", "codex_local"]
    assert [work[machine] for machine in ordered] == sorted([work[machine] for machine in ordered], reverse=True)


def test_changing_rl_rate_changes_rl_assignment_without_relaxing_reference_rules():
    profiles = list(default_machine_profiles())
    baseline = _plan()
    changed = [
        replace(profile, rl_effective_steps_per_second=profile.rl_effective_steps_per_second * 0.1)
        if profile.machine_id == "citi_gojira" else profile
        for profile in profiles
    ]
    updated = allocate_waves(_rows(), profiles=changed, estimate_steps=_cost)
    baseline_rl = [(row["run_id"], row["machine_id"]) for row in baseline["runs"] if row["policy_configuration"] == RL_TREATMENT]
    updated_rl = [(row["run_id"], row["machine_id"]) for row in updated["runs"] if row["policy_configuration"] == RL_TREATMENT]
    assert baseline_rl != updated_rl
    assert all(row["policy_configuration"] == REFERENCE_TREATMENT for row in updated["runs"] if row["machine_id"] in {"alisha_pc", "dewa_macbook"})


def test_allocation_is_deterministic_and_rl_slots_never_exceed_budget():
    first, second = _plan(), _plan()
    assert first == second
    profiles = {profile.machine_id: profile for profile in default_machine_profiles()}
    for row in first["runs"]:
        if row["policy_configuration"] == RL_TREATMENT:
            assert row["machine_slot_index"] < profiles[row["machine_id"]].max_rl_workers
    by_slot = defaultdict(list)
    for row in first["runs"]:
        by_slot[(row["stage_first_requested"], row["machine_id"], row["machine_slot_index"])].append(row)
    for scheduled in by_slot.values():
        previous_finish = 0.0
        for row in sorted(scheduled, key=lambda item: item["projected_slot_start_seconds"]):
            assert row["projected_slot_start_seconds"] >= previous_finish
            previous_finish = row["projected_slot_finish_seconds"]


def test_impossible_rl_eligibility_fails_before_assignment():
    no_rl = [
        MachineAllocationProfile(
            machine_id="reference-only", max_workers=1,
            allowed_treatments=(REFERENCE_TREATMENT,), eligible_stages=(1,),
            reference_effective_steps_per_second=1.0,
        )
    ]
    with pytest.raises(RuntimeError, match="no eligible slots"):
        allocate_stage(
            [{"condition_key": "rl", "policy_configuration": RL_TREATMENT, "stage_first_requested": 1, "cost": 1.0}],
            stage=1, profiles=no_rl, estimate_steps=_cost,
        )
