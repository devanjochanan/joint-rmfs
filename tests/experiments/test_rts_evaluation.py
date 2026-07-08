import pytest
from pathlib import Path
import json

from src.rmfs.experiments.evaluation.controller import (
    run_rts_paired_evaluation,
    assert_paired_plan,
)
from src.rmfs.experiments.evaluation.seed_pack import build_seed_pack
from src.rmfs.orchestration.run_spec import RunSpec

def test_paired_spec_generation_dry_run(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    checkpoint_dir = repo_root / "data/models/rts/batch_000014/checkpoint"

    # Run in dry-run mode
    summary = run_rts_paired_evaluation(
        repo_root=repo_root,
        checkpoint_dir=checkpoint_dir,
        zone_ids=("auto",),
        output_root=tmp_path,
        replications=3,
        seed_base=100,
        simulated_seconds=10.0,
        robot_count=20,
        order_rate=500,
        charging_enabled=True,
        max_workers=1,
        dry_run=True,
    )

    # 1. Verify summary output fields
    assert summary["status"] == "dry_run"
    assert summary["replications"] == 3
    assert summary["robot_count"] == 20
    assert summary["order_rate"] == 500
    assert summary["charging_enabled"] is True

    # 2. Check generated files
    eval_run_id = summary["eval_run_id"]
    run_root = tmp_path / eval_run_id
    assert (run_root / "eval_config.json").exists()
    assert (run_root / "seed_pack.json").exists()
    assert (run_root / "worker_specs.json").exists()
    assert (run_root / "eval_summary.json").exists()

    # Load worker specs
    with (run_root / "worker_specs.json").open() as fh:
        specs_data = json.load(fh)
    
    assert len(specs_data) == 6

    # Verify interleaved treatment and shared seeds
    for idx in range(3):
        current_spec = specs_data[idx * 2]
        rl_spec = specs_data[idx * 2 + 1]

        assert current_spec["rts_policy_mode"] == "current"
        assert rl_spec["rts_policy_mode"] == "rts_rl_explicit"

        assert current_spec["rts_random_seed"] == rl_spec["rts_random_seed"]
        assert current_spec["worker_id"] == idx + 1
        assert rl_spec["worker_id"] == idx + 1

        # Assert no other differences besides expected identity keys
        expected_diffs = {
            "run_id", "runtime_root", "rts_policy_mode", "rts_rollout_enabled",
            "rts_policy_checkpoint_dir", "rts_policy_checkpoint_id", "rts_policy_action_mode",
            "rts_torch_threads", "rts_torch_interop_threads", "timestamp",
            "policy_configuration", "rts_checkpoint_sha256"
        }
        for k in current_spec:
            if k not in expected_diffs:
                assert current_spec[k] == rl_spec[k], f"Mismatch in field: {k}"


def test_assert_paired_plan_validation():
    # Helper to build valid RunSpecs
    def make_base_spec(run_id, policy_mode, seed, idx=1, ticks=66667):
        return RunSpec(
            run_id=run_id,
            ticks=ticks,
            runtime_root=Path("/tmp"),
            repo_root=Path("/tmp"),
            rts_policy_mode=policy_mode,
            rts_rollout_enabled=(policy_mode == "rts_rl_explicit"),
            rts_random_seed=seed,
            robot_count=20,
            order_rate_per_hour=500,
            charging_enabled=True,
            pps_mode="heuristic",
            kpi_schema_version="sensitivity_full_kpi.v1",
            run_profile="training",
            order_generation_mode="shuffled_historical_cycle",
            replication=idx,
            campaign_id="test_campaign",
            machine_id="local",
            stage_first_requested=1,
            pps_model_sha256="none",
            rts_checkpoint_sha256="none",
            policy_configuration=policy_mode,
        )

    seed_pack = build_seed_pack(seed_base=42, replications=2, netlogo_steps_per_run=66667, purpose="test")
    seeds = [s["seed"] for s in seed_pack["seeds"]]

    # 1. Valid interleaved plan
    specs_valid = [
        make_base_spec("current_001", "current", seeds[0], 1),
        make_base_spec("rts_rl_001", "rts_rl_explicit", seeds[0], 1),
        make_base_spec("current_002", "current", seeds[1], 2),
        make_base_spec("rts_rl_002", "rts_rl_explicit", seeds[1], 2),
    ]
    # Should not raise AssertionError
    assert_paired_plan(specs_valid, seed_pack, replications=2, steps=66667)

    # 2. Invalid ordering
    specs_invalid_order = [
        make_base_spec("current_001", "current", seeds[0], 1),
        make_base_spec("current_002", "current", seeds[1], 2),
        make_base_spec("rts_rl_001", "rts_rl_explicit", seeds[0], 1),
        make_base_spec("rts_rl_002", "rts_rl_explicit", seeds[1], 2),
    ]
    with pytest.raises(AssertionError, match="Expected specs.*to be rts_rl_explicit"):
        assert_paired_plan(specs_invalid_order, seed_pack, replications=2, steps=66667)

    # 3. Mismatched ticks
    specs_bad_ticks = [
        make_base_spec("current_001", "current", seeds[0], 1),
        make_base_spec("rts_rl_001", "rts_rl_explicit", seeds[0], 1),
        make_base_spec("current_002", "current", seeds[1], 2),
        make_base_spec("rts_rl_002", "rts_rl_explicit", seeds[1], 2, ticks=1000),
    ]
    with pytest.raises(AssertionError, match="Expected 66667 steps, got 1000"):
        assert_paired_plan(specs_bad_ticks, seed_pack, replications=2, steps=66667)
