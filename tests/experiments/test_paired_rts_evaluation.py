"""Regression coverage for the balanced local RTS comparison campaign."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.rmfs.experiments.evaluation import paired_campaign as campaign


def _worker_summary(spec):
    return {
        "status": "success",
        "run_id": spec.run_id,
        "seed": spec.campaign_seed,
        "replication": spec.replication,
        "policy_configuration": spec.policy_configuration,
        "campaign_id": spec.campaign_id,
        "simulated_seconds": float(spec.ticks) * 0.15,
        "kpi_schema_version": spec.kpi_schema_version,
        "repo_commit": spec.commit,
        "kpi": {
            "kpi_complete": True,
            "kpi_complete_strict": True,
            "kpi_completion_status": "completed_strict",
            "orders_completed": 1,
        },
    }


def test_paired_campaign_runs_equal_terminal_waves_and_retains_only_compact_outputs(tmp_path, monkeypatch):
    seed_pack = {
        "seed_pack_id": "paired-test-pack",
        "seed_base": 91,
        "replications": 4,
        "netlogo_steps_per_run": 10,
        "seeds": [
            {"replication": 1, "seed": 101},
            {"replication": 2, "seed": 102},
            {"replication": 3, "seed": 103},
            {"replication": 4, "seed": 104},
        ],
    }
    pack_path = tmp_path / "pack.json"
    pack_path.write_text(json.dumps(seed_pack), encoding="utf-8")
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()

    monkeypatch.setattr(
        campaign,
        "load_policy_from_checkpoint",
        lambda *_args, **_kwargs: SimpleNamespace(policy_checkpoint_id="batch_000014"),
    )
    monkeypatch.setattr(campaign, "resolve_ablation", lambda _name: SimpleNamespace(name="full", hash="full-hash"))
    monkeypatch.setattr(campaign, "git_value", lambda *_args: "test-value")

    launched_waves = []

    def fake_run_specs(specs, *, max_workers, before_launch, on_run_complete, **_kwargs):
        pending = list(specs)
        completed = []
        while pending:
            active = []
            while pending and len(active) < max_workers:
                spec = pending[0]
                if not before_launch(spec, len(active)):
                    break
                active.append(pending.pop(0))
            assert len(active) == max_workers
            launched_waves.append([spec.policy_configuration for spec in active])
            for spec in active:
                spec.runtime_root.mkdir(parents=True, exist_ok=True)
                summary = _worker_summary(spec)
                (spec.runtime_root / "worker_summary.json").write_text(json.dumps(summary), encoding="utf-8")
                on_run_complete(spec, 0)
                completed.append({"spec": spec, "return_code": 0, "worker_summary": summary})
        return completed

    monkeypatch.setattr(campaign, "run_specs", fake_run_specs)
    result = campaign.run_paired_rts_rl_vs_nearest_evaluation(
        repo_root=tmp_path,
        checkpoint_dir=checkpoint,
        zone_ids=("auto",),
        seed_pack_path=pack_path,
        output_root=tmp_path / "output",
        max_workers=4,
        charging_mode="enabled",
        machine_id="dewan_wsl",
    )

    campaign_root = tmp_path / "output" / result["campaign_id"]
    assert result["valid"] is True
    assert result["completed_per_policy"] == {"current": 4, "rts_rl_explicit": 4}
    assert launched_waves == [
        ["current", "rts_rl_explicit", "current", "rts_rl_explicit"],
        ["current", "rts_rl_explicit", "current", "rts_rl_explicit"],
    ]
    assert not (campaign_root / "workers").exists()
    assert (campaign_root / "run_outcomes.jsonl").exists()
    specs = json.loads((campaign_root / "worker_specs.json").read_text(encoding="utf-8"))
    nearest = next(spec for spec in specs if spec["run_id"] == "nearest_001")
    rts_rl = next(spec for spec in specs if spec["run_id"] == "rts_rl_001")
    assert nearest["rts_rollout_enabled"] is False
    assert nearest["rts_policy_action_mode"] == "sample"
    assert rts_rl["rts_rollout_enabled"] is True
    assert rts_rl["rts_policy_action_mode"] == "greedy"
    assert nearest["machine_id"] == rts_rl["machine_id"] == "dewan_wsl"
    rows = json.loads((campaign_root / "full_kpi_summary.json").read_text(encoding="utf-8"))
    assert len(rows) == 8


def test_paired_campaign_resume_allows_worker_and_thread_reconfiguration(tmp_path, monkeypatch):
    seed_pack = {
        "seed_pack_id": "resume-test-pack",
        "seed_base": 91,
        "replications": 2,
        "netlogo_steps_per_run": 10,
        "seeds": [{"replication": 1, "seed": 101}, {"replication": 2, "seed": 102}],
    }
    pack_path = tmp_path / "pack.json"
    pack_path.write_text(json.dumps(seed_pack), encoding="utf-8")
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    monkeypatch.setattr(
        campaign,
        "load_policy_from_checkpoint",
        lambda *_args, **_kwargs: SimpleNamespace(policy_checkpoint_id="batch_000014"),
    )
    monkeypatch.setattr(campaign, "resolve_ablation", lambda _name: SimpleNamespace(name="full", hash="full-hash"))
    monkeypatch.setattr(campaign, "git_value", lambda *_args: "test-value")

    initial = campaign.run_paired_rts_rl_vs_nearest_evaluation(
        repo_root=tmp_path,
        checkpoint_dir=checkpoint,
        zone_ids=("auto",),
        seed_pack_path=pack_path,
        output_root=tmp_path / "output",
        max_workers=4,
        rts_torch_threads=2,
        rts_torch_interop_threads=1,
        charging_mode="enabled",
        dry_run=True,
    )
    monkeypatch.setattr(campaign, "git_value", lambda *_args: "updated-test-value")
    with pytest.raises(ValueError, match="repo_commit"):
        campaign.run_paired_rts_rl_vs_nearest_evaluation(
            repo_root=tmp_path,
            checkpoint_dir=checkpoint,
            zone_ids=("auto",),
            seed_pack_path=pack_path,
            output_root=tmp_path / "output",
            max_workers=2,
            rts_torch_threads=1,
            rts_torch_interop_threads=1,
            charging_mode="enabled",
            machine_id="dewan_wsl",
            resume_campaign_id=initial["campaign_id"],
            dry_run=True,
        )
    resumed = campaign.run_paired_rts_rl_vs_nearest_evaluation(
        repo_root=tmp_path,
        checkpoint_dir=checkpoint,
        zone_ids=("auto",),
        seed_pack_path=pack_path,
        output_root=tmp_path / "output",
        max_workers=2,
        rts_torch_threads=1,
        rts_torch_interop_threads=1,
        charging_mode="enabled",
        machine_id="dewan_wsl",
        resume_campaign_id=initial["campaign_id"],
        allow_resume_repo_commit_mismatch=True,
        dry_run=True,
    )

    campaign_root = tmp_path / "output" / initial["campaign_id"]
    original_config = json.loads((campaign_root / "campaign_config.json").read_text(encoding="utf-8"))
    resume_operations = [json.loads(line) for line in (campaign_root / "resume_operations.jsonl").read_text(encoding="utf-8").splitlines()]
    assert resumed["campaign_id"] == initial["campaign_id"]
    assert resumed["resumed"] is True
    assert original_config["max_workers"] == 4
    assert resume_operations[-1]["current_max_workers"] == 2
    assert resume_operations[-1]["current_rts_torch_threads"] == 1
    assert resume_operations[-1]["previous_repo_commit"] == "test-value"
    assert resume_operations[-1]["current_repo_commit"] == "updated-test-value"
    assert resume_operations[-1]["repo_commit_mismatch_allowed"] is True
