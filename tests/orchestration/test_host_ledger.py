import json

from src.rmfs.orchestration.host_ledger import HostLedger
from src.rmfs.orchestration.run_spec import RunSpec
from src.rmfs.orchestration.source_identity import SourceIdentity


def _condition(key="c1", stage=1):
    return {"condition_key": key, "run_id": f"run-{key}", "stage_first_requested": stage}


def test_campaign_seed_round_trip(tmp_path):
    spec = RunSpec(run_id="x", ticks=1, runtime_root=tmp_path / "run", repo_root=tmp_path,
                   run_profile="gui", campaign_seed=42)
    assert RunSpec.from_json_dict(spec.to_json_dict()).campaign_seed == 42


def test_ledger_tracks_multiple_active_conditions_and_attempt_history(tmp_path):
    ledger = HostLedger.create("host", "campaign", "manifest", [_condition("a"), _condition("b")])
    assert ledger.next_condition()["condition_key"] == "a"
    ledger.start_condition("a")
    assert ledger.next_condition()["condition_key"] == "b"
    ledger.start_condition("b")
    ledger.mark_failed("a", {"error_message": "transient"})
    ledger.mark_completed("b", {"result_path": "b/worker_summary.json"})
    assert ledger.state_for("a")["status"] == "failed_retryable"
    assert ledger.state_for("b")["status"] == "completed_strict"
    assert len(ledger.state_for("a")["attempts"]) == 1
    path = tmp_path / "ledger.json"
    ledger.save(path)
    loaded = HostLedger.load(path)
    assert loaded.next_condition()["condition_key"] == "a"


def test_stage_filter_is_applied_before_selection():
    ledger = HostLedger.create("host", "campaign", "manifest", [_condition("s1", 1), _condition("s2", 2)])
    assert ledger.next_condition(eligible_stages={2})["condition_key"] == "s2"
    assert ledger.state_for("s1")["status"] == "pending"


def test_v1_ledger_migrates_without_losing_completed_outputs(tmp_path):
    path = tmp_path / "old.json"
    path.write_text(json.dumps({
        "ledger_schema_version": "host_ledger.v1", "host_id": "h", "campaign_id": "c",
        "manifest_sha256": "m", "assigned_conditions": [_condition()],
        "completed_conditions": ["c1"], "failed_conditions": [],
    }))
    ledger = HostLedger.load(path)
    assert ledger.state_for("c1")["status"] == "completed_strict"


def test_source_identity_does_not_require_git_metadata(tmp_path):
    (tmp_path / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    identity = SourceIdentity.compute(tmp_path)
    assert identity.source_tree_hash
    assert identity.git_commit is None
