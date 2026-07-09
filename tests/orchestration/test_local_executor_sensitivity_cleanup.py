import inspect
import json
import sys
import types
from dataclasses import replace
from pathlib import Path

from src.rmfs.app import netlogo_api
from src.rmfs.orchestration.local_executor import (
    FAILED_RECLAIMABLE_RUN_ARTIFACTS,
    SENSITIVITY_KPI_SCHEMA_VERSION,
    SUCCESS_RECLAIMABLE_RUN_ARTIFACTS,
    derive_sensitivity_kpi_payload,
    expected_worker_files,
    reclaim_completed_run_artifacts_with_stats,
    reclaim_failed_run_artifacts_with_stats,
    reclaim_interrupted_run_artifacts_with_stats,
    run_worker,
    run_specs,
)
from src.rmfs.orchestration.run_spec import RunSpec


class _Pipe:
    def read(self, _size=-1):
        return b""

    def close(self):
        return None


class _FakeProcess:
    instances = []

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.stdout = _Pipe()
        self.stderr = _Pipe()
        self.returncode = None
        _FakeProcess.instances.append(self)

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        self.returncode = 0
        return 0


class _OrderManager:
    orders = []


class _PodManager:
    pods = []


class _Warehouse:
    def __init__(self):
        self.order_manager = _OrderManager()
        self.pod_manager = _PodManager()
        self._objects = [
            types.SimpleNamespace(
                object_type="robot",
                travel_distances=0,
                loaded_travel_distance=0,
                empty_travel_distance=0,
                completed_cycle_count=0,
                completed_cycle_duration_sum=0,
                fixed_load_energy_consumption=0,
            )
        ]
        self.station_manager = types.SimpleNamespace(picking_stations=[], replenishment_stations=[])
        self.job_queue = []
        self.total_energy = 0
        self.stop_and_go = 0
        self.total_turning = 0
        self._tick = 0

    def get_movable_objects(self):
        return self._objects


def _spec(tmp_path: Path, run_id: str = "run") -> RunSpec:
    return RunSpec(
        run_id=run_id,
        ticks=1,
        runtime_root=tmp_path / run_id,
        repo_root=Path.cwd(),
        input_root=Path.cwd() / "data" / "input",
        python_executable="python",
        commit="commit123",
        bootstrap_n_orders=1,
        campaign_id="campaign_abc",
        allocation_patch_id="allocation_patch_0001_abc",
        simulation_semantics_id="sensitivity_simulation_semantics.v1",
        machine_id="codex_local",
        stage_first_requested=1,
        kpi_schema_version=SENSITIVITY_KPI_SCHEMA_VERSION,
        policy_configuration="all_off",
        replication=1,
        campaign_seed=42,
        rts_checkpoint_sha256="a" * 64,
        pps_model_sha256="p" * 64,
    )


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_generated_order_meta(spec: RunSpec) -> None:
    _write_json(
        spec.runtime_root / "generated_order_meta.json",
        {
            "profile": spec.run_profile,
            "order_generation_mode": spec.order_generation_mode,
            "full_raw_order_replay": False,
            "seed": spec.rts_random_seed,
            "arrival_time_unit": "simulated_seconds",
            "generated_unique_orders": 0,
            "generated_order_lines": 0,
            "generated_max_arrival": 0,
        },
    )


def test_run_spec_defaults_and_expected_files_respect_final_state(tmp_path: Path):
    spec = _spec(tmp_path)
    payload = spec.to_json_dict()

    assert spec.persist_final_state is False
    assert RunSpec.from_json_dict(payload).persist_final_state is False
    assert "netlogo.state" not in expected_worker_files(persist_final_state=False)
    assert "netlogo.state" in expected_worker_files(persist_final_state=True)


def test_headless_session_default_changed_but_finalizer_default_is_legacy():
    session_default = inspect.signature(netlogo_api.HeadlessSimulationSession).parameters["persist_final_state"].default
    finalizer_default = inspect.signature(netlogo_api.finalize_headless_run).parameters["persist_final_state"].default

    assert session_default is False
    assert finalizer_default is True


def test_legacy_setup_tick_still_use_file_backed_state():
    assert "persist_initial_state=True" in inspect.getsource(netlogo_api.setup)
    assert "_load_universe" in inspect.getsource(netlogo_api.tick)
    assert "_persist_universe" in inspect.getsource(netlogo_api.tick)
    assert "_load_universe" in inspect.getsource(netlogo_api.console_tick)
    assert "_persist_universe" in inspect.getsource(netlogo_api.console_tick)


def test_run_worker_resident_state_persistence_is_opt_in(tmp_path: Path, monkeypatch):
    seen_persistence = []
    context_holder = {}

    class StepResult:
        payload = ["tick", 0, 0, 0, 0, 0]
        status = "ok"
        steps_executed = 1

    class FakeSession:
        def __init__(self, persist_final_state=False):
            self.persist_final_state = persist_final_state
            self.finalized = False
            self.warehouse = _Warehouse()
            seen_persistence.append(persist_final_state)

        def setup(self):
            return ["setup"]

        def step(self):
            return StepResult()

        def finalize(self, reason, success):
            self.finalized = True
            if self.persist_final_state:
                context_holder["ctx"].state_path.write_text("state", encoding="utf-8")
            return {"reason": reason, "success": success, "runtime_invariants": {}}

    fake_netlogo = types.SimpleNamespace(
        HeadlessSimulationSession=FakeSession,
        SimulationTermination=types.SimpleNamespace(
            MAXIMUM_HORIZON="maximum_horizon",
            WORKER_EXCEPTION="worker_exception",
        ),
        configure_run_context=lambda ctx: context_holder.update({"ctx": ctx}),
        reset_run_context=lambda: None,
        set_sim_seed=lambda _seed: None,
    )
    monkeypatch.setitem(sys.modules, "netlogo", fake_netlogo)

    for persist in (False, True):
        spec = replace(
            _spec(tmp_path, f"persist_{persist}"),
            persist_final_state=persist,
            robot_count=1,
        )
        _write_generated_order_meta(spec)

        assert run_worker(spec) == 0
        summary = json.loads((spec.runtime_root / "worker_summary.json").read_text(encoding="utf-8"))

        assert summary["status"] == "success"
        assert summary["persist_final_state"] is persist
        assert summary["kpi_complete"] is True
        assert (spec.runtime_root / "netlogo.state").exists() is persist

    assert seen_persistence == [False, True]


def test_all_off_sensitivity_kpi_is_complete_without_rts_checkpoint(tmp_path: Path):
    spec = _spec(tmp_path)
    payload = derive_sensitivity_kpi_payload(
        spec,
        _Warehouse(),
        finalization={"reason": "completed", "runtime_invariants": {}},
        generated_order_contract={"generated_unique_orders": 0},
    )

    assert payload["rts_checkpoint_id"] == "not_applicable"
    assert payload["kpi_complete"] is True


def test_success_cleanup_preserves_summary_spec_and_embeds_rollout_summary(tmp_path: Path):
    run_root = tmp_path / "run"
    run_root.mkdir()
    _write_json(run_root / "run_spec.json", {"run_id": "run"})
    _write_json(run_root / "worker_summary.json", {"status": "success"})
    _write_json(run_root / "rts_rollout_summary.json", {"action_counts": {"1": 2}})
    for name in SUCCESS_RECLAIMABLE_RUN_ARTIFACTS:
        if name != "rts_rollout_summary.json":
            (run_root / name).write_text("scratch", encoding="utf-8")

    stats = reclaim_completed_run_artifacts_with_stats(run_root)
    summary = json.loads((run_root / "worker_summary.json").read_text(encoding="utf-8"))

    assert stats["bytes"] > 0
    assert (run_root / "run_spec.json").exists()
    assert (run_root / "worker_summary.json").exists()
    assert summary["rts_rollout_summary"] == {"action_counts": {"1": 2}}
    for name in SUCCESS_RECLAIMABLE_RUN_ARTIFACTS:
        assert not (run_root / name).exists()


def test_failed_and_interrupted_cleanup_preserves_diagnostics(tmp_path: Path):
    for helper, status in (
        (reclaim_failed_run_artifacts_with_stats, "failure"),
        (reclaim_interrupted_run_artifacts_with_stats, "running"),
    ):
        run_root = tmp_path / status
        run_root.mkdir()
        _write_json(run_root / "run_spec.json", {"run_id": status})
        _write_json(run_root / "worker_summary.json", {"status": status})
        for name in FAILED_RECLAIMABLE_RUN_ARTIFACTS:
            (run_root / name).write_text("scratch", encoding="utf-8")
        for name in ("worker_status.json", "worker_stdout.log", "worker_stderr.log", "rts_rollout_summary.json"):
            (run_root / name).write_text("diagnostic", encoding="utf-8")

        stats = helper(run_root)

        assert stats["bytes"] > 0
        assert (run_root / "run_spec.json").exists()
        assert (run_root / "worker_summary.json").exists()
        for name in FAILED_RECLAIMABLE_RUN_ARTIFACTS:
            assert not (run_root / name).exists()
        for name in ("worker_status.json", "worker_stdout.log", "worker_stderr.log", "rts_rollout_summary.json"):
            assert (run_root / name).exists()


def test_run_specs_rechecks_launch_guard_after_cleanup(tmp_path: Path, monkeypatch):
    _FakeProcess.instances = []
    specs = [_spec(tmp_path, "first"), _spec(tmp_path, "second")]
    launch_checks = []
    completions = []

    def before_launch(spec, active_count):
        launch_checks.append((spec.run_id, active_count, list(completions)))
        return active_count == 0

    def on_complete(spec, return_code):
        completions.append((spec.run_id, return_code))

    monkeypatch.setattr("src.rmfs.orchestration.local_executor.subprocess.Popen", _FakeProcess)

    completed = run_specs(
        specs,
        max_workers=2,
        before_launch=before_launch,
        on_run_complete=on_complete,
    )

    assert [item["spec"].run_id for item in completed] == ["first", "second"]
    assert completions == [("first", 0), ("second", 0)]
    assert launch_checks[:2] == [
        ("first", 0, []),
        ("second", 1, []),
    ]
    assert launch_checks[-1] == ("second", 0, [("first", 0)])
