#!/usr/bin/env python3
"""Direct RunSpec worker smoke for the all-off bounded vertical slice."""

from __future__ import annotations

import datetime
import json
import sys
import tempfile
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))

    from src.rmfs.orchestration.local_executor import (
        changed_root_files,
        git_value,
        load_worker_summary,
        run_specs,
        snapshot_root,
        worker_environment_overrides,
    )
    from src.rmfs.orchestration.run_spec import RunSpec

    output_root = Path(tempfile.mkdtemp(prefix="rmfs_all_off_vertical_slice_"))
    runtime_root = output_root / "run_001"
    before = snapshot_root(repo_root)
    branch = git_value(repo_root, "rev-parse", "--abbrev-ref", "HEAD")
    commit = git_value(repo_root, "rev-parse", "HEAD")

    spec = RunSpec(
        run_id="run_001",
        ticks=3,
        runtime_root=runtime_root,
        repo_root=repo_root,
        branch=branch,
        commit=commit,
        python_executable=sys.executable,
        timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        run_profile="smoke",
        run_horizon_ticks=3,
        bootstrap_n_orders=1,
        demand_horizon_ticks=3,
        demand_buffer_ticks=0,
        order_generation_mode="controlled_count",
        full_raw_order_replay=False,
        pod_location_mode="fixed",
        robot_count=15,
        expected_picking_station_count=3,
        expected_replenishment_station_count=1,
        rts_policy_mode="current",
        rts_rollout_enabled=False,
        committed_next_reservations_enabled=False,
        pps_mode="heuristic",
        pps_model_path=None,
        charging_enabled=False,
        robot_task_allocator="legacy_nearest",
        regret_k=None,
        detail_db=False,
        debug_trace=False,
        keep_runtime_artifacts=True,
    )

    try:
        round_trip = RunSpec.from_json_dict(spec.to_json_dict())
        assert round_trip.robot_count == 15
        assert round_trip.expected_picking_station_count == 3
        assert round_trip.expected_replenishment_station_count == 1
        assert round_trip.pps_mode == "heuristic"
        assert round_trip.pps_model_path is None
        assert round_trip.charging_enabled is False

        env = worker_environment_overrides(spec)
        assert env["RMFS_NUM_ROBOTS"] == "15"
        assert env["PPS_MODE"] == "heuristic"
        assert env["PPS_RL_ENABLED"] == "0"
        assert env["RMFS_CHARGING_ENABLED"] == "0"
        assert "PPS_RL_MODEL_PATH" not in env

        try:
            RunSpec(
                run_id="invalid",
                ticks=1,
                runtime_root=output_root / "invalid",
                repo_root=repo_root,
                robot_count=0,
            ).to_json_dict()
            raise AssertionError("invalid robot_count did not fail")
        except ValueError as exc:
            assert "robot_count" in str(exc)

        completed = run_specs([spec], max_workers=1, progress=False)
        assert len(completed) == 1
        assert completed[0]["return_code"] == 0

        summary = load_worker_summary(runtime_root)
        assert summary["status"] == "success", json.dumps(summary, indent=2)
        assert summary["requested_robot_count"] == 15
        assert summary["realized_robot_count"] == 15
        assert summary["expected_picking_station_count"] == 3
        assert summary["realized_picking_station_count"] == 3
        assert summary["expected_replenishment_station_count"] == 1
        assert summary["realized_replenishment_station_count"] == 1
        assert summary["pps_mode"] == "heuristic"
        assert summary["pps_rl_enabled"] is False
        assert summary["rts_policy_mode"] == "current"
        assert summary["rts_rollout_enabled"] is False
        assert summary["robot_task_allocator"] == "legacy_nearest"
        assert (runtime_root / "worker_summary.json").exists()

        changed = changed_root_files(before, snapshot_root(repo_root))
        assert changed == [], f"root-sensitive files changed: {changed}"

        print(json.dumps({
            "status": "success",
            "output_root": str(output_root),
            "worker_summary": str(runtime_root / "worker_summary.json"),
            "requested_robot_count": summary["requested_robot_count"],
            "realized_robot_count": summary["realized_robot_count"],
            "realized_picking_station_count": summary["realized_picking_station_count"],
            "realized_replenishment_station_count": summary["realized_replenishment_station_count"],
        }, indent=2))
        return 0
    except Exception:
        print(f"validation output retained at {output_root}", file=sys.stderr)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
