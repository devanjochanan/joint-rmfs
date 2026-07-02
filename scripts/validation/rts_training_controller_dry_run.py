#!/usr/bin/env python3
"""Dry-run smoke for the RTS on-policy training controller."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from io import StringIO
from unittest.mock import patch, MagicMock

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.training.rts_train_controller import main as controller_main


def init_checkpoint(path: Path) -> None:
    subprocess.check_call(
        [
            sys.executable,
            "scripts/training/init_rts_checkpoint.py",
            "--checkpoint-dir",
            str(path),
            "--zone-ids",
            "zone1",
            "--policy-checkpoint-id",
            "dry_run_smoke_checkpoint",
        ],
        cwd=REPO_ROOT,
    )


def main():
    # 1. Standard dry-run configuration propagation checks
    output_root = REPO_ROOT / "data" / "runtime" / "rts_training" / "phase9_dry_run_smoke"
    shutil.rmtree(output_root, ignore_errors=True)
    output_root.mkdir(parents=True, exist_ok=True)
    controller_main(
        [
            "--artifact-label",
            "phase9_dry_run_smoke",
            "--output-root",
            str(output_root),
            "--batches",
            "1",
            "--workers",
            "2",
            "--netlogo-steps-per-run",
            "3",
            "--seed",
            "42",
            "--no-progress",
            "--no-tensorboard",
            "--dry-run",
        ]
    )
    run_root = output_root / "phase9_dry_run_smoke"
    assert (run_root / "training_config.json").exists()
    
    # Verify default config has debug_worker_logs=False
    with (run_root / "training_config.json").open() as fh:
        cfg = json.load(fh)
    assert cfg.get("debug_worker_logs") is False

    assert (run_root / "batch_000001" / "rollout_input" / "active_checkpoint_ref.json").exists()
    assert not (run_root / "batch_000001" / "rollout_input" / "cycle_reference.json").exists()
    assert (run_root / "batch_000001" / "workers" / "run_001" / "run_spec.json").exists()
    assert (run_root / "batch_000001" / "workers" / "run_002" / "run_spec.json").exists()
    shutil.rmtree(output_root, ignore_errors=True)

    # 1b. Resume dry-run appends additional batches after latest.json
    shutil.rmtree(output_root, ignore_errors=True)
    output_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_dir = Path(tmpdir) / "checkpoint"
        init_checkpoint(checkpoint_dir)
        run_root = output_root / "phase9_dry_run_smoke"
        (run_root / "batch_000001").mkdir(parents=True, exist_ok=True)
        with (run_root / "latest.json").open("w") as fh:
            json.dump({"batch_id": 1, "checkpoint_dir": str(checkpoint_dir), "policy_checkpoint_id": "dry_run_smoke_checkpoint"}, fh)
        controller_main(
            [
                "--artifact-label",
                "phase9_dry_run_smoke",
                "--output-root",
                str(output_root),
                "--batches",
                "2",
                "--workers",
                "1",
                "--netlogo-steps-per-run",
                "3",
                "--seed",
                "42",
                "--no-progress",
                "--no-tensorboard",
                "--resume-latest",
                "--dry-run",
            ]
        )
        assert (run_root / "batch_000001").exists()
        assert (run_root / "batch_000002" / "batch_summary.json").exists()
        assert (run_root / "batch_000003" / "batch_summary.json").exists()
    shutil.rmtree(output_root, ignore_errors=True)

    # 2. Dry-run debug_worker_logs configuration propagation checks
    shutil.rmtree(output_root, ignore_errors=True)
    output_root.mkdir(parents=True, exist_ok=True)
    controller_main(
        [
            "--artifact-label",
            "phase9_dry_run_smoke",
            "--output-root",
            str(output_root),
            "--batches",
            "1",
            "--workers",
            "2",
            "--netlogo-steps-per-run",
            "3",
            "--seed",
            "42",
            "--no-progress",
            "--no-tensorboard",
            "--dry-run",
            "--debug-worker-logs",
        ]
    )
    run_root = output_root / "phase9_dry_run_smoke"
    with (run_root / "training_config.json").open() as fh:
        cfg = json.load(fh)
    assert cfg.get("debug_worker_logs") is True
    shutil.rmtree(output_root, ignore_errors=True)

    # 3. Unit test: Subprocess failure with debug-worker-logs disabled (default)
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        checkpoint_dir = tmp_path / "checkpoint"
        init_checkpoint(checkpoint_dir)
        real_output_root = REPO_ROOT / "data" / "runtime" / "rts_training" / "test_worker_logs_default"
        shutil.rmtree(real_output_root, ignore_errors=True)
        real_output_root.mkdir(parents=True, exist_ok=True)
        
        mock_p = MagicMock()
        mock_p.poll.return_value = 1
        
        with patch("subprocess.Popen", return_value=mock_p) as mock_popen:
            stderr_capture = StringIO()
            with patch("sys.stderr", stderr_capture):
                try:
                    controller_main(
                        [
                            "--artifact-label",
                            "test_failed_default",
                            "--output-root",
                            str(real_output_root),
                            "--batches",
                            "1",
                            "--workers",
                            "2",
                            "--netlogo-steps-per-run",
                            "3",
                            "--seed",
                            "42",
                            "--no-progress",
                            "--no-tensorboard",
                            "--execute",
                            "--initial-checkpoint-dir",
                            str(checkpoint_dir),
                            "--zone-ids",
                            "zone1",
                        ]
                    )
                    assert False, "Should have raised RuntimeError"
                except RuntimeError as exc:
                    assert "failed with exit code 1" in str(exc)
            
            mock_popen.assert_called()
            worker_calls = [
                call for call in mock_popen.call_args_list
                if (call.args and call.args[0] and call.args[0][0] == sys.executable) or
                   (call.kwargs.get("args") and call.kwargs["args"][0] == sys.executable)
            ]
            assert len(worker_calls) == 2
            for call in worker_calls:
                kwargs = call.kwargs
                assert kwargs.get("stdout") == subprocess.DEVNULL
                assert kwargs.get("stderr") == subprocess.DEVNULL
            
            report_msg = stderr_capture.getvalue()
            if "Worker 2 failed with exit code 1." not in report_msg:
                print("ACTUAL REPORT MSG:\n", report_msg, file=sys.__stderr__)
            assert "Worker 2 failed with exit code 1." in report_msg
            assert "Worker output logs were not persisted; rerun with --debug-worker-logs for stdout/stderr capture." in report_msg
            
            run_root = real_output_root / "test_failed_default"
            worker_1_dir = run_root / "batch_000001" / "workers" / "run_001"
            assert not (worker_1_dir / "worker_stdout.log").exists()
            assert not (worker_1_dir / "worker_stderr.log").exists()

        shutil.rmtree(real_output_root, ignore_errors=True)

    # 4. Unit test: Subprocess failure with debug-worker-logs enabled
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        checkpoint_dir = tmp_path / "checkpoint"
        init_checkpoint(checkpoint_dir)
        real_output_root = REPO_ROOT / "data" / "runtime" / "rts_training" / "test_worker_logs_debug"
        shutil.rmtree(real_output_root, ignore_errors=True)
        real_output_root.mkdir(parents=True, exist_ok=True)
        
        mock_p = MagicMock()
        mock_p.poll.return_value = 1
        
        with patch("subprocess.Popen", return_value=mock_p) as mock_popen:
            stderr_capture = StringIO()
            with patch("sys.stderr", stderr_capture):
                try:
                    controller_main(
                        [
                            "--artifact-label",
                            "test_failed_debug",
                            "--output-root",
                            str(real_output_root),
                            "--batches",
                            "1",
                            "--workers",
                            "2",
                            "--netlogo-steps-per-run",
                            "3",
                            "--seed",
                            "42",
                            "--no-progress",
                            "--no-tensorboard",
                            "--execute",
                            "--initial-checkpoint-dir",
                            str(checkpoint_dir),
                            "--zone-ids",
                            "zone1",
                            "--debug-worker-logs",
                        ]
                    )
                    assert False, "Should have raised RuntimeError"
                except RuntimeError as exc:
                    assert "failed with exit code 1" in str(exc)
            
            mock_popen.assert_called()
            worker_calls = [
                call for call in mock_popen.call_args_list
                if (call.args and call.args[0] and call.args[0][0] == sys.executable) or
                   (call.kwargs.get("args") and call.kwargs["args"][0] == sys.executable)
            ]
            assert len(worker_calls) == 2
            for call in worker_calls:
                kwargs = call.kwargs
                assert kwargs.get("stdout") != subprocess.DEVNULL
                assert kwargs.get("stderr") != subprocess.DEVNULL
            
            report_msg = stderr_capture.getvalue()
            assert "Worker 2 failed with exit code 1." in report_msg
            assert "Worker Stdout Log:" in report_msg
            assert "Worker Stderr Log:" in report_msg
            
            run_root = real_output_root / "test_failed_debug"
            worker_1_dir = run_root / "batch_000001" / "workers" / "run_001"
            assert (worker_1_dir / "worker_stdout.log").exists()
            assert (worker_1_dir / "worker_stderr.log").exists()

        shutil.rmtree(real_output_root, ignore_errors=True)

    print("rts training controller dry run ok")


if __name__ == "__main__":
    main()
