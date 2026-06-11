"""RTS on-policy PPO training controller spine."""

from __future__ import annotations

import datetime
import json
import subprocess
from pathlib import Path
import shutil
import sys
from typing import Any

import torch

from src.rmfs.orchestration.run_spec import RunSpec
from src.rmfs.rl.rts.training.checkpoint import load_training_checkpoint, save_training_checkpoint
from src.rmfs.rl.rts.training.config import RTSTrainingConfig
from src.rmfs.rl.rts.training.on_policy_dataset import build_on_policy_ppo_batch, build_on_policy_training_steps
from src.rmfs.rl.rts.training.policy_loader import load_policy_from_checkpoint

from .device import resolve_rts_torch_device
from .metrics import append_jsonl, atomic_write_json
from .on_policy_config import RTSOnPolicyTrainingConfig, validate_on_policy_training_config
from .progress import progress_bar, resolve_progress_enabled
from .seeding import derive_worker_seed
from .tensorboard import RTSTensorBoardLogger


def run_on_policy_training_controller(
    *,
    config: RTSOnPolicyTrainingConfig,
    repo_root: Path,
    initial_checkpoint_dir: Path | None = None,
    resume_latest: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    validate_on_policy_training_config(config, require_cycle_reference_exists=True)
    run_root = Path(config.output_root) / config.artifact_label
    run_root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(run_root / "training_config.json", config.to_json_dict())
    tb = RTSTensorBoardLogger(run_root / "tb", enabled=config.tensorboard_enabled)
    progress_enabled = resolve_progress_enabled(config.progress)
    latest = _read_latest(run_root) if resume_latest else None
    active_checkpoint_dir = Path(initial_checkpoint_dir) if initial_checkpoint_dir else None
    active_checkpoint_id = _checkpoint_id(active_checkpoint_dir) if active_checkpoint_dir else "dry_run_uninitialized"
    if latest:
        active_checkpoint_dir = Path(latest["checkpoint_dir"])
        active_checkpoint_id = _checkpoint_id(active_checkpoint_dir)
    batch_summaries = []
    for batch_id in progress_bar(range(1, config.batches + 1), enabled=progress_enabled, desc="batches"):
        batch_dir = run_root / f"batch_{batch_id:06d}"
        rollout_input = batch_dir / "rollout_input"
        workers_dir = batch_dir / "workers"
        rollout_input.mkdir(parents=True, exist_ok=True)
        workers_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(config.cycle_reference_path, rollout_input / "cycle_reference.json")
        active_ref = {
            "policy_checkpoint_dir": str(active_checkpoint_dir) if active_checkpoint_dir else None,
            "policy_checkpoint_id": active_checkpoint_id,
        }
        atomic_write_json(rollout_input / "active_checkpoint_ref.json", active_ref)
        worker_specs = []
        # Prefer explicit config.zone_ids
        if config.zone_ids:
            active_zone_ids = config.zone_ids
        else:
            # Try to load from checkpoint metadata
            active_zone_ids = _zone_ids_from_checkpoint_metadata(active_checkpoint_dir)
            # As last resort for dry-run/smoke, infer from checkpoint feature names
            if not active_zone_ids and active_checkpoint_dir:
                try:
                    active_zone_ids = _zone_ids_from_checkpoint(active_checkpoint_dir)
                except Exception:
                    active_zone_ids = None

        if not dry_run:
            if not active_zone_ids:
                raise RuntimeError("non-dry training execution requires zone_ids to be explicitly determined")
            if any(not str(z).strip() for z in active_zone_ids) or len(set(active_zone_ids)) != len(active_zone_ids):
                raise ValueError("zone_ids must be nonblank and unique for non-dry training")

        for worker_index in range(config.workers):
            run_id = f"run_{worker_index + 1:03d}"
            worker_root = workers_dir / run_id
            spec = RunSpec(
                run_id=run_id,
                ticks=config.netlogo_steps_per_run,
                runtime_root=worker_root,
                repo_root=repo_root,
                python_executable=sys.executable,
                timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                rts_policy_mode="rts_rl_explicit",
                rts_rollout_enabled=True,
                rts_reward_reference_path=str(rollout_input / "cycle_reference.json"),
                rts_random_seed=derive_worker_seed(config.seed, batch_id, worker_index),
                rts_zone_ids=list(active_zone_ids or ()),
                rts_policy_checkpoint_dir=str(active_checkpoint_dir) if active_checkpoint_dir else None,
                rts_policy_checkpoint_id=active_checkpoint_id,
                rts_policy_action_mode=config.policy_action_mode,
                rts_policy_device=config.worker_device,
            )
            worker_root.mkdir(parents=True, exist_ok=True)
            atomic_write_json(worker_root / "run_spec.json", spec.to_json_dict())
            worker_specs.append(spec.to_json_dict())
        if dry_run:
            summary = {
                "batch_id": batch_id,
                "status": "dry_run",
                "netlogo_steps_per_run": config.netlogo_steps_per_run,
                "workers": config.workers,
                "active_checkpoint_id": active_checkpoint_id,
                "worker_specs_written": len(worker_specs),
                "trainable_step_count": 0,
                "ppo_update": None,
                "latest_updated": False,
                "zone_ids": list(active_zone_ids or ()),
            }
            atomic_write_json(batch_dir / "batch_summary.json", summary)
            append_jsonl(run_root / "training_events.jsonl", {"event_type": "batch_dry_run", **summary})
            append_jsonl(run_root / "batch_metrics.jsonl", summary)
            tb.log_scalars(
                {
                    "time/netlogo_steps_per_run": config.netlogo_steps_per_run,
                    "rollout/trainable_step_count": 0,
                    "checkpoint/batch_id": batch_id,
                    "checkpoint/latest_updated": 0,
                },
                batch_id,
            )
            batch_summaries.append(summary)
            continue

        if active_checkpoint_dir is None:
            raise RuntimeError("non-dry on-policy training requires an active checkpoint")
        for spec in worker_specs:
            subprocess.check_call(
                [
                    sys.executable,
                    "-m",
                    "src.rmfs.orchestration.local_executor",
                    "worker",
                    "--spec",
                    str(Path(spec["runtime_root"]) / "run_spec.json"),
                ],
                cwd=repo_root,
            )
        events = _load_rollout_shards(workers_dir)
        dataset = build_on_policy_training_steps(events, required_policy_checkpoint_id=active_checkpoint_id)
        if dataset.summary["trainable_step_count"] < config.min_trainable_steps:
            summary = {
                "batch_id": batch_id,
                "status": "skipped_insufficient_trainable_steps",
                "active_checkpoint_id": active_checkpoint_id,
                "dataset_summary": dataset.summary,
                "latest_updated": False,
                "zone_ids": list(active_zone_ids or ()),
            }
            atomic_write_json(batch_dir / "batch_summary.json", summary)
            append_jsonl(run_root / "training_events.jsonl", {"event_type": "batch_skipped", **summary})
            append_jsonl(run_root / "batch_metrics.jsonl", summary)
            batch_summaries.append(summary)
            continue

        loaded = load_policy_from_checkpoint(active_checkpoint_dir, device=_controller_device(config.device))
        optimizer = torch.optim.Adam(loaded.model.parameters(), lr=_learning_rate(loaded.metadata))
        load_training_checkpoint(active_checkpoint_dir, model=loaded.model, optimizer=optimizer, device=_controller_device(config.device))
        train_config = _ppo_training_config(config, loaded.metadata)
        ppo_batch = build_on_policy_ppo_batch(dataset, gamma=train_config.gamma, gae_lambda=train_config.gae_lambda)
        from src.rmfs.rl.rts.training.ppo import run_ppo_update

        update_result = run_ppo_update(loaded.model, optimizer, ppo_batch, train_config, _controller_device(config.device))
        checkpoint_dir = save_training_checkpoint(
            model=loaded.model,
            optimizer=optimizer,
            config=train_config,
            batch_id=batch_id,
            dataset_summary=dataset.summary,
            ppo_update_result=update_result,
            action_feature_names=ppo_batch.action_feature_names,
            stock_feature_names=ppo_batch.stock_feature_names,
            cycle_reference_path=rollout_input / "cycle_reference.json",
        )
        active_checkpoint_dir = checkpoint_dir
        active_checkpoint_id = _checkpoint_id(checkpoint_dir)
        summary = {
            "batch_id": batch_id,
            "status": "updated",
            "netlogo_steps_per_run": config.netlogo_steps_per_run,
            "workers": config.workers,
            "active_checkpoint_id": active_checkpoint_id,
            "worker_specs_written": len(worker_specs),
            "trainable_step_count": dataset.summary["trainable_step_count"],
            "dataset_summary": dataset.summary,
            "ppo_update": update_result.to_json_dict(),
            "checkpoint_dir": str(checkpoint_dir),
            "latest_updated": True,
            "zone_ids": list(active_zone_ids or ()),
        }
        atomic_write_json(batch_dir / "batch_summary.json", summary)
        append_jsonl(run_root / "training_events.jsonl", {"event_type": "batch_updated", **summary})
        append_jsonl(run_root / "batch_metrics.jsonl", summary)
        tb.log_scalars(
            {
                "train/ppo_total_loss": update_result.total_loss_mean,
                "train/policy_loss": update_result.policy_loss_mean,
                "train/value_loss": update_result.value_loss_mean,
                "train/entropy": update_result.entropy_mean,
                "train/approx_kl": update_result.approx_kl_mean,
                "train/clip_fraction": update_result.clip_fraction_mean,
                "train/optimizer_steps": update_result.optimizer_steps,
                "time/netlogo_steps_per_run": config.netlogo_steps_per_run,
                "rollout/trainable_step_count": dataset.summary["trainable_step_count"],
                "rollout/decision_count": dataset.summary["decision_count"],
                "rollout/outcome_count": dataset.summary["outcome_count"],
                "rollout/rejected_non_on_policy_count": dataset.summary["rejected_non_on_policy_count"],
                "rollout/rejected_checkpoint_mismatch_count": dataset.summary["rejected_checkpoint_mismatch_count"],
                "rollout/avg_reward": dataset.summary["avg_reward"],
                "checkpoint/batch_id": batch_id,
                "checkpoint/latest_updated": 1,
            },
            batch_id,
        )
        batch_summaries.append(summary)
    tb.close()
    if dry_run:
        status = "dry_run"
    else:
        updated_any = any(b["status"] == "updated" for b in batch_summaries)
        if updated_any:
            status = "completed"
        else:
            status = "completed_with_skips"

    return {
        "status": status,
        "run_root": str(run_root),
        "batches": batch_summaries,
    }


def _read_latest(run_root: Path) -> dict[str, Any] | None:
    path = run_root / "latest.json"
    if not path.exists():
        return None
    with path.open() as fh:
        return json.load(fh)


def _checkpoint_id(checkpoint_dir: Path | None) -> str:
    if checkpoint_dir is None:
        return "dry_run_uninitialized"
    parent = checkpoint_dir.parent
    return parent.name if parent.name.startswith("batch_") else checkpoint_dir.name


def _load_rollout_shards(workers_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(Path(workers_dir).glob("run_*/rts_rollout.jsonl")):
        with path.open() as fh:
            for line in fh:
                if line.strip():
                    rows.append(json.loads(line))
    return rows


def _zone_ids_from_checkpoint(checkpoint_dir: Path | None) -> tuple[str, ...] | None:
    if checkpoint_dir is None:
        return None
    loaded = load_policy_from_checkpoint(checkpoint_dir, device="cpu")
    names = loaded.feature_schema.get("action_feature_names", []) or []
    zones = [
        name.removeprefix("next_retrieval_zone_one_hot__")
        for name in names
        if str(name).startswith("next_retrieval_zone_one_hot__")
    ]
    if not zones:
        raise RuntimeError("could not infer RTS zone_ids from checkpoint feature schema")
    return tuple(zones)


def _controller_device(device: str) -> str:
    return resolve_rts_torch_device(device)


def _zone_ids_from_checkpoint_metadata(checkpoint_dir: Path | None) -> tuple[str, ...] | None:
    if checkpoint_dir is None:
        return None
    try:
        with (checkpoint_dir / "metadata.json").open() as fh:
            metadata = json.load(fh)
        config_data = metadata.get("training_config", {}) or {}
        zones = config_data.get("zone_ids")
        if zones:
            return tuple(str(z) for z in zones)
    except Exception:
        pass
    return None


def _learning_rate(metadata: dict[str, Any]) -> float:
    return float((metadata.get("training_config") or {}).get("learning_rate", 1e-4))


def _ppo_training_config(config: RTSOnPolicyTrainingConfig, metadata: dict[str, Any]) -> RTSTrainingConfig:
    base = metadata.get("training_config") or {}
    return RTSTrainingConfig(
        artifact_label=config.artifact_label,
        output_root=config.output_root,
        seed=config.seed,
        learning_rate=float(base.get("learning_rate", 1e-4)),
        gamma=float(base.get("gamma", 0.99)),
        gae_lambda=float(base.get("gae_lambda", 0.95)),
        ppo_epochs=config.ppo_epochs,
        minibatch_size=config.minibatch_size,
        hidden_sizes=tuple(base.get("hidden_sizes", (64, 64))),
        stock_hidden_sizes=tuple(base.get("stock_hidden_sizes", (32, 32))),
        stock_embedding_dim=int(base.get("stock_embedding_dim", 16)),
        tensorboard_enabled=config.tensorboard_enabled,
    )
