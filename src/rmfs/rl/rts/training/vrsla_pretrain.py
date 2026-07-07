"""VRSLA teacher-row loading, behavior cloning, and checkpoint save helpers."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from src.rmfs.rl.rts.action_space import decode_action
from src.rmfs.rl.rts.model import RTSMaskedActorCritic
from src.rmfs.rl.rts.stock_features import STOCK_FEATURE_NAMES
from src.rmfs.rl.rts.training.checkpoint import load_training_checkpoint, save_training_checkpoint
from src.rmfs.rl.rts.training.config import RTSTrainingConfig
from src.rmfs.rl.rts.training.metrics import atomic_write_json
from src.rmfs.rl.rts.training.policy_loader import load_policy_from_checkpoint


@dataclass(frozen=True)
class VRSLATeacherBatch:
    X_actions: np.ndarray
    M_actions: np.ndarray
    X_stock: np.ndarray
    M_stock: np.ndarray
    teacher_action_indices: np.ndarray
    action_feature_names: tuple[str, ...]
    stock_feature_names: tuple[str, ...]
    zone_ids_by_row: tuple[tuple[str, ...], ...]
    teacher_branches: tuple[str, ...]
    teacher_zone_ids: tuple[str, ...]
    run_keys: tuple[str, ...]
    rows: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class VRSLABehaviorCloningResult:
    epochs_completed: int
    best_epoch: int
    train_loss_initial: float
    train_loss_final: float
    validation_loss_best: float
    overall_action_agreement: float
    zone_agreement: float
    branch_agreement: float
    invalid_action_rate: float

    def to_json_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def load_teacher_rows(paths: Sequence[Path]) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for path in sorted(Path(p) for p in paths):
        if not path.exists():
            continue
        with path.open() as fh:
            for line in fh:
                if line.strip():
                    rows.append(json.loads(line))
    return tuple(rows)


def teacher_batch_from_rows(rows: Sequence[Mapping[str, Any]]) -> VRSLATeacherBatch:
    payloads = [dict(row) for row in rows]
    if not payloads:
        raise ValueError("VRSLA behavior cloning requires at least one teacher row")
    action_names = tuple(str(name) for name in payloads[0].get("action_feature_names", []) or ())
    stock_names = tuple(str(name) for name in payloads[0].get("stock_feature_names", []) or ())
    if not action_names:
        raise ValueError("teacher rows are missing action_feature_names")
    if not stock_names:
        stock_names = tuple(STOCK_FEATURE_NAMES)
    for row in payloads:
        if tuple(str(name) for name in row.get("action_feature_names", []) or ()) != action_names:
            raise ValueError("teacher action feature schemas must match")
        if tuple(str(name) for name in row.get("stock_feature_names", []) or ()) not in {stock_names, ()}:
            raise ValueError("teacher stock feature schemas must match")
    action_arrays = [np.asarray(row.get("X_actions"), dtype=np.float32) for row in payloads]
    action_masks = [np.asarray(row.get("M_actions"), dtype=np.int64) for row in payloads]
    stock_arrays = [np.asarray(row.get("X_stock"), dtype=np.float32) for row in payloads]
    stock_masks = [np.asarray(row.get("M_stock"), dtype=np.int64) for row in payloads]
    max_actions = max(arr.shape[0] for arr in action_arrays)
    max_stock = max((arr.shape[0] for arr in stock_arrays), default=1)
    action_dim = len(action_names)
    stock_dim = len(stock_names)
    n = len(payloads)
    X_actions = np.zeros((n, max_actions, action_dim), dtype=np.float32)
    M_actions = np.zeros((n, max_actions), dtype=np.int64)
    X_stock = np.zeros((n, max(1, max_stock), stock_dim), dtype=np.float32)
    M_stock = np.zeros((n, max(1, max_stock)), dtype=np.int64)
    selected = np.zeros((n,), dtype=np.int64)
    zone_ids_by_row: list[tuple[str, ...]] = []
    branches: list[str] = []
    zones: list[str] = []
    run_keys: list[str] = []
    for index, row in enumerate(payloads):
        xa = action_arrays[index]
        ma = action_masks[index]
        xs = stock_arrays[index]
        ms = stock_masks[index]
        if xa.ndim != 2 or xa.shape[1] != action_dim:
            raise ValueError("teacher X_actions row has wrong shape")
        if ma.ndim != 1 or ma.shape[0] != xa.shape[0]:
            raise ValueError("teacher M_actions row has wrong shape")
        if xs.ndim != 2 or xs.shape[1] != stock_dim:
            raise ValueError("teacher X_stock row has wrong shape")
        if ms.ndim != 1 or ms.shape[0] != xs.shape[0]:
            raise ValueError("teacher M_stock row has wrong shape")
        teacher_index = int(row.get("teacher_action_index"))
        if teacher_index < 0 or teacher_index >= ma.shape[0] or int(ma[teacher_index]) <= 0:
            raise ValueError("teacher_action_index must be valid under M_actions")
        X_actions[index, : xa.shape[0], :] = xa
        M_actions[index, : ma.shape[0]] = ma
        if xs.shape[0]:
            X_stock[index, : xs.shape[0], :] = xs
            M_stock[index, : ms.shape[0]] = ms
        selected[index] = teacher_index
        row_zones = tuple(str(zone) for zone in row.get("zone_ids", []) or ())
        zone_ids_by_row.append(row_zones)
        branches.append(str(row.get("teacher_branch", "")))
        zones.append(str(row.get("teacher_zone_id", "")))
        run_keys.append(_run_key(row))
    return VRSLATeacherBatch(
        X_actions=X_actions,
        M_actions=M_actions,
        X_stock=X_stock,
        M_stock=M_stock,
        teacher_action_indices=selected,
        action_feature_names=action_names,
        stock_feature_names=stock_names,
        zone_ids_by_row=tuple(zone_ids_by_row),
        teacher_branches=tuple(branches),
        teacher_zone_ids=tuple(zones),
        run_keys=tuple(run_keys),
        rows=tuple(payloads),
    )


def run_behavior_cloning(
    batch: VRSLATeacherBatch,
    *,
    config: RTSTrainingConfig,
    device: str | torch.device = "cpu",
    max_epochs: int = 50,
    patience: int = 5,
) -> tuple[RTSMaskedActorCritic, VRSLABehaviorCloningResult]:
    device = torch.device(device)
    model = RTSMaskedActorCritic(
        action_feature_dim=batch.X_actions.shape[-1],
        stock_feature_dim=batch.X_stock.shape[-1],
        hidden_sizes=config.hidden_sizes,
        stock_hidden_sizes=config.stock_hidden_sizes,
        stock_embedding_dim=config.stock_embedding_dim,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config.learning_rate))
    train_idx, val_idx = _split_by_run(batch.run_keys)
    tensors = _batch_tensors(batch, device)
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    best_val = float("inf")
    best_epoch = 0
    bad_epochs = 0
    train_initial = None
    train_final = None
    for epoch in range(1, int(max_epochs) + 1):
        model.train()
        loss = _masked_ce_loss(model, tensors, train_idx)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.max_grad_norm))
        optimizer.step()
        train_loss = float(loss.detach().cpu())
        if train_initial is None:
            train_initial = train_loss
        train_final = train_loss
        model.eval()
        with torch.no_grad():
            val_loss = float(_masked_ce_loss(model, tensors, val_idx).detach().cpu())
        if val_loss + 1e-8 < best_val:
            best_val = val_loss
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= int(patience):
                break
    model.load_state_dict(best_state)
    metrics = _agreement_metrics(model, batch, tensors)
    result = VRSLABehaviorCloningResult(
        epochs_completed=epoch,
        best_epoch=best_epoch,
        train_loss_initial=float(train_initial if train_initial is not None else 0.0),
        train_loss_final=float(train_final if train_final is not None else 0.0),
        validation_loss_best=float(best_val),
        **metrics,
    )
    return model, result


def save_behavior_cloning_checkpoint(
    *,
    model: RTSMaskedActorCritic,
    config: RTSTrainingConfig,
    batch_id: int,
    teacher_batch: VRSLATeacherBatch,
    result: VRSLABehaviorCloningResult,
    teacher_batch_count: int = 10,
) -> Path:
    fresh_ppo_optimizer = torch.optim.Adam(model.parameters(), lr=float(config.learning_rate))
    dataset_summary = {
        "teacher_row_count": len(teacher_batch.rows),
        "trainable_step_count": 0,
        "avg_reward": 0.0,
        "teacher_batch_count": int(teacher_batch_count),
        "supervised_target": "teacher_action_index",
    }
    checkpoint_dir = save_training_checkpoint(
        model=model,
        optimizer=fresh_ppo_optimizer,
        config=config,
        batch_id=batch_id,
        dataset_summary=dataset_summary,
        ppo_update_result={"behavior_cloning": result.to_json_dict()},
        action_feature_names=teacher_batch.action_feature_names,
        stock_feature_names=teacher_batch.stock_feature_names,
        lineage_metadata={
            "initialization_method": "vrsla_behavior_cloning",
            "teacher_policy": "vrsla_event_driven",
            "teacher_batch_count": int(teacher_batch_count),
            "pod_velocity_semantics": "distinct_order_union",
            "slot_score_semantics": "directed_loaded_cycle_distance",
            "replenishment_teacher_rule": "seeded_50_50_when_both_masked_valid",
            "critic_pretrained": False,
            **_score_identity_summary(teacher_batch.rows),
        },
    )
    metadata_path = checkpoint_dir / "metadata.json"
    with metadata_path.open() as fh:
        metadata = json.load(fh)
    metadata.update(
        {
            "initialization_method": "vrsla_behavior_cloning",
            "teacher_policy": "vrsla_event_driven",
            "teacher_batch_count": int(teacher_batch_count),
            "pod_velocity_semantics": "distinct_order_union",
            "slot_score_semantics": "directed_loaded_cycle_distance",
            "replenishment_teacher_rule": "seeded_50_50_when_both_masked_valid",
            "critic_pretrained": False,
            "behavior_cloning_result": result.to_json_dict(),
            **_score_identity_summary(teacher_batch.rows),
        }
    )
    atomic_write_json(metadata_path, metadata)
    loaded = load_policy_from_checkpoint(checkpoint_dir, device="cpu")
    verifier = torch.optim.Adam(loaded.model.parameters(), lr=float(config.learning_rate))
    load_training_checkpoint(checkpoint_dir, model=loaded.model, optimizer=verifier, device="cpu")
    return checkpoint_dir


def _batch_tensors(batch: VRSLATeacherBatch, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "X_actions": torch.as_tensor(batch.X_actions, dtype=torch.float32, device=device),
        "M_actions": torch.as_tensor(batch.M_actions, dtype=torch.int64, device=device),
        "X_stock": torch.as_tensor(batch.X_stock, dtype=torch.float32, device=device),
        "M_stock": torch.as_tensor(batch.M_stock, dtype=torch.int64, device=device),
        "selected": torch.as_tensor(batch.teacher_action_indices, dtype=torch.int64, device=device),
    }


def _masked_ce_loss(model: RTSMaskedActorCritic, tensors: Mapping[str, torch.Tensor], indexes: np.ndarray) -> torch.Tensor:
    idx = torch.as_tensor(indexes, dtype=torch.long, device=tensors["X_actions"].device)
    logits, _values = model(
        tensors["X_actions"][idx],
        tensors["M_actions"][idx],
        tensors["X_stock"][idx],
        tensors["M_stock"][idx],
    )
    selected = tensors["selected"][idx]
    mask = tensors["M_actions"][idx]
    if torch.any(mask[torch.arange(mask.shape[0], device=mask.device), selected] <= 0):
        raise ValueError("teacher target is masked invalid")
    masked_logits = logits.masked_fill(mask <= 0, torch.finfo(logits.dtype).min)
    return F.cross_entropy(masked_logits, selected)


def _split_by_run(run_keys: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    groups = sorted(set(str(key) for key in run_keys))
    indexes_by_group = {group: [idx for idx, key in enumerate(run_keys) if str(key) == group] for group in groups}
    if len(groups) < 2:
        indexes = np.arange(len(run_keys), dtype=np.int64)
        return indexes, indexes
    val_count = max(1, len(groups) // 5)
    val_groups = set(groups[-val_count:])
    train = [idx for group in groups if group not in val_groups for idx in indexes_by_group[group]]
    val = [idx for group in groups if group in val_groups for idx in indexes_by_group[group]]
    return np.asarray(train, dtype=np.int64), np.asarray(val, dtype=np.int64)


def _agreement_metrics(model: RTSMaskedActorCritic, batch: VRSLATeacherBatch, tensors: Mapping[str, torch.Tensor]) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        logits, _values = model(tensors["X_actions"], tensors["M_actions"], tensors["X_stock"], tensors["M_stock"])
        masked_logits = logits.masked_fill(tensors["M_actions"] <= 0, torch.finfo(logits.dtype).min)
        pred = masked_logits.argmax(dim=-1).detach().cpu().numpy().astype(np.int64)
    selected = batch.teacher_action_indices
    invalid = [
        1.0 if batch.M_actions[index, pred[index]] <= 0 else 0.0
        for index in range(len(pred))
    ]
    zones_ok = []
    branches_ok = []
    for index, predicted in enumerate(pred):
        action = decode_action(int(predicted), batch.zone_ids_by_row[index])
        zones_ok.append(1.0 if action.zone_id == batch.teacher_zone_ids[index] else 0.0)
        branches_ok.append(1.0 if action.branch == batch.teacher_branches[index] else 0.0)
    return {
        "overall_action_agreement": float(np.mean(pred == selected)) if len(selected) else 0.0,
        "zone_agreement": float(np.mean(zones_ok)) if zones_ok else 0.0,
        "branch_agreement": float(np.mean(branches_ok)) if branches_ok else 0.0,
        "invalid_action_rate": float(np.mean(invalid)) if invalid else 0.0,
    }


def _run_key(row: Mapping[str, Any]) -> str:
    for key in ("run_id", "seed", "worker", "decision_event_id"):
        value = row.get(key)
        if value is not None:
            return f"{key}:{value}"
    return "unknown"


def _score_identity_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    keys = (
        "pod_velocity_score_cache_identity",
        "historical_source_hash",
        "pod_sku_allocation_identity",
        "slot_score_cache_identity",
    )
    summary = {}
    for key in keys:
        values = sorted({str(row.get(key)) for row in rows if row.get(key) is not None})
        summary[key] = values[0] if len(values) == 1 else values
    return summary
