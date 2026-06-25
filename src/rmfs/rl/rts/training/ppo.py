"""Synthetic PPO update helpers for RTS-RL."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .config import RTSTrainingConfig
from .metrics import mean_or_none
from .rollout_dataset import RTSPaddedTrainingBatch


@dataclass(frozen=True)
class RTSPPORolloutBatch:
    X_actions: np.ndarray
    M_actions: np.ndarray
    X_stock: np.ndarray
    M_stock: np.ndarray
    selected_action_indices: np.ndarray
    old_log_probs: np.ndarray
    old_values: np.ndarray
    rewards: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    action_feature_names: tuple[str, ...]
    stock_feature_names: tuple[str, ...]
    returns: np.ndarray
    advantages: np.ndarray


@dataclass(frozen=True)
class PPOUpdateResult:
    optimizer_steps: int
    total_loss_mean: float
    policy_loss_mean: float
    value_loss_mean: float
    entropy_mean: float
    approx_kl_mean: float | None
    clip_fraction_mean: float | None

    def to_json_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def masked_categorical_from_logits(logits: torch.Tensor, action_mask: torch.Tensor) -> torch.distributions.Categorical:
    if logits.shape != action_mask.shape:
        raise ValueError("logits and action_mask must align")
    if torch.any(action_mask.sum(dim=-1) <= 0):
        raise ValueError("all-invalid RTS action mask")
    masked_logits = logits.masked_fill(action_mask <= 0, torch.finfo(logits.dtype).min)
    return torch.distributions.Categorical(logits=masked_logits)


def compute_log_probs_values(model, padded_batch: RTSPaddedTrainingBatch | RTSPPORolloutBatch, device: str | torch.device):
    tensors = _batch_tensors(padded_batch, device)
    logits, values = model(tensors["X_actions"], tensors["M_actions"], tensors["X_stock"], tensors["M_stock"])
    dist = masked_categorical_from_logits(logits, tensors["M_actions"])
    _validate_selected(tensors["selected"], tensors["M_actions"])
    log_probs = dist.log_prob(tensors["selected"])
    entropy = dist.entropy()
    return log_probs, values, entropy


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    terminated: np.ndarray,
    truncated: np.ndarray,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    rewards = np.asarray(rewards, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    terminated = np.asarray(terminated, dtype=np.bool_)
    truncated = np.asarray(truncated, dtype=np.bool_)
    advantages = np.zeros_like(rewards, dtype=np.float32)
    last_advantage = 0.0
    for index in reversed(range(len(rewards))):
        if index == len(rewards) - 1:
            next_value = 0.0
            next_nonterminal = 0.0
        else:
            next_value = float(values[index + 1])
            next_nonterminal = 0.0 if bool(terminated[index] or truncated[index]) else 1.0
        delta = float(rewards[index]) + float(gamma) * next_value * next_nonterminal - float(values[index])
        last_advantage = delta + float(gamma) * float(gae_lambda) * next_nonterminal * last_advantage
        advantages[index] = last_advantage
    returns = advantages + values
    return advantages.astype(np.float32), returns.astype(np.float32)


def build_synthetic_ppo_smoke_batch(
    model,
    padded_batch: RTSPaddedTrainingBatch,
    device: str | torch.device,
    gamma: float,
    gae_lambda: float,
) -> RTSPPORolloutBatch:
    """This helper exists only for synthetic PPO math validation. It is not a training data path. Do not use it to train from current_probe, random_valid, heuristic, or offline rollout rows."""
    model.eval()
    with torch.no_grad():
        old_log_probs, old_values, _entropy = compute_log_probs_values(model, padded_batch, device)
    old_values_np = old_values.detach().cpu().numpy().astype(np.float32)
    advantages, returns = compute_gae(
        padded_batch.rewards,
        old_values_np,
        padded_batch.terminated,
        padded_batch.truncated,
        gamma,
        gae_lambda,
    )
    if advantages.size > 1:
        std = float(advantages.std())
        if std > 1e-8:
            advantages = ((advantages - advantages.mean()) / std).astype(np.float32)
    return RTSPPORolloutBatch(
        X_actions=padded_batch.X_actions,
        M_actions=padded_batch.M_actions,
        X_stock=padded_batch.X_stock,
        M_stock=padded_batch.M_stock,
        selected_action_indices=padded_batch.selected_action_indices,
        old_log_probs=old_log_probs.detach().cpu().numpy().astype(np.float32),
        old_values=old_values_np,
        rewards=padded_batch.rewards,
        terminated=padded_batch.terminated,
        truncated=padded_batch.truncated,
        action_feature_names=padded_batch.action_feature_names,
        stock_feature_names=padded_batch.stock_feature_names,
        returns=returns.astype(np.float32),
        advantages=advantages.astype(np.float32),
    )


def run_ppo_update(
    model,
    optimizer,
    batch: RTSPPORolloutBatch,
    config: RTSTrainingConfig,
    device: str | torch.device,
) -> PPOUpdateResult:
    model.train()
    n = int(batch.rewards.shape[0])
    indexes = np.arange(n)
    total_losses: list[float] = []
    policy_losses: list[float] = []
    value_losses: list[float] = []
    entropies: list[float] = []
    kls: list[float] = []
    clips: list[float] = []
    steps = 0
    for _epoch in range(int(config.ppo_epochs)):
        np.random.shuffle(indexes)
        for start in range(0, n, int(config.minibatch_size)):
            mb = indexes[start : start + int(config.minibatch_size)]
            mini = _slice_batch(batch, mb)
            tensors = _batch_tensors(mini, device)
            logits, values = model(tensors["X_actions"], tensors["M_actions"], tensors["X_stock"], tensors["M_stock"])
            dist = masked_categorical_from_logits(logits, tensors["M_actions"])
            _validate_selected(tensors["selected"], tensors["M_actions"])
            new_log_probs = dist.log_prob(tensors["selected"])
            ratio = torch.exp(new_log_probs - tensors["old_log_probs"])
            unclipped = ratio * tensors["advantages"]
            clipped = torch.clamp(ratio, 1.0 - config.clip_epsilon, 1.0 + config.clip_epsilon) * tensors["advantages"]
            policy_loss = -torch.min(unclipped, clipped).mean()
            value_loss = F.mse_loss(values, tensors["returns"])
            entropy = dist.entropy().mean()
            total_loss = policy_loss + config.value_loss_coef * value_loss - config.entropy_coef * entropy
            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            with torch.no_grad():
                approx_kl = (tensors["old_log_probs"] - new_log_probs).mean()
                clip_fraction = ((ratio - 1.0).abs() > config.clip_epsilon).to(torch.float32).mean()
            total_losses.append(float(total_loss.detach().cpu()))
            policy_losses.append(float(policy_loss.detach().cpu()))
            value_losses.append(float(value_loss.detach().cpu()))
            entropies.append(float(entropy.detach().cpu()))
            kls.append(float(approx_kl.detach().cpu()))
            clips.append(float(clip_fraction.detach().cpu()))
            steps += 1
    return PPOUpdateResult(
        optimizer_steps=steps,
        total_loss_mean=float(mean_or_none(total_losses) or 0.0),
        policy_loss_mean=float(mean_or_none(policy_losses) or 0.0),
        value_loss_mean=float(mean_or_none(value_losses) or 0.0),
        entropy_mean=float(mean_or_none(entropies) or 0.0),
        approx_kl_mean=mean_or_none(kls),
        clip_fraction_mean=mean_or_none(clips),
    )


def _batch_tensors(batch: RTSPaddedTrainingBatch | RTSPPORolloutBatch, device: str | torch.device) -> dict[str, torch.Tensor]:
    payload = {
        "X_actions": torch.as_tensor(batch.X_actions, dtype=torch.float32, device=device),
        "M_actions": torch.as_tensor(batch.M_actions, dtype=torch.int64, device=device),
        "X_stock": torch.as_tensor(batch.X_stock, dtype=torch.float32, device=device),
        "M_stock": torch.as_tensor(batch.M_stock, dtype=torch.int64, device=device),
        "selected": torch.as_tensor(batch.selected_action_indices, dtype=torch.int64, device=device),
    }
    if isinstance(batch, RTSPPORolloutBatch):
        payload.update(
            {
                "old_log_probs": torch.as_tensor(batch.old_log_probs, dtype=torch.float32, device=device),
                "advantages": torch.as_tensor(batch.advantages, dtype=torch.float32, device=device),
                "returns": torch.as_tensor(batch.returns, dtype=torch.float32, device=device),
            }
        )
    return payload


def _validate_selected(selected: torch.Tensor, mask: torch.Tensor) -> None:
    if torch.any(selected < 0) or torch.any(selected >= mask.shape[1]):
        raise ValueError("selected action outside action range")
    row_indexes = torch.arange(mask.shape[0], device=mask.device)
    if torch.any(mask[row_indexes, selected] <= 0):
        raise ValueError("selected action is masked invalid")


def _slice_batch(batch: RTSPPORolloutBatch, indexes: np.ndarray) -> RTSPPORolloutBatch:
    return RTSPPORolloutBatch(
        X_actions=batch.X_actions[indexes],
        M_actions=batch.M_actions[indexes],
        X_stock=batch.X_stock[indexes],
        M_stock=batch.M_stock[indexes],
        selected_action_indices=batch.selected_action_indices[indexes],
        old_log_probs=batch.old_log_probs[indexes],
        old_values=batch.old_values[indexes],
        rewards=batch.rewards[indexes],
        terminated=batch.terminated[indexes],
        truncated=batch.truncated[indexes],
        action_feature_names=batch.action_feature_names,
        stock_feature_names=batch.stock_feature_names,
        returns=batch.returns[indexes],
        advantages=batch.advantages[indexes],
    )

