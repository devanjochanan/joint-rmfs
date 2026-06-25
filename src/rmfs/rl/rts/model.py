"""Masked RTS-RL actor-critic model architecture."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

from .stock_encoder import RTSStockSetEncoder


class RTSMaskedActorCritic(nn.Module):
    def __init__(
        self,
        *,
        action_feature_dim: int,
        stock_feature_dim: int,
        hidden_sizes: Sequence[int] = (64, 64),
        stock_hidden_sizes: Sequence[int] = (32, 32),
        stock_embedding_dim: int = 16,
    ) -> None:
        super().__init__()
        if int(action_feature_dim) <= 0:
            raise ValueError("action_feature_dim must be positive")
        if int(stock_feature_dim) <= 0:
            raise ValueError("stock_feature_dim must be positive")
        hidden = tuple(int(size) for size in hidden_sizes)
        if not hidden or any(size <= 0 for size in hidden):
            raise ValueError("hidden_sizes must be positive")
        self.stock_encoder = RTSStockSetEncoder(
            stock_input_dim=int(stock_feature_dim),
            hidden_sizes=stock_hidden_sizes,
            embedding_dim=stock_embedding_dim,
        )
        layers: list[nn.Module] = []
        previous = int(action_feature_dim) + int(stock_embedding_dim)
        for size in hidden:
            layers.append(nn.Linear(previous, size))
            layers.append(nn.ReLU())
            previous = size
        self.row_encoder = nn.Sequential(*layers)
        self.policy_logits_head = nn.Linear(previous, 1)
        self.value_head = nn.Linear(previous, 1)

    def forward(
        self,
        action_features: torch.Tensor,
        action_mask: torch.Tensor,
        stock_features: torch.Tensor,
        stock_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if action_features.ndim != 3:
            raise ValueError("action_features must have shape [batch, actions, features]")
        if action_mask.ndim != 2:
            raise ValueError("action_mask must have shape [batch, actions]")
        if stock_features.ndim != 3:
            raise ValueError("stock_features must have shape [batch, rows, features]")
        if stock_mask.ndim != 2:
            raise ValueError("stock_mask must have shape [batch, rows]")
        if action_features.shape[:2] != action_mask.shape:
            raise ValueError("action feature and mask dimensions must align")
        if stock_features.shape[:2] != stock_mask.shape:
            raise ValueError("stock feature and mask dimensions must align")
        if action_features.shape[0] != stock_features.shape[0]:
            raise ValueError("action and stock batch dimensions must align")
        batch, actions, _width = action_features.shape
        stock_embedding = self.stock_encoder(stock_features, stock_mask)
        expanded_stock = stock_embedding.unsqueeze(1).expand(-1, actions, -1)
        row_inputs = torch.cat((action_features, expanded_stock), dim=-1)
        encoded = self.row_encoder(row_inputs.reshape(batch * actions, -1)).reshape(batch, actions, -1)
        logits = self.policy_logits_head(encoded).squeeze(-1)
        valid = action_mask > 0
        masked_encoded = encoded * valid.unsqueeze(-1).to(dtype=encoded.dtype)
        counts = valid.sum(dim=1, keepdim=True).clamp(min=1).to(dtype=encoded.dtype)
        pooled = masked_encoded.sum(dim=1) / counts
        values = self.value_head(pooled).squeeze(-1)
        return logits, values


def build_rts_actor_critic_model(
    *,
    action_feature_dim: int,
    stock_feature_dim: int,
    hidden_sizes: Sequence[int] = (64, 64),
) -> RTSMaskedActorCritic:
    return RTSMaskedActorCritic(
        action_feature_dim=action_feature_dim,
        stock_feature_dim=stock_feature_dim,
        hidden_sizes=hidden_sizes,
    )
