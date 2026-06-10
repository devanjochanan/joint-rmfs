"""PyTorch stock-row set encoder for RTS-RL."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn


class RTSStockSetEncoder(nn.Module):
    def __init__(
        self,
        *,
        stock_input_dim: int,
        hidden_sizes: Sequence[int] = (32, 32),
        embedding_dim: int = 16,
    ) -> None:
        super().__init__()
        if int(stock_input_dim) <= 0:
            raise ValueError("stock_input_dim must be positive")
        hidden = tuple(int(size) for size in hidden_sizes)
        if not hidden or any(size <= 0 for size in hidden):
            raise ValueError("stock encoder hidden sizes must be positive")
        if int(embedding_dim) <= 0:
            raise ValueError("stock embedding_dim must be positive")
        layers: list[nn.Module] = []
        previous = int(stock_input_dim)
        for size in hidden:
            layers.append(nn.Linear(previous, size))
            layers.append(nn.ReLU())
            previous = size
        self.row_encoder = nn.Sequential(*layers)
        self.row_projection = nn.Linear(previous, int(embedding_dim))
        self.rho_projection = nn.Sequential(
            nn.Linear((2 * int(embedding_dim)) + 1, int(embedding_dim)),
            nn.ReLU(),
        )
        self.embedding_dim = int(embedding_dim)

    def forward(self, stock_rows: torch.Tensor, stock_mask: torch.Tensor) -> torch.Tensor:
        if stock_rows.ndim != 3:
            raise ValueError("stock_rows must have shape [batch, rows, features]")
        if stock_mask.ndim != 2:
            raise ValueError("stock_mask must have shape [batch, rows]")
        if stock_rows.shape[:2] != stock_mask.shape:
            raise ValueError("stock row and mask dimensions must align")
        batch_size, row_count, _width = stock_rows.shape
        if row_count:
            encoded = self.row_encoder(stock_rows.reshape(batch_size * row_count, -1))
            embeddings = self.row_projection(encoded).reshape(batch_size, row_count, self.embedding_dim)
        else:
            embeddings = stock_rows.new_zeros((batch_size, 0, self.embedding_dim))
        valid = stock_mask > 0
        counts = valid.sum(dim=1, keepdim=True).to(dtype=stock_rows.dtype)
        masked = embeddings.masked_fill(~valid.unsqueeze(-1), 0.0)
        mean_pool = masked.sum(dim=1) / counts.clamp(min=1.0)
        if row_count:
            floor = torch.finfo(embeddings.dtype).min
            max_pool = embeddings.masked_fill(~valid.unsqueeze(-1), floor).max(dim=1).values
            max_pool = torch.where(counts > 0.0, max_pool, torch.zeros_like(max_pool))
        else:
            max_pool = stock_rows.new_zeros((batch_size, self.embedding_dim))
        return self.rho_projection(torch.cat((mean_pool, max_pool, counts), dim=-1))
