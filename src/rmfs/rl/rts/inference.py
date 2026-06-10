"""Inference helpers for masked RTS-RL action selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from .action_space import decode_action, validate_action_mask


@dataclass(frozen=True)
class RTSInferenceResult:
    selected_action_index: int
    action: Any
    logits: list[float]
    probabilities: list[float]
    value_estimate: float


def resolve_torch_device(device: str = "auto") -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def masked_logits(logits: torch.Tensor, action_mask: torch.Tensor, floor: float = -1.0e9) -> torch.Tensor:
    if logits.shape != action_mask.shape:
        raise ValueError("logits and action_mask must have matching shapes")
    if not bool((action_mask > 0).any().detach().cpu().item()):
        raise ValueError("cannot mask logits: action mask has no valid actions")
    return logits.masked_fill(action_mask <= 0, floor)


def masked_softmax(logits: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
    return torch.softmax(masked_logits(logits, action_mask), dim=-1)


def sample_valid_action(logits: torch.Tensor, action_mask: torch.Tensor, generator: torch.Generator | None = None) -> int:
    probs = masked_softmax(logits, action_mask)
    if probs.ndim != 1:
        raise ValueError("sample_valid_action expects 1D logits")
    return int(torch.multinomial(probs, num_samples=1, generator=generator).item())


def select_greedy_action(logits: torch.Tensor, action_mask: torch.Tensor) -> int:
    masked = masked_logits(logits, action_mask)
    if masked.ndim != 1:
        raise ValueError("select_greedy_action expects 1D logits")
    return int(torch.argmax(masked).item())


def run_inference(
    model,
    *,
    action_features: np.ndarray,
    action_mask: np.ndarray,
    stock_features: np.ndarray,
    stock_mask: np.ndarray,
    zone_ids: tuple[str, ...],
    mode: str = "greedy",
    device: str = "auto",
) -> RTSInferenceResult:
    validate_action_mask(zone_ids, action_mask, require_valid=True)
    torch_device = resolve_torch_device(device)
    model = model.to(torch_device)
    model.eval()
    action_tensor = torch.as_tensor(action_features, dtype=torch.float32, device=torch_device).unsqueeze(0)
    action_mask_tensor = torch.as_tensor(action_mask, dtype=torch.int64, device=torch_device).unsqueeze(0)
    stock_tensor = torch.as_tensor(stock_features, dtype=torch.float32, device=torch_device).unsqueeze(0)
    stock_mask_tensor = torch.as_tensor(stock_mask, dtype=torch.int64, device=torch_device).unsqueeze(0)
    with torch.no_grad():
        logits, values = model(action_tensor, action_mask_tensor, stock_tensor, stock_mask_tensor)
        logits_row = logits.squeeze(0)
        mask_row = action_mask_tensor.squeeze(0)
        probs = masked_softmax(logits_row, mask_row)
        selected = sample_valid_action(logits_row, mask_row) if mode == "sample" else select_greedy_action(logits_row, mask_row)
    return RTSInferenceResult(
        selected_action_index=selected,
        action=decode_action(selected, zone_ids),
        logits=[float(x) for x in logits_row.detach().cpu().tolist()],
        probabilities=[float(x) for x in probs.detach().cpu().tolist()],
        value_estimate=float(values.squeeze(0).detach().cpu().item()),
    )
