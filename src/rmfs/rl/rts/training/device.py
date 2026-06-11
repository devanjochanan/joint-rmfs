"""Device auto-resolution helper for RTS-RL training/inference."""

from __future__ import annotations

import torch


def resolve_rts_torch_device(requested: str, *, prefer_cuda: bool = True) -> str:
    if requested == "auto":
        return "cuda" if prefer_cuda and torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("requested cuda but torch.cuda.is_available() is False")
    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        return "cuda"
    raise ValueError("device must be auto, cpu, or cuda")
