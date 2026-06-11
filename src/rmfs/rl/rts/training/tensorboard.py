"""Optional TensorBoard logging for RTS training controller metrics."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping


class RTSTensorBoardLogger:
    def __init__(self, log_dir: Path, *, enabled: bool = False):
        self.enabled = bool(enabled)
        self.writer = None
        if self.enabled:
            try:
                from torch.utils.tensorboard import SummaryWriter
            except Exception as exc:
                raise RuntimeError("TensorBoard requested but SummaryWriter import failed") from exc
            self.writer = SummaryWriter(log_dir=str(log_dir))

    def log_scalars(self, scalars: Mapping[str, Any], step: int) -> None:
        if not self.writer:
            return
        for key, value in scalars.items():
            if value is None:
                continue
            try:
                self.writer.add_scalar(key, float(value), int(step))
            except Exception:
                continue

    def close(self) -> None:
        if self.writer:
            self.writer.close()

