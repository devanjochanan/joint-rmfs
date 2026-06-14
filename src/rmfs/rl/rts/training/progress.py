"""TQDM progress helpers for RTS training controller code."""

from __future__ import annotations

import sys
import time
from typing import Iterable

from tqdm.auto import tqdm


def resolve_progress_enabled(progress: bool | None) -> bool:
    if progress is None:
        return bool(sys.stdout.isatty())
    return bool(progress)


def progress_bar(iterable: Iterable, *, enabled: bool, **kwargs):
    if not enabled:
        return iterable
    return tqdm(iterable, **kwargs)


class RTSTrainingProgressBar:
    def __init__(
        self,
        *,
        enabled: bool,
        batch_id: int,
        workers: int,
        progress_target: int,
        latest_checkpoint_id: str,
        update_time: float = 0.0,
    ) -> None:
        self.enabled = enabled
        self.batch_id = batch_id
        self.workers = workers
        self.progress_target = progress_target
        self.latest_checkpoint_id = latest_checkpoint_id
        self.update_time = update_time
        self.start_time = time.perf_counter()
        
        self.bar = None
        if self.enabled:
            try:
                from tqdm import tqdm
                self.bar = tqdm(
                    total=self.progress_target,
                    desc=f"[batch {self.batch_id:06d}]",
                    unit="step",
                    file=getattr(sys, "__stderr__", sys.stderr),
                    dynamic_ncols=True,
                    leave=True,
                )
            except Exception:
                self.bar = None

    def update_live(
        self,
        *,
        progress_current: int,
        workers_done: int,
        worker_failed_count: int,
        wait_time: float,
        straggler_ratio: float,
    ) -> None:
        if not self.enabled:
            return
        
        elapsed = time.perf_counter() - self.start_time
        steps_per_sec = progress_current / elapsed if elapsed > 0.0 else 0.0
        
        postfix = (
            f"batch={self.batch_id} | "
            f"workers={workers_done}/{self.workers} | "
            f"fail={worker_failed_count} | "
            f"prog={progress_current}/{self.progress_target} | "
            f"step/s={steps_per_sec:.1f} | "
            f"upd={self.update_time:.1f}s | "
            f"wait={wait_time:.1f}s | "
            f"strag={straggler_ratio:.2f}x | "
            f"chk={self.latest_checkpoint_id}"
        )
        
        if self.bar is not None:
            self.bar.n = min(progress_current, self.progress_target)
            self.bar.set_postfix_str(postfix)
            self.bar.refresh()

    def close(self) -> None:
        if self.bar is not None:
            self.bar.close()
        self.bar = None
