"""Lightweight runtime timing helpers for RMFS.

Timing is disabled by default. When enabled, sections accumulate compact
wall-clock totals that can be written as JSON at the end of a smoke or worker
run.
"""

from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Iterator


TRUE_VALUES = {"1", "true", "yes", "on"}


def _env_enabled() -> bool:
    return os.environ.get("RMFS_TIMING", "").strip().lower() in TRUE_VALUES


@dataclass
class TimingRecorder:
    enabled: bool = False
    output_path: Path | None = None
    sections: dict[str, dict[str, float | int]] = field(default_factory=dict)
    counters: dict[str, int] = field(default_factory=dict)
    _lock: Lock = field(default_factory=Lock, repr=False)

    @contextmanager
    def section(self, name: str) -> Iterator[None]:
        if not self.enabled:
            yield
            return
        start = time.perf_counter()
        try:
            yield
        finally:
            self.record(name, time.perf_counter() - start)

    def record(self, name: str, seconds: float) -> None:
        if not self.enabled:
            return
        with self._lock:
            item = self.sections.setdefault(name, {"count": 0, "seconds": 0.0})
            item["count"] = int(item["count"]) + 1
            item["seconds"] = float(item["seconds"]) + float(seconds)

    def increment(self, name: str, value: int = 1) -> None:
        if not self.enabled:
            return
        with self._lock:
            self.counters[name] = self.counters.get(name, 0) + int(value)

    def summary(self) -> dict[str, object]:
        sections = {}
        with self._lock:
            for name, item in sorted(self.sections.items()):
                count = int(item["count"])
                seconds = float(item["seconds"])
                sections[name] = {
                    "count": count,
                    "seconds": seconds,
                    "avg_seconds": seconds / count if count else 0.0,
                }
            counters = dict(sorted(self.counters.items()))
        return {
            "enabled": self.enabled,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "sections": sections,
            "counters": counters,
        }

    def write(self, path: str | Path | None = None) -> Path | None:
        target = Path(path) if path is not None else self.output_path
        if target is None:
            return None
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.summary(), indent=2), encoding="utf-8")
        return target


_RECORDER = TimingRecorder(enabled=_env_enabled())


def configure_timing(
    enabled: bool | None = None,
    output_path: str | Path | None = None,
) -> TimingRecorder:
    """Configure the process-wide timing recorder."""

    global _RECORDER
    active = _env_enabled() if enabled is None else bool(enabled)
    target = Path(output_path) if output_path is not None else None
    _RECORDER = TimingRecorder(enabled=active, output_path=target)
    return _RECORDER


def get_timing_recorder() -> TimingRecorder:
    return _RECORDER


def is_timing_enabled() -> bool:
    return _RECORDER.enabled


@contextmanager
def timed(name: str) -> Iterator[None]:
    with _RECORDER.section(name):
        yield


def increment_counter(name: str, value: int = 1) -> None:
    _RECORDER.increment(name, value)


def write_timing_summary(path: str | Path | None = None) -> Path | None:
    return _RECORDER.write(path)
