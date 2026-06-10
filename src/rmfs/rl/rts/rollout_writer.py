"""Worker-local JSONL writer for RTS rollout events."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


class RTSRolloutWriter:
    def __init__(self, path: Path, *, enabled: bool = True, max_events: int | None = None):
        self.path = Path(path)
        self.enabled = bool(enabled)
        self.max_events = max_events
        self.events: list[dict[str, Any]] = []
        self._written = 0
        self._closed = False
        self._fh = None

    def write_decision(self, row: Mapping[str, Any]) -> None:
        self._write(row)

    def write_outcome(self, row: Mapping[str, Any]) -> None:
        self._write(row)

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None
        self._closed = True

    def _write(self, row: Mapping[str, Any]) -> None:
        if not self.enabled:
            return
        if self.max_events is not None and self._written >= int(self.max_events):
            return
        if self._closed:
            raise RuntimeError("cannot write RTS rollout event after writer close")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self._fh is None:
            self._fh = self.path.open("a")
        event = dict(row)
        self._fh.write(json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n")
        self._fh.flush()
        self.events.append(event)
        self._written += 1

