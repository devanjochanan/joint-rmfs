"""Worker-local JSONL writer for RTS rollout events."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping


class RTSRolloutWriter:
    def __init__(self, path: Path, *, enabled: bool = True, max_events: int | None = None):
        self.path = Path(path)
        self.enabled = bool(enabled)
        self.max_events = max_events
        self.events: list[dict[str, Any]] = []
        self._written = 0
        self._unflushed = 0
        self.flush_cadence = self._resolve_flush_cadence()
        self._closed = False
        self._fh = None

    def write_decision(self, row: Mapping[str, Any]) -> None:
        self._write(row)

    def write_outcome(self, row: Mapping[str, Any]) -> None:
        self._write(row)

    def close(self) -> None:
        if self._fh is not None:
            self._fh.flush()
            self._fh.close()
            self._fh = None
        self._closed = True

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_fh"] = None
        state["events"] = []
        state["_unflushed"] = 0
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._fh = None

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
        self.events.append(event)
        self._written += 1
        self._unflushed += 1
        if self._unflushed >= self.flush_cadence or self._is_terminal_outcome(event):
            self._fh.flush()
            self._unflushed = 0

    @staticmethod
    def _resolve_flush_cadence() -> int:
        value = os.environ.get("RMFS_RTS_ROLLOUT_FLUSH_CADENCE", "").strip()
        if not value:
            value = "25" if os.environ.get("RMFS_DEBUG_TRACE", "").strip().lower() in {"1", "true", "yes", "on"} else "100"
        try:
            return max(1, int(value))
        except ValueError:
            return 100

    @staticmethod
    def _is_terminal_outcome(event: Mapping[str, Any]) -> bool:
        if event.get("event_type") != "outcome":
            return False
        status = str(event.get("paper_cycle_status") or "")
        if status == "complete" or status.startswith("censored"):
            return True
        return str(event.get("outcome_status") or "").startswith("paper_cycle_")

