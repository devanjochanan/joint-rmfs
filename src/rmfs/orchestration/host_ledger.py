"""Persistent, per-condition ledger for autonomous campaign hosts."""

from __future__ import annotations

import datetime as _dt
import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

LEDGER_SCHEMA_VERSION = "host_ledger.v2"
TERMINAL_STATES = {
    "completed_strict", "completed_with_warnings", "failed_final", "quarantined",
}
RUNNABLE_STATES = {"pending", "failed_retryable"}


def _now_utc() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def _atomic_write_json(path: Path, payload: Any, *, attempts: int = 7) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    delay = 0.05
    last_exc: OSError | None = None
    for attempt in range(attempts):
        try:
            tmp.replace(path)
            return
        except OSError as exc:
            if getattr(exc, "winerror", None) not in {32, 33} and getattr(exc, "errno", None) not in {13, 16}:
                raise
            last_exc = exc
            if attempt + 1 < attempts:
                time.sleep(delay)
                delay = min(delay * 2.0, 1.0)
    raise RuntimeError(f"failed to replace {path} after {attempts} attempts: {last_exc}") from last_exc


def _new_state() -> dict[str, Any]:
    return {
        "status": "pending", "attempt_count": 0, "active_attempt_id": None,
        "started_at": None, "completed_at": None, "result_path": None,
        "failure_reason": None, "warning_count": 0, "export_status": "not_exported",
        "attempts": [],
    }


@dataclass
class HostLedger:
    host_id: str
    campaign_id: str
    manifest_sha256: str
    assigned_conditions: list[dict[str, Any]]
    condition_states: dict[str, dict[str, Any]] = field(default_factory=dict)
    local_result_index: dict[str, dict[str, Any]] = field(default_factory=dict)
    export_status: dict[str, Any] = field(default_factory=dict)
    source_tree_hash: str | None = None
    immutable_hashes: dict[str, str] = field(default_factory=dict)
    kpi_schema_version: str | None = None
    ledger_schema_version: str = LEDGER_SCHEMA_VERSION
    last_heartbeat: str = ""
    created_at: str = ""
    updated_at: str = ""

    @classmethod
    def create(cls, host_id: str, campaign_id: str, manifest_sha256: str,
               assigned_conditions: list[dict[str, Any]], *, kpi_schema_version: str | None = None,
               source_tree_hash: str | None = None, immutable_hashes: dict[str, str] | None = None) -> "HostLedger":
        now = _now_utc()
        states = {str(c["condition_key"]): _new_state() for c in assigned_conditions}
        return cls(host_id, campaign_id, manifest_sha256, list(assigned_conditions), states,
                   source_tree_hash=source_tree_hash, immutable_hashes=dict(immutable_hashes or {}),
                   kpi_schema_version=kpi_schema_version, created_at=now, updated_at=now, last_heartbeat=now)

    @classmethod
    def load(cls, path: Path) -> "HostLedger":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        schema = data.get("ledger_schema_version", "host_ledger.v1")
        if schema not in {"host_ledger.v1", LEDGER_SCHEMA_VERSION}:
            raise RuntimeError(f"incompatible ledger schema: {schema!r}")
        if schema == "host_ledger.v1":
            states = {str(c["condition_key"]): _new_state() for c in data.get("assigned_conditions", [])}
            completed = set(data.get("completed_conditions", []))
            for key in completed:
                if key in states:
                    states[key].update(status="completed_strict", completed_at=data.get("updated_at"))
            for failed in data.get("failed_conditions", []):
                key = str(failed.get("condition_key", ""))
                if key not in states:
                    continue
                states[key].update(
                    status="failed_final" if failed.get("exhausted_retries") else "failed_retryable",
                    attempt_count=int(failed.get("attempt", 0)), failure_reason=failed.get("error_message"),
                    attempts=[dict(failed)],
                )
            active = data.get("current_condition")
            if active in states and states[active]["status"] not in TERMINAL_STATES:
                states[active]["status"] = "failed_retryable"
                states[active]["failure_reason"] = "interrupted_before_ledger_v2_migration"
            data["condition_states"] = states
        return cls(
            host_id=str(data["host_id"]), campaign_id=str(data["campaign_id"]),
            manifest_sha256=str(data["manifest_sha256"]), assigned_conditions=list(data.get("assigned_conditions", [])),
            condition_states=dict(data.get("condition_states", {})),
            local_result_index=dict(data.get("local_result_index", {})), export_status=dict(data.get("export_status", {})),
            source_tree_hash=data.get("source_tree_hash"), immutable_hashes=dict(data.get("immutable_hashes", {})),
            kpi_schema_version=data.get("kpi_schema_version"), last_heartbeat=str(data.get("last_heartbeat", "")),
            created_at=str(data.get("created_at", "")), updated_at=str(data.get("updated_at", "")),
        )

    def save(self, path: Path) -> None:
        self.updated_at = _now_utc()
        _atomic_write_json(path, asdict(self))

    def _condition_by_key(self, key: str) -> dict[str, Any] | None:
        return next((c for c in self.assigned_conditions if c.get("condition_key") == key), None)

    def state_for(self, key: str) -> dict[str, Any]:
        return self.condition_states.setdefault(key, _new_state())

    @property
    def completed_conditions(self) -> list[str]:
        return [key for key, state in self.condition_states.items() if state.get("status") in {"completed_strict", "completed_with_warnings"}]

    @property
    def failed_conditions(self) -> list[dict[str, Any]]:
        return [dict(condition_key=key, **state) for key, state in self.condition_states.items() if state.get("status", "").startswith("failed")]

    @property
    def retry_counts(self) -> dict[str, int]:
        return {key: int(state.get("attempt_count", 0)) for key, state in self.condition_states.items()}

    def next_condition(self, *, eligible_stages: set[int] | None = None) -> dict[str, Any] | None:
        """Return one eligible non-active condition; never mutates a skipped row."""
        for condition in self.assigned_conditions:
            key = str(condition.get("condition_key", ""))
            state = self.state_for(key)
            if state.get("status") not in RUNNABLE_STATES:
                continue
            if eligible_stages is not None and int(condition.get("stage_first_requested", 0)) not in eligible_stages:
                continue
            return condition
        return None

    def start_condition(self, key: str) -> str:
        state = self.state_for(key)
        if state.get("status") not in RUNNABLE_STATES:
            raise RuntimeError(f"condition {key} is not runnable: {state.get('status')}")
        attempt_id = uuid.uuid4().hex
        state.update(status="running", active_attempt_id=attempt_id, started_at=_now_utc(), failure_reason=None)
        self.last_heartbeat = _now_utc()
        return attempt_id

    def mark_completed(self, key: str, outcome_summary: dict[str, Any], *, warning_count: int = 0) -> None:
        state = self.state_for(key)
        attempt_id = state.get("active_attempt_id")
        state.update(status="completed_with_warnings" if warning_count else "completed_strict",
                     completed_at=_now_utc(), active_attempt_id=None, warning_count=int(warning_count),
                     result_path=outcome_summary.get("result_path"), failure_reason=None)
        state["attempts"].append({"attempt_id": attempt_id, "status": state["status"], "ended_at": _now_utc(), **outcome_summary})
        self.local_result_index[key] = {"status": state["status"], **outcome_summary}
        self.last_heartbeat = _now_utc()

    def mark_failed(self, key: str, error_info: dict[str, Any], *, max_retries: int = 2, quarantined: bool = False) -> None:
        state = self.state_for(key)
        attempt_id = state.get("active_attempt_id")
        count = int(state.get("attempt_count", 0)) + 1
        terminal = quarantined or count > max_retries
        status = "quarantined" if quarantined else "failed_final" if terminal else "failed_retryable"
        state.update(status=status, attempt_count=count, active_attempt_id=None,
                     failure_reason=error_info.get("error_message") or error_info.get("reason"))
        state["attempts"].append({"attempt_id": attempt_id, "status": status, "ended_at": _now_utc(), **error_info})
        self.last_heartbeat = _now_utc()

    def mark_exported(self, key: str, export_name: str) -> None:
        self.state_for(key)["export_status"] = export_name

    def reconcile_with_outputs(self, results_root: Path, *, classify_fn: Any | None = None) -> dict[str, Any]:
        reconciled: list[str] = []
        uncertain: list[dict[str, Any]] = []
        for condition in self.assigned_conditions:
            key = str(condition.get("condition_key", ""))
            state = self.state_for(key)
            if state.get("status") in TERMINAL_STATES:
                continue
            run_dir = Path(results_root) / str(condition.get("run_id", ""))
            summary_path, spec_path = run_dir / "worker_summary.json", run_dir / "run_spec.json"
            if not summary_path.exists():
                if state.get("status") == "running":
                    attempt_id = state.get("active_attempt_id")
                    state.update(status="failed_retryable", active_attempt_id=None,
                                 attempt_count=int(state.get("attempt_count", 0)) + 1,
                                 failure_reason="interrupted_or_missing_output")
                    state["attempts"].append({"attempt_id": attempt_id, "status": "failed_retryable",
                                              "ended_at": _now_utc(), "reason": "interrupted_or_missing_output"})
                continue
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                spec = json.loads(spec_path.read_text(encoding="utf-8")) if spec_path.exists() else None
            except Exception as exc:
                uncertain.append({"condition_key": key, "reason": f"unreadable_output: {exc}"})
                self.mark_failed(key, {"reason": "unreadable_output", "error_message": str(exc)}, quarantined=True)
                continue
            result = classify_fn(condition, spec, summary) if classify_fn else "completed_strict"
            if result in {"completed_strict", "completed_with_warnings"}:
                self.mark_completed(key, {"run_id": condition.get("run_id"), "result_path": str(summary_path), "reconciled": True}, warning_count=int(result == "completed_with_warnings"))
                reconciled.append(key)
            elif result == "quarantined":
                self.mark_failed(key, {"reason": "structurally_uncertain"}, quarantined=True)
                uncertain.append({"condition_key": key, "reason": result})
            else:
                self.mark_failed(key, {"reason": "terminal_output_not_usable", "error_message": summary.get("error_message", result)})
        return {"reconciled": reconciled, "still_pending": [c.get("condition_key") for c in self.assigned_conditions if self.state_for(str(c.get("condition_key"))).get("status") in RUNNABLE_STATES], "failed_reconcile": uncertain}

    def summary(self) -> dict[str, Any]:
        counts: dict[str, int] = {}
        for state in self.condition_states.values():
            status = state.get("status", "pending")
            counts[status] = counts.get(status, 0) + 1
        return {"host_id": self.host_id, "campaign_id": self.campaign_id, "total_assigned": len(self.assigned_conditions), "states": counts, "active_conditions": sorted(key for key, state in self.condition_states.items() if state.get("status") == "running"), "last_heartbeat": self.last_heartbeat, "kpi_schema_version": self.kpi_schema_version}
