"""Per-host assignment ledger for autonomous campaign execution.

Maintains the full lifecycle of condition assignments on a single machine:
assigned → started → completed / failed (with retries).

The ledger is a single JSON file that survives reboots.  On restart the
``reconcile_with_outputs`` method scans the local result directory and
reconciles the ledger with reality.
"""

from __future__ import annotations

import datetime as _dt
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

LEDGER_SCHEMA_VERSION = "host_ledger.v1"


def _now_utc() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def _atomic_write_json(path: Path, payload: Any, *, attempts: int = 7) -> None:
    """Write *payload* to *path* atomically via a .tmp intermediate."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    delay = 0.05
    last_exc: OSError | None = None
    for attempt in range(attempts):
        try:
            tmp.replace(path)
            return
        except OSError as exc:
            winerror = getattr(exc, "winerror", None)
            errno_ = getattr(exc, "errno", None)
            if winerror not in {32, 33} and errno_ not in {13, 16}:
                raise
            last_exc = exc
            if attempt >= attempts - 1:
                break
            time.sleep(delay)
            delay = min(delay * 2.0, 1.0)
    raise RuntimeError(
        f"failed to replace {path} after {attempts} attempts: "
        f"{type(last_exc).__name__}: {last_exc}"
    ) from last_exc


@dataclass
class HostLedger:
    """Persistent per-host assignment and execution ledger."""

    host_id: str
    campaign_id: str
    manifest_sha256: str
    assigned_conditions: list[dict[str, Any]]
    completed_conditions: list[str] = field(default_factory=list)
    failed_conditions: list[dict[str, Any]] = field(default_factory=list)
    current_condition: str | None = None
    retry_counts: dict[str, int] = field(default_factory=dict)
    last_heartbeat: str = ""
    local_result_index: dict[str, dict[str, Any]] = field(default_factory=dict)
    export_status: dict[str, Any] = field(default_factory=dict)
    source_tree_hash: str | None = None
    kpi_schema_version: str | None = None
    ledger_schema_version: str = LEDGER_SCHEMA_VERSION
    created_at: str = ""
    updated_at: str = ""

    # ── constructors ──────────────────────────────────────────────

    @classmethod
    def create(
        cls,
        host_id: str,
        campaign_id: str,
        manifest_sha256: str,
        assigned_conditions: list[dict[str, Any]],
        *,
        kpi_schema_version: str | None = None,
        source_tree_hash: str | None = None,
    ) -> "HostLedger":
        now = _now_utc()
        return cls(
            host_id=host_id,
            campaign_id=campaign_id,
            manifest_sha256=manifest_sha256,
            assigned_conditions=list(assigned_conditions),
            kpi_schema_version=kpi_schema_version,
            source_tree_hash=source_tree_hash,
            created_at=now,
            updated_at=now,
            last_heartbeat=now,
        )

    @classmethod
    def load(cls, path: Path) -> "HostLedger":
        """Load a ledger from a JSON file, validating schema version."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        schema = data.get("ledger_schema_version", "")
        if schema != LEDGER_SCHEMA_VERSION:
            raise RuntimeError(
                f"incompatible ledger schema: expected {LEDGER_SCHEMA_VERSION}, "
                f"got {schema!r}"
            )
        return cls(
            host_id=str(data["host_id"]),
            campaign_id=str(data["campaign_id"]),
            manifest_sha256=str(data["manifest_sha256"]),
            assigned_conditions=list(data.get("assigned_conditions", [])),
            completed_conditions=list(data.get("completed_conditions", [])),
            failed_conditions=list(data.get("failed_conditions", [])),
            current_condition=data.get("current_condition"),
            retry_counts=dict(data.get("retry_counts", {})),
            last_heartbeat=str(data.get("last_heartbeat", "")),
            local_result_index=dict(data.get("local_result_index", {})),
            export_status=dict(data.get("export_status", {})),
            source_tree_hash=data.get("source_tree_hash"),
            kpi_schema_version=data.get("kpi_schema_version"),
            ledger_schema_version=LEDGER_SCHEMA_VERSION,
            created_at=str(data.get("created_at", "")),
            updated_at=str(data.get("updated_at", "")),
        )

    # ── persistence ───────────────────────────────────────────────

    def save(self, path: Path) -> None:
        """Atomically write the ledger to *path*."""
        self.updated_at = _now_utc()
        _atomic_write_json(path, asdict(self))

    # ── condition lifecycle ───────────────────────────────────────

    def _condition_by_key(self, condition_key: str) -> dict[str, Any] | None:
        for cond in self.assigned_conditions:
            if cond.get("condition_key") == condition_key:
                return cond
        return None

    def _permanently_failed_keys(self) -> set[str]:
        return {
            entry["condition_key"]
            for entry in self.failed_conditions
            if entry.get("exhausted_retries", False)
        }

    def next_condition(self) -> dict[str, Any] | None:
        """Select the next unfinished, non-exhausted condition.

        Priority:
        1. If ``current_condition`` is set and not completed/permanently-failed,
           resume it (the last attempt may have been interrupted).
        2. Otherwise pick the first assigned condition that is not completed
           and not permanently failed.
        """
        done = set(self.completed_conditions)
        perm_failed = self._permanently_failed_keys()
        skip = done | perm_failed

        # Resume interrupted current condition
        if self.current_condition and self.current_condition not in skip:
            cond = self._condition_by_key(self.current_condition)
            if cond is not None:
                return cond

        for cond in self.assigned_conditions:
            key = cond.get("condition_key", "")
            if key not in skip:
                return cond
        return None

    def mark_started(self, condition_key: str) -> None:
        """Record that execution of *condition_key* has begun."""
        self.current_condition = condition_key
        self.last_heartbeat = _now_utc()

    def mark_completed(
        self,
        condition_key: str,
        outcome_summary: dict[str, Any],
    ) -> None:
        """Record successful completion of *condition_key*."""
        if condition_key not in self.completed_conditions:
            self.completed_conditions.append(condition_key)
        self.local_result_index[condition_key] = {
            "status": "completed",
            "completed_at": _now_utc(),
            **outcome_summary,
        }
        if self.current_condition == condition_key:
            self.current_condition = None
        self.last_heartbeat = _now_utc()

    def mark_failed(
        self,
        condition_key: str,
        error_info: dict[str, Any],
        *,
        max_retries: int = 2,
    ) -> None:
        """Record a failure for *condition_key*.

        If the retry budget is exhausted the condition is marked as
        permanently failed and will be skipped by ``next_condition``.
        """
        current_count = self.retry_counts.get(condition_key, 0) + 1
        self.retry_counts[condition_key] = current_count
        exhausted = current_count > max_retries

        entry = {
            "condition_key": condition_key,
            "attempt": current_count,
            "exhausted_retries": exhausted,
            "failed_at": _now_utc(),
            **error_info,
        }
        # Replace any prior entry for the same key or append
        self.failed_conditions = [
            e for e in self.failed_conditions
            if e.get("condition_key") != condition_key
        ]
        self.failed_conditions.append(entry)

        if self.current_condition == condition_key:
            self.current_condition = None
        self.last_heartbeat = _now_utc()

    def heartbeat(self) -> None:
        """Update the heartbeat timestamp."""
        self.last_heartbeat = _now_utc()

    # ── reconciliation ────────────────────────────────────────────

    def reconcile_with_outputs(
        self,
        results_root: Path,
        *,
        validate_fn: Any | None = None,
    ) -> dict[str, Any]:
        """Scan *results_root* for completed runs and update the ledger.

        For each assigned condition whose ``run_id`` has a
        ``worker_summary.json`` with ``status == "success"`` and
        ``kpi_complete == true``, the condition is marked completed.

        Parameters
        ----------
        results_root:
            Path to ``<campaign_root>/<host_id>/runs/``.
        validate_fn:
            Optional callable ``(condition, spec_dict, summary_dict) -> bool``
            for strict identity checks.  Defaults to checking status/kpi only.

        Returns
        -------
        dict with keys ``reconciled``, ``already_completed``, ``still_pending``,
        ``failed_reconcile``.
        """
        results_root = Path(results_root)
        reconciled: list[str] = []
        already_completed = list(self.completed_conditions)
        failed_reconcile: list[dict[str, Any]] = []

        for cond in self.assigned_conditions:
            key = cond.get("condition_key", "")
            if key in self.completed_conditions:
                continue
            run_id = cond.get("run_id", "")
            if not run_id:
                continue
            run_dir = results_root / run_id
            summary_path = run_dir / "worker_summary.json"
            spec_path = run_dir / "run_spec.json"
            if not summary_path.exists():
                continue
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
            except Exception:
                failed_reconcile.append({"condition_key": key, "reason": "unreadable_summary"})
                continue

            # Basic validity checks
            if summary.get("status") != "success":
                continue
            if not summary.get("kpi_complete", False):
                continue

            # Optional strict validation
            if validate_fn is not None:
                spec_dict = None
                if spec_path.exists():
                    try:
                        spec_dict = json.loads(spec_path.read_text(encoding="utf-8"))
                    except Exception:
                        pass
                try:
                    if not validate_fn(cond, spec_dict, summary):
                        failed_reconcile.append({
                            "condition_key": key,
                            "reason": "strict_validation_failed",
                        })
                        continue
                except Exception as exc:
                    failed_reconcile.append({
                        "condition_key": key,
                        "reason": f"validation_error: {type(exc).__name__}: {exc}",
                    })
                    continue

            self.mark_completed(key, {
                "reconciled": True,
                "run_id": run_id,
                "seed": summary.get("seed"),
                "status": summary.get("status"),
            })
            reconciled.append(key)

        still_pending = [
            cond.get("condition_key", "")
            for cond in self.assigned_conditions
            if cond.get("condition_key", "") not in set(self.completed_conditions)
            and cond.get("condition_key", "") not in self._permanently_failed_keys()
        ]
        return {
            "reconciled": reconciled,
            "already_completed": already_completed,
            "still_pending": still_pending,
            "failed_reconcile": failed_reconcile,
        }

    # ── summary ───────────────────────────────────────────────────

    def summary(self) -> dict[str, Any]:
        """Return a human-readable summary of ledger state."""
        perm_failed = self._permanently_failed_keys()
        return {
            "host_id": self.host_id,
            "campaign_id": self.campaign_id,
            "total_assigned": len(self.assigned_conditions),
            "completed": len(self.completed_conditions),
            "permanently_failed": len(perm_failed),
            "retryable_failed": len([
                e for e in self.failed_conditions
                if not e.get("exhausted_retries", False)
            ]),
            "pending": len(self.assigned_conditions) - len(self.completed_conditions) - len(perm_failed),
            "current_condition": self.current_condition,
            "last_heartbeat": self.last_heartbeat,
            "kpi_schema_version": self.kpi_schema_version,
        }
