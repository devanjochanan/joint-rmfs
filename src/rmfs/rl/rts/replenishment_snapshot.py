"""Read-only replenishment eligibility snapshot for RTS-RL state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


REPLENISHMENT_SNAPSHOT_VERSION = "rts_replenishment_snapshot.v1"


@dataclass(frozen=True)
class RTSReplenishmentSnapshot:
    eligible: bool
    eligible_skus: tuple[int, ...]
    local_low_skus: tuple[int, ...]
    global_low_skus: tuple[int, ...]
    pod_critical_fill_score: float
    reason: str

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "replenishment_snapshot_version": REPLENISHMENT_SNAPSHOT_VERSION,
            "eligible": bool(self.eligible),
            "eligible_skus": list(self.eligible_skus),
            "local_low_skus": list(self.local_low_skus),
            "global_low_skus": list(self.global_low_skus),
            "pod_critical_fill_score": float(self.pod_critical_fill_score),
            "reason": self.reason,
        }


def build_replenishment_snapshot(warehouse: Any, pod: Any) -> RTSReplenishmentSnapshot:
    if pod is None or not getattr(pod, "skus", None):
        return RTSReplenishmentSnapshot(False, (), (), (), 1.0, "missing_or_empty_pod")
    local_low: list[Any] = []
    global_low: list[Any] = []
    eligible: list[Any] = []
    fill_ratios: list[float] = []
    pod_manager = getattr(warehouse, "pod_manager", None)
    global_data = getattr(pod_manager, "skus_data", {}) or {}
    for sku_id, details in sorted((getattr(pod, "skus", {}) or {}).items(), key=lambda item: str(item[0])):
        local_fill = _ratio(details.get("current_qty", 0.0), details.get("limit_qty", 0.0))
        local_threshold = _float(details.get("threshold", 0.0))
        if local_fill <= local_threshold:
            local_low.append(_sku_key(sku_id))
        g = global_data.get(sku_id)
        if g is None:
            try:
                g = global_data.get(int(sku_id))
            except Exception:
                g = None
        if g is None:
            continue
        global_fill = _ratio(g.get("current_global_qty", 0.0), g.get("max_global_qty", 0.0))
        if global_fill <= _float(g.get("global_threshold_inv_level", 0.0)):
            global_low.append(_sku_key(sku_id))
            eligible.append(_sku_key(sku_id))
            fill_ratios.append(local_fill)
    if not eligible:
        return RTSReplenishmentSnapshot(False, (), tuple(local_low), tuple(global_low), 1.0, "no_global_low_skus")
    score = sum(fill_ratios) / len(fill_ratios) if fill_ratios else 1.0
    threshold = _float(getattr(warehouse, "pod_replenishment_threshold", 0.5))
    is_eligible = score < threshold
    return RTSReplenishmentSnapshot(
        is_eligible,
        tuple(sorted(set(eligible), key=str)) if is_eligible else (),
        tuple(sorted(set(local_low), key=str)),
        tuple(sorted(set(global_low), key=str)),
        max(0.0, min(1.0, score)),
        "eligible" if is_eligible else "critical_fill_score_above_threshold",
    )


def _ratio(numerator: Any, denominator: Any) -> float:
    denom = _float(denominator)
    if denom <= 0.0:
        return 0.0
    return max(0.0, min(1.0, _float(numerator) / denom))


def _float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _sku_key(value: Any) -> Any:
    try:
        return int(value)
    except Exception:
        return str(value)
