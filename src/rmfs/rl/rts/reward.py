"""RTS-RL reward helpers focused on robot cycle time."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

from .action_space import REPLENISH_STORE, STORE

REWARD_CONTRACT_VERSION = "rts_rl_reward.v1"


@dataclass(frozen=True)
class RTSRewardReference:
    reference_overall_cycle_time: float
    reference_avg_storage_cycle_time: float
    reference_avg_replenish_cycle_time: float
    alpha: float = 0.0
    source: str = "current_repo_ledger_or_reference"
    source_run_id: str | None = None
    semantics: str = "realized_robot_cycle_time"


@dataclass(frozen=True)
class RTSRewardComponents:
    selected_action_branch: str
    cycle_time: float
    cycle_time_source: str
    queue_time_estimate: float = 0.0
    replenishment_process_time: float = 0.0
    stock_factor: float = 0.0


@dataclass(frozen=True)
class RTSRewardResult:
    reference_available: bool
    reward_computed: bool
    reward_value: float | None
    base_term: float | None
    cycle_penalty_term: float | None
    replenishment_bonus_term: float | None
    normalized_cycle_time: float | None
    alpha: float | None
    is_estimate: bool
    components: RTSRewardComponents
    reference: RTSRewardReference | None

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "reward_contract_version": REWARD_CONTRACT_VERSION,
            "reference_available": self.reference_available,
            "reward_computed": self.reward_computed,
            "reward_value": self.reward_value,
            "base_term": self.base_term,
            "cycle_penalty_term": self.cycle_penalty_term,
            "replenishment_bonus_term": self.replenishment_bonus_term,
            "normalized_cycle_time": self.normalized_cycle_time,
            "alpha": self.alpha,
            "is_estimate": self.is_estimate,
            "components": asdict(self.components),
            "reference": asdict(self.reference) if self.reference is not None else None,
        }


def build_reward_components_from_realized_cycle(
    *,
    selected_action_branch: str,
    realized_cycle_time: float,
    queue_time_estimate: float = 0.0,
    replenishment_process_time: float = 0.0,
    stock_factor: float = 0.0,
) -> RTSRewardComponents:
    return _build_components(
        selected_action_branch=selected_action_branch,
        cycle_time=realized_cycle_time,
        cycle_time_source="realized",
        queue_time_estimate=queue_time_estimate,
        replenishment_process_time=replenishment_process_time,
        stock_factor=stock_factor,
    )


def build_reward_components_from_estimated_cycle(
    *,
    selected_action_branch: str,
    estimated_cycle_time: float,
    queue_time_estimate: float = 0.0,
    replenishment_process_time: float = 0.0,
    stock_factor: float = 0.0,
) -> RTSRewardComponents:
    return _build_components(
        selected_action_branch=selected_action_branch,
        cycle_time=estimated_cycle_time,
        cycle_time_source="estimated",
        queue_time_estimate=queue_time_estimate,
        replenishment_process_time=replenishment_process_time,
        stock_factor=stock_factor,
    )


def compute_reward(
    components: RTSRewardComponents,
    reference: RTSRewardReference | None,
) -> RTSRewardResult:
    if reference is None:
        return RTSRewardResult(
            reference_available=False,
            reward_computed=False,
            reward_value=None,
            base_term=None,
            cycle_penalty_term=None,
            replenishment_bonus_term=None,
            normalized_cycle_time=None,
            alpha=None,
            is_estimate=components.cycle_time_source == "estimated",
            components=components,
            reference=None,
        )
    validate_reward_reference(reference)
    branch_reference = (
        reference.reference_avg_storage_cycle_time
        if components.selected_action_branch == STORE
        else reference.reference_avg_replenish_cycle_time
    )
    normalized_cycle = _normalized_cycle_time(
        overall_reference=reference.reference_overall_cycle_time,
        branch_reference=branch_reference,
        cycle_time=components.cycle_time,
    )
    base = float(reference.reference_overall_cycle_time)
    penalty = -float(normalized_cycle)
    bonus = 0.0
    if components.selected_action_branch == REPLENISH_STORE:
        bonus = (
            max(0.0, float(reference.alpha))
            * max(0.0, float(components.queue_time_estimate + components.replenishment_process_time))
            * max(0.0, float(components.stock_factor))
        )
    return RTSRewardResult(
        reference_available=True,
        reward_computed=True,
        reward_value=float(base + penalty + bonus),
        base_term=base,
        cycle_penalty_term=penalty,
        replenishment_bonus_term=bonus,
        normalized_cycle_time=float(normalized_cycle),
        alpha=float(reference.alpha),
        is_estimate=components.cycle_time_source == "estimated",
        components=components,
        reference=reference,
    )


def validate_reward_reference(reference: RTSRewardReference) -> None:
    values = (
        reference.reference_overall_cycle_time,
        reference.reference_avg_storage_cycle_time,
        reference.reference_avg_replenish_cycle_time,
    )
    if any(not math.isfinite(float(value)) or float(value) <= 0.0 for value in values):
        raise ValueError("RTS reward reference cycle times must be finite and positive")
    if reference.semantics != "realized_robot_cycle_time":
        raise ValueError("RTS reward reference semantics must be realized_robot_cycle_time")


def _build_components(
    *,
    selected_action_branch: str,
    cycle_time: float,
    cycle_time_source: str,
    queue_time_estimate: float,
    replenishment_process_time: float,
    stock_factor: float,
) -> RTSRewardComponents:
    branch = str(selected_action_branch).strip()
    if branch not in {STORE, REPLENISH_STORE}:
        raise ValueError(f"unsupported RTS reward branch: {selected_action_branch!r}")
    normalized_cycle_time = float(cycle_time)
    if not math.isfinite(normalized_cycle_time) or normalized_cycle_time <= 0.0:
        raise ValueError("RTS reward cycle time must be finite and positive")
    return RTSRewardComponents(
        selected_action_branch=branch,
        cycle_time=normalized_cycle_time,
        cycle_time_source=cycle_time_source,
        queue_time_estimate=max(0.0, float(queue_time_estimate)),
        replenishment_process_time=max(0.0, float(replenishment_process_time)),
        stock_factor=max(0.0, float(stock_factor)),
    )


def _normalized_cycle_time(*, overall_reference: float, branch_reference: float, cycle_time: float) -> float:
    return float(cycle_time * (overall_reference / branch_reference))
