"""Active job-queue robot task allocation with regret-k selection."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable, Sequence


DEFAULT_ROBOT_TASK_ALLOCATOR = "regret_k"
LEGACY_NEAREST_ALLOCATOR = "legacy_nearest"
TASK_ALLOCATOR_SCOPE = "active_job_queue"
DEFAULT_REGRET_K = 2


@dataclass(frozen=True)
class AllocationResult:
    job: Any
    robot: Any
    queue_index: int
    cheapest_cost: float
    regret: float
    feasible_robot_count: int
    allocator: str
    regret_k: int | None


def scheduler_metadata(
    *,
    robot_task_allocator: str = DEFAULT_ROBOT_TASK_ALLOCATOR,
    regret_k: int | None = DEFAULT_REGRET_K,
    committed_next_reservations_enabled: bool = False,
) -> dict[str, Any]:
    allocator = _normalize_allocator(robot_task_allocator)
    return {
        "robot_task_allocator": allocator,
        "regret_k": _normalize_k(regret_k) if allocator == DEFAULT_ROBOT_TASK_ALLOCATOR else None,
        "task_allocator_scope": TASK_ALLOCATOR_SCOPE,
        "committed_next_reservations_enabled": bool(committed_next_reservations_enabled),
    }


def select_active_job_queue_assignment(
    *,
    jobs: Sequence[Any],
    robots: Sequence[Any],
    cost_fn: Callable[[Any, Any], float | int | None],
    robot_task_allocator: str = DEFAULT_ROBOT_TASK_ALLOCATOR,
    regret_k: int | None = DEFAULT_REGRET_K,
    job_id_fn: Callable[[Any], Any] | None = None,
    robot_id_fn: Callable[[Any], Any] | None = None,
) -> AllocationResult | None:
    allocator = _normalize_allocator(robot_task_allocator)
    k = _normalize_k(regret_k)
    if allocator == LEGACY_NEAREST_ALLOCATOR:
        return _select_legacy_nearest(
            jobs=jobs,
            robots=robots,
            cost_fn=cost_fn,
            robot_id_fn=robot_id_fn,
        )
    if allocator != DEFAULT_ROBOT_TASK_ALLOCATOR:
        raise ValueError(f"unsupported robot_task_allocator: {robot_task_allocator!r}")
    return _select_regret_k(
        jobs=jobs,
        robots=robots,
        cost_fn=cost_fn,
        regret_k=k,
        job_id_fn=job_id_fn,
        robot_id_fn=robot_id_fn,
    )


def _select_legacy_nearest(
    *,
    jobs: Sequence[Any],
    robots: Sequence[Any],
    cost_fn: Callable[[Any, Any], float | int | None],
    robot_id_fn: Callable[[Any], Any] | None,
) -> AllocationResult | None:
    for queue_index, job in enumerate(jobs):
        if job is None:
            continue
        scored = _scored_robots(job, robots, cost_fn=cost_fn, robot_id_fn=robot_id_fn)
        if not scored:
            continue
        cost, _robot_key, robot = scored[0]
        return AllocationResult(
            job=job,
            robot=robot,
            queue_index=queue_index,
            cheapest_cost=cost,
            regret=0.0,
            feasible_robot_count=len(scored),
            allocator=LEGACY_NEAREST_ALLOCATOR,
            regret_k=None,
        )
    return None


def _select_regret_k(
    *,
    jobs: Sequence[Any],
    robots: Sequence[Any],
    cost_fn: Callable[[Any, Any], float | int | None],
    regret_k: int,
    job_id_fn: Callable[[Any], Any] | None,
    robot_id_fn: Callable[[Any], Any] | None,
) -> AllocationResult | None:
    winner: tuple[tuple[float, float, int, str, str], AllocationResult] | None = None
    for queue_index, job in enumerate(jobs):
        if job is None:
            continue
        scored = _scored_robots(job, robots, cost_fn=cost_fn, robot_id_fn=robot_id_fn)
        if not scored:
            continue
        cheapest_cost, robot_key, robot = scored[0]
        regret = 0.0
        for cost, _candidate_key, _candidate_robot in scored[1:min(regret_k, len(scored))]:
            regret += float(cost - cheapest_cost)
        result = AllocationResult(
            job=job,
            robot=robot,
            queue_index=queue_index,
            cheapest_cost=cheapest_cost,
            regret=float(regret),
            feasible_robot_count=len(scored),
            allocator=DEFAULT_ROBOT_TASK_ALLOCATOR,
            regret_k=regret_k,
        )
        # tuple is minimized, so negate regret to prefer higher regret.
        rank = (
            -float(regret),
            float(cheapest_cost),
            int(queue_index),
            _stable_text(job_id_fn(job) if job_id_fn else _default_job_id(job)),
            robot_key,
        )
        if winner is None or rank < winner[0]:
            winner = (rank, result)
    return winner[1] if winner else None


def _scored_robots(
    job: Any,
    robots: Sequence[Any],
    *,
    cost_fn: Callable[[Any, Any], float | int | None],
    robot_id_fn: Callable[[Any], Any] | None,
) -> list[tuple[float, str, Any]]:
    scored: list[tuple[float, str, Any]] = []
    for index, robot in enumerate(robots):
        try:
            raw_cost = cost_fn(job, robot)
        except Exception:
            continue
        if raw_cost is None:
            continue
        try:
            cost = float(raw_cost)
        except Exception:
            continue
        if not math.isfinite(cost):
            continue
        robot_key = _stable_text(robot_id_fn(robot) if robot_id_fn else _default_robot_id(robot, index))
        scored.append((cost, robot_key, robot))
    scored.sort(key=lambda item: (item[0], item[1]))
    return scored


def _normalize_allocator(value: str) -> str:
    allocator = str(value or DEFAULT_ROBOT_TASK_ALLOCATOR).strip()
    return allocator or DEFAULT_ROBOT_TASK_ALLOCATOR


def _normalize_k(value: int | None) -> int:
    if value is None:
        return DEFAULT_REGRET_K
    k = int(value)
    if k < 1:
        raise ValueError("regret_k must be >= 1")
    return k


def _stable_text(value: Any) -> str:
    return "" if value is None else str(value)


def _default_job_id(job: Any) -> Any:
    for attr in ("job_id", "id", "pod", "pod_id", "station_id"):
        value = getattr(job, attr, None)
        if value is not None:
            return value
    return repr(job)


def _default_robot_id(robot: Any, index: int) -> Any:
    for attr in ("_id", "id", "robot_id"):
        value = getattr(robot, attr, None)
        if value is not None:
            return value
    return index
