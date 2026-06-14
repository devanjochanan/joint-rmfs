#!/usr/bin/env python3
"""Smoke tests for active job-queue regret-k task allocation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.rmfs.decisions.task_allocation import select_active_job_queue_assignment


@dataclass(frozen=True)
class Job:
    job_id: str


@dataclass(frozen=True)
class Robot:
    robot_id: str


def pick(costs, jobs, robots, *, mode="regret_k", k=2):
    return select_active_job_queue_assignment(
        jobs=jobs,
        robots=robots,
        cost_fn=lambda job, robot: costs.get((job.job_id, robot.robot_id)),
        robot_task_allocator=mode,
        regret_k=k,
        job_id_fn=lambda job: job.job_id,
        robot_id_fn=lambda robot: robot.robot_id,
    )


def main():
    jobs = [Job("low_regret"), Job("high_regret")]
    robots = [Robot("r1"), Robot("r2")]
    costs = {
        ("low_regret", "r1"): 2.0,
        ("low_regret", "r2"): 3.0,
        ("high_regret", "r1"): 1.0,
        ("high_regret", "r2"): 10.0,
    }
    allocation = pick(costs, jobs, robots)
    assert allocation is not None
    assert allocation.job.job_id == "high_regret"
    assert allocation.robot.robot_id == "r1"
    assert allocation.cheapest_cost == 1.0
    assert allocation.regret == 9.0

    tie_jobs = [Job("a"), Job("b")]
    tie_costs = {
        ("a", "r1"): 1.0,
        ("a", "r2"): 5.0,
        ("b", "r1"): 1.0,
        ("b", "r2"): 5.0,
    }
    first = pick(tie_costs, tie_jobs, robots)
    second = pick(tie_costs, tie_jobs, robots)
    assert first == second
    assert first.job.job_id == "a"
    assert first.robot.robot_id == "r1"

    legacy_jobs = [Job("first_queue_job"), Job("higher_regret_later")]
    legacy_costs = {
        ("first_queue_job", "r1"): 8.0,
        ("first_queue_job", "r2"): 2.0,
        ("higher_regret_later", "r1"): 1.0,
        ("higher_regret_later", "r2"): 50.0,
    }
    legacy = pick(legacy_costs, legacy_jobs, robots, mode="legacy_nearest")
    assert legacy is not None
    assert legacy.job.job_id == "first_queue_job"
    assert legacy.robot.robot_id == "r2"
    assert legacy.regret_k is None

    infeasible = pick(
        {
            ("blocked", "r1"): math.inf,
            ("blocked", "r2"): None,
            ("reachable", "r1"): 4.0,
            ("reachable", "r2"): None,
        },
        [Job("blocked"), Job("reachable")],
        robots,
    )
    assert infeasible is not None
    assert infeasible.job.job_id == "reachable"
    assert infeasible.robot.robot_id == "r1"
    assert infeasible.feasible_robot_count == 1
    assert infeasible.regret == 0.0

    three_robots = [Robot("r1"), Robot("r2"), Robot("r3")]
    k_jobs = [Job("wins_at_k2"), Job("wins_at_k3")]
    k_costs = {
        ("wins_at_k2", "r1"): 1.0,
        ("wins_at_k2", "r2"): 50.0,
        ("wins_at_k2", "r3"): 51.0,
        ("wins_at_k3", "r1"): 1.0,
        ("wins_at_k3", "r2"): 2.0,
        ("wins_at_k3", "r3"): 100.0,
    }
    assert pick(k_costs, k_jobs, three_robots, k=2).job.job_id == "wins_at_k2"
    assert pick(k_costs, k_jobs, three_robots, k=3).job.job_id == "wins_at_k3"

    print("regret-k allocator smoke ok")


if __name__ == "__main__":
    main()
