"""Deterministic, treatment-aware allocation for sensitivity campaign hosts.

This module deliberately knows nothing about worker execution, devices, or model
loading.  It only creates static, capacity-safe assignment plans from condition
metadata and measured (or explicitly marked fallback) throughput values.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable, Iterable


REFERENCE_TREATMENT = "all_off"
RL_TREATMENT = "all_on_rl"
TREATMENTS = (REFERENCE_TREATMENT, RL_TREATMENT)
CRITICAL_STAGES = (1, 2, 3)
BEST_EFFORT_STAGES = (4,)


@dataclass(frozen=True)
class MachineAllocationProfile:
    """Static machine capability and throughput profile.

    Rates are aggregate machine throughput at the configured worker count.
    Slot rates are derived by dividing by the relevant slot count exactly once.
    """

    machine_id: str
    max_workers: int
    allowed_treatments: tuple[str, ...]
    eligible_stages: tuple[int, ...]
    reference_effective_steps_per_second: float
    rl_effective_steps_per_second: float | None = None
    max_rl_workers: int = 0
    rl_rate_source: str = "fallback_reference_measured"

    def __post_init__(self) -> None:
        if self.max_workers < 1:
            raise ValueError(f"{self.machine_id}: max_workers must be positive")
        if self.max_rl_workers < 0 or self.max_rl_workers > self.max_workers:
            raise ValueError(f"{self.machine_id}: max_rl_workers must be within [0, max_workers]")
        if self.reference_effective_steps_per_second <= 0:
            raise ValueError(f"{self.machine_id}: reference throughput must be positive")
        if RL_TREATMENT in self.allowed_treatments:
            if self.max_rl_workers == 0 or not self.rl_effective_steps_per_second or self.rl_effective_steps_per_second <= 0:
                raise ValueError(f"{self.machine_id}: RL eligibility requires a positive RL rate and slot budget")
        elif self.max_rl_workers:
            raise ValueError(f"{self.machine_id}: non-RL profile cannot expose RL slots")

    def supports(self, treatment: str, stage: int) -> bool:
        return treatment in self.allowed_treatments and int(stage) in self.eligible_stages

    def slot_count_for(self, treatment: str) -> int:
        if treatment == RL_TREATMENT:
            return self.max_rl_workers
        return self.max_workers

    def aggregate_rate_for(self, treatment: str) -> float:
        if treatment == RL_TREATMENT:
            if not self.rl_effective_steps_per_second:
                raise ValueError(f"{self.machine_id}: no RL throughput configured")
            return float(self.rl_effective_steps_per_second)
        return float(self.reference_effective_steps_per_second)

    def slot_rate_for(self, treatment: str) -> float:
        count = self.slot_count_for(treatment)
        if count < 1:
            raise ValueError(f"{self.machine_id}: no compatible {treatment} slots")
        return self.aggregate_rate_for(treatment) / float(count)


def default_machine_profiles() -> tuple[MachineAllocationProfile, ...]:
    """Campaign allocation policy using existing aggregate measured rates.

    The eligible-host RL rates are intentionally fallbacks to the existing
    measured aggregate throughput.  They are not GPU benchmarks and do not
    modify CPU-pinned inference behavior.
    """
    all_stages = (1, 2, 3, 4)
    both = (REFERENCE_TREATMENT, RL_TREATMENT)
    return (
        MachineAllocationProfile("win_lukman", 8, both, all_stages, 414.64, 414.64, 8),
        MachineAllocationProfile("win_admin", 8, both, all_stages, 395.66, 395.66, 8),
        MachineAllocationProfile("citi_angiebow", 4, both, all_stages, 363.25, 363.25, 4),
        MachineAllocationProfile("codex_local", 8, both, all_stages, 300.0, 300.0, 8),
        MachineAllocationProfile("alisha_pc", 8, (REFERENCE_TREATMENT,), all_stages, 330.0, None, 0, "not_applicable"),
        MachineAllocationProfile("dewa_macbook", 8, (REFERENCE_TREATMENT,), all_stages, 260.0, None, 0, "not_applicable"),
        MachineAllocationProfile("citi_gojira", 24, both, all_stages, 850.0, 850.0, 24),
    )


def allocation_config_rows(profiles: Iterable[MachineAllocationProfile]) -> list[dict[str, Any]]:
    return [
        {
            **asdict(profile),
            "allowed_treatments": list(profile.allowed_treatments),
            "eligible_stages": list(profile.eligible_stages),
            "rate_semantics": "aggregate_machine_throughput_at_configured_worker_count",
        }
        for profile in sorted(profiles, key=lambda item: item.machine_id)
    ]


def allocation_wave_for_stage(stage: int) -> str:
    if int(stage) in CRITICAL_STAGES:
        return "critical"
    if int(stage) in BEST_EFFORT_STAGES:
        return "best_effort"
    raise ValueError(f"unknown campaign stage: {stage}")


def _condition_treatment(row: dict[str, Any]) -> str:
    value = str(row.get("policy_configuration", row.get("treatment", "")))
    if value not in TREATMENTS:
        raise ValueError(f"unsupported treatment {value!r}")
    return value


def _condition_stage(row: dict[str, Any]) -> int:
    stage = int(row.get("stage_first_requested", row.get("stage", 0)))
    if stage not in (*CRITICAL_STAGES, *BEST_EFFORT_STAGES):
        raise ValueError(f"invalid stage {stage!r}")
    return stage


def _condition_identity(row: dict[str, Any]) -> tuple[str, int, int, int, str]:
    return (
        str(row.get("condition_key", "")),
        int(row.get("robot_count", 0)),
        int(row.get("order_rate", 0)),
        int(row.get("replication", 0)),
        str(row.get("run_id", "")),
    )


def _compatible_slots(
    profiles: Iterable[MachineAllocationProfile], treatment: str, stage: int,
) -> list[dict[str, Any]]:
    slots: list[dict[str, Any]] = []
    for profile in profiles:
        if not profile.supports(treatment, stage):
            continue
        for index in range(profile.slot_count_for(treatment)):
            slots.append({"profile": profile, "slot_index": index, "assigned_seconds": 0.0, "assigned_count": 0})
    return slots


def _required_participants(
    *, stage: int, treatment: str, profiles: tuple[MachineAllocationProfile, ...],
) -> list[str]:
    """Small explicit participation requirements from the launch policy."""
    if treatment == REFERENCE_TREATMENT and stage == 1:
        # This is intentionally the smallest participation constraint: one
        # compatible critical condition each for the Mac and Alisha.
        return [machine_id for machine_id in ("alisha_pc", "dewa_macbook") if any(p.machine_id == machine_id and p.supports(treatment, stage) for p in profiles)]
    if treatment == REFERENCE_TREATMENT and stage in {2, 4}:
        # Central and best-effort reference waves distribute work across every
        # compatible host.
        return [p.machine_id for p in profiles if p.supports(treatment, stage)]
    if treatment == RL_TREATMENT and stage == 3:
        # Central RL wave must exercise every compatible RL host.
        return [p.machine_id for p in profiles if p.supports(treatment, stage)]
    return []


def allocate_stage(
    rows: Iterable[dict[str, Any]],
    *,
    stage: int,
    profiles: Iterable[MachineAllocationProfile],
    estimate_steps: Callable[[dict[str, Any]], float],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Allocate one stage with constrained-first, treatment-aware LPT slots."""
    stage = int(stage)
    profiles = tuple(sorted(profiles, key=lambda item: item.machine_id))
    candidates = [dict(row) for row in rows if _condition_stage(row) == stage]
    eligible_by_treatment = {
        treatment: [profile.machine_id for profile in profiles if profile.supports(treatment, stage)]
        for treatment in TREATMENTS
    }
    slots_by_treatment = {
        treatment: _compatible_slots(profiles, treatment, stage)
        for treatment in TREATMENTS
    }
    for row in candidates:
        treatment = _condition_treatment(row)
        if not slots_by_treatment[treatment]:
            raise RuntimeError(f"stage {stage}: no eligible slots for {treatment} condition {_condition_identity(row)}")

    def ordering(row: dict[str, Any]) -> tuple[Any, ...]:
        treatment = _condition_treatment(row)
        return (
            len(slots_by_treatment[treatment]),
            -float(estimate_steps(row)),
            _condition_identity(row),
        )

    allocated: list[dict[str, Any]] = []
    per_machine: dict[str, dict[str, Any]] = {
        profile.machine_id: {"runs": {treatment: 0 for treatment in TREATMENTS}, "estimated_steps": {treatment: 0.0 for treatment in TREATMENTS}, "slot_finish_seconds": [0.0] * profile.max_workers}
        for profile in profiles
    }
    # Slots represent the shared physical worker budget.  RL-capable slots are
    # a subset of normal slots, so all_off work sees the same running finish time
    # after constrained RL work is placed.
    shared_slots = {
        profile.machine_id: [0.0] * profile.max_workers
        for profile in profiles
    }
    assigned_participants = {treatment: set() for treatment in TREATMENTS}
    required_participants = {
        treatment: _required_participants(stage=stage, treatment=treatment, profiles=profiles)
        for treatment in TREATMENTS
    }

    for row in sorted(candidates, key=ordering):
        treatment = _condition_treatment(row)
        steps = float(estimate_steps(row))
        compatible = slots_by_treatment[treatment]
        missing_participants = [
            machine_id for machine_id in required_participants[treatment]
            if machine_id not in assigned_participants[treatment]
        ]
        if missing_participants:
            # Preserve constrained-first ordering, then give the explicit
            # required host a single compatible condition before normal LPT.
            compatible = [slot for slot in compatible if slot["profile"].machine_id == missing_participants[0]]
        choices = []
        for slot in compatible:
            profile = slot["profile"]
            index = int(slot["slot_index"])
            start_seconds = shared_slots[profile.machine_id][index]
            if treatment == RL_TREATMENT:
                # RL preference is based on measured aggregate machine
                # throughput.  This prevents a lower-capacity host with a
                # slightly faster individual slot from displacing a stronger
                # RL host merely because it has fewer slots.
                finish = (
                    per_machine[profile.machine_id]["estimated_steps"][treatment] + steps
                ) / profile.aggregate_rate_for(treatment)
                slot_finish = start_seconds + steps / profile.slot_rate_for(treatment)
            else:
                finish = start_seconds + steps / profile.slot_rate_for(treatment)
                slot_finish = finish
            choices.append((finish, slot_finish, sum(per_machine[profile.machine_id]["runs"].values()), profile.machine_id, index, slot))
        _finish, _slot_finish, _count, _machine_id, index, selected = min(choices, key=lambda item: item[:5])
        profile = selected["profile"]
        duration = steps / profile.slot_rate_for(treatment)
        scheduled_start = shared_slots[profile.machine_id][index]
        shared_slots[profile.machine_id][index] += duration
        selected["assigned_seconds"] = shared_slots[profile.machine_id][index]
        selected["assigned_count"] += 1
        per_machine[profile.machine_id]["runs"][treatment] += 1
        assigned_participants[treatment].add(profile.machine_id)
        per_machine[profile.machine_id]["estimated_steps"][treatment] += steps
        per_machine[profile.machine_id]["slot_finish_seconds"][index] = shared_slots[profile.machine_id][index]
        allocated.append({
            **row,
            "machine_id": profile.machine_id,
            "machine_slot_index": index,
            "allocation_wave": allocation_wave_for_stage(stage),
            "estimated_backend_steps": steps,
            "estimated_duration_seconds": duration,
            "projected_slot_start_seconds": scheduled_start,
            "projected_slot_finish_seconds": shared_slots[profile.machine_id][index],
            "allocation_rate_steps_per_second": profile.slot_rate_for(treatment),
            "allocation_rate_source": profile.rl_rate_source if treatment == RL_TREATMENT else "reference_measured",
        })

    projected = {
        machine_id: max(details["slot_finish_seconds"], default=0.0)
        for machine_id, details in per_machine.items()
    }
    summary = {
        "stage": stage,
        "allocation_wave": allocation_wave_for_stage(stage),
        "new_runs": len(allocated),
        "eligible_machines_by_treatment": eligible_by_treatment,
        "assigned_runs_by_machine_and_treatment": {machine_id: details["runs"] for machine_id, details in per_machine.items()},
        "assigned_estimated_steps_by_machine_and_treatment": {machine_id: details["estimated_steps"] for machine_id, details in per_machine.items()},
        "projected_finish_seconds_by_machine": projected,
        "projected_stage_makespan_seconds": max(projected.values(), default=0.0),
    }
    return allocated, summary


def allocate_waves(
    rows: Iterable[dict[str, Any]],
    *,
    profiles: Iterable[MachineAllocationProfile],
    estimate_steps: Callable[[dict[str, Any]], float],
) -> dict[str, Any]:
    """Allocate stages lexicographically and expose independent launch waves."""
    source = [dict(row) for row in rows]
    if len(source) != 720:
        raise AssertionError(f"expected 720 campaign conditions, got {len(source)}")
    allocated: list[dict[str, Any]] = []
    stage_summaries: dict[str, Any] = {}
    for stage in (*CRITICAL_STAGES, *BEST_EFFORT_STAGES):
        stage_allocated, summary = allocate_stage(source, stage=stage, profiles=profiles, estimate_steps=estimate_steps)
        allocated.extend(stage_allocated)
        stage_summaries[str(stage)] = summary
    critical = [row for row in allocated if row["allocation_wave"] == "critical"]
    best_effort = [row for row in allocated if row["allocation_wave"] == "best_effort"]
    validate_allocation(allocated, profiles)
    return {
        "runs": allocated,
        "critical_wave": critical,
        "best_effort_wave": best_effort,
        "stages": stage_summaries,
    }


def validate_allocation(rows: Iterable[dict[str, Any]], profiles: Iterable[MachineAllocationProfile]) -> None:
    rows, profiles = list(rows), tuple(profiles)
    by_machine = {profile.machine_id: profile for profile in profiles}
    if len(rows) != 720:
        raise AssertionError(f"total conditions must be 720, got {len(rows)}")
    by_stage = {stage: [row for row in rows if _condition_stage(row) == stage] for stage in (*CRITICAL_STAGES, *BEST_EFFORT_STAGES)}
    if {stage: len(items) for stage, items in by_stage.items()} != {1: 18, 2: 39, 3: 39, 4: 624}:
        raise AssertionError("stage counts changed")
    if len([row for row in rows if row.get("allocation_wave") == "critical"]) != 96:
        raise AssertionError("critical wave must contain 96 conditions")
    if len([row for row in rows if row.get("allocation_wave") == "best_effort"]) != 624:
        raise AssertionError("best-effort wave must contain 624 conditions")
    for row in rows:
        profile = by_machine.get(row.get("machine_id"))
        if profile is None or not profile.supports(_condition_treatment(row), _condition_stage(row)):
            raise AssertionError(f"ineligible assignment: {row.get('condition_key')}")
        if _condition_treatment(row) == RL_TREATMENT and int(row.get("machine_slot_index", -1)) >= profile.max_rl_workers:
            raise AssertionError(f"RL assignment exceeds RL slot budget: {row.get('condition_key')}")
    for forbidden in ("alisha_pc", "dewa_macbook"):
        if any(row["machine_id"] == forbidden and _condition_treatment(row) == RL_TREATMENT for row in rows):
            raise AssertionError(f"{forbidden} received an RL condition")
    for participant in ("alisha_pc", "dewa_macbook"):
        if not any(row["machine_id"] == participant and _condition_treatment(row) == REFERENCE_TREATMENT and row["allocation_wave"] == "critical" for row in rows):
            raise AssertionError(f"{participant} lacks a critical compatible assignment")
