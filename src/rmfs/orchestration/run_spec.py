"""Run specification for local isolated RMFS workers."""

from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    ticks: int
    runtime_root: Path
    repo_root: Path
    input_root: Path | None = None
    branch: str | None = None
    commit: str | None = None
    python_executable: str | None = None
    timestamp: str | None = None
    debug_trace: bool = False
    trace_cadence: int = 1000
    trace_first_n: int = 0
    rts_policy_mode: str = "current"
    rts_rollout_enabled: bool = False
    rts_zone_ids: list[str] | None = None
    rts_reward_reference_path: str | None = None
    rts_random_seed: int | None = None
    rts_max_events: int | None = None
    rts_policy_checkpoint_dir: str | None = None
    rts_policy_checkpoint_id: str | None = None
    rts_policy_action_mode: str = "sample"
    rts_policy_device: str = "cpu"

    def to_json_dict(self):
        data = asdict(self)
        data["runtime_root"] = str(self.runtime_root)
        data["repo_root"] = str(self.repo_root)
        data["input_root"] = str(self.input_root) if self.input_root is not None else None
        return data

    @classmethod
    def from_json_dict(cls, data):
        payload = dict(data)
        payload["runtime_root"] = Path(payload["runtime_root"])
        payload["repo_root"] = Path(payload["repo_root"])
        payload["input_root"] = Path(payload["input_root"]) if payload.get("input_root") else None
        return cls(**payload)
