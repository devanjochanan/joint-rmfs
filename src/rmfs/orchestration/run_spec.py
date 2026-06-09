"""Run specification for local isolated RMFS workers."""

from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    ticks: int
    runtime_root: Path
    repo_root: Path
    branch: str | None = None
    commit: str | None = None
    python_executable: str | None = None
    timestamp: str | None = None

    def to_json_dict(self):
        data = asdict(self)
        data["runtime_root"] = str(self.runtime_root)
        data["repo_root"] = str(self.repo_root)
        return data

    @classmethod
    def from_json_dict(cls, data):
        payload = dict(data)
        payload["runtime_root"] = Path(payload["runtime_root"])
        payload["repo_root"] = Path(payload["repo_root"])
        return cls(**payload)
