"""Runtime path context for RMFS simulator runs."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path


BASE_INPUT_FILES = {
    "items.csv",
    "pods.csv",
    "generated_pod.csv",
    "raw_order.csv",
}

DICTIONARY_INPUT_FILES = {
    "items_dictionary.csv",
    "pods_dictionary.csv",
    "items_slots_configuration.csv",
}


def _find_repo_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / ".git").exists() and (candidate / "netlogo.py").exists():
            return candidate
    return current


def _resolve_existing(candidates: list[Path], description: str) -> Path:
    for candidate in candidates:
        if candidate.exists():
            if len(candidates) > 1 and candidate == candidates[-1]:
                warnings.warn(
                    f"Using legacy root {description} at {candidate}; "
                    "move canonical inputs under data/input.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            return candidate
    return candidates[0]


def _input_file(root: Path, input_root: Path, filename: str) -> Path:
    if filename in DICTIONARY_INPUT_FILES:
        return _resolve_existing(
            [
                input_root / "dictionaries" / filename,
                input_root / filename,
                root / "data" / "input" / "dictionaries" / filename,
                root / filename,
            ],
            filename,
        )
    if filename in BASE_INPUT_FILES:
        return _resolve_existing(
            [
                input_root / filename,
                input_root / "base" / filename,
                root / "data" / "input" / "base" / filename,
                root / filename,
            ],
            filename,
        )
    return input_root / filename


@dataclass(frozen=True)
class RunContext:
    repo_root: Path
    input_root: Path
    dictionaries_root: Path
    runtime_root: Path
    output_root: Path
    model_root: Path
    state_file: Path
    sqlite_db: Path
    assign_order_csv: Path
    pod_info_csv: Path
    skus_data_csv: Path
    sorted_skus_data_csv: Path
    generated_order_csv: Path
    generated_backlog_csv: Path
    generated_database_order_csv: Path
    generated_order_meta_json: Path
    generated_pod_csv: Path
    pods_csv: Path
    raw_order_csv: Path
    items_csv: Path
    items_dictionary_csv: Path
    pods_dictionary_csv: Path
    items_slots_configuration_csv: Path
    saved_models_dir: Path

    @classmethod
    def default(cls, repo_root=None, input_root=None, runtime_root=None):
        root = Path(repo_root).resolve() if repo_root is not None else _find_repo_root()
        inputs = Path(input_root) if input_root is not None else root / "data" / "input" / "base"
        if not inputs.is_absolute():
            inputs = root / inputs
        inputs = inputs.resolve()
        runtime = Path(runtime_root) if runtime_root is not None else root / "data" / "runtime" / "latest"
        if not runtime.is_absolute():
            runtime = root / runtime
        runtime = runtime.resolve()
        return cls._build(root=root, input_root=inputs, runtime_root=runtime, output_root=root / "data" / "output")

    @classmethod
    def isolated(cls, runtime_root, repo_root=None, input_root=None):
        root = Path(repo_root).resolve() if repo_root is not None else _find_repo_root()
        runtime = Path(runtime_root)
        if not runtime.is_absolute():
            runtime = root / runtime
        runtime = runtime.resolve()
        inputs = Path(input_root) if input_root is not None else root / "data" / "input" / "base"
        if not inputs.is_absolute():
            inputs = root / inputs
        inputs = inputs.resolve()
        return cls._build(root=root, input_root=inputs, runtime_root=runtime, output_root=root / "data" / "output")

    @classmethod
    def with_input_root(cls, input_root, repo_root=None, runtime_root=None):
        return cls.default(repo_root=repo_root, input_root=input_root, runtime_root=runtime_root)

    @classmethod
    def legacy_root(cls, repo_root=None):
        root = Path(repo_root).resolve() if repo_root is not None else _find_repo_root()
        return cls._build(root=root, input_root=root, runtime_root=root, output_root=root / "output")

    @classmethod
    def _build(cls, root: Path, input_root: Path, runtime_root: Path, output_root: Path):
        dictionaries_root = root / "data" / "input" / "dictionaries"
        model_root = root / "data" / "models"
        return cls(
            repo_root=root,
            input_root=input_root,
            dictionaries_root=dictionaries_root,
            runtime_root=runtime_root,
            output_root=output_root,
            model_root=model_root,
            state_file=runtime_root / "netlogo.state",
            sqlite_db=runtime_root / "warehouse.db",
            assign_order_csv=runtime_root / "assign_order.csv",
            pod_info_csv=runtime_root / "pod_info.csv",
            skus_data_csv=runtime_root / "skus_data.csv",
            sorted_skus_data_csv=runtime_root / "sorted_skus_data.csv",
            generated_order_csv=runtime_root / "generated_order.csv",
            generated_backlog_csv=runtime_root / "generated_backlog.csv",
            generated_database_order_csv=runtime_root / "generated_database_order.csv",
            generated_order_meta_json=runtime_root / "generated_order_meta.json",
            generated_pod_csv=_input_file(root, input_root, "generated_pod.csv"),
            pods_csv=_input_file(root, input_root, "pods.csv"),
            raw_order_csv=_input_file(root, input_root, "raw_order.csv"),
            items_csv=_input_file(root, input_root, "items.csv"),
            items_dictionary_csv=_input_file(root, dictionaries_root, "items_dictionary.csv"),
            pods_dictionary_csv=_input_file(root, dictionaries_root, "pods_dictionary.csv"),
            items_slots_configuration_csv=_input_file(root, dictionaries_root, "items_slots_configuration.csv"),
            saved_models_dir=model_root,
        )

    @property
    def state_path(self):
        return self.state_file

    @property
    def sqlite_db_path(self):
        return self.sqlite_db

    def ensure_runtime_dirs(self):
        self.runtime_root.mkdir(parents=True, exist_ok=True)
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.model_root.mkdir(parents=True, exist_ok=True)
        for path in (
            self.state_file,
            self.sqlite_db,
            self.assign_order_csv,
            self.pod_info_csv,
            self.skus_data_csv,
            self.sorted_skus_data_csv,
            self.generated_order_csv,
            self.generated_backlog_csv,
            self.generated_database_order_csv,
            self.generated_order_meta_json,
        ):
            path.parent.mkdir(parents=True, exist_ok=True)

    def inventory_paths(self):
        return {
            "assign_order_csv": str(self.assign_order_csv),
            "pod_info_csv": str(self.pod_info_csv),
            "generated_order_csv": str(self.generated_order_csv),
        }
