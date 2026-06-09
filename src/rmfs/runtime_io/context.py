"""Runtime path context for RMFS simulator runs."""

from dataclasses import dataclass
from pathlib import Path


def _find_repo_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / ".git").exists() and (candidate / "netlogo.py").exists():
            return candidate
    return current


@dataclass(frozen=True)
class RunContext:
    repo_root: Path
    input_root: Path
    runtime_root: Path
    output_root: Path
    state_file: Path
    sqlite_db: Path
    assign_order_csv: Path
    pod_info_csv: Path
    skus_data_csv: Path
    sorted_skus_data_csv: Path
    generated_order_csv: Path
    generated_backlog_csv: Path
    generated_database_order_csv: Path
    generated_pod_csv: Path
    pods_csv: Path
    items_csv: Path
    saved_models_dir: Path

    @classmethod
    def default(cls, repo_root=None):
        root = Path(repo_root).resolve() if repo_root is not None else _find_repo_root()
        return cls(
            repo_root=root,
            input_root=root,
            runtime_root=root,
            output_root=root / "output",
            state_file=root / "netlogo.state",
            sqlite_db=root / "warehouse.db",
            assign_order_csv=root / "assign_order.csv",
            pod_info_csv=root / "pod_info.csv",
            skus_data_csv=root / "skus_data.csv",
            sorted_skus_data_csv=root / "sorted_skus_data.csv",
            generated_order_csv=root / "generated_order.csv",
            generated_backlog_csv=root / "generated_backlog.csv",
            generated_database_order_csv=root / "generated_database_order.csv",
            generated_pod_csv=root / "generated_pod.csv",
            pods_csv=root / "pods.csv",
            items_csv=root / "items.csv",
            saved_models_dir=root / "saved_models",
        )

    @classmethod
    def isolated(cls, runtime_root, repo_root=None):
        root = Path(repo_root).resolve() if repo_root is not None else _find_repo_root()
        runtime = Path(runtime_root)
        if not runtime.is_absolute():
            runtime = root / runtime
        runtime = runtime.resolve()
        default = cls.default(root)
        return cls(
            repo_root=root,
            input_root=root,
            runtime_root=runtime,
            output_root=runtime / "output",
            state_file=runtime / "netlogo.state",
            sqlite_db=runtime / "warehouse.db",
            assign_order_csv=runtime / "assign_order.csv",
            pod_info_csv=runtime / "pod_info.csv",
            skus_data_csv=runtime / "skus_data.csv",
            sorted_skus_data_csv=runtime / "sorted_skus_data.csv",
            generated_order_csv=default.generated_order_csv,
            generated_backlog_csv=default.generated_backlog_csv,
            generated_database_order_csv=default.generated_database_order_csv,
            generated_pod_csv=default.generated_pod_csv,
            pods_csv=default.pods_csv,
            items_csv=default.items_csv,
            saved_models_dir=default.saved_models_dir,
        )

    def ensure_runtime_dirs(self):
        self.runtime_root.mkdir(parents=True, exist_ok=True)
        self.output_root.mkdir(parents=True, exist_ok=True)
        for path in (
            self.state_file,
            self.sqlite_db,
            self.assign_order_csv,
            self.pod_info_csv,
            self.skus_data_csv,
            self.sorted_skus_data_csv,
        ):
            path.parent.mkdir(parents=True, exist_ok=True)

    def inventory_paths(self):
        return {
            "assign_order_csv": str(self.assign_order_csv),
            "pod_info_csv": str(self.pod_info_csv),
            "generated_order_csv": str(self.generated_order_csv),
        }
