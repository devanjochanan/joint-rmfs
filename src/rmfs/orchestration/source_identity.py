"""Git-free source-tree identity for frozen campaign snapshots.

Produces a deterministic hash of key source files so that a host
without a Git repository can still prove which code version it ran.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import datetime as _dt


# Default globs that define the scientific identity of the source tree.
DEFAULT_SOURCE_GLOBS = (
    "**/*.py",
    "simulation.nlogo",
    "configs/**/*.json",
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compute_source_tree_hash(
    root: Path,
    globs: tuple[str, ...] | list[str] = DEFAULT_SOURCE_GLOBS,
) -> str:
    """Walk *root* matching *globs* and return a deterministic SHA-256.

    The hash is computed over sorted ``(relative_posix_path, file_sha256)``
    pairs, so it is stable across platforms and independent of file
    modification timestamps.
    """
    root = Path(root).resolve()
    entries: list[tuple[str, str]] = []
    seen: set[Path] = set()
    for pattern in globs:
        for path in sorted(root.glob(pattern)):
            resolved = path.resolve()
            if not resolved.is_file() or resolved in seen:
                continue
            # Skip hidden directories and __pycache__
            rel = resolved.relative_to(root)
            parts = rel.parts
            if any(part.startswith(".") or part == "__pycache__" for part in parts):
                continue
            seen.add(resolved)
            entries.append((rel.as_posix(), _file_sha256(resolved)))
    entries.sort()
    tree_bytes = json.dumps(entries, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(tree_bytes).hexdigest()


def _git_value(root: Path, *args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=root, text=True, stderr=subprocess.DEVNULL,
        ).strip() or None
    except Exception:
        return None


def _git_is_dirty(root: Path) -> bool | None:
    try:
        output = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=root, text=True, stderr=subprocess.DEVNULL,
        )
        return bool(output.strip())
    except Exception:
        return None


@dataclass(frozen=True)
class SourceIdentity:
    """Identity of the source tree at a point in time."""

    source_tree_hash: str
    computed_at: str
    file_count: int
    git_commit: str | None = None
    git_branch: str | None = None
    git_dirty: bool | None = None

    @classmethod
    def compute(
        cls,
        root: Path,
        globs: tuple[str, ...] | list[str] = DEFAULT_SOURCE_GLOBS,
    ) -> "SourceIdentity":
        root = Path(root).resolve()
        tree_hash = compute_source_tree_hash(root, globs)
        # Count files
        seen: set[Path] = set()
        for pattern in globs:
            for path in root.glob(pattern):
                resolved = path.resolve()
                if resolved.is_file():
                    rel = resolved.relative_to(root)
                    parts = rel.parts
                    if not any(p.startswith(".") or p == "__pycache__" for p in parts):
                        seen.add(resolved)
        return cls(
            source_tree_hash=tree_hash,
            computed_at=_dt.datetime.now(_dt.timezone.utc).isoformat(),
            file_count=len(seen),
            git_commit=_git_value(root, "rev-parse", "HEAD"),
            git_branch=_git_value(root, "rev-parse", "--abbrev-ref", "HEAD"),
            git_dirty=_git_is_dirty(root),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
