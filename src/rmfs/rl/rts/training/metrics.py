"""Small JSON and numeric helpers for RTS training artifacts."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Mapping


def finite_float(value: Any, default: float | None = None) -> float | None:
    try:
        result = float(value)
    except Exception:
        return default
    if not math.isfinite(result):
        return default
    return result


def mean_or_none(values: Iterable[Any]) -> float | None:
    finite = [float(value) for value in (finite_float(v) for v in values) if value is not None]
    return (sum(finite) / len(finite)) if finite else None


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value if not isinstance(value, float) or math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    if hasattr(value, "to_json_dict"):
        return json_safe(value.to_json_dict())
    if hasattr(value, "tolist"):
        return json_safe(value.tolist())
    if hasattr(value, "__dict__"):
        return json_safe(value.__dict__)
    return str(value)


def write_json(path: Path, payload: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w") as fh:
        json.dump(json_safe(payload), fh, indent=2)


def append_jsonl(path: Path, payload: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("a") as fh:
        fh.write(json.dumps(json_safe(payload), sort_keys=True, separators=(",", ":")) + "\n")


def atomic_write_json(path: Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as fh:
        json.dump(json_safe(payload), fh, indent=2)
    os.replace(tmp, path)

