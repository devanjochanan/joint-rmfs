"""Deterministic experiment identity helpers."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping


def canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def short_hash(payload: Mapping[str, Any], *, length: int = 12) -> str:
    if int(length) < 1:
        raise ValueError("length must be positive")
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()[: int(length)]


def make_scenario_id(config: Mapping[str, Any]) -> str:
    return f"scenario_{short_hash(config)}"


def make_experiment_run_id(config: Mapping[str, Any]) -> str:
    return f"rtsrl_{short_hash(config)}"

