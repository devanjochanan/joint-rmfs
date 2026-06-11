"""TQDM progress helpers for RTS training controller code."""

from __future__ import annotations

import sys
from typing import Iterable

from tqdm.auto import tqdm


def resolve_progress_enabled(progress: bool | None) -> bool:
    if progress is None:
        return bool(sys.stdout.isatty())
    return bool(progress)


def progress_bar(iterable: Iterable, *, enabled: bool, **kwargs):
    if not enabled:
        return iterable
    return tqdm(iterable, **kwargs)

