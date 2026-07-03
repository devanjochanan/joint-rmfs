#!/usr/bin/env python3
"""Summarize committed-next eligibility diagnostics from RTS rollout JSONL files."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Summarize RTS committed-next debug diagnostics.")
    parser.add_argument("path", help="Run root, worker root, or a single rts_rollout.jsonl file.")
    args = parser.parse_args(argv)

    paths = _rollout_paths(Path(args.path))
    if not paths:
        raise SystemExit(f"no rts_rollout.jsonl files found under {args.path}")

    decisions = 0
    no_next_decisions = 0
    real_next_decisions = 0
    zero_pool_decisions = 0
    queue_lengths = []
    pool_sizes = []
    rejection_counts = Counter()

    for path in paths:
        with path.open() as fh:
            for line in fh:
                if not line.strip():
                    continue
                event = json.loads(line)
                if event.get("event_type") != "decision":
                    continue
                decisions += 1
                if event.get("selected_proposal_has_next_job") is True:
                    real_next_decisions += 1
                else:
                    no_next_decisions += 1
                pool_size = int(event.get("eligible_job_pool_size") or 0)
                pool_sizes.append(pool_size)
                if pool_size == 0:
                    zero_pool_decisions += 1
                queue_lengths.append(int(event.get("eligible_job_queue_length") or 0))
                rejection_counts.update(dict(event.get("eligible_job_rejection_counts_by_reason") or {}))

    print(f"rollout_files: {len(paths)}")
    print(f"decisions: {decisions}")
    print(f"decisions_with_real_next_job: {real_next_decisions}")
    print(f"decisions_without_next_job: {no_next_decisions}")
    print(f"decisions_with_zero_eligible_pool: {zero_pool_decisions}")
    print(f"mean_job_queue_length: {_mean(queue_lengths):.3f}")
    print(f"mean_eligible_pool_size: {_mean(pool_sizes):.3f}")
    print("rejection_counts_by_reason:")
    for reason, count in rejection_counts.most_common():
        print(f"  {reason}: {count}")
    return 0


def _rollout_paths(path: Path) -> list[Path]:
    if path.is_file():
        return [path] if path.name == "rts_rollout.jsonl" else []
    return sorted(path.glob("**/rts_rollout.jsonl"))


def _mean(values: list[int]) -> float:
    return float(sum(values)) / float(len(values)) if values else 0.0


if __name__ == "__main__":
    raise SystemExit(main())
