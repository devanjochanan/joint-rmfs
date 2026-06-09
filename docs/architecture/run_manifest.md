# Run Manifest

Phase 4 adds `run_manifest.json` to local executor runs so short isolated runs can be audited later.

The manifest records repository identity, Python runtime, output roots, input snapshot paths, worker counts, tick counts, debug trace settings, aggregate summary mode, policy labels, and the root-sensitive files guarded by the controller.

This manifest is metadata only. It does not change simulator behavior, prove paper fidelity, prove behavior equivalence, or claim performance, throughput, or congestion improvement.

Worker summaries remain aggregate-only by default. `debug_trace.jsonl` remains opt-in through `--debug-trace`.

Phase 4 is not BehaviorSpace, PPO, DoE, training, checkpointing, TensorBoard, or DuckDB.
