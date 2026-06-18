# RTS-RL Evaluation Protocol

Phase 10 adds evaluation planning infrastructure only. Evaluation is dry-run by default and does not launch simulator workers.

Default evaluation mode is:

```text
rts_rl_explicit greedy
```

`random_valid` is not enabled by default. Baseline `current` can be represented in metadata but is not the default RTS-RL checkpoint evaluation mode.

Seed packs are deterministic JSON files under `data/runtime/eval_seed_packs/` by default. They record `seed_pack_id`, seed base, replications, `netlogo_steps_per_run`, and per-replication seeds.

The dry-run evaluation controller writes:

- `eval_config.json`
- `worker_specs.json`
- `eval_summary.json`

Best checkpoint selection writes a metadata pointer only:

```text
best_checkpoint.json
```

It does not copy checkpoint directories and does not mutate `latest.json`, checkpoints, model weights, optimizer state, metadata, or feature schemas.

---

## Evaluation Hardening Updates
- **Best-Checkpoint Selection**: Selection rules use later checkpoint indices (e.g. higher batch numbers) only as the final tie-breaker.
- **Cycle-Reference Proposal Validation**: Proposals can optionally validate evaluation summaries to ensure completeness (only accepting `success`/`completed` or explicitly `valid=true` statuses, and checking that failed replications are within bounds) before generating proposals.
- **True Long-Format Export**: `eval_metrics_long.csv` exports metrics in a true long-format layout (columns: `eval_run_id`, `metric_name`, `metric_value`).


