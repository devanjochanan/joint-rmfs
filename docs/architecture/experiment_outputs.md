# Experiment Outputs

Phase 10 experiment outputs are structured JSON, JSONL, SQLite, and CSV files.

SQLite ledger:

```text
data/output/rmfs_experiments.sqlite
```

CSV exports default to:

```text
data/output/exports/
```

Evaluation seed packs and dry-run outputs default under `data/runtime/**`, which is ignored runtime space.

Cycle-reference update is proposal-only in Phase 10. Proposal files require manual approval and do not overwrite current cycle-reference files.

Long runs, DoE, benchmarks, ablations, and final experiment ledgers beyond SQLite ingestion/export are deferred.

