# RTS-RL Evaluation Protocol Validation

Run:

```bash
/home/dewan/torch-gpu/bin/python scripts/validation/eval_seed_pack_smoke.py
/home/dewan/torch-gpu/bin/python scripts/validation/rts_eval_controller_dry_run.py
/home/dewan/torch-gpu/bin/python scripts/validation/best_checkpoint_selection_smoke.py
/home/dewan/torch-gpu/bin/python scripts/validation/cycle_reference_update_proposal_smoke.py
```

The evaluation controller smoke is dry-run only. It creates evaluation specs and summaries, then cleans up. It does not mutate checkpoints or `latest.json`.

### Verifications Checked:
- **Best-Checkpoint Selection**: Verifies that when metrics are tied, the selector correctly breaks ties in favor of later checkpoints (e.g. `batch_000074` wins over `batch_000020`), and that mean-style metric aliases are correctly normalized.
- **Cycle-Reference Proposal Gate**: Asserts that `success`/`completed` or explicitly `valid=true` statuses are accepted, while `dry_run`, `failed`, `partial`, and `skipped` evaluations are rejected, and that the original reference file remains completely unchanged.


