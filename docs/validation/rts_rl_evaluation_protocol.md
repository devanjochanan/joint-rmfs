# RTS-RL Evaluation Protocol Validation

Run:

```bash
/home/dewan/torch-gpu/bin/python scripts/validation/eval_seed_pack_smoke.py
/home/dewan/torch-gpu/bin/python scripts/validation/rts_eval_controller_dry_run.py
/home/dewan/torch-gpu/bin/python scripts/validation/best_checkpoint_selection_smoke.py
/home/dewan/torch-gpu/bin/python scripts/validation/cycle_reference_update_proposal_smoke.py
```

The evaluation controller smoke is dry-run only. It creates evaluation specs and summaries, then cleans up. It does not mutate checkpoints or `latest.json`.

