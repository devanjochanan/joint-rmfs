# RMFS Scenario Inputs

Bundled RMFS scenario inputs for the `joint-rmfs` runtime.

Each scenario folder contains the files activated together by
`scripts/data/activate_scenario.py`, the headless runner, or the thin NetLogo
API compatibility wrapper.

- `items.csv`
- `pods.csv`
- `generated_pod.csv` (optional but now supported)
- `raw_order.csv` (optional but now supported)

This keeps scenario switching self-contained inside `joint-rmfs` and avoids
depending on external copies under `_full_postt_parallel_runs`.

Available bundled scenarios currently include:

- `cindy_s1`
- `cindy_s2`
- `cindy_s3`
- `my_scenario`
- `scenario4_sij`

The runtime normalizes these source files when activating a scenario so older
semicolon-delimited exports or trailing summary rows do not break setup.

For the recent `_full_postt_parallel_runs` integration, scenario bundles are
typically refreshed via:

```powershell
& ".\.rmfs\Scripts\python.exe" joint-rmfs\scripts\data\sync_full_postt_scenarios.py
```

Important behavior:

- activating a scenario currently copies the selected bundle back into
  `data/input/base/`
- therefore the "base" input reflects the last activated scenario

For the full handoff note about this integration, see:

- `docs/current/full_postt_joint_rmfs_handoff.md`
