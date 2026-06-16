Bundled RMFS scenario inputs for the `joint-rmfs` runtime.

Each scenario folder contains the pair of files activated together by
`netlogo.setup("<scenario_name>")`:

- `items.csv`
- `pods.csv`

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
