# Validation Protocol

Validation ensures code stability, compile-readiness, and that dependency boundaries remain sound across edits.

## Validation Commands

Operators should run these checks regularly:

* **Fast Validation Check**:
  ```bash
  /home/dewan/torch-gpu/bin/python scripts/run/rmfs.py validate --fast
  ```
* **Full Validation Suite**:
  ```bash
  /home/dewan/torch-gpu/bin/python scripts/run/rmfs.py validate
  ```
* **Operator CLI Smoke Check**:
  ```bash
  /home/dewan/torch-gpu/bin/python scripts/validation/operator_cli_smoke.py
  ```

## High-Level Checks Performed

The validation suite verifies the following components without launching expensive simulations:

1. **Compile & Import Health**: Runs Python syntax checks (`compileall`) across all model, source, and script paths to catch syntax errors or broken imports.
2. **RunContext Path Routing**: Verifies `RunContext` is directing inputs to `data/input/base` and sandbox files to `data/runtime/tmp/` instead of polluting the repository root.
3. **Layout Randomization Contract**: Verifies that pod layout shuffling maps to the designated seeds and preserves grid geometry bounds.
4. **Detail DB Toggle**: Ensures detail SQLite database writes can be successfully toggled off to avoid unneeded disk IO during headless rolls.
5. **NetLogo Import Dependency Boundary**: Confirms that importing the basic NetLogo bridge does not trigger imports of Gymnasium or Stable-Baselines.
6. **Order Generation Policy**: Validates order count throttling and ensures full raw-order CSV replays are disabled unless explicitly requested.
7. **Local Executor Contract**: Validates the subprocess orchestration interface for running headless simulation workers.
8. **Bounded Heuristic Setup/Tick**: Runs a tiny heuristic simulation (1-10 ticks) to assert the model loop resolves without runtime exceptions.
9. **Operator CLI Wrapper**: Asserts that config JSONs parsing and dry-run profile resolutions match CLI arguments.
10. **Scenario Dry-Run**: Validates scenario activation dry-runs.
11. **Cleanup Dry-Run**: Runs the cleanup tool in dry-run mode to ensure artifact scanning behaves correctly.

## Limitations & Scope

> [!NOTE]
> - These tests are purely structural sanity checks.
> - They do not evaluate or claim full behavioral equivalence between the heuristic and trained policy models.
> - They do not make or claim any performance, congestion, or throughput improvements.
