# RMFS App Module

This folder is part of the future refactored structure for the Rika RMFS simulation.

* **Status**: Active bridge boundary plus future app scaffold.
* **Rules**:
  * `netlogo_api.py` contains the active NetLogo bridge implementation moved from root `netlogo.py` in Phase 4.
  * Root `netlogo.py` remains the NetLogo-facing compatibility shim and re-exports the public bridge API.
  * Do not move additional active behavior code into this folder without an explicit package-refactor phase.
* **Current Purpose**: NetLogo bridge implementation boundary.
* **Future Purpose**: App initialization, CLI run configurations, and additional NetLogo bridge interfaces.
* **Future Owner**: Team / Shared
