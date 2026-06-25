# RMFS Charging Decisions Module

This module manages charging configurations, placement strategies, and battery threshold policies.

* **Status**: Active. Code has been relocated from legacy script locations.
* **Owner**: Salsa (charging / energy / charger layout)

## Structure

- `__init__.py`: Package entry point exporting config loaders and policies.
- `types.py`: Dataclasses representing `ChargingConfig`, `ChargerPosition`, and `ChargingThresholdPolicy`.
- `config.py`: Functions for loading, validating, and saving Salsa's charging configuration (`salsa_charging_config.json`).
- `placement.py`: Placement algorithm that derives optimal charging spots using picker corridor tiers (dwell-priority) and storage area depot cells (affinity propagation). Can be executed standalone from repo root:
  ```bash
  python -m src.rmfs.decisions.charging.placement
  ```
- `policy.py`: Logic helpers determining when a robot should charge, is fully charged, or should interrupt charging early due to picking demand.
