# RMFS Order Generation Module

This module manages order stream generation and pod SKU allocations.

* **Status**: Active. Logic has been relocated from legacy model and runtime locations.
* **Owner**: Lukman / Luki (order generation / pod-SKU allocation)

## Structure

- `__init__.py`: Package entry point exporting generators and configuration loaders.
- `bootstrap.py`: Logic to resample and generate order streams from raw transactional datasets (`raw_order.csv`).
- `policy.py`: Resolvers mapping default simulation configuration profiles (smoke, training, gui, etc.) to target order limits, buffers, and time horizons.
- `pod_sku.py`: `PodGenerator` class, responsible for configuring slot mappings, allocating inventory levels across classes, and generating `pods.csv` and `items.csv`.
