# RMFS Order Generation Module

This module manages order stream generation and pod SKU allocations.

* **Status**: Active. Logic has been relocated from legacy model and runtime locations.
* **Owner**: Lukman / Luki (order generation / pod-SKU allocation)

## Structure

- `__init__.py`: Package entry point exporting generators and configuration loaders.
- `bootstrap.py`: Logic to resample and generate order streams from raw transactional datasets (`raw_order.csv`).
- `policy.py`: Resolvers mapping default simulation configuration profiles (smoke, training, gui, etc.) to target order limits, buffers, and time horizons.
- `pod_sku.py`: `PodGenerator` class, responsible for configuring slot mappings, allocating inventory levels across classes, and generating `pods.csv` and `items.csv`.
## Shared PPS Order Stream

NetLogo GUI, `run_pps_backend_episode.py`, and PPS RL training use the same
order-generation mechanism:

- every valid historical order appears exactly once;
- each order keeps its original SKU lines and quantities;
- the complete order sequence is randomly shuffled from
  `data/input/base/raw_order.csv` for each setup or training episode;
- random exponential interarrival gaps are generated per simulated hour;
- `order_cycle_time` means orders per simulated hour (default: 500);
- the same seed reproduces both sequence and arrival times;
- a missing or zero GUI/backend seed selects and reports a fresh random seed;
- PPS training uses its per-episode seed, so successive episodes get different
  order sequences while remaining reproducible from their recorded seeds.

Set the rate in the NetLogo `order_cycle_time` input, or pass
`--order-cycle-time <orders-per-hour>` to either the backend runner or training
script.
