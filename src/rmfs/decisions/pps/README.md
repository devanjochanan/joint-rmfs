# RMFS PPS (Pick Pod Selection) Module

This module manages pick pod selection algorithms, including heuristics (pile-on, demand) and reinforcement learning (PPO) runtime integration.

* **Status**: Active. Logic has been relocated from legacy model and runtime locations.
* **Owner**: Devan (PPS / pick pod selection)

## Structure

- `__init__.py`: Package entry point exporting mode helpers, model paths, heuristics, and runtime strategy setters.
- `types.py`: Enum definitions, including `PPSMode` (ppo, random, heuristic, demand).
- `modes.py`: Normalization helper for PPS modes.
- `model_paths.py`: Path resolution for stable-baselines3 model checkpoints.
- `heuristic.py`: Heuristic selection algorithms (`find_best_pod`, `find_pod_with_the_highest_pile_on`, `find_pod_with_the_highest_demand`) decoupled from the `Inventory` model class.
- `runtime.py`: Runtime simulation integrations, strategy configuration, and model loading.
