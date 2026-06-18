"""Order generation module — Lukman/Luki order generation and pod-SKU allocation.

Provides:
- Bootstrap order generation from raw order data (bootstrap.py)
- Order generation policy resolution (policy.py)
- Pod/SKU allocation and item slot configuration (pod_sku.py)
"""

from .bootstrap import config_orders, generate_orders_from_raw_bootstrap
from .policy import resolve_order_generation_policy
from .pod_sku import PodGenerator

__all__ = [
    "config_orders",
    "generate_orders_from_raw_bootstrap",
    "resolve_order_generation_policy",
    "PodGenerator",
]
