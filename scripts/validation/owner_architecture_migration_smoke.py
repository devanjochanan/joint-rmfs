"""Smoke test to verify owner feature migration and architecture compliance.

Tests imports of charging, PPS, and order generation modules, ensuring
legacy modules are deleted/redirected and new APIs work correctly.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def check_legacy_paths():
    print("[SMOKE] Checking legacy paths...")
    legacy_files = [
        "model/order_generator.py",
        "model/pod_generator.py",
        "model/item_pod_generator.py",
        "src/rmfs/runtime_io/order_generation.py",
        "src/rmfs/rl/pps/model_paths.py",
    ]
    for path_str in legacy_files:
        full_path = REPO_ROOT / path_str
        if full_path.exists():
            print(f"[ERROR] Legacy path still exists: {path_str}")
            sys.exit(1)
        else:
            print(f"  OK: {path_str} is absent.")


def check_imports():
    print("[SMOKE] Verifying module imports...")
    try:
        # Charging
        import src.rmfs.decisions.charging as charging
        from src.rmfs.decisions.charging.types import ChargingConfig
        from src.rmfs.decisions.charging.policy import DEFAULT_POLICY
        print("  OK: decisions.charging imported successfully.")

        # PPS
        import src.rmfs.decisions.pps as pps
        from src.rmfs.decisions.pps.types import PPSMode
        from src.rmfs.decisions.pps.heuristic import find_best_pod
        print("  OK: decisions.pps imported successfully.")

        # Order Generation
        import src.rmfs.order_generation as og
        from src.rmfs.order_generation.bootstrap import config_orders
        from src.rmfs.order_generation.pod_sku import PodGenerator
        print("  OK: order_generation imported successfully.")

        # Netlogo compatibility bridge
        import netlogo
        print("  OK: netlogo compatibility bridge imported successfully.")

    except Exception as e:
        print(f"[ERROR] Import verification failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main():
    print("[SMOKE] Starting architecture migration validation...")
    check_legacy_paths()
    check_imports()
    print("[SMOKE] All checks passed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
