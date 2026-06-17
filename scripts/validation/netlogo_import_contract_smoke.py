"""Validate that normal NetLogo import does not require PPS training deps."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> int:
    sys.modules.pop("gymnasium", None)
    import netlogo  # noqa: F401
    from src.rmfs.rl.pps import get_default_pps_model_path

    gymnasium_imported = "gymnasium" in sys.modules
    print(f"gymnasium imported: {gymnasium_imported}")
    print(f"default PPS model path: {get_default_pps_model_path()}")
    if gymnasium_imported:
        raise SystemExit("import netlogo should not import gymnasium")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
