"""Redirect — placement logic has moved to src.rmfs.decisions.charging.placement.

Run:  python -m src.rmfs.decisions.charging.placement
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.rmfs.decisions.charging.placement import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
