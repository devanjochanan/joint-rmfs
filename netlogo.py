"""NetLogo Python bridge -- compatibility shim.

Root module kept importable as ``import netlogo`` for backward
compatibility with simulation.nlogo and local scripts (e.g.
profile_netlogo.py).

All implementation now lives in src.rmfs.app.netlogo_api.
This file must remain at the repository root so that NetLogo's
``py:run "import netlogo"`` continues to work unchanged.
"""

import sys
import os

# Ensure the repository root is on sys.path so that engine/** and
# model/** imports inside netlogo_api resolve correctly regardless of
# the current working directory.
_repo_root = os.path.dirname(os.path.abspath(__file__))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from src.rmfs.app.netlogo_api import *  # noqa: F401,F403