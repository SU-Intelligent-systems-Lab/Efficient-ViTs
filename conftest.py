"""
Pytest bootstrap for the Light-weight ViTs library.

Placing this file at the package root makes pytest add this directory to
``sys.path`` before collecting tests, so the top-level packages (``models``,
``data``, ``training``, ``inference``) import cleanly regardless of the
directory pytest is invoked from. Run the suite with::

    pytest "Light-wight ViTs/tests"
"""

from __future__ import annotations

import os
import sys

# Ensure the package root (this file's directory) is importable.
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
