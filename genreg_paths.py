"""genreg_paths - one import puts every project package dir on
sys.path so cross-line bare imports (radial core, lm modules) resolve
from anywhere: repo root scripts, package scripts run directly, or
Flask. Add new package dirs here as lines move into folders.
"""
import os as _os
import sys as _sys

_ROOT = _os.path.dirname(_os.path.abspath(__file__))
for _d in ("", "lm", "radial"):
    _p = _os.path.join(_ROOT, _d) if _d else _ROOT
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
