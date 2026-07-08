"""Shared import shim for dispatched jobs that need genreg_train.wordpipe/
genelib/etc. without triggering genreg_train/__init__.py's eager import of an
unrelated subsystem (trainer.py -> engine_api.py -> requires
project/genreg-engine-main, a different RL-engine project not deployed to
compute nodes). Import this FIRST, before any `from genreg_train import ...`:

    import _pkg_stub   # relies on the script's own dir being on sys.path,
                       # which Python does automatically for `python script.py`
    from genreg_train import wordpipe as wp
"""
import os
import sys
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if "genreg_train" not in sys.modules:
    _pkg = types.ModuleType("genreg_train")
    _pkg.__path__ = [os.path.join(ROOT, "genreg_train")]
    sys.modules["genreg_train"] = _pkg
