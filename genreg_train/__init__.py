"""genreg_train — wire the genreg-engine to the GENREG game worlds (Snake, 2048).

The submodules use flat absolute imports (engine-style), so importing this package
puts its own directory on sys.path first, then loads them as top-level modules.
Flask does ``from genreg_train import Trainer, parse_config``.
"""
import os
import sys

_DIR = os.path.dirname(os.path.abspath(__file__))
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

from trainer import Trainer, TrainConfig, parse_config, create_trainer   # noqa: E402
from envs import ENVS                                    # noqa: E402
import runstore                                          # noqa: E402,F401

__all__ = ["Trainer", "TrainConfig", "parse_config", "create_trainer", "ENVS", "runstore"]
