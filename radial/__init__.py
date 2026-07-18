"""The radial core + radial-line experiments (evo engines, stack,
seed-stack cifar/mnist, anim radial trainer). Shared by every line -
consumers import by bare name via genreg_paths.
"""
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
