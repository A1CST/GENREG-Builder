"""The LM line - scripts for the word-level radial language model
(training, cranks, steering, persistence, decode experiments, live
inference). Modules import each other by bare name: this package dir
is added to sys.path on import, and every script adds the REPO ROOT
so shared core (radial_stack, radial_evo*, radial_lm, zetifile) and
data dirs (radial_data/, corpora/) resolve from either location.
"""
import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
