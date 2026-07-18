"""Standalone infra CLIs: job runner, terminal daemon, pod
watchdog. Run as python tools/<x>.py."""
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
