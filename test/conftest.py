"""Shared pytest configuration: hardware-facing import guards.

The ``keyboard`` package registers a macOS event listener at import time,
which aborts the interpreter outright unless the process has root/Input
Monitoring permission. That kills test collection on development Macs (there
is no way to catch the abort), so stub the module out before anything imports
``shield_das``. CI runs on Linux and keeps the real module.
"""

import sys
from unittest.mock import MagicMock

if sys.platform == "darwin" and "keyboard" not in sys.modules:
    sys.modules["keyboard"] = MagicMock()
