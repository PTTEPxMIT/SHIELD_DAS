"""Tests for the run_monitor extraction.

The monitoring helpers moved from ``live_dashboard`` to ``run_monitor`` so
non-UI consumers can use them without importing Dash. Behaviour is covered by
``test_live_dashboard.py`` (which exercises them through the re-exports);
here we pin down that the extraction preserved the public surface.
"""

import importlib
import sys

from shield_das import live_dashboard, run_monitor

MOVED_NAMES = [
    "DATA_CSV_NAME",
    "METADATA_NAME",
    "TIMESTAMP_COLUMN",
    "LOCAL_TEMPERATURE_COLUMN",
    "IncrementalRunReader",
    "_convert_gauge_voltage",
    "_decimation_indices",
    "_parse_timestamp",
    "_read_metadata",
    "find_active_run",
]


def test_live_dashboard_reexports_are_the_same_objects():
    """Importing a moved name from either module yields the same object."""
    for name in MOVED_NAMES:
        assert getattr(live_dashboard, name) is getattr(run_monitor, name), name


def test_run_monitor_does_not_import_dash():
    """run_monitor must stay usable without the Dash/Plotly stack."""
    saved = {
        name: sys.modules.pop(name)
        for name in list(sys.modules)
        if name == "shield_das.run_monitor" or name.split(".")[0] in ("dash", "plotly")
    }
    try:
        importlib.import_module("shield_das.run_monitor")
        assert not any(name.split(".")[0] in ("dash", "plotly") for name in sys.modules)
    finally:
        sys.modules.update(saved)
