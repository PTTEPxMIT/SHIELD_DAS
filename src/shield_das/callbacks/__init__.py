"""Callback modules for the SHIELD Data Acquisition System.

This package contains all Dash callbacks organized by functional area:
- dataset_callbacks: Dataset management (add, delete, name/color changes)
- plot_control_callbacks: Plot settings (scale, range, error bars, valve times)
- ui_callbacks: UI interactions (collapse/expand sections)
- live_data_callbacks: Live data updates and resampling
- export_callbacks: Download/export functionality
"""

from .dataset_callbacks import register_dataset_callbacks
from .export_callbacks import register_export_callbacks
from .live_data_callbacks import register_live_data_callbacks
from .plot_control_callbacks import register_plot_control_callbacks
from .states import PLOT_CONTROL_STATES
from .ui_callbacks import register_ui_callbacks

__all__ = [
    "PLOT_CONTROL_STATES",
    "register_dataset_callbacks",
    "register_export_callbacks",
    "register_live_data_callbacks",
    "register_plot_control_callbacks",
    "register_ui_callbacks",
]


def register_all_callbacks(plotter):
    """Register all callbacks for the DataPlotter instance.

    Args:
        plotter: DataPlotter instance with app and datasets

    This is a convenience function that registers all callback modules.
    """
    register_dataset_callbacks(plotter)
    register_plot_control_callbacks(plotter)
    register_ui_callbacks(plotter)
    register_live_data_callbacks(plotter)
    register_export_callbacks(plotter)
