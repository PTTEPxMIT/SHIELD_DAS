from importlib import metadata

try:
    __version__ = metadata.version("shield-das")
except metadata.PackageNotFoundError:
    __version__ = "unknown"

# Analysis functions for standalone use
from .analysis import (
    average_pressure_after_increase,
    calculate_error_on_pressure_reading,
    calculate_flux_from_sample,
    calculate_permeability_from_flux,
    evaluate_permeability_values,
    fit_permeability_data,
    voltage_to_pressure,
    voltage_to_temperature,
)

# Core data acquisition and recording
from .data_plotter import DataPlotter
from .data_recorder import DataRecorder
from .dataset import Dataset

# Live dashboard for the run currently being recorded
from .live_dashboard import (
    DashboardState,
    IncrementalRunReader,
    LiveDashboardConfig,
    build_traces,
    create_app,
    find_active_run,
    update_dashboard,
)
from .pressure_gauge import (
    Baratron626D_Gauge,
    CVM211_Gauge,
    PressureGauge,
    WGM701_Gauge,
)
from .thermocouple import Thermocouple

# Automatic upload of completed runs to SHIELD-Data
from .uploader import (
    UploaderConfig,
    find_completed_runs,
    normalize_run,
    push_run,
    sweep,
)

__all__ = [
    "Baratron626D_Gauge",
    "CVM211_Gauge",
    "DashboardState",
    "DataPlotter",
    "DataRecorder",
    "Dataset",
    "IncrementalRunReader",
    "LiveDashboardConfig",
    "PressureGauge",
    "Thermocouple",
    "UploaderConfig",
    "WGM701_Gauge",
    "average_pressure_after_increase",
    "build_traces",
    "calculate_error_on_pressure_reading",
    "calculate_flux_from_sample",
    "calculate_permeability_from_flux",
    "create_app",
    "evaluate_permeability_values",
    "find_active_run",
    "find_completed_runs",
    "fit_permeability_data",
    "normalize_run",
    "push_run",
    "sweep",
    "update_dashboard",
    "voltage_to_pressure",
    "voltage_to_temperature",
]
