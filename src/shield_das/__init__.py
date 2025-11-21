from importlib import metadata

try:
    __version__ = metadata.version("shield-das")
except Exception:
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
from .pressure_gauge import (
    Baratron626D_Gauge,
    CVM211_Gauge,
    PressureGauge,
    WGM701_Gauge,
)
from .thermocouple import Thermocouple

__all__ = [
    "Baratron626D_Gauge",
    "CVM211_Gauge",
    "DataPlotter",
    "DataRecorder",
    "Dataset",
    "PressureGauge",
    "Thermocouple",
    "WGM701_Gauge",
    "average_pressure_after_increase",
    "calculate_error_on_pressure_reading",
    "calculate_flux_from_sample",
    "calculate_permeability_from_flux",
    "evaluate_permeability_values",
    "fit_permeability_data",
    "voltage_to_pressure",
    "voltage_to_temperature",
]
