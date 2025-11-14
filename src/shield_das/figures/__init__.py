"""Figure generation classes for the SHIELD Data Acquisition System.

This module provides base and specialized classes for generating interactive plots.
"""

from .base_graph import BaseGraph
from .permeability_plot import PermeabilityPlot
from .pressure_plot import PressurePlot
from .temperature_plot import TemperaturePlot

__all__ = ["BaseGraph", "PermeabilityPlot", "PressurePlot", "TemperaturePlot"]
