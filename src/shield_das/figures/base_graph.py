"""Base class for interactive plot generation.

This module provides the BaseGraph abstract base class that defines the common
interface and functionality for all plot types in the SHIELD DAS system.
"""

from abc import ABC, abstractmethod

import plotly.graph_objs as go
from plotly_resampler import FigureResampler


class BaseGraph(ABC):
    """Abstract base class for generating interactive plots.

    This class defines the common interface and shared functionality for all
    plot types. Subclasses must implement the generate() method to create
    specific plot types.

    Attributes:
        datasets: List of dataset dictionaries containing plot data
        figure_resampler: FigureResampler instance for efficient rendering
        plot_id: Unique identifier for this plot (used for callbacks)
        plot_parameters: Dictionary containing plot settings:
            - show_error_bars: Whether to display error bars
            - show_valve_times: Whether to display valve markers
            - x_scale: X-axis scale type ("linear" or "log")
            - y_scale: Y-axis scale type ("linear" or "log")
            - x_min: Minimum x-axis value (None for auto)
            - x_max: Maximum x-axis value (None for auto)
            - y_min: Minimum y-axis value (None for auto)
            - y_max: Maximum y-axis value (None for auto)
    """

    def __init__(self, datasets: list, plot_id: str):
        """Initialize the base graph.

        Args:
            datasets: List of dataset dictionaries with plot data
            plot_id: Unique identifier for this plot
        """
        self.datasets = datasets
        self.plot_id = plot_id
        self.figure_resampler: FigureResampler | None = None

        # Plot parameters with defaults
        self.plot_parameters = {
            "show_error_bars": False,
            "show_valve_times": False,
            "x_scale": "linear",
            "y_scale": "linear",
            "x_min": None,
            "x_max": None,
            "y_min": None,
            "y_max": None,
        }

    def _create_figure_resampler(self) -> FigureResampler:
        """Create a FigureResampler instance for efficient plotting.

        Returns:
            FigureResampler: Configured resampler instance
        """
        return FigureResampler(
            go.Figure(),
            default_n_shown_samples=1000,
            show_dash_kwargs={"mode": "inline"},
        )

    def _apply_axis_settings(self) -> None:
        """Apply axis scaling and range settings from plot_parameters."""
        fig = self.figure_resampler
        if fig is None:
            return

        # Get parameters
        x_scale = self.plot_parameters.get("x_scale", "linear")
        y_scale = self.plot_parameters.get("y_scale", "linear")
        x_min = self.plot_parameters.get("x_min")
        x_max = self.plot_parameters.get("x_max")
        y_min = self.plot_parameters.get("y_min")
        y_max = self.plot_parameters.get("y_max")

        # Apply axis scaling
        fig.update_xaxes(type=x_scale)
        fig.update_yaxes(type=y_scale)

        # Apply axis ranges if provided and valid
        if (
            x_min is not None
            and x_max is not None
            and x_min < x_max
            and (x_scale != "log" or (x_min > 0 and x_max > 0))
        ):
            fig.update_xaxes(range=[x_min, x_max])

        if (
            y_min is not None
            and y_max is not None
            and y_min < y_max
            and (y_scale != "log" or (y_min > 0 and y_max > 0))
        ):
            fig.update_yaxes(range=[y_min, y_max])

    def _clean_trace_names(self) -> None:
        """Remove [R] annotations from plotly-resampler trace names."""
        if self.figure_resampler is None:
            return

        for trace in self.figure_resampler.data:
            if trace.name and trace.name.endswith(" [R]"):
                trace.name = trace.name[:-4]

    @abstractmethod
    def generate(self) -> go.Figure:
        """Generate the plot using current instance attributes.

        Subclasses should use self.show_error_bars, self.show_valve_times,
        self.x_scale, self.y_scale, and axis range attributes to configure
        the plot.

        Returns:
            plotly.graph_objs.Figure: The generated plot
        """
        pass
