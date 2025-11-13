"""Pressure plot generation for the SHIELD Data Acquisition System.

This module provides the PressurePlot class for generating upstream and
downstream pressure plots with error bars and valve time markers.
"""

import math

import numpy as np
import plotly.graph_objects as go

from .base_graph import BaseGraph


class PressurePlot(BaseGraph):
    """Generates pressure plots (upstream or downstream).

    This class creates interactive pressure vs time plots with support for
    error bars, valve time markers, and configurable axis scaling.

    Attributes:
        plot_type: Either "upstream" or "downstream"
        pressure_attr: Dataset attribute name for pressure data
        error_attr: Dataset attribute name for error data
        time_attr: Dataset attribute name for time data
    """

    def __init__(self, datasets: list, plot_id: str, plot_type: str = "upstream"):
        """Initialize the pressure plot.

        Args:
            datasets: List of Dataset objects containing plot data
            plot_id: Unique identifier for this plot (e.g., "upstream-plot")
            plot_type: Either "upstream" or "downstream"
        """
        super().__init__(datasets, plot_id)
        self.plot_type = plot_type

        # Set attribute names based on plot type
        # Note: time_data is shared between upstream and downstream
        self.time_attr = "time_data"
        if plot_type == "upstream":
            self.pressure_attr = "upstream_pressure"
            self.error_attr = "upstream_error"
        else:  # downstream
            self.pressure_attr = "downstream_pressure"
            self.error_attr = "downstream_error"

    def generate(self) -> go.Figure:
        """Generate the pressure plot using instance attributes.

        Returns:
            plotly.graph_objs.Figure: The generated pressure plot
        """
        # Create FigureResampler instance
        self.figure_resampler = self._create_figure_resampler()

        # Add data traces for each dataset
        for dataset in self.datasets:
            # Skip datasets with no data
            time_data = getattr(dataset, self.time_attr, None)
            pressure_data = getattr(dataset, self.pressure_attr, None)

            if time_data is None or pressure_data is None:
                continue

            if len(time_data) == 0 or len(pressure_data) == 0:
                continue

            self._add_dataset_trace(dataset)

        # Configure layout
        self._configure_layout()

        # Apply axis settings and ranges
        self._apply_axis_settings()
        self._apply_axis_ranges()

        # Clean up trace names (remove [R] annotations from resampler)
        self._clean_trace_names()

        return self.figure_resampler

    def _add_dataset_trace(self, dataset) -> None:
        """Add a single dataset's trace to the figure.

        Args:
            dataset: Dataset object with data to plot
        """
        # Get data arrays
        time_data = np.ascontiguousarray(getattr(dataset, self.time_attr))
        pressure_data = np.ascontiguousarray(getattr(dataset, self.pressure_attr))
        pressure_error = np.ascontiguousarray(getattr(dataset, self.error_attr))
        colour = dataset.colour

        # Ensure consistent array lengths
        min_len = min(len(time_data), len(pressure_data), len(pressure_error))
        if len(time_data) != min_len or len(pressure_data) != min_len:
            time_data = time_data[:min_len]
            pressure_data = pressure_data[:min_len]
            pressure_error = pressure_error[:min_len]

        # Create scatter trace
        scatter_kwargs = {
            "mode": "lines+markers",
            "name": dataset.name,
            "line": dict(color=colour, width=1.5),
            "marker": dict(size=3),
        }

        # Add error bars if enabled
        show_error_bars = self.plot_parameters.get("show_error_bars", False)
        show_valve_times = self.plot_parameters.get("show_valve_times", False)

        if show_error_bars:
            scatter_kwargs["error_y"] = dict(
                type="data",
                array=pressure_error,
                visible=True,
                color=colour,
                thickness=1.5,
                width=3,
            )
            # Add trace without downsampling to preserve error bars
            scatter_kwargs["x"] = time_data
            scatter_kwargs["y"] = pressure_data
            self.figure_resampler.add_trace(go.Scatter(**scatter_kwargs))
        else:
            # Use plotly-resampler for automatic downsampling
            self.figure_resampler.add_trace(
                go.Scatter(**scatter_kwargs),
                hf_x=time_data,
                hf_y=pressure_data,
            )

        # Add valve time vertical lines
        if show_valve_times:
            self._add_valve_markers(dataset, colour)

    def _add_valve_markers(self, dataset, colour: str) -> None:
        """Add vertical lines marking valve operation times.

        Args:
            dataset: Dataset with valve time data
            colour: Color for the marker lines
        """
        valve_times = dataset.valve_times
        for valve_event, valve_time in valve_times.items():
            self.figure_resampler.add_vline(
                x=valve_time,
                line_dash="dash",
                line_color=colour,
                line_width=1,
                annotation_text=valve_event.replace("_", " ").title(),
                annotation_position="top",
                annotation_textangle=0,
                annotation_font_size=8,
            )

    def _configure_layout(self) -> None:
        """Configure the plot layout."""
        self.figure_resampler.update_layout(
            height=500,
            xaxis_title="Time (s)",
            yaxis_title="Pressure (Torr)",
            template="plotly_white",
            margin=dict(l=60, r=30, t=40, b=60),
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5
            ),
        )

    def _apply_axis_ranges(self) -> None:
        """Apply axis ranges, computing from data if not provided."""
        # Get parameters from dict
        x_scale = self.plot_parameters.get("x_scale", "linear")
        y_scale = self.plot_parameters.get("y_scale", "linear")
        x_min = self.plot_parameters.get("x_min")
        x_max = self.plot_parameters.get("x_max")
        y_min = self.plot_parameters.get("y_min")
        y_max = self.plot_parameters.get("y_max")

        # X-axis range
        if x_scale == "log":
            xmin_lin, xmax_lin = self._get_log_range(
                x_min,
                x_max,
                self.time_attr,
                default_min=1e-12,
                default_max=1e-6,
            )
            self.figure_resampler.update_xaxes(
                range=[math.log10(xmin_lin), math.log10(xmax_lin)]
            )
        else:
            if x_min is not None and x_max is not None and x_min < x_max:
                self.figure_resampler.update_xaxes(range=[x_min, x_max])
            else:
                vals = self._collect_values(self.time_attr)
                if vals:
                    self.figure_resampler.update_xaxes(range=[min(vals), max(vals)])

        # Y-axis range
        if y_scale == "log":
            ymin_lin, ymax_lin = self._get_log_range(
                y_min,
                y_max,
                self.pressure_attr,
                default_min=1e-12,
                default_max=1e-6,
            )
            self.figure_resampler.update_yaxes(
                range=[math.log10(ymin_lin), math.log10(ymax_lin)]
            )
        else:
            if y_min is not None and y_max is not None and y_min < y_max:
                self.figure_resampler.update_yaxes(range=[y_min, y_max])
            else:
                vals = self._collect_values(self.pressure_attr)
                if vals:
                    self.figure_resampler.update_yaxes(range=[min(vals), max(vals)])

    def _get_log_range(
        self,
        user_min: float | None,
        user_max: float | None,
        attr_name: str,
        default_min: float,
        default_max: float,
    ) -> tuple[float, float]:
        """Get range for logarithmic axis.

        Args:
            user_min: User-provided minimum (or None)
            user_max: User-provided maximum (or None)
            attr_name: Dataset attribute name to read data from
            default_min: Default minimum if no data available
            default_max: Default maximum if no data available

        Returns:
            tuple: (min_value, max_value) for the axis
        """
        if (
            user_min is not None
            and user_max is not None
            and user_min > 0
            and user_max > 0
            and user_min < user_max
        ):
            return float(user_min), float(user_max)

        # Collect positive values from datasets
        pos_vals = []
        for ds in self.datasets:
            try:
                vals = np.asarray(getattr(ds, attr_name), dtype=float)
                vals = vals[vals > 0]
                if vals.size:
                    pos_vals.extend(vals.tolist())
            except Exception:
                continue

        if pos_vals:
            min_val = float(min(pos_vals))
            max_val = float(max(pos_vals))
            if min_val >= max_val:
                max_val = min_val * 10.0
            return min_val, max_val

        return default_min, default_max

    def _collect_values(self, attr_name: str) -> list[float]:
        """Collect all values for a given attribute across datasets.

        Args:
            attr_name: Dataset attribute name

        Returns:
            list: All values collected
        """
        vals = []
        for ds in self.datasets:
            try:
                v = getattr(ds, attr_name)
                vals.extend([float(x) for x in v])
            except Exception:
                continue
        return vals
