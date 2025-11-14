"""Permeability plot generation for the SHIELD Data Acquisition System.

This module provides the PermeabilityPlot class for generating Arrhenius plots
showing permeability vs inverse temperature with HTM reference data.
"""

import plotly.graph_objects as go

from ..analysis import evaluate_permeability_values, fit_permeability_data
from ..helpers import import_htm_data
from .base_graph import BaseGraph


class PermeabilityPlot(BaseGraph):
    """Generate permeability (Arrhenius) plots with fit line and HTM reference data.

    This class creates interactive plots showing:
    - Permeability vs inverse temperature (1000/T)
    - Arrhenius fit line through experimental data
    - HTM reference data for comparison
    - Error bars on experimental measurements
    """

    def generate(self) -> go.Figure:
        """Generate permeability plot with Arrhenius fit and HTM reference data.

        Creates a plot of permeability vs 1000/T (K⁻¹) with:
        - Experimental data points (with error bars if enabled)
        - Arrhenius fit line (red dashed)
        - HTM reference data for comparison

        Returns:
            plotly.graph_objs.Figure: The generated permeability (Arrhenius) plot
        """
        # Create figure with downsampling
        self.figure_resampler = self._create_figure_resampler()

        # Get show_error_bars from plot_parameters
        show_error_bars = self.plot_parameters.get("show_error_bars", True)

        # Check if we have any datasets
        if not self.datasets:
            self._add_no_data_message()
            self._configure_layout()
            self._apply_axis_settings()
            return self.figure_resampler

        # Evaluate permeability values from datasets
        try:
            (
                temps,
                perms,
                x_error,
                y_error,
                error_lower,
                error_upper,
            ) = evaluate_permeability_values(self.datasets)
        except Exception:
            self._add_no_data_message()
            self._configure_layout()
            return self.figure_resampler

        # Check if we got any data
        if len(temps) == 0:
            self._add_no_data_message()
            self._configure_layout()
            return self.figure_resampler

        # Add data points with error bars
        self._add_data_points(temps, perms, error_lower, error_upper, show_error_bars)

        # Add Arrhenius fit line if we have enough points
        if len(temps) >= 2:
            self._add_fit_line(temps, perms)

        # Add HTM reference data
        self._add_htm_data()

        # Configure layout and axes
        self._configure_layout()
        self._apply_axis_settings()
        self._clean_trace_names()

        return self.figure_resampler

    def _add_no_data_message(self) -> None:
        """Add centered annotation when no datasets have permeability data."""
        self.figure_resampler.add_annotation(
            text="No data available",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=20, color="gray"),
        )

    def _add_data_points(
        self,
        temps: list[float],
        perms: list,
        error_lower: list[float],
        error_upper: list[float],
        show_error_bars: bool,
    ) -> None:
        """Add permeability data points with optional error bars.

        Plots experimental permeability measurements as scatter points on
        the Arrhenius plot (permeability vs 1000/T).

        Args:
            temps: Temperature values in Kelvin
            perms: Permeability values (ufloat objects or floats)
            error_lower: Lower error bar magnitudes for each point
            error_upper: Upper error bar magnitudes for each point
            show_error_bars: Whether to display asymmetric error bars on points
        """
        for i, (temp, perm) in enumerate(zip(temps, perms)):
            inv_temp = 1000 / temp
            perm_val = perm.n if hasattr(perm, "n") else perm

            # Prepare error bar dict if needed
            error_y_dict = None
            if show_error_bars:
                error_y_dict = dict(
                    type="data",
                    symmetric=False,
                    array=[error_upper[i]],
                    arrayminus=[error_lower[i]],
                )

            self.figure_resampler.add_trace(
                go.Scatter(
                    x=[inv_temp],
                    y=[perm_val],
                    mode="markers",
                    marker=dict(size=10),
                    error_y=error_y_dict,
                    showlegend=False,
                ),
            )

    def _add_fit_line(self, temps: list[float], perms: list) -> None:
        """Add Arrhenius fit line to the plot.

        Fits experimental data to Arrhenius equation and plots the resulting
        fit line as a red dashed line.

        Args:
            temps: Temperature values in Kelvin
            perms: Permeability values (ufloat objects or floats)
        """
        # Calculate fit using analysis module
        fit_x, fit_y = fit_permeability_data(temps, perms)

        # Add fit line as dashed red trace
        self.figure_resampler.add_trace(
            go.Scatter(
                x=fit_x,
                y=fit_y,
                mode="lines",
                line=dict(color="red", dash="dash"),
                name="Arrhenius Fit",
            ),
            hf_x=fit_x,
            hf_y=fit_y,
        )

    def _configure_layout(self) -> None:
        """Configure the plot layout with axis labels and styling.

        Sets up the Arrhenius plot appearance including:
        - Axis titles with proper units
        - White template for clean appearance
        - Hover mode for data point inspection
        """
        self.figure_resampler.update_layout(
            xaxis_title="1000/T (K⁻¹)",
            yaxis_title="Permeability (mol/(m·s·Pa^0.5))",
            template="plotly_white",
            hovermode="closest",
        )

    def _clean_trace_names(self) -> None:
        """Remove [R] annotations from plotly-resampler trace names.

        FigureResampler adds [R] suffix to resampled traces. This method
        removes those annotations for cleaner legend entries.
        """
        for trace in self.figure_resampler.data:
            if trace.name and trace.name.endswith(" [R]"):
                trace.name = trace.name[:-4]

    def _add_htm_data(self, material: str = "316l_steel") -> None:
        """Add HTM reference data to the plot for comparison.

        Loads and plots reference permeability data from the HTM database
        to allow comparison with experimental measurements.

        Args:
            material: Material identifier in HTM database (default: "316l_steel")
        """
        # Load HTM reference data
        htm_x, htm_y, htm_labels = import_htm_data(material)

        # Add each HTM dataset as a separate trace
        for x, y, label in zip(htm_x, htm_y, htm_labels):
            x_plot = 1000 / x  # Convert temperature to 1000/T
            self.figure_resampler.add_trace(go.Scatter(x=x_plot, y=y, name=label))
