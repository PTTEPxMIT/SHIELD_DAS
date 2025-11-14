"""Temperature plot generation for the SHIELD Data Acquisition System.

This module provides the TemperaturePlot class for generating temperature
plots showing furnace setpoint and thermocouple data.
"""

from typing import TYPE_CHECKING

import numpy as np
import plotly.graph_objects as go

from .base_graph import BaseGraph

if TYPE_CHECKING:
    from ..dataset import Dataset


class TemperaturePlot(BaseGraph):
    """Generates temperature plots showing furnace and thermocouple data.

    This class creates interactive temperature vs time plots with support for
    multiple datasets, showing both furnace setpoint and thermocouple readings.
    """

    def generate(self) -> go.Figure:
        """Generate the temperature plot.

        Note: Temperature plots don't use error bars or valve times.

        Returns:
            plotly.graph_objs.Figure: The generated temperature plot
        """
        # Create FigureResampler instance
        fig = self._create_figure_resampler()
        self.figure_resampler = fig

        # Track which datasets have/don't have temperature data
        datasets_with_temp = []
        datasets_without_temp = []

        for dataset in self.datasets:
            if dataset.thermocouple_data is None:
                datasets_without_temp.append(dataset.name)
            else:
                datasets_with_temp.append(dataset)

        # Plot temperature data for datasets that have it
        for dataset in datasets_with_temp:
            self._add_temperature_traces(fig, dataset)

        # Configure layout
        self._configure_layout(fig)

        # Apply axis settings from plot_parameters
        self._apply_axis_settings()

        # Add annotation for datasets without temperature data
        if datasets_without_temp:
            self._add_missing_data_annotation(
                fig, datasets_without_temp, len(datasets_with_temp) > 0
            )

        # Clean up trace names
        self._clean_trace_names()

        return fig

    def _add_temperature_traces(self, fig: go.Figure, dataset: "Dataset") -> None:
        """Add temperature traces for a dataset.

        Adds both furnace setpoint (dashed line) and thermocouple reading
        (solid line) to the temperature plot.

        Args:
            fig: Figure to add traces to
            dataset: Dataset object with temperature data (thermocouple_data,
                local_temperature_data, time_data attributes)
        """
        time_data = np.ascontiguousarray(dataset.time_data)
        local_temp = np.ascontiguousarray(dataset.local_temperature_data)
        thermocouple_temp = np.ascontiguousarray(dataset.thermocouple_data)
        colour = dataset.colour
        thermocouple_name = dataset.thermocouple_name or "Thermocouple"

        # Add furnace setpoint trace (dashed line)
        fig.add_trace(
            go.Scatter(
                mode="lines",
                name=f"{dataset.name} - Furnace setpoint (C)",
                line=dict(color=colour, width=1.5, dash="dash"),
            ),
            hf_x=time_data,
            hf_y=local_temp,
        )

        # Add thermocouple temperature trace (solid line)
        fig.add_trace(
            go.Scatter(
                mode="lines",
                name=f"{dataset.name} - {thermocouple_name} (C)",
                line=dict(color=colour, width=2),
            ),
            hf_x=time_data,
            hf_y=thermocouple_temp,
        )

    def _configure_layout(self, fig: go.Figure) -> None:
        """Configure the plot layout.

        Args:
            fig: Figure to configure (kept for compatibility with call sites)
        """
        self.figure_resampler.update_layout(
            height=500,
            xaxis_title="Time (s)",
            yaxis_title="Temperature (°C)",
            template="plotly_white",
            margin=dict(l=60, r=30, t=40, b=60),
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5
            ),
        )

    def _add_missing_data_annotation(
        self, fig: go.Figure, missing_datasets: list[str], has_some_data: bool
    ) -> None:
        """Add annotation for datasets without temperature data.

        Args:
            fig: Figure to add annotation to
            missing_datasets: List of dataset names without temperature data
            has_some_data: Whether any datasets have temperature data
        """
        if not has_some_data:
            # No temperature data at all - show centered message
            annotation_text = "No temperature data available"
            x_pos, y_pos = 0.5, 0.5
            font_size = 16
            x_anchor = "center"
            y_anchor = "middle"
        else:
            # Some datasets have data, show smaller message in corner
            if len(missing_datasets) == 1:
                annotation_text = f"No temperature data for: {missing_datasets[0]}"
            else:
                annotation_text = (
                    f"No temperature data for: {', '.join(missing_datasets)}"
                )
            x_pos, y_pos = 0.98, 0.02
            font_size = 10
            x_anchor = "right"
            y_anchor = "bottom"

        fig.add_annotation(
            text=annotation_text,
            xref="paper",
            yref="paper",
            x=x_pos,
            y=y_pos,
            xanchor=x_anchor,
            yanchor=y_anchor,
            showarrow=False,
            font=dict(size=font_size, color="gray"),
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="gray",
            borderwidth=1,
            borderpad=4,
        )

    def _clean_trace_names(self) -> None:
        """Remove [R] annotations from trace names added by resampler."""
        for trace in self.figure_resampler.data:
            if hasattr(trace, "name") and trace.name and "[R]" in trace.name:
                trace.name = trace.name.replace("[R] ", "").replace("[R]", "")
