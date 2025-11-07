import io
import math
import os
import threading
import webbrowser
import zipfile

import dash
import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objects as go
from plotly_resampler import FigureResampler

from . import layout_components as lc
from .analysis import evaluate_permeability_values, fit_permeability_data
from .callbacks import register_all_callbacks
from .dataset import Dataset
from .dataset_table import DatasetTable
from .helpers import import_htm_data


class DataPlotter:
    """Plotter UI for pressure gauge datasets using Dash.

    Provides a Dash app that displays upstream and downstream pressure
    plots for multiple datasets. Datasets are stored in `self.datasets` as a list.

    Args:
        dataset_paths: List of folder paths containing datasets to load on
        dataset_names: List of names corresponding to each dataset path
        port: Port number for Dash app (default: 8050)

    Attributes:
        dataset_paths: List of folder paths containing datasets to load on
        dataset_names: List of names corresponding to each dataset path
        port: Port number for Dash app (default: 8050)
        app: Dash app instance
        datasets: List of Dataset instances for plotting
        figure_resamplers: Dictionary of FigureResampler instances for each plot
    """

    # Type hints / attributes
    dataset_paths: list[str]
    dataset_names: list[str]
    port: int

    app: dash.Dash
    datasets: list[Dataset]
    upstream_datasets: list[dict]
    downstream_datasets: list[dict]

    def __init__(self, dataset_paths=None, dataset_names=None, port=8050):
        self.dataset_paths = dataset_paths or []
        self.dataset_names = dataset_names or []
        self.port = port

        # Initialize Dash app``
        self.app = dash.Dash(
            __name__,
            external_stylesheets=[
                dbc.themes.BOOTSTRAP,
                "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css",
            ],
        )
        # set the browser tab title
        self.app.title = "SHIELD Data Visualisation"

        # Store datasets as a list
        self.datasets = []

        # Store FigureResampler instances for callback registration
        self.figure_resamplers = {}

    @property
    def dataset_paths(self) -> list[str]:
        return self._dataset_paths

    @dataset_paths.setter
    def dataset_paths(self, value: list[str]):
        # if value not a list of strings raise ValueError
        if not isinstance(value, list) or not all(
            isinstance(item, str) for item in value
        ):
            raise ValueError("dataset_paths must be a list of strings")

        # Check if all dataset paths exist
        for dataset_path in value:
            if not os.path.exists(dataset_path):
                raise ValueError(f"Dataset path does not exist: {dataset_path}")

        # check all dataset paths are unique
        if len(value) != len(set(value)):
            raise ValueError("dataset_paths must contain unique paths")

        # check csv files exist in each dataset path
        for dataset_path in value:
            csv_files = [
                f for f in os.listdir(dataset_path) if f.lower().endswith(".csv")
            ]
            if not csv_files:
                raise FileNotFoundError(
                    f"No data CSV files found in dataset path: {dataset_path}"
                )

        # check that run_metadata.json exists in each dataset path
        for dataset_path in value:
            metadata_file = os.path.join(dataset_path, "run_metadata.json")
            if not os.path.exists(metadata_file):
                raise FileNotFoundError(
                    f"No run_metadata.json file found in dataset path: {dataset_path}"
                )

        self._dataset_paths = value

    @property
    def dataset_names(self) -> list[str]:
        return self._dataset_names

    @dataset_names.setter
    def dataset_names(self, value: list[str]):
        # if value not a list of strings raise ValueError
        if not isinstance(value, list) or not all(
            isinstance(item, str) for item in value
        ):
            raise ValueError("dataset_names must be a list of strings")

        # Check if dataset_names length matches dataset_paths length
        if len(value) != len(self.dataset_paths):
            raise ValueError(
                f"dataset_names length ({len(value)}) must match dataset_paths "
                f"length ({len(self.dataset_paths)})"
            )

        # Check if all dataset names are unique
        if len(value) != len(set(value)):
            raise ValueError("dataset_names must contain unique names")

        self._dataset_names = value

    def load_data(self, dataset_path: str, dataset_name: str):
        """
        Load and process data from specified data path using Dataset class.

        Only supports metadata version 1.3. Will raise ValueError for other versions.

        Args:
            dataset_path: Path to dataset folder containing run_metadata.json
            dataset_name: Name to assign to this dataset

        Raises:
            ValueError: If metadata version is not 1.3
            FileNotFoundError: If metadata file or data files are missing
        """
        # Create Dataset instance
        dataset = Dataset(path=dataset_path, name=dataset_name)

        # Assign color
        dataset.colour = self.get_next_color(len(self.datasets))

        # Process the data (loads from files)
        dataset.process_data()

        # Add to list
        self.datasets.append(dataset)

    def get_next_color(self, index: int) -> str:
        """
        Get a color for the dataset based on its index.

        Args:
            index: Index of the dataset

        Returns:
            str: Color hex code
        """
        colors = [
            "#000000",  # Black
            "#DF1AD2",  # Magenta
            "#779BE7",  # Light Blue
            "#49B6FF",  # Blue
            "#254E70",  # Dark Blue
            "#0CCA4A",  # Green
            "#929487",  # Gray
            "#A1B0AB",  # Light Gray
        ]
        return colors[index % len(colors)]

    def create_layout(self):
        """Create the main Dash layout using component builders.

        Returns:
            dbc.Container: The complete Dash layout
        """
        from .layout_components import create_download_components, create_hidden_stores

        return dbc.Container(
            [
                lc.create_header(),
                lc.create_dataset_management_card(self.create_dataset_table()),
                *create_hidden_stores(),
                lc.create_pressure_plots_row(
                    self._generate_upstream_plot(),
                    self._generate_downstream_plot(),
                ),
                lc.create_plot_controls_row(),
                lc.create_temperature_plot_card(self._generate_temperature_plot()),
                lc.create_permeability_plot_card(self._generate_permeability_plot()),
                lc.create_bottom_spacing(),
                *create_download_components(),
                lc.create_live_data_interval(),
            ],
            fluid=True,
        )

    def create_dataset_table(self):
        """Create the dataset table using DatasetTable class.

        Returns:
            html.Div: The complete dataset table component
        """
        table = DatasetTable(self.datasets)
        return table.create()

    def register_callbacks(self):
        """Register all Dash callbacks for the application.

        Callbacks are organized into modules by feature area:
        - dataset_callbacks: Dataset CRUD operations
        - ui_callbacks: UI collapse/expand interactions
        - plot_control_callbacks: Plot settings (scale, range, error bars)
        - live_data_callbacks: Live data toggle and updates
        - export_callbacks: Plot export and interactive zoom/pan
        """
        register_all_callbacks(self)

    def _generate_both_plots(
        self,
        show_error_bars_upstream=True,
        show_error_bars_downstream=True,
        show_valve_times_upstream=False,
        show_valve_times_downstream=False,
        **kwargs,
    ):
        """Helper method to generate both upstream and downstream plots
        with common parameters"""
        return [
            self._generate_upstream_plot(
                show_error_bars=show_error_bars_upstream,
                show_valve_times=show_valve_times_upstream,
                **kwargs,
            ),
            self._generate_downstream_plot(
                show_error_bars=show_error_bars_downstream,
                show_valve_times=show_valve_times_downstream,
                **kwargs,
            ),
            self._generate_temperature_plot(),
        ]

    def _generate_upstream_plot(
        self,
        show_error_bars=True,
        show_valve_times=False,
        x_scale=None,
        y_scale=None,
        x_min=None,
        x_max=None,
        y_min=None,
        y_max=None,
    ):
        """Generate the upstream pressure plot"""
        # Use FigureResampler with parameters to hide resampling annotations
        fig = FigureResampler(
            go.Figure(),
            show_dash_kwargs={"mode": "disabled"},
            show_mean_aggregation_size=False,
            verbose=False,
        )

        # Store the FigureResampler instance
        self.figure_resamplers["upstream-plot"] = fig

        # Iterate through datasets and obtain the upstream data
        for i, dataset in enumerate(self.datasets):
            time_data = np.ascontiguousarray(dataset.time_data)
            pressure_data = np.ascontiguousarray(dataset.upstream_pressure)
            pressure_error = np.ascontiguousarray(dataset.upstream_error)
            colour = dataset.colour

            # Debug: Check array lengths
            if len(time_data) != len(pressure_data):
                print(
                    f"WARNING: Dataset {dataset.name}: time_data length={len(time_data)}, pressure_data length={len(pressure_data)}"
                )
                print("  Trimming to minimum length")
                min_len = min(len(time_data), len(pressure_data))
                time_data = time_data[:min_len]
                pressure_data = pressure_data[:min_len]
                pressure_error = (
                    pressure_error[:min_len]
                    if len(pressure_error) > min_len
                    else pressure_error
                )

            # Create scatter trace
            scatter_kwargs = {
                "mode": "lines+markers",
                "name": dataset.name,
                "line": dict(color=colour, width=1.5),
                "marker": dict(size=3),
            }

            # Add error bars if enabled
            if show_error_bars:
                scatter_kwargs["error_y"] = dict(
                    type="data",
                    array=pressure_error,
                    visible=True,
                    color=colour,
                    thickness=1.5,
                    width=3,
                )

            # Use plotly-resampler for automatic downsampling
            fig.add_trace(
                go.Scatter(**scatter_kwargs), hf_x=time_data, hf_y=pressure_data
            )

            # Add valve time vertical lines
            if show_valve_times:
                valve_times = dataset.valve_times
                for valve_event, valve_time in valve_times.items():
                    fig.add_vline(
                        x=valve_time,
                        line_dash="dash",
                        line_color=colour,
                        line_width=1,
                        annotation_text=valve_event.replace("_", " ").title(),
                        annotation_position="top",
                        annotation_textangle=0,
                        annotation_font_size=8,
                    )

        # Configure the layout
        fig.update_layout(
            height=500,
            xaxis_title="Time (s)",
            yaxis_title="Pressure (Torr)",
            template="plotly_white",
            margin=dict(l=60, r=30, t=40, b=60),
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5
            ),
        )

        # Apply axis scaling
        x_axis_type = x_scale if x_scale else "linear"
        y_axis_type = y_scale if y_scale else "linear"

        fig.update_xaxes(type=x_axis_type)
        fig.update_yaxes(type=y_axis_type)

        # Determine x-axis range from data (or use provided bounds when valid)
        if x_axis_type == "log":
            if (
                x_min is not None
                and x_max is not None
                and x_min > 0
                and x_max > 0
                and x_min < x_max
            ):
                xmin_lin, xmax_lin = float(x_min), float(x_max)
            else:
                pos_vals = []
                for ds in self.datasets:
                    try:
                        vals = np.asarray(ds.time_data, dtype=float)
                        vals = vals[vals > 0]
                        if vals.size:
                            pos_vals.extend(vals.tolist())
                    except Exception:
                        continue
                if pos_vals:
                    xmin_lin = float(min(pos_vals))
                    xmax_lin = float(max(pos_vals))
                    if xmin_lin >= xmax_lin:
                        xmax_lin = xmin_lin * 10.0
                else:
                    xmin_lin, xmax_lin = 1e-12, 1e-6

            fig.update_xaxes(range=[math.log10(xmin_lin), math.log10(xmax_lin)])
        else:
            if x_min is not None and x_max is not None and x_min < x_max:
                fig.update_xaxes(range=[x_min, x_max])
            else:
                # derive from data
                vals = []
                for ds in self.datasets:
                    try:
                        v = ds.time_data
                        vals.extend([float(x) for x in v])
                    except Exception:
                        continue
                if vals:
                    fig.update_xaxes(range=[min(vals), max(vals)])

        # Determine y-axis range from upstream data (or use provided bounds when valid)
        if y_axis_type == "log":
            if (
                y_min is not None
                and y_max is not None
                and y_min > 0
                and y_max > 0
                and y_min < y_max
            ):
                ymin_lin, ymax_lin = float(y_min), float(y_max)
            else:
                pos_vals = []
                for ds in self.datasets:
                    try:
                        vals = np.asarray(ds.upstream_pressure, dtype=float)
                        vals = vals[vals > 0]
                        if vals.size:
                            pos_vals.extend(vals.tolist())
                    except Exception:
                        continue
                if pos_vals:
                    ymin_lin = float(min(pos_vals))
                    ymax_lin = float(max(pos_vals))
                    if ymin_lin >= ymax_lin:
                        ymax_lin = ymin_lin * 10.0
                else:
                    ymin_lin, ymax_lin = 1e-12, 1e-6

            fig.update_yaxes(range=[math.log10(ymin_lin), math.log10(ymax_lin)])
        else:
            if y_min is not None and y_max is not None and y_min < y_max:
                fig.update_yaxes(range=[y_min, y_max])
            else:
                vals = []
                for ds in self.datasets:
                    try:
                        v = ds.upstream_pressure
                        vals.extend([float(x) for x in v])
                    except Exception:
                        continue
                if vals:
                    fig.update_yaxes(range=[min(vals), max(vals)])

        # Clean up trace names to remove [R] annotations
        for trace in fig.data:
            if hasattr(trace, "name") and trace.name and "[R]" in trace.name:
                trace.name = trace.name.replace("[R] ", "").replace("[R]", "")

        return fig

    def _generate_downstream_plot(
        self,
        show_error_bars=True,
        show_valve_times=False,
        x_scale=None,
        y_scale=None,
        x_min=None,
        x_max=None,
        y_min=None,
        y_max=None,
    ):
        """Generate the downstream pressure plot"""
        # Use FigureResampler with parameters to hide resampling annotations
        fig = FigureResampler(
            go.Figure(),
            show_dash_kwargs={"mode": "disabled"},
            show_mean_aggregation_size=False,
            verbose=False,
        )

        # Store the FigureResampler instance
        self.figure_resamplers["downstream-plot"] = fig

        # Iterate through datasets and obtain the downstream data
        for i, dataset in enumerate(self.datasets):
            time_data = np.ascontiguousarray(dataset.time_data)
            pressure_data = np.ascontiguousarray(dataset.downstream_pressure)
            pressure_error = np.ascontiguousarray(dataset.downstream_error)
            colour = dataset.colour

            # Debug: Check array lengths
            if len(time_data) != len(pressure_data):
                print(
                    f"WARNING: Dataset {dataset.name}: "
                    f"time_data length={len(time_data)}, "
                    f"pressure_data length={len(pressure_data)}"
                )
                print("  Trimming to minimum length")
                min_len = min(len(time_data), len(pressure_data))
                time_data = time_data[:min_len]
                pressure_data = pressure_data[:min_len]
                if len(pressure_error) > min_len:
                    pressure_error = pressure_error[:min_len]

            # Create scatter trace
            scatter_kwargs = {
                "mode": "lines+markers",
                "name": dataset.name,
                "line": dict(color=colour, width=1.5),
                "marker": dict(size=3),
            }

            # Add error bars if enabled
            if show_error_bars:
                scatter_kwargs["error_y"] = dict(
                    type="data",
                    array=pressure_error,
                    visible=True,
                    color=colour,
                    thickness=1.5,
                    width=3,
                )

            # Use plotly-resampler for automatic downsampling
            fig.add_trace(
                go.Scatter(**scatter_kwargs), hf_x=time_data, hf_y=pressure_data
            )

            # Add valve time vertical lines
            if show_valve_times:
                valve_times = dataset.valve_times
                for valve_event, valve_time in valve_times.items():
                    fig.add_vline(
                        x=valve_time,
                        line_dash="dash",
                        line_color=colour,
                        line_width=1,
                        annotation_text=valve_event.replace("_", " ").title(),
                        annotation_position="top",
                        annotation_textangle=0,
                        annotation_font_size=8,
                    )

        # Configure the layout
        fig.update_layout(
            height=500,
            xaxis_title="Time (s)",
            yaxis_title="Pressure (Torr)",
            template="plotly_white",
            margin=dict(l=60, r=30, t=40, b=60),
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5
            ),
        )
        # Apply axis scaling
        x_axis_type = x_scale if x_scale else "linear"
        y_axis_type = y_scale if y_scale else "linear"

        fig.update_xaxes(type=x_axis_type)
        fig.update_yaxes(type=y_axis_type)

        # Apply axis ranges if specified
        if x_min is not None and x_max is not None:
            if x_axis_type == "log":

                def _safe_x_log_range_ds(xmin_val, xmax_val):
                    try:
                        if xmin_val is not None and xmax_val is not None:
                            if xmin_val > 0 and xmax_val > 0 and xmin_val < xmax_val:
                                return math.log10(xmin_val), math.log10(xmax_val)
                    except Exception:
                        pass

                    pos_vals = []
                    for dataset in self.datasets:
                        try:
                            vals = dataset.time_data
                            for v in vals:
                                if v is not None and v > 0:
                                    pos_vals.append(float(v))
                        except Exception:
                            continue

                    if pos_vals:
                        xmin_p = min(pos_vals)
                        xmax_p = max(pos_vals)
                        if xmin_p <= 0:
                            xmin_p = min(x for x in pos_vals if x > 0)
                        if xmin_p >= xmax_p:
                            xmax_p = xmin_p * 10.0
                        return math.log10(xmin_p), math.log10(xmax_p)

                    return -12.0, -6.0

                x_min_log, x_max_log = _safe_x_log_range_ds(x_min, x_max)
                fig.update_xaxes(range=[x_min_log, x_max_log])
            else:
                fig.update_xaxes(range=[x_min, x_max])

        if y_min is not None and y_max is not None:
            if y_axis_type == "log":
                # For log scale, ensure positive bounds; if provided bounds are
                # non-positive, derive a safe range from the data.

                def _safe_log_range_ds(
                    downstream_or_upstream: str, y_min_val, y_max_val
                ):
                    try:
                        if y_min_val is not None and y_max_val is not None:
                            if (
                                y_min_val > 0
                                and y_max_val > 0
                                and y_min_val < y_max_val
                            ):
                                return y_min_val, y_max_val
                    except Exception:
                        pass

                    pos_vals = []
                    for dataset in self.datasets:
                        try:
                            if downstream_or_upstream == "downstream_data":
                                vals = dataset.downstream_pressure
                            else:
                                vals = dataset.upstream_pressure
                            for v in vals:
                                if v is not None and v > 0:
                                    pos_vals.append(float(v))
                        except Exception:
                            continue

                    if pos_vals:
                        ymin_p = min(pos_vals)
                        ymax_p = max(pos_vals)
                        if ymin_p <= 0:
                            ymin_p = min(x for x in pos_vals if x > 0)
                        if ymin_p >= ymax_p:
                            ymax_p = ymin_p * 10.0
                        return ymin_p, ymax_p

                    return 1e-12, 1e-6

                y_min_use, y_max_use = _safe_log_range_ds(
                    "downstream_data", y_min, y_max
                )
                fig.update_yaxes(range=[math.log10(y_min_use), math.log10(y_max_use)])
            else:
                fig.update_yaxes(range=[y_min, y_max])

        # Clean up trace names to remove [R] annotations
        for trace in fig.data:
            if hasattr(trace, "name") and trace.name and "[R]" in trace.name:
                trace.name = trace.name.replace("[R] ", "").replace("[R]", "")

        return fig

    def _generate_temperature_plot(self):
        """Generate the temperature plot for datasets with thermocouple data.

        Shows temperature traces for datasets with data, and displays a message
        for datasets without temperature data in the bottom right corner.
        """
        # Use FigureResampler with parameters to hide resampling annotations
        fig = FigureResampler(
            go.Figure(),
            show_dash_kwargs={"mode": "disabled"},
            show_mean_aggregation_size=False,
            verbose=False,
        )

        # Store the FigureResampler instance
        self.figure_resamplers["temperature-plot"] = fig

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
            time_data = np.ascontiguousarray(dataset.time_data)
            local_temp = np.ascontiguousarray(dataset.local_temperature_data)
            thermocouple_temp = np.ascontiguousarray(dataset.thermocouple_data)
            colour = dataset.colour
            thermocouple_name = dataset.thermocouple_name or "Thermocouple"

            # Add local temperature trace
            fig.add_trace(
                go.Scatter(
                    mode="lines",
                    name=f"{dataset.name} - Furnace setpoint (C)",
                    line=dict(color=colour, width=1.5, dash="dash"),
                ),
                hf_x=time_data,
                hf_y=local_temp,
            )

            # Add thermocouple temperature trace
            fig.add_trace(
                go.Scatter(
                    mode="lines",
                    name=f"{dataset.name} - {thermocouple_name} (C)",
                    line=dict(color=colour, width=2),
                ),
                hf_x=time_data,
                hf_y=thermocouple_temp,
            )

        # Configure the layout
        fig.update_layout(
            height=400,
            xaxis_title="Time (s)",
            yaxis_title="Temperature (°C)",
            template="plotly_white",
            margin=dict(l=60, r=30, t=40, b=60),
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5
            ),
        )

        # Add annotation for datasets without temperature data
        if datasets_without_temp:
            if len(datasets_with_temp) == 0:
                # No temperature data at all - show centered message
                annotation_text = "No temperature data available"
                x_pos, y_pos = 0.5, 0.5
                font_size = 16
            else:
                # Some datasets have data, show smaller message in corner
                if len(datasets_without_temp) == 1:
                    annotation_text = (
                        f"No temperature data for: {datasets_without_temp[0]}"
                    )
                else:
                    annotation_text = (
                        f"No temperature data for: {', '.join(datasets_without_temp)}"
                    )
                x_pos, y_pos = 0.98, 0.02
                font_size = 10

            fig.add_annotation(
                text=annotation_text,
                xref="paper",
                yref="paper",
                x=x_pos,
                y=y_pos,
                xanchor="right" if x_pos > 0.5 else "center",
                yanchor="bottom" if y_pos < 0.5 else "middle",
                showarrow=False,
                font=dict(size=font_size, color="gray"),
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor="gray",
                borderwidth=1,
                borderpad=4,
            )

        # Clean up trace names to remove [R] annotations
        for trace in fig.data:
            if hasattr(trace, "name") and trace.name and "[R]" in trace.name:
                trace.name = trace.name.replace("[R] ", "").replace("[R]", "")

        return fig

    def _generate_permeability_plot(self):
        """Generate permeability plot with HTM reference and measured data."""
        fig = FigureResampler(
            go.Figure(),
            show_dash_kwargs={"mode": "disabled"},
            show_mean_aggregation_size=False,
            verbose=False,
        )
        self.figure_resamplers["permeability-plot"] = fig

        # Add HTM reference data
        htm_x, htm_y, htm_labels = import_htm_data("316l_steel")
        for x, y, label in zip(htm_x, htm_y, htm_labels):
            fig.add_trace(go.Scatter(x=1000 / x, y=y, name=label))

        # Calculate and plot permeability for each dataset
        temps, perms, x_error, y_error, error_lower, error_upper = (
            evaluate_permeability_values(self.datasets)
        )

        # Add error bars (no visible markers, just the bars)
        fig.add_trace(
            go.Scatter(
                x=x_error,
                y=y_error,
                mode="markers",
                name="Error Range",
                marker=dict(size=0.1, color="black", opacity=0),
                error_y=dict(
                    type="data",
                    symmetric=False,
                    array=error_upper,
                    arrayminus=error_lower,
                    color="black",
                    thickness=2,
                    width=6,
                ),
                showlegend=False,
            )
        )

        # Add individual data points on top (no legend)
        # Extract nominal values from ufloat objects
        perm_values = np.array([p.n if hasattr(p, "n") else p for p in perms])
        fig.add_trace(
            go.Scatter(
                x=1000 / np.array(temps),
                y=perm_values,
                mode="markers",
                marker=dict(size=6, color="black"),
                showlegend=False,
            )
        )

        # Fit a line through all data points (in log space for permeability)
        fit_x, fit_y = fit_permeability_data(temps, perms)

        fig.add_trace(
            go.Scatter(
                x=fit_x,
                y=fit_y,
                mode="lines",
                name="SHIELD data",
                line=dict(color="black", width=2, dash="solid"),
                showlegend=True,
            )
        )

        # Configure layout
        fig.update_layout(
            xaxis_title="1000/T (K-1)",
            yaxis_title="Permeability (m-1 s-1 Pa-0.5)",
            yaxis_type="log",
            hovermode="closest",
            template="plotly_white",
            legend=dict(
                orientation="v", yanchor="top", y=0.99, xanchor="right", x=0.99
            ),
        )

        # Configure y-axis to show exponent at top (matplotlib style)
        fig.update_yaxes(exponentformat="e", showexponent="all")

        # Clean up resampler annotations
        for trace in fig.data:
            if hasattr(trace, "name") and trace.name and "[R]" in trace.name:
                trace.name = trace.name.replace("[R] ", "").replace("[R]", "")

        return fig

    def _create_dataset_download(self, dataset_path: str):
        """Package original dataset files for download based on metadata version.

        For version '0.0' zip all CSV files in the folder and return as bytes.
        For version '1.0' return the single CSV file named in run_info.data_filename

        Returns a dict suitable for dcc.send_bytes or dcc.send_file style use.
        """

        # Zip the entire dataset folder (all files and subfolders), preserving
        # the relative directory structure inside the archive.
        mem_zip = io.BytesIO()
        with zipfile.ZipFile(mem_zip, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            for root, _dirs, files in os.walk(dataset_path):
                for fname in files:
                    file_path = os.path.join(root, fname)
                    try:
                        arcname = os.path.relpath(file_path, dataset_path)
                        zf.write(file_path, arcname)
                    except Exception:
                        # Skip files we fail to read/write into the archive
                        continue
        mem_zip.seek(0)
        # Use a normalized basename in case dataset_path ends with a slash
        base = os.path.basename(os.path.normpath(dataset_path))
        return dict(
            content=mem_zip.getvalue(),
            filename=f"{base}.zip",
            type="application/zip",
        )

    def start(self):
        """Process data and start the Dash web server"""

        # Process data
        for dataset_path, dataset_name in zip(self.dataset_paths, self.dataset_names):
            self.load_data(dataset_path, dataset_name)

        # Setup the app layout
        self.app.layout = self.create_layout()

        # Add custom CSS for hover effects
        self.app.index_string = hover_css

        custom_favicon_link = (
            '<link rel="icon" href="/assets/shield.svg" type="image/svg+xml">'
        )
        self.app.index_string = hover_css.replace("{%favicon%}", custom_favicon_link)

        # Register callbacks
        self.register_callbacks()

        print(f"Starting dashboard on http://localhost:{self.port}")

        # Open web browser after a short delay
        threading.Timer(
            0.1, lambda: webbrowser.open(f"http://127.0.0.1:{self.port}")
        ).start()

        # Run the server directly (blocking)
        self.app.run(debug=False, host="127.0.0.1", port=self.port)


hover_css = """
    <!DOCTYPE html>
    <html>
        <head>
            {%metas%}
            <title>{%title%}</title>
            {%favicon%}
            {%css%}
            <style>
                .dataset-name-input:hover {
                    border-color: #007bff !important;
                    box-shadow: 0 0 0 0.2rem rgba(0, 123, 255, 0.25) !important;
                    transform: scale(1.01) !important;
                }

                .dataset-name-input:focus {
                    border-color: #007bff !important;
                    box-shadow: 0 0 0 0.2rem rgba(0, 123, 255, 0.25) !important;
                    outline: 0 !important;
                }

                .color-picker-input:hover {
                    border-color: #007bff !important;
                    box-shadow: 0 0 0 2px rgba(0, 123, 255, 0.4) !important;
                    transform: none !important;
                }

                .color-picker-input:focus {
                    border-color: #007bff !important;
                    box-shadow: 0 0 0 2px rgba(0, 123, 255, 0.4) !important;
                    outline: 0 !important;
                }
            </style>
        </head>
        <body>
            {%app_entry%}
            <footer>
                {%config%}
                {%scripts%}
                {%renderer%}
            </footer>
        </body>
    </html>
    """
