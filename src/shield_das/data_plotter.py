import io
import json
import math
import os
import threading
import webbrowser
import zipfile
from datetime import datetime

import dash
import dash_bootstrap_components as dbc
import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from dash import ALL, dcc, html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from plotly_resampler import FigureResampler

from .analysis import (
    calculate_error_on_pressure_reading,
    evaluate_permeability_values,
    fit_permeability_data,
    voltage_to_pressure,
    voltage_to_temperature,
)
from .dataset import Dataset
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

    # Helper constants for repeated callback state patterns
    PLOT_CONTROL_STATES = [
        State("show-error-bars-upstream", "value"),
        State("show-error-bars-downstream", "value"),
        State("show-valve-times-upstream", "value"),
        State("show-valve-times-downstream", "value"),
    ]

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

        # Initialize Dash app
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
        from .layout_components import (
            create_bottom_spacing,
            create_dataset_management_card,
            create_download_components,
            create_header,
            create_hidden_stores,
            create_live_data_interval,
            create_permeability_plot_card,
            create_plot_controls_row,
            create_pressure_plots_row,
            create_temperature_plot_card,
        )

        return dbc.Container(
            [
                create_header(),
                create_dataset_management_card(self.create_dataset_table()),
                *create_hidden_stores(),
                create_pressure_plots_row(
                    self._generate_upstream_plot(),
                    self._generate_downstream_plot(),
                ),
                create_plot_controls_row(),
                create_temperature_plot_card(self._generate_temperature_plot()),
                create_permeability_plot_card(self._generate_permeability_plot()),
                create_bottom_spacing(),
                *create_download_components(),
                create_live_data_interval(),
            ],
            fluid=True,
        )

    def create_dataset_table(self):
        """Create a table showing folder-level datasets with editable name and color"""
        # Create table rows
        rows = []

        # Header row
        header_row = html.Tr(
            [
                html.Th(
                    "Dataset Name",
                    style={
                        "text-align": "left",
                        "width": "43.75%",
                        "padding": "2px",
                        "font-weight": "normal",
                    },
                ),
                html.Th(
                    "Dataset Path",
                    style={
                        "text-align": "left",
                        "width": "43.75%",
                        "padding": "2px",
                        "font-weight": "normal",
                    },
                ),
                html.Th(
                    "Live",
                    style={
                        "text-align": "center",
                        "width": "2.5%",
                        "padding": "2px",
                        "font-weight": "normal",
                    },
                ),
                html.Th(
                    "Colour",
                    style={
                        "text-align": "center",
                        "width": "5%",
                        "padding": "2px",
                        "font-weight": "normal",
                    },
                ),
                html.Th(
                    "",
                    style={
                        "text-align": "center",
                        "width": "2.5%",
                        "padding": "2px",
                        "font-weight": "normal",
                    },
                ),
                html.Th(
                    "",
                    style={
                        "text-align": "center",
                        "width": "2.5%",
                        "padding": "2px",
                        "font-weight": "normal",
                    },
                ),
            ]
        )
        rows.append(header_row)

        # Add dataset rows
        for i, dataset in enumerate(self.datasets):
            row = html.Tr(
                [
                    html.Td(
                        [
                            dcc.Input(
                                id={"type": "dataset-name", "index": i},
                                value=dataset.name,
                                style={
                                    "width": "95%",
                                    "border": "1px solid #ccc",
                                    "padding": "4px",
                                    "border-radius": "4px",
                                    "transition": "all 0.2s ease",
                                },
                                className="dataset-name-input",
                            )
                        ],
                        style={"padding": "2px", "border": "none"},
                    ),
                    html.Td(
                        [
                            html.Div(
                                [
                                    html.Span(
                                        dataset.path,
                                        style={
                                            "font-family": "monospace",
                                            "font-size": "0.9em",
                                            "color": "#666",
                                            "word-break": "break-all",
                                        },
                                        title=dataset.path,  # Full path on hover
                                    )
                                ],
                                style={
                                    "width": "100%",
                                    "padding": "4px",
                                    "min-height": "1.5em",  # Match input field height
                                    "display": "flex",
                                    "align-items": "center",
                                },
                            )
                        ],
                        style={"padding": "4px", "border": "none"},
                    ),
                    html.Td(
                        [
                            html.Div(
                                [
                                    dbc.Checkbox(
                                        id={"type": "dataset-live-data", "index": i},
                                        value=dataset.live_data,
                                        style={
                                            "transform": "scale(1.2)",
                                            "display": "inline-block",
                                        },
                                    ),
                                ],
                                style={
                                    "margin-left": "15px",
                                },
                            )
                        ],
                        style={
                            "padding": "4px",
                            "text-align": "center",
                            "border": "none",
                        },
                    ),
                    html.Td(
                        [
                            dcc.Input(
                                id={"type": "dataset-color", "index": i},
                                type="color",
                                value=dataset.colour,
                                style={
                                    "width": "32px",
                                    "height": "32px",
                                    "border": "2px solid transparent",
                                    "border-radius": "4px",
                                    "cursor": "pointer",
                                    "transition": "all 0.2s ease",
                                    "padding": "0",
                                    "outline": "none",
                                },
                                className="color-picker-input",
                            ),
                        ],
                        style={
                            "text-align": "center",
                            "padding": "4px",
                            "border": "none",
                        },
                    ),
                    html.Td(
                        [
                            html.Div(
                                [
                                    html.Button(
                                        html.Img(
                                            src="/assets/download.svg",
                                            style={
                                                "width": "16px",
                                                "height": "16px",
                                            },
                                        ),
                                        id={"type": "download-dataset", "index": i},
                                        className="btn btn-outline-primary btn-sm",
                                        style={
                                            "width": "32px",
                                            "height": "32px",
                                            "padding": "0",
                                            "border-radius": "4px",
                                            "font-size": "14px",
                                            "line-height": "1",
                                            "display": "flex",
                                            "align-items": "center",
                                            "justify-content": "center",
                                        },
                                        title=f"Download {dataset.name}",
                                    ),
                                ],
                                style={
                                    "margin-left": "15px",  # Add left margin
                                },
                            )
                        ],
                        style={
                            "text-align": "center",
                            "padding": "4px",
                            "vertical-align": "middle",
                            "border": "none",
                        },
                    ),
                    html.Td(
                        [
                            html.Div(
                                [
                                    html.Button(
                                        html.Img(
                                            src="/assets/delete.svg",
                                            style={
                                                "width": "16px",
                                                "height": "16px",
                                            },
                                        ),
                                        id={"type": "delete-dataset", "index": i},
                                        className="btn btn-outline-danger btn-sm",
                                        style={
                                            "width": "32px",
                                            "height": "32px",
                                            "padding": "0",
                                            "border-radius": "4px",
                                            "font-size": "14px",
                                            "line-height": "1",
                                            "display": "flex",
                                            "align-items": "center",
                                            "justify-content": "center",
                                        },
                                        title=f"Delete {dataset.name}",
                                    ),
                                ],
                                style={
                                    "margin-left": "15px",  # Add left margin
                                },
                            )
                        ],
                        style={
                            "text-align": "center",
                            "padding": "4px",
                            "vertical-align": "middle",
                            "border": "none",
                        },
                    ),
                ]
            )
            rows.append(row)

        # Create the table
        table = html.Table(
            rows,
            className="table table-striped table-hover",
            style={
                "margin": "0",
                "border": "1px solid #dee2e6",
                "border-radius": "8px",
                "overflow": "hidden",
            },
        )

        return html.Div([table])

    def register_callbacks(self):
        # No longer needed - self.datasets is now a list, not a dict
        # Helper functions kept for backward compatibility during transition
        def _keys_list():
            # Return list of indices for the datasets list
            return list(range(len(self.datasets)))

        def _iter_datasets():
            return self.datasets

        # Callback for dataset name changes
        @self.app.callback(
            [
                Output("dataset-table-container", "children", allow_duplicate=True),
                Output("upstream-plot", "figure", allow_duplicate=True),
                Output("downstream-plot", "figure", allow_duplicate=True),
            ],
            [Input({"type": "dataset-name", "index": ALL}, "value")],
            self.PLOT_CONTROL_STATES,
            prevent_initial_call=True,
        )
        def update_dataset_names(
            names,
            show_error_bars_upstream,
            show_error_bars_downstream,
            show_valve_times_upstream,
            show_valve_times_downstream,
        ):
            # Build current names list for comparison to avoid double application
            current_names = [ds.name for ds in self.datasets]

            # If nothing changed, skip to avoid duplicate updates
            if list(names) == current_names:
                raise PreventUpdate

            # Update only entries that changed
            for i, name in enumerate(names):
                if i < len(self.datasets) and name and name != current_names[i]:
                    self.datasets[i].name = name

            # Return updated table and plots
            plots = self._generate_both_plots(
                show_error_bars_upstream=show_error_bars_upstream,
                show_error_bars_downstream=show_error_bars_downstream,
                show_valve_times_upstream=show_valve_times_upstream,
                show_valve_times_downstream=show_valve_times_downstream,
            )
            return [self.create_dataset_table(), *plots]

        # Callback for dataset color changes
        @self.app.callback(
            [
                Output("dataset-table-container", "children", allow_duplicate=True),
                Output("upstream-plot", "figure", allow_duplicate=True),
                Output("downstream-plot", "figure", allow_duplicate=True),
                Output("temperature-plot", "figure", allow_duplicate=True),
            ],
            [Input({"type": "dataset-color", "index": ALL}, "value")],
            prevent_initial_call=True,
        )
        def update_dataset_colors(colors):
            for i, color in enumerate(colors):
                if i < len(self.datasets) and color:
                    self.datasets[i].colour = color

            # Return updated table and plots
            return [
                self.create_dataset_table(),
                self._generate_upstream_plot(),
                self._generate_downstream_plot(),
                self._generate_temperature_plot(),
            ]

        # Callback to handle collapse/expand of dataset management
        @self.app.callback(
            [
                Output("collapse-dataset", "is_open"),
                Output("collapse-dataset-button", "children"),
            ],
            [Input("collapse-dataset-button", "n_clicks")],
            [State("collapse-dataset", "is_open")],
            prevent_initial_call=True,
        )
        def toggle_dataset_collapse(n_clicks, is_open):
            if n_clicks:
                new_state = not is_open
                # Change icon based on state
                if new_state:
                    icon = html.I(className="fas fa-chevron-down")
                else:
                    icon = html.I(className="fas fa-chevron-up")
                return new_state, icon
            return is_open, html.I(className="fas fa-chevron-up")

        # Callbacks to handle collapse/expand of upstream plot controls
        @self.app.callback(
            [
                Output("collapse-upstream-controls", "is_open"),
                Output("collapse-upstream-controls-button", "children"),
            ],
            [Input("collapse-upstream-controls-button", "n_clicks")],
            [State("collapse-upstream-controls", "is_open")],
            prevent_initial_call=True,
        )
        def toggle_upstream_controls_collapse(n_clicks, is_open):
            if n_clicks:
                new_state = not is_open
                # Change icon based on state
                if new_state:
                    icon = html.I(className="fas fa-chevron-down")
                else:
                    icon = html.I(className="fas fa-chevron-up")
                return new_state, icon
            return is_open, html.I(className="fas fa-chevron-up")

        # Callbacks to handle collapse/expand of downstream plot controls
        @self.app.callback(
            [
                Output("collapse-downstream-controls", "is_open"),
                Output("collapse-downstream-controls-button", "children"),
            ],
            [Input("collapse-downstream-controls-button", "n_clicks")],
            [State("collapse-downstream-controls", "is_open")],
            prevent_initial_call=True,
        )
        def toggle_downstream_controls_collapse(n_clicks, is_open):
            if n_clicks:
                new_state = not is_open
                # Change icon based on state
                if new_state:
                    icon = html.I(className="fas fa-chevron-down")
                else:
                    icon = html.I(className="fas fa-chevron-up")
                return new_state, icon
            return is_open, html.I(className="fas fa-chevron-up")

        # Callback for upstream plot settings changes
        @self.app.callback(
            [
                Output("upstream-plot", "figure", allow_duplicate=True),
                Output("upstream-settings-store", "data"),
            ],
            [
                Input("upstream-x-scale", "value"),
                Input("upstream-y-scale", "value"),
                Input("upstream-x-min", "value"),
                Input("upstream-x-max", "value"),
                Input("upstream-y-min", "value"),
                Input("upstream-y-max", "value"),
                Input("show-error-bars-upstream", "value"),
                Input("show-valve-times-upstream", "value"),
            ],
            [
                State("upstream-plot", "figure"),
                State("upstream-settings-store", "data"),
            ],
            prevent_initial_call=True,
        )
        def update_upstream_plot_settings(
            x_scale,
            y_scale,
            x_min,
            x_max,
            y_min,
            y_max,
            show_error_bars_upstream,
            show_valve_times_upstream,
            current_fig,
            store_data,
        ):
            # Helper to extract axis ranges from an existing figure
            def _extract_axis_range(fig, axis_name):
                if not fig:
                    return None, None
                layout = fig.get("layout", {})
                # common axis keys
                for key in (axis_name, f"{axis_name}1"):
                    ax = layout.get(key)
                    if isinstance(ax, dict):
                        r = ax.get("range")
                        if r and len(r) == 2:
                            return r[0], r[1]
                return None, None

            # Simpler behavior: always reset the figure when scale or error-bar
            # options change. Do not preserve current zoom. If inputs are None
            # pass None so the generator autosizes.
            x_min_use = x_min if x_min is not None else None
            x_max_use = x_max if x_max is not None else None
            y_min_use = y_min if y_min is not None else None
            y_max_use = y_max if y_max is not None else None

            # Generate updated upstream plot with new settings (use keywords)
            # Update store data with current scale settings
            new_store = {"y_scale": y_scale}

            return [
                self._generate_upstream_plot(
                    show_error_bars=bool(show_error_bars_upstream),
                    show_valve_times=bool(show_valve_times_upstream),
                    x_scale=x_scale,
                    y_scale=y_scale,
                    x_min=x_min_use,
                    x_max=x_max_use,
                    y_min=y_min_use,
                    y_max=y_max_use,
                ),
                new_store,
            ]

        # Callback for downstream plot settings changes
        @self.app.callback(
            [
                Output("downstream-plot", "figure", allow_duplicate=True),
                Output("downstream-settings-store", "data"),
            ],
            [
                Input("downstream-x-scale", "value"),
                Input("downstream-y-scale", "value"),
                Input("downstream-x-min", "value"),
                Input("downstream-x-max", "value"),
                Input("downstream-y-min", "value"),
                Input("downstream-y-max", "value"),
                Input("show-error-bars-downstream", "value"),
                Input("show-valve-times-downstream", "value"),
            ],
            [
                State("downstream-plot", "figure"),
                State("downstream-settings-store", "data"),
            ],
            prevent_initial_call=True,
        )
        def update_downstream_plot_settings(
            x_scale,
            y_scale,
            x_min,
            x_max,
            y_min,
            y_max,
            show_error_bars_downstream,
            show_valve_times_downstream,
            current_fig,
            store_data,
        ):
            # Helper to extract axis ranges from an existing figure
            def _extract_axis_range(fig, axis_name):
                if not fig:
                    return None, None
                layout = fig.get("layout", {})
                for key in (axis_name, f"{axis_name}1"):
                    ax = layout.get(key)
                    if isinstance(ax, dict):
                        r = ax.get("range")
                        if r and len(r) == 2:
                            return r[0], r[1]
                return None, None

            # Simpler behavior: always reset the figure when scale or error-bar
            # options change. Do not preserve current zoom. If inputs are None
            # pass None so the generator autosizes.
            x_min_use = x_min if x_min is not None else None
            x_max_use = x_max if x_max is not None else None
            y_min_use = y_min if y_min is not None else None
            y_max_use = y_max if y_max is not None else None

            new_store = {"y_scale": y_scale}

            # Generate updated downstream plot with new settings (use keywords)
            return [
                self._generate_downstream_plot(
                    show_error_bars=bool(show_error_bars_downstream),
                    show_valve_times=bool(show_valve_times_downstream),
                    x_scale=x_scale,
                    y_scale=y_scale,
                    x_min=x_min_use,
                    x_max=x_max_use,
                    y_min=y_min_use,
                    y_max=y_max_use,
                ),
                new_store,
            ]

        # Callbacks to update min values based on scale mode
        @self.app.callback(
            [Output("upstream-x-min", "value")],
            [Input("upstream-x-scale", "value")],
            prevent_initial_call=True,
        )
        def update_upstream_x_min(x_scale):
            return [0 if x_scale == "linear" else None]

        @self.app.callback(
            [Output("upstream-y-min", "value")],
            [Input("upstream-y-scale", "value")],
            prevent_initial_call=True,
        )
        def update_upstream_y_min(y_scale):
            return [0 if y_scale == "linear" else None]

        @self.app.callback(
            [Output("downstream-x-min", "value")],
            [Input("downstream-x-scale", "value")],
            prevent_initial_call=True,
        )
        def update_downstream_x_min(x_scale):
            return [0 if x_scale == "linear" else None]

        @self.app.callback(
            [Output("downstream-y-min", "value")],
            [Input("downstream-y-scale", "value")],
            prevent_initial_call=True,
        )
        def update_downstream_y_min(y_scale):
            return [0 if y_scale == "linear" else None]

        # Callback for adding new dataset
        @self.app.callback(
            [
                Output("dataset-table-container", "children", allow_duplicate=True),
                Output("upstream-plot", "figure", allow_duplicate=True),
                Output("downstream-plot", "figure", allow_duplicate=True),
                Output("new-dataset-path", "value"),
                Output("add-dataset-status", "children"),
            ],
            [Input("add-dataset-button", "n_clicks")],
            [State("new-dataset-path", "value")],
            prevent_initial_call=True,
        )
        def add_new_dataset(n_clicks, new_path):
            if not n_clicks or not new_path:
                return [
                    self.create_dataset_table(),
                    self._generate_upstream_plot(),
                    self._generate_downstream_plot(),
                    new_path or "",
                    "",
                ]

            # Check if path exists and contains valid data
            if not os.path.exists(new_path):
                return [
                    self.create_dataset_table(),
                    self._generate_upstream_plot(),
                    self._generate_downstream_plot(),
                    new_path,
                    dbc.Alert(
                        "Path does not exist.",
                        color="danger",
                        dismissable=True,
                        duration=3000,
                    ),
                ]

            # Validate metadata exists before attempting load
            metadata_path = os.path.join(new_path, "run_metadata.json")
            if not os.path.exists(metadata_path):
                return [
                    self.create_dataset_table(),
                    self._generate_upstream_plot(),
                    self._generate_downstream_plot(),
                    new_path,
                    dbc.Alert(
                        "run_metadata.json not found in dataset folder.",
                        color="danger",
                        dismissable=True,
                        duration=5000,
                    ),
                ]

            # Determine a sensible dataset name (basename of folder)
            dataset_name = (
                os.path.basename(new_path) or f"dataset_{len(self.datasets) + 1}"
            )

            # Attempt to load dataset; report any error to the UI instead of
            # letting the callback raise and appear to 'do nothing'.
            try:
                self.load_data(new_path, dataset_name)
            except Exception as e:
                # Report the error to the user
                msg = str(e) or "Unknown error while loading dataset"
                print(f"Error adding dataset from {new_path}: {msg}")
                return [
                    self.create_dataset_table(),
                    self._generate_upstream_plot(),
                    self._generate_downstream_plot(),
                    new_path,
                    dbc.Alert(
                        f"Failed to add dataset: {msg}",
                        color="danger",
                        dismissable=True,
                        duration=7000,
                    ),
                ]

            return [
                self.create_dataset_table(),
                self._generate_upstream_plot(),
                self._generate_downstream_plot(),
                "",  # Clear the input field
                dbc.Alert(
                    f"Dataset added successfully from {new_path}",
                    color="success",
                    dismissable=True,
                    duration=3000,
                ),
            ]

        # Callback for deleting datasets
        @self.app.callback(
            [
                Output("dataset-table-container", "children"),
                Output("upstream-plot", "figure"),
                Output("downstream-plot", "figure"),
                Output("temperature-plot", "figure"),
            ],
            [Input({"type": "delete-dataset", "index": ALL}, "n_clicks")],
            prevent_initial_call=True,
        )
        def delete_dataset(n_clicks_list):
            # Check if any delete button was clicked
            if not n_clicks_list or not any(n_clicks_list):
                raise PreventUpdate

            # Find which button was clicked
            ctx = dash.callback_context
            if not ctx.triggered:
                raise PreventUpdate

            # Extract the index from the triggered button
            button_id = ctx.triggered[0]["prop_id"]

            try:
                button_data = json.loads(button_id.split(".")[0])
                delete_index = int(button_data["index"])
            except (json.JSONDecodeError, KeyError, IndexError, ValueError):
                raise PreventUpdate

            # Map positional index to dataset key and remove
            keys = _keys_list()
            if 0 <= delete_index < len(keys):
                key = keys[delete_index]
                deleted = self.datasets.pop(key, None)
                if deleted:
                    print(f"Deleted dataset: {deleted.get('name')}")

            # Return updated components
            return [
                self.create_dataset_table(),
                self._generate_upstream_plot(),
                self._generate_downstream_plot(),
                self._generate_temperature_plot(),
            ]

        # Callback for downloading datasets
        @self.app.callback(
            Output("download-dataset-output", "data", allow_duplicate=True),
            [Input({"type": "download-dataset", "index": ALL}, "n_clicks")],
            prevent_initial_call=True,
        )
        def download_dataset(n_clicks_list):
            # Check if any download button was clicked
            if not n_clicks_list or not any(n_clicks_list):
                raise PreventUpdate

            # Find which button was clicked
            ctx = dash.callback_context
            if not ctx.triggered:
                raise PreventUpdate

            # Extract the index from the triggered button
            button_id = ctx.triggered[0]["prop_id"]
            try:
                button_data = json.loads(button_id.split(".")[0])
                download_index = int(button_data["index"])
            except (json.JSONDecodeError, KeyError, IndexError, ValueError):
                raise PreventUpdate

            # Map positional index to dataset key
            keys = _keys_list()
            dataset = None
            if 0 <= download_index < len(keys):
                key = keys[download_index]
                dataset = self.datasets.get(key)

            if dataset:
                # Prefer explicit dataset folder path
                dataset_path = dataset.get("dataset_path") or dataset.get("folder")
                if dataset_path and os.path.exists(dataset_path):
                    packaged = self._create_dataset_download(dataset_path)
                    if packaged is not None:
                        # If packaged content is binary (zip or bytes), use dcc.send_bytes
                        content = packaged.get("content")
                        filename = packaged.get("filename")
                        if isinstance(content, (bytes, bytearray)):
                            return dcc.send_bytes(
                                lambda f, data=content: f.write(data), filename
                            )
                        # If content is text (string), return as dict like before
                        return dict(
                            content=content,
                            filename=filename,
                            type=packaged.get("type", "text/csv"),
                        )

            raise PreventUpdate

        # Callback for exporting upstream plot
        @self.app.callback(
            Output("download-upstream-plot", "data", allow_duplicate=True),
            [Input("export-upstream-plot", "n_clicks")],
            [State("upstream-plot", "figure")],
            prevent_initial_call=True,
        )
        def export_upstream_plot(n_clicks, current_fig):
            if not n_clicks:
                raise PreventUpdate

            # Use the current rendered figure (as dict) and convert to HTML
            if not current_fig:
                raise PreventUpdate

            # Convert dict->plotly Figure then to HTML
            try:
                fig = go.Figure(current_fig)
                html_str = fig.to_html(include_plotlyjs="inline")
            except Exception:
                raise PreventUpdate

            return dict(
                content=html_str,
                filename="upstream_plot.html",
                type="text/html",
            )

        # Callback for exporting downstream plot
        @self.app.callback(
            Output("download-downstream-plot", "data", allow_duplicate=True),
            [Input("export-downstream-plot", "n_clicks")],
            prevent_initial_call=True,
        )
        def export_downstream_plot(n_clicks):
            if not n_clicks:
                raise PreventUpdate

            # Generate the downstream plot with FULL DATA (no resampling)
            fig = self._generate_downstream_plot_full_data(False)

            # Convert to HTML
            html_str = fig.to_html(include_plotlyjs="inline")

            return dict(
                content=html_str,
                filename="downstream_plot_full_data.html",
                type="text/html",
            )

        # Callback for toggling add dataset section
        @self.app.callback(
            [
                Output("collapse-add-dataset", "is_open"),
                Output("add-dataset-icon", "className"),
            ],
            [Input("toggle-add-dataset", "n_clicks")],
            [State("collapse-add-dataset", "is_open")],
            prevent_initial_call=True,
        )
        def toggle_add_dataset_section(n_clicks, is_open):
            # Only handle actual clicks
            if n_clicks is None:
                raise PreventUpdate

            # Toggle the collapse state
            new_is_open = not bool(is_open)
            new_icon_class = "fas fa-minus" if new_is_open else "fas fa-plus"
            print(
                f"toggle_add_dataset_section: clicked={n_clicks}, "
                f"is_open={is_open} -> {new_is_open}, icon={new_icon_class}"
            )
            return new_is_open, new_icon_class

        # Callback for handling live data checkbox changes
        @self.app.callback(
            [
                Output("upstream-plot", "figure", allow_duplicate=True),
                Output("downstream-plot", "figure", allow_duplicate=True),
                Output("temperature-plot", "figure", allow_duplicate=True),
                Output("live-data-interval", "disabled"),
            ],
            [Input({"type": "dataset-live-data", "index": ALL}, "value")],
            self.PLOT_CONTROL_STATES,
            prevent_initial_call=True,
        )
        def handle_live_data_toggle(
            live_data_values,
            show_error_bars_upstream,
            show_error_bars_downstream,
            show_valve_times_upstream,
            show_valve_times_downstream,
        ):
            # Update the live_data flag for each dataset
            for i, is_live in enumerate(live_data_values):
                if i < len(self.datasets):
                    self.datasets[i].live_data = bool(is_live)

            # Check if any dataset has live data enabled
            any_live_data = any(live_data_values) if live_data_values else False

            # Regenerate plots with updated data
            plots = self._generate_both_plots(
                show_error_bars_upstream=show_error_bars_upstream,
                show_error_bars_downstream=show_error_bars_downstream,
                show_valve_times_upstream=show_valve_times_upstream,
                show_valve_times_downstream=show_valve_times_downstream,
            )
            return [*plots, not any_live_data]  # Disable interval if no live data

        # Callback for periodic live data updates
        @self.app.callback(
            [
                Output("upstream-plot", "figure", allow_duplicate=True),
                Output("downstream-plot", "figure", allow_duplicate=True),
                Output("temperature-plot", "figure", allow_duplicate=True),
            ],
            [Input("live-data-interval", "n_intervals")],
            self.PLOT_CONTROL_STATES,
            prevent_initial_call=True,
        )
        def update_live_data(
            n_intervals,
            show_error_bars_upstream,
            show_error_bars_downstream,
            show_valve_times_upstream,
            show_valve_times_downstream,
        ):
            # Check if any dataset has live data enabled
            has_live_data = any(dataset.live_data for dataset in self.datasets)

            if not has_live_data:
                raise PreventUpdate

            # Reload data for live datasets by calling process_data() again
            for dataset in self.datasets:
                if dataset.live_data:
                    # Re-process the data to get updated values
                    dataset.process_data()

            # Regenerate plots with updated data
            return self._generate_both_plots(
                show_error_bars_upstream=show_error_bars_upstream,
                show_error_bars_downstream=show_error_bars_downstream,
                show_valve_times_upstream=show_valve_times_upstream,
                show_valve_times_downstream=show_valve_times_downstream,
            )

        # FigureResampler callbacks for interactive zooming/panning
        @self.app.callback(
            Output("upstream-plot", "figure", allow_duplicate=True),
            Input("upstream-plot", "relayoutData"),
            prevent_initial_call=True,
        )
        def update_upstream_plot_on_relayout(relayoutData):
            if "upstream-plot" in self.figure_resamplers and relayoutData:
                fig_resampler = self.figure_resamplers["upstream-plot"]

                # Check if this is an autoscale/reset zoom event (double-click)
                if (
                    "autosize" in relayoutData
                    or "xaxis.autorange" in relayoutData
                    or "yaxis.autorange" in relayoutData
                    or relayoutData.get("xaxis.autorange") is True
                    or relayoutData.get("yaxis.autorange") is True
                ):
                    # Regenerate the full plot for autoscale
                    return self._generate_upstream_plot()
                else:
                    # Normal zoom/pan - use resampling
                    return fig_resampler.construct_update_data_patch(relayoutData)
            return dash.no_update

        @self.app.callback(
            Output("downstream-plot", "figure", allow_duplicate=True),
            Input("downstream-plot", "relayoutData"),
            prevent_initial_call=True,
        )
        def update_downstream_plot_on_relayout(relayoutData):
            if "downstream-plot" in self.figure_resamplers and relayoutData:
                fig_resampler = self.figure_resamplers["downstream-plot"]

                # Check if this is an autoscale/reset zoom event (double-click)
                if (
                    "autosize" in relayoutData
                    or "xaxis.autorange" in relayoutData
                    or "yaxis.autorange" in relayoutData
                    or relayoutData.get("xaxis.autorange") is True
                    or relayoutData.get("yaxis.autorange") is True
                ):
                    # Regenerate the full plot for autoscale
                    return self._generate_downstream_plot()
                else:
                    # Normal zoom/pan - use resampling
                    return fig_resampler.construct_update_data_patch(relayoutData)
            return dash.no_update

        @self.app.callback(
            Output("temperature-plot", "figure", allow_duplicate=True),
            Input("temperature-plot", "relayoutData"),
            prevent_initial_call=True,
        )
        def update_temperature_plot_on_relayout(relayoutData):
            if "temperature-plot" in self.figure_resamplers and relayoutData:
                fig_resampler = self.figure_resamplers["temperature-plot"]

                # Check if this is an autoscale/reset zoom event (double-click)
                if (
                    "autosize" in relayoutData
                    or "xaxis.autorange" in relayoutData
                    or "yaxis.autorange" in relayoutData
                    or relayoutData.get("xaxis.autorange") is True
                    or relayoutData.get("yaxis.autorange") is True
                ):
                    # Regenerate the full plot for autoscale
                    return self._generate_temperature_plot()
                else:
                    # Normal zoom/pan - use resampling
                    return fig_resampler.construct_update_data_patch(relayoutData)
            return dash.no_update

        @self.app.callback(
            Output("permeability-plot", "figure", allow_duplicate=True),
            Input("permeability-plot", "relayoutData"),
            prevent_initial_call=True,
        )
        def update_permeability_plot_on_relayout(relayoutData):
            if "permeability-plot" in self.figure_resamplers and relayoutData:
                fig_resampler = self.figure_resamplers["permeability-plot"]

                # Check if this is an autoscale/reset zoom event (double-click)
                if (
                    "autosize" in relayoutData
                    or "xaxis.autorange" in relayoutData
                    or "yaxis.autorange" in relayoutData
                    or relayoutData.get("xaxis.autorange") is True
                    or relayoutData.get("yaxis.autorange") is True
                ):
                    # Regenerate the full plot for autoscale
                    return self._generate_permeability_plot()
                else:
                    # Normal zoom/pan - use resampling
                    return fig_resampler.construct_update_data_patch(relayoutData)
            return dash.no_update

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
        """Generate the temperature plot for v1.2 datasets"""
        # Use FigureResampler with parameters to hide resampling annotations
        fig = FigureResampler(
            go.Figure(),
            show_dash_kwargs={"mode": "disabled"},
            show_mean_aggregation_size=False,
            verbose=False,
        )

        # Store the FigureResampler instance
        self.figure_resamplers["temperature-plot"] = fig

        # Iterate through datasets and check for datasets with temperature data
        has_temperature_data = False
        for i, dataset in enumerate(self.datasets):
            # Check if dataset has temperature data
            if dataset.thermocouple_data is None:
                continue

            has_temperature_data = True

            time_data = np.ascontiguousarray(dataset.time_data)
            local_temp = np.ascontiguousarray(dataset.local_temperature_data)
            thermocouple_temp = np.ascontiguousarray(dataset.thermocouple_data)
            colour = dataset.colour
            thermocouple_name = dataset.thermocouple_name or "Thermocouple"

            # Add local temperature trace
            fig.add_trace(
                go.Scatter(
                    mode="lines",
                    name="Furnace setpoint (C)",
                    line=dict(color=colour, width=1.5, dash="dash"),
                ),
                hf_x=time_data,
                hf_y=local_temp,
            )

            # Add thermocouple temperature trace
            fig.add_trace(
                go.Scatter(
                    mode="lines",
                    name=f"{thermocouple_name} (C)",
                    line=dict(color=colour, width=2),
                ),
                hf_x=time_data,
                hf_y=thermocouple_temp,
            )

        # Configure the layout
        if has_temperature_data:
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
        else:
            # No temperature data available, show message
            fig.add_annotation(
                text="No temperature data available (requires v1.2 datasets)",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=16, color="gray"),
            )
            fig.update_layout(
                height=400,
                xaxis_title="Time (s)",
                yaxis_title="Temperature (°C)",
                template="plotly_white",
                margin=dict(l=60, r=30, t=40, b=60),
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
