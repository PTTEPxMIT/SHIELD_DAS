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

from .helpers import calculate_error, voltage_to_pressure


class DataPlotter:
    """Plotter UI for pressure gauge datasets using Dash.

    Provides a Dash app that displays upstream and downstream pressure
    plots for multiple datasets. Datasets are stored in `self.datasets`.

    Args:
        dataset_paths: List of folder paths containing datasets to load on
        dataset_names: List of names corresponding to each dataset path
        port: Port number for Dash app (default: 8050)

    Attributes:
        dataset_paths: List of folder paths containing datasets to load on
        dataset_names: List of names corresponding to each dataset path
        port: Port number for Dash app (default: 8050)
        app: Dash app instance
        datasets: Dictionary of loaded datasets for plotting
        figure_resamplers: Dictionary of FigureResampler instances for each plot
    """

    # Type hints / attributes
    dataset_paths: list[str]
    dataset_names: list[str]
    port: int

    app: dash.Dash
    datasets = dict
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

        # Store datasets
        self.datasets = {}

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
        Load and process data from all specified data path.
        """

        # Read metadata file
        metadata_path = os.path.join(dataset_path, "run_metadata.json")
        with open(metadata_path) as f:
            metadata = json.load(f)

        # Process CSV data based on version
        time_data, upstream_data, downstream_data = self.process_csv_data(
            metadata, dataset_path
        )

        # Create datasets for plotting
        self.create_dataset(
            dataset_path, dataset_name, time_data, upstream_data, downstream_data
        )

    def process_csv_data(
        self, metadata: dict, data_folder: str
    ) -> tuple[npt.NDArray, dict, dict]:
        """
        Process CSV data based on metadata version.

        Args:
            metadata: Parsed JSON metadata dictionary
            data_folder: Path to folder containing CSV data files
            gauge_instances: List of gauge instances to populate with data
        """

        version = metadata.get("version")

        if version == "0.0":
            time_data, upstream_data, downstream_data = self.process_csv_v0_0(
                metadata, data_folder
            )
        elif version == "1.0":
            time_data, upstream_data, downstream_data = self.process_csv_v1_0(
                metadata, data_folder
            )
        else:
            raise NotImplementedError(
                f"Unsupported metadata version: {version}. "
                f"Only versions '0.0' and '1.0' are supported."
            )

        return time_data, upstream_data, downstream_data

    def process_csv_v0_0(
        self, metadata: dict, data_folder: str
    ) -> tuple[npt.NDArray, dict, dict]:
        """
        Process CSV data for metadata version 0.0 (multiple CSV files).

        Args:
            metadata: Parsed JSON metadata dictionary
            data_folder: Path to folder containing CSV data files
        """
        # just use first gauge for time data
        csv_path = os.path.join(data_folder, metadata["gauges"][0]["filename"])
        data = np.genfromtxt(csv_path, delimiter=",", names=True)
        time_data = data["RelativeTime"]

        # Load upstream and downstream data for each gauge
        for gauge in metadata["gauges"]:
            if gauge["type"] == "Baratron626D_Gauge":
                if gauge["gauge_location"] == "upstream":
                    csv_path = os.path.join(data_folder, gauge["filename"])
                    data = np.genfromtxt(csv_path, delimiter=",", names=True)
                    upstream_pressure_data = data["Pressure_Torr"]
                if gauge["gauge_location"] == "downstream":
                    csv_path = os.path.join(data_folder, gauge["filename"])
                    data = np.genfromtxt(csv_path, delimiter=",", names=True)
                    downstream_pressure_data = data["Pressure_Torr"]

        upstream_error = calculate_error(upstream_pressure_data)
        downstream_error = calculate_error(downstream_pressure_data)

        upstream_data = {
            "pressure_data": upstream_pressure_data,
            "error_data": upstream_error,
        }
        downstream_data = {
            "pressure_data": downstream_pressure_data,
            "error_data": downstream_error,
        }

        return time_data, upstream_data, downstream_data

    def process_csv_v1_0(
        self, metadata: dict, data_folder: str
    ) -> tuple[npt.NDArray, dict, dict]:
        """
        Process CSV data for metadata version 1.0 (single CSV file).

        Args:
            metadata: Parsed JSON metadata dictionary
            data_folder: Path to folder containing CSV data file
        """

        # Expect a single CSV file specified in metadata->run_info->data_filename
        data_filename = metadata.get("run_info", {}).get("data_filename")
        if not data_filename:
            raise ValueError("Missing data_filename in run_info for v1.0 metadata")

        csv_path = os.path.join(data_folder, data_filename)
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        # Read structured CSV; allow text fields for timestamps
        data = np.genfromtxt(
            csv_path, delimiter=",", names=True, dtype=None, encoding="utf-8"
        )

        # Convert RealTimestamp strings to relative time floats
        if "RealTimestamp" not in data.dtype.names:
            raise ValueError("RealTimestamp column not found in v1.0 CSV")

        dt_objects = [
            datetime.strptime(t, "%Y-%m-%d %H:%M:%S.%f") for t in data["RealTimestamp"]
        ]
        time_data = np.array(
            [(dt - dt_objects[0]).total_seconds() for dt in dt_objects]
        )

        upstream_pressure_data = None
        downstream_pressure_data = None

        # Iterate gauges and extract Baratron voltage columns where present
        for gauge in metadata.get("gauges", []):
            gname = gauge.get("name")
            gtype = gauge.get("type")
            loc = gauge.get("gauge_location")

            # We only handle Baratron gauges (voltage -> pressure) here
            if gtype == "Baratron626D_Gauge":
                col_name = f"{gname}_Voltage_V"

                volt_vals = np.array(data[col_name], dtype=float)

                #  Convert voltage to pressure using helper
                pressure_vals = voltage_to_pressure(
                    volt_vals, full_scale_torr=float(gauge["full_scale_torr"])
                )

                if loc == "upstream":
                    upstream_pressure_data = pressure_vals
                else:
                    downstream_pressure_data = pressure_vals

        upstream_error = calculate_error(upstream_pressure_data)
        downstream_error = calculate_error(downstream_pressure_data)

        upstream_data = {
            "pressure_data": upstream_pressure_data,
            "error_data": upstream_error,
        }
        downstream_data = {
            "pressure_data": downstream_pressure_data,
            "error_data": downstream_error,
        }

        return time_data, upstream_data, downstream_data

    def create_dataset(
        self,
        dataset_path: str,
        dataset_name: str,
        time_data: npt.NDArray,
        upstream_data: dict,
        downstream_data: dict,
    ):
        """
        Create dataset dictionaries from gauge instances for plotting.

        Args:
            upstream_gauges: List of gauge instances with upstream location
            downstream_gauges: List of gauge instances with downstream location
            data_folder: Path to folder containing the data
        """

        dataset_color = self.get_next_color(len(self.datasets))

        dataset = {
            "name": dataset_name,
            "colour": dataset_color,
            "dataset_path": dataset_path,
            "live_data": False,
            "time_data": time_data,
            "upstream_data": upstream_data,
            "downstream_data": downstream_data,
        }

        # Add the folder dataset to our list
        i = len(self.datasets) + 1
        self.datasets[f"dataset_{i}"] = dataset

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
        return dbc.Container(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            html.H1(
                                "SHIELD Data Visualisation",
                                className="text-center",
                                style={
                                    "fontSize": "3.5rem",
                                    "fontWeight": "standard",
                                    "marginTop": "2rem",
                                    "marginBottom": "2rem",
                                    "color": "#2c3e50",
                                },
                            ),
                            width=12,
                        ),
                    ],
                    className="mb-4",
                ),
                # Dataset Management Card at the top
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Card(
                                    [
                                        dbc.CardHeader(
                                            dbc.Row(
                                                [
                                                    dbc.Col(
                                                        "Dataset Management",
                                                        className="d-flex align-items-center",
                                                    ),
                                                    dbc.Col(
                                                        dbc.Button(
                                                            html.I(
                                                                className="fas fa-chevron-up"
                                                            ),
                                                            id="collapse-dataset-button",
                                                            color="light",
                                                            size="sm",
                                                            className="ms-auto",
                                                            style={
                                                                "border": "1px solid #dee2e6",
                                                                "background-color": "#f8f9fa",
                                                                "box-shadow": "0 1px 3px rgba(0,0,0,0.1)",
                                                                "width": "30px",
                                                                "height": "30px",
                                                                "padding": "0",
                                                                "display": "flex",
                                                                "align-items": "center",
                                                                "justify-content": "center",
                                                            },
                                                        ),
                                                        width="auto",
                                                        className="d-flex justify-content-end",
                                                    ),
                                                ],
                                                className="g-0 align-items-center",
                                            )
                                        ),
                                        dbc.Collapse(
                                            dbc.CardBody(
                                                [
                                                    # Dataset table
                                                    html.Div(
                                                        id="dataset-table-container",
                                                        children=self.create_dataset_table(),
                                                    ),
                                                    # Collapsible Add Dataset Section
                                                    html.Div(
                                                        [
                                                            # Separator with centered plus button
                                                            html.Div(
                                                                [
                                                                    html.Hr(
                                                                        style={
                                                                            "flex": "1",
                                                                            "margin": "0",
                                                                            "border-top": (
                                                                                "1px solid "
                                                                                "#dee2e6"
                                                                            ),
                                                                        }
                                                                    ),
                                                                    dbc.Button(
                                                                        html.I(
                                                                            id="add-dataset-icon",
                                                                            className=(
                                                                                "fas fa-plus"
                                                                            ),
                                                                        ),
                                                                        id="toggle-add-dataset",
                                                                        color="light",
                                                                        size="sm",
                                                                        style={
                                                                            "margin": (
                                                                                "0 10px"
                                                                            ),
                                                                            "border-radius": (
                                                                                "50%"
                                                                            ),
                                                                            "width": "32px",
                                                                            "height": "32px",
                                                                            "padding": "0",
                                                                            "border": (
                                                                                "1px solid "
                                                                                "#dee2e6"
                                                                            ),
                                                                        },
                                                                        title=(
                                                                            "Add new dataset"
                                                                        ),
                                                                    ),
                                                                    html.Hr(
                                                                        style={
                                                                            "flex": "1",
                                                                            "margin": "0",
                                                                            "border-top": (
                                                                                "1px solid "
                                                                                "#dee2e6"
                                                                            ),
                                                                        }
                                                                    ),
                                                                ],
                                                                style={
                                                                    "display": "flex",
                                                                    "align-items": (
                                                                        "center"
                                                                    ),
                                                                    "margin": (
                                                                        "20px 0 15px 0"
                                                                    ),
                                                                },
                                                            ),
                                                            # Collapsible add dataset form
                                                            dbc.Collapse(
                                                                [
                                                                    dbc.Row(
                                                                        [
                                                                            dbc.Col(
                                                                                [
                                                                                    dbc.Input(
                                                                                        id="new-dataset-path",
                                                                                        type="text",
                                                                                        placeholder=(
                                                                                            "Enter dataset "
                                                                                            "folder path..."
                                                                                        ),
                                                                                        style={
                                                                                            "margin-bottom": (
                                                                                                "10px"
                                                                                            )
                                                                                        },
                                                                                    ),
                                                                                ],
                                                                                width=9,
                                                                            ),
                                                                            dbc.Col(
                                                                                [
                                                                                    dbc.Button(
                                                                                        [
                                                                                            html.I(
                                                                                                className=(
                                                                                                    "fas fa-plus me-2"
                                                                                                )
                                                                                            ),
                                                                                            (
                                                                                                "Add Dataset"
                                                                                            ),
                                                                                        ],
                                                                                        id="add-dataset-button",
                                                                                        color="primary",
                                                                                        style={
                                                                                            "width": (
                                                                                                "100%"
                                                                                            )
                                                                                        },
                                                                                    ),
                                                                                ],
                                                                                width=3,
                                                                            ),
                                                                        ],
                                                                        className="g-2",
                                                                    ),
                                                                    # Status message for add dataset
                                                                    html.Div(
                                                                        id="add-dataset-status",
                                                                        style={
                                                                            "margin-top": (
                                                                                "10px"
                                                                            )
                                                                        },
                                                                    ),
                                                                ],
                                                                id="collapse-add-dataset",
                                                                is_open=False,
                                                            ),
                                                        ]
                                                    ),
                                                ]
                                            ),
                                            id="collapse-dataset",
                                            is_open=True,
                                        ),
                                    ]
                                ),
                            ],
                            width=12,
                        ),
                    ],
                    className="mb-3",
                ),
                # Hidden store to trigger plot updates
                dcc.Store(id="datasets-store"),
                # Hidden stores for plot settings
                dcc.Store(id="upstream-settings-store", data={}),
                dcc.Store(id="downstream-settings-store", data={}),
                # Status message for upload feedback (floating)
                html.Div(
                    id="upload-status",
                    style={
                        "position": "fixed",
                        "top": "20px",
                        "right": "20px",
                        "zIndex": "9999",
                        "maxWidth": "400px",
                        "minWidth": "300px",
                    },
                ),
                # Dual plots for upstream and downstream pressure
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Card(
                                    [
                                        dbc.CardHeader("Upstream Pressure"),
                                        dbc.CardBody(
                                            [
                                                dcc.Graph(
                                                    id="upstream-plot",
                                                    figure=self._generate_upstream_plot(),
                                                )
                                            ]
                                        ),
                                    ]
                                ),
                            ],
                            width=6,
                        ),
                        dbc.Col(
                            [
                                dbc.Card(
                                    [
                                        dbc.CardHeader("Downstream Pressure"),
                                        dbc.CardBody(
                                            [
                                                dcc.Graph(
                                                    id="downstream-plot",
                                                    figure=self._generate_downstream_plot(),
                                                )
                                            ]
                                        ),
                                    ]
                                ),
                            ],
                            width=6,
                        ),
                    ]
                ),
                # Plot controls section - Dual controls for upstream and downstream
                dbc.Row(
                    [
                        # Upstream Plot Controls
                        dbc.Col(
                            [
                                dbc.Card(
                                    [
                                        dbc.CardHeader(
                                            dbc.Row(
                                                [
                                                    dbc.Col(
                                                        "Upstream Plot Controls",
                                                        className="d-flex align-items-center",
                                                    ),
                                                    dbc.Col(
                                                        dbc.Button(
                                                            html.I(
                                                                className="fas fa-chevron-up"
                                                            ),
                                                            id="collapse-upstream-controls-button",
                                                            color="light",
                                                            size="sm",
                                                            className="ms-auto",
                                                            style={
                                                                "border": "1px solid #dee2e6",
                                                                "background-color": "#f8f9fa",
                                                                "box-shadow": "0 1px 3px rgba(0,0,0,0.1)",
                                                                "width": "30px",
                                                                "height": "30px",
                                                                "padding": "0",
                                                                "display": "flex",
                                                                "align-items": "center",
                                                                "justify-content": "center",
                                                            },
                                                        ),
                                                        width="auto",
                                                        className="d-flex justify-content-end",
                                                    ),
                                                ],
                                                className="g-0 align-items-center",
                                            )
                                        ),
                                        dbc.Collapse(
                                            dbc.CardBody(
                                                [
                                                    dbc.Row(
                                                        [
                                                            # X-axis controls
                                                            dbc.Col(
                                                                [
                                                                    html.H6(
                                                                        "X-Axis",
                                                                        className="mb-2",
                                                                    ),
                                                                    dbc.Row(
                                                                        [
                                                                            dbc.Col(
                                                                                [
                                                                                    dbc.Label(
                                                                                        "Scale:"
                                                                                    ),
                                                                                    dbc.RadioItems(
                                                                                        id="upstream-x-scale",
                                                                                        options=[
                                                                                            {
                                                                                                "label": "Linear",
                                                                                                "value": "linear",
                                                                                            },
                                                                                            {
                                                                                                "label": "Log",
                                                                                                "value": "log",
                                                                                            },
                                                                                        ],
                                                                                        value="linear",
                                                                                        inline=True,
                                                                                    ),
                                                                                ],
                                                                                width=12,
                                                                            ),
                                                                        ],
                                                                        className="mb-2",
                                                                    ),
                                                                    dbc.Row(
                                                                        [
                                                                            dbc.Col(
                                                                                [
                                                                                    dbc.Label(
                                                                                        "Min:"
                                                                                    ),
                                                                                    dbc.Input(
                                                                                        id="upstream-x-min",
                                                                                        type="number",
                                                                                        placeholder="Auto",
                                                                                        value=0,
                                                                                        size="sm",
                                                                                    ),
                                                                                ],
                                                                                width=6,
                                                                            ),
                                                                            dbc.Col(
                                                                                [
                                                                                    dbc.Label(
                                                                                        "Max:"
                                                                                    ),
                                                                                    dbc.Input(
                                                                                        id="upstream-x-max",
                                                                                        type="number",
                                                                                        placeholder="Auto",
                                                                                        size="sm",
                                                                                    ),
                                                                                ],
                                                                                width=6,
                                                                            ),
                                                                        ]
                                                                    ),
                                                                ],
                                                                width=6,
                                                            ),
                                                            # Y-axis controls
                                                            dbc.Col(
                                                                [
                                                                    html.H6(
                                                                        "Y-Axis",
                                                                        className="mb-2",
                                                                    ),
                                                                    dbc.Row(
                                                                        [
                                                                            dbc.Col(
                                                                                [
                                                                                    dbc.Label(
                                                                                        "Scale:"
                                                                                    ),
                                                                                    dbc.RadioItems(
                                                                                        id="upstream-y-scale",
                                                                                        options=[
                                                                                            {
                                                                                                "label": "Linear",
                                                                                                "value": "linear",
                                                                                            },
                                                                                            {
                                                                                                "label": "Log",
                                                                                                "value": "log",
                                                                                            },
                                                                                        ],
                                                                                        value="linear",
                                                                                        inline=True,
                                                                                    ),
                                                                                ],
                                                                                width=12,
                                                                            ),
                                                                        ],
                                                                        className="mb-2",
                                                                    ),
                                                                    dbc.Row(
                                                                        [
                                                                            dbc.Col(
                                                                                [
                                                                                    dbc.Label(
                                                                                        "Min:"
                                                                                    ),
                                                                                    dbc.Input(
                                                                                        id="upstream-y-min",
                                                                                        type="number",
                                                                                        placeholder="Auto",
                                                                                        value=0,
                                                                                        size="sm",
                                                                                    ),
                                                                                ],
                                                                                width=6,
                                                                            ),
                                                                            dbc.Col(
                                                                                [
                                                                                    dbc.Label(
                                                                                        "Max:"
                                                                                    ),
                                                                                    dbc.Input(
                                                                                        id="upstream-y-max",
                                                                                        type="number",
                                                                                        placeholder="Auto",
                                                                                        size="sm",
                                                                                    ),
                                                                                ],
                                                                                width=6,
                                                                            ),
                                                                        ]
                                                                    ),
                                                                ],
                                                                width=6,
                                                            ),
                                                        ]
                                                    ),
                                                    # Options Row for Upstream
                                                    dbc.Row(
                                                        [
                                                            dbc.Col(
                                                                [
                                                                    html.H6(
                                                                        "Options",
                                                                        className="mb-2 mt-3",
                                                                    ),
                                                                    # removed: show gauge names option
                                                                    dbc.Checkbox(
                                                                        id="show-error-bars-upstream",
                                                                        label="Show error bars",
                                                                        value=True,
                                                                        className="mb-2",
                                                                    ),
                                                                    html.Hr(
                                                                        className="my-2"
                                                                    ),
                                                                    dbc.Button(
                                                                        [
                                                                            html.I(
                                                                                className="fas fa-download me-2"
                                                                            ),
                                                                            "Export Upstream Plot",
                                                                        ],
                                                                        id="export-upstream-plot",
                                                                        color="outline-secondary",
                                                                        size="sm",
                                                                        className="w-100",
                                                                    ),
                                                                ],
                                                                width=12,
                                                            ),
                                                        ]
                                                    ),
                                                ]
                                            ),
                                            id="collapse-upstream-controls",
                                            is_open=False,
                                        ),
                                    ]
                                ),
                            ],
                            width=6,
                        ),
                        # Downstream Plot Controls
                        dbc.Col(
                            [
                                dbc.Card(
                                    [
                                        dbc.CardHeader(
                                            dbc.Row(
                                                [
                                                    dbc.Col(
                                                        "Downstream Plot Controls",
                                                        className="d-flex align-items-center",
                                                    ),
                                                    dbc.Col(
                                                        dbc.Button(
                                                            html.I(
                                                                className="fas fa-chevron-up"
                                                            ),
                                                            id="collapse-downstream-controls-button",
                                                            color="light",
                                                            size="sm",
                                                            className="ms-auto",
                                                            style={
                                                                "border": "1px solid #dee2e6",
                                                                "background-color": "#f8f9fa",
                                                                "box-shadow": "0 1px 3px rgba(0,0,0,0.1)",
                                                                "width": "30px",
                                                                "height": "30px",
                                                                "padding": "0",
                                                                "display": "flex",
                                                                "align-items": "center",
                                                                "justify-content": "center",
                                                            },
                                                        ),
                                                        width="auto",
                                                        className="d-flex justify-content-end",
                                                    ),
                                                ],
                                                className="g-0 align-items-center",
                                            )
                                        ),
                                        dbc.Collapse(
                                            dbc.CardBody(
                                                [
                                                    dbc.Row(
                                                        [
                                                            # X-axis controls
                                                            dbc.Col(
                                                                [
                                                                    html.H6(
                                                                        "X-Axis",
                                                                        className="mb-2",
                                                                    ),
                                                                    dbc.Row(
                                                                        [
                                                                            dbc.Col(
                                                                                [
                                                                                    dbc.Label(
                                                                                        "Scale:"
                                                                                    ),
                                                                                    dbc.RadioItems(
                                                                                        id="downstream-x-scale",
                                                                                        options=[
                                                                                            {
                                                                                                "label": "Linear",
                                                                                                "value": "linear",
                                                                                            },
                                                                                            {
                                                                                                "label": "Log",
                                                                                                "value": "log",
                                                                                            },
                                                                                        ],
                                                                                        value="linear",
                                                                                        inline=True,
                                                                                    ),
                                                                                ],
                                                                                width=12,
                                                                            ),
                                                                        ],
                                                                        className="mb-2",
                                                                    ),
                                                                    dbc.Row(
                                                                        [
                                                                            dbc.Col(
                                                                                [
                                                                                    dbc.Label(
                                                                                        "Min:"
                                                                                    ),
                                                                                    dbc.Input(
                                                                                        id="downstream-x-min",
                                                                                        type="number",
                                                                                        placeholder="Auto",
                                                                                        value=0,
                                                                                        size="sm",
                                                                                    ),
                                                                                ],
                                                                                width=6,
                                                                            ),
                                                                            dbc.Col(
                                                                                [
                                                                                    dbc.Label(
                                                                                        "Max:"
                                                                                    ),
                                                                                    dbc.Input(
                                                                                        id="downstream-x-max",
                                                                                        type="number",
                                                                                        placeholder="Auto",
                                                                                        size="sm",
                                                                                    ),
                                                                                ],
                                                                                width=6,
                                                                            ),
                                                                        ]
                                                                    ),
                                                                ],
                                                                width=6,
                                                            ),
                                                            # Y-axis controls
                                                            dbc.Col(
                                                                [
                                                                    html.H6(
                                                                        "Y-Axis",
                                                                        className="mb-2",
                                                                    ),
                                                                    dbc.Row(
                                                                        [
                                                                            dbc.Col(
                                                                                [
                                                                                    dbc.Label(
                                                                                        "Scale:"
                                                                                    ),
                                                                                    dbc.RadioItems(
                                                                                        id="downstream-y-scale",
                                                                                        options=[
                                                                                            {
                                                                                                "label": "Linear",
                                                                                                "value": "linear",
                                                                                            },
                                                                                            {
                                                                                                "label": "Log",
                                                                                                "value": "log",
                                                                                            },
                                                                                        ],
                                                                                        value="linear",
                                                                                        inline=True,
                                                                                    ),
                                                                                ],
                                                                                width=12,
                                                                            ),
                                                                        ],
                                                                        className="mb-2",
                                                                    ),
                                                                    dbc.Row(
                                                                        [
                                                                            dbc.Col(
                                                                                [
                                                                                    dbc.Label(
                                                                                        "Min:"
                                                                                    ),
                                                                                    dbc.Input(
                                                                                        id="downstream-y-min",
                                                                                        type="number",
                                                                                        placeholder="Auto",
                                                                                        value=0,
                                                                                        size="sm",
                                                                                    ),
                                                                                ],
                                                                                width=6,
                                                                            ),
                                                                            dbc.Col(
                                                                                [
                                                                                    dbc.Label(
                                                                                        "Max:"
                                                                                    ),
                                                                                    dbc.Input(
                                                                                        id="downstream-y-max",
                                                                                        type="number",
                                                                                        placeholder="Auto",
                                                                                        size="sm",
                                                                                    ),
                                                                                ],
                                                                                width=6,
                                                                            ),
                                                                        ]
                                                                    ),
                                                                ],
                                                                width=6,
                                                            ),
                                                        ]
                                                    ),
                                                    # Options Row for Downstream
                                                    dbc.Row(
                                                        [
                                                            dbc.Col(
                                                                [
                                                                    html.H6(
                                                                        "Options",
                                                                        className="mb-2 mt-3",
                                                                    ),
                                                                    # removed: show gauge names option
                                                                    dbc.Checkbox(
                                                                        id="show-error-bars-downstream",
                                                                        label="Show error bars",
                                                                        value=True,
                                                                        className="mb-2",
                                                                    ),
                                                                    html.Hr(
                                                                        className="my-2"
                                                                    ),
                                                                    dbc.Button(
                                                                        [
                                                                            html.I(
                                                                                className="fas fa-download me-2"
                                                                            ),
                                                                            "Export Downstream Plot",
                                                                        ],
                                                                        id="export-downstream-plot",
                                                                        color="outline-secondary",
                                                                        size="sm",
                                                                        className="w-100",
                                                                    ),
                                                                ],
                                                                width=12,
                                                            ),
                                                        ]
                                                    ),
                                                ]
                                            ),
                                            id="collapse-downstream-controls",
                                            is_open=False,
                                        ),
                                    ]
                                ),
                            ],
                            width=6,
                        ),
                    ],
                    className="mt-3",
                ),
                # Add whitespace at the bottom of the page
                dbc.Row(
                    [
                        dbc.Col(
                            html.Div(style={"height": "100px"}),
                            width=12,
                        ),
                    ],
                ),
                # Download component for dataset downloads
                dcc.Download(id="download-dataset-output"),
                # Download components for plot exports
                dcc.Download(id="download-upstream-plot"),
                dcc.Download(id="download-downstream-plot"),
                # Interval component for live data updates
                dcc.Interval(
                    id="live-data-interval",
                    interval=1000,  # Update every 1 second
                    n_intervals=0,
                    disabled=True,  # Start disabled, enable when needed
                ),
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
        for i, dataset in enumerate(self.datasets.keys()):
            row = html.Tr(
                [
                    html.Td(
                        [
                            dcc.Input(
                                id={"type": "dataset-name", "index": i},
                                value=self.datasets[f"{dataset}"]["name"],
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
                                        self.datasets[f"{dataset}"]["dataset_path"],
                                        style={
                                            "font-family": "monospace",
                                            "font-size": "0.9em",
                                            "color": "#666",
                                            "word-break": "break-all",
                                        },
                                        title=self.datasets[f"{dataset}"][
                                            "dataset_path"
                                        ],  # Full path on hover
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
                                        value=self.datasets[f"{dataset}"].get(
                                            "live_data", False
                                        ),
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
                                value=self.datasets[f"{dataset}"]["colour"],
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
                                        title=f"Download {self.datasets[f'{dataset}']['name']}",
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
                                        title=f"Delete {self.datasets[f'{dataset}']['name']}",
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
        # Helpers to work with self.datasets (dict) and legacy lists if present
        def _keys_list():
            # Return stable ordered list of keys for indexing by position
            return list(self.datasets.keys())

        def _iter_datasets():
            return self.datasets.values()

        # Callback for dataset name changes
        @self.app.callback(
            [
                Output("dataset-table-container", "children", allow_duplicate=True),
                Output("upstream-plot", "figure", allow_duplicate=True),
                Output("downstream-plot", "figure", allow_duplicate=True),
            ],
            [Input({"type": "dataset-name", "index": ALL}, "value")],
            [
                State("show-error-bars-upstream", "value"),
                State("show-error-bars-downstream", "value"),
            ],
            prevent_initial_call=True,
        )
        def update_dataset_names(
            names,
            show_error_bars_upstream,
            show_error_bars_downstream,
        ):
            # Map positional indices from the UI to dataset keys
            keys = _keys_list()

            # Build current names list for comparison to avoid double application
            current_names = []
            for i in range(len(keys)):
                current_names.append(self.datasets[keys[i]].get("name", ""))

            # If nothing changed, skip to avoid duplicate updates
            if list(names) == current_names:
                raise PreventUpdate

            # Update only entries that changed
            for i, name in enumerate(names):
                if i < len(keys) and name and name != current_names[i]:
                    key = keys[i]
                    self.datasets[key]["name"] = name

            # Return updated table and plots
            return [
                self.create_dataset_table(),
                # upstream: _generate_upstream_plot takes show_error_bars as first arg
                self._generate_upstream_plot(show_error_bars_upstream),
                # show_gauge_names option removed; pass show_error_bars by keyword
                self._generate_downstream_plot(
                    show_error_bars=show_error_bars_downstream
                ),
            ]

        # Callback for dataset color changes
        @self.app.callback(
            [
                Output("dataset-table-container", "children", allow_duplicate=True),
                Output("upstream-plot", "figure", allow_duplicate=True),
                Output("downstream-plot", "figure", allow_duplicate=True),
            ],
            [Input({"type": "dataset-color", "index": ALL}, "value")],
            prevent_initial_call=True,
        )
        def update_dataset_colors(colors):
            keys = _keys_list()
            for i, color in enumerate(colors):
                if i < len(keys) and color:
                    key = keys[i]
                    # new-style datasets use 'colour' key
                    self.datasets[key]["colour"] = color

            # Return updated table and plots
            return [
                self.create_dataset_table(),
                self._generate_upstream_plot(),
                self._generate_downstream_plot(),
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
            import json

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
            import json

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
                Output("live-data-interval", "disabled"),
            ],
            [Input({"type": "dataset-live-data", "index": ALL}, "value")],
            [
                State("show-error-bars-upstream", "value"),
                State("show-error-bars-downstream", "value"),
            ],
            prevent_initial_call=True,
        )
        def handle_live_data_toggle(
            live_data_values, show_error_bars_upstream, show_error_bars_downstream
        ):
            # Update the live_data flag for each dataset using keys mapping
            keys = _keys_list()
            for i, is_live in enumerate(live_data_values):
                if i < len(keys):
                    key = keys[i]
                    self.datasets[key]["live_data"] = bool(is_live)

            # Check if any dataset has live data enabled
            any_live_data = any(live_data_values) if live_data_values else False

            # Regenerate plots with updated data
            return [
                self._generate_upstream_plot(show_error_bars_upstream),
                self._generate_downstream_plot(
                    show_error_bars=show_error_bars_downstream
                ),
                not any_live_data,  # Disable interval if no live data
            ]

        # Callback for periodic live data updates
        @self.app.callback(
            [
                Output("upstream-plot", "figure", allow_duplicate=True),
                Output("downstream-plot", "figure", allow_duplicate=True),
            ],
            [Input("live-data-interval", "n_intervals")],
            [
                State("show-error-bars-upstream", "value"),
                State("show-error-bars-downstream", "value"),
            ],
            prevent_initial_call=True,
        )
        def update_live_data(
            n_intervals, show_error_bars_upstream, show_error_bars_downstream
        ):
            # Check if any dataset has live data enabled
            datasets_iter = list(_iter_datasets())
            has_live_data = any(
                dataset.get("live_data", False) for dataset in datasets_iter
            )

            if not has_live_data:
                raise PreventUpdate

            # Reload data for live datasets by updating existing datasets in place
            for key in list(self.datasets.keys()):
                dataset = self.datasets[key]
                if dataset.get("live_data", False):
                    # Get dataset info
                    dataset_path = dataset.get("dataset_path") or dataset.get("folder")
                    dataset_name = dataset.get("name")

                    if dataset_path and dataset_name:
                        # Read metadata file
                        metadata_path = os.path.join(dataset_path, "run_metadata.json")
                        with open(metadata_path) as f:
                            metadata = json.load(f)

                        # Process CSV data based on version
                        (
                            time_data,
                            upstream_data,
                            downstream_data,
                        ) = self.process_csv_data(metadata, dataset_path)

                        # Update existing dataset in place
                        # (preserve name, colour, live_data)
                        dataset.update(
                            {
                                "time_data": time_data,
                                "upstream_data": upstream_data,
                                "downstream_data": downstream_data,
                            }
                        )

            # Regenerate plots with updated data;
            return [
                self._generate_upstream_plot(show_error_bars_upstream),
                self._generate_downstream_plot(
                    show_error_bars=show_error_bars_downstream
                ),
            ]

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

    def _generate_upstream_plot(
        self,
        show_error_bars=True,
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
        for dataset_name in self.datasets.keys():
            time_data = self.datasets[f"{dataset_name}"]["time_data"]
            time_data = np.ascontiguousarray(time_data)
            pressure_data = self.datasets[f"{dataset_name}"]["upstream_data"][
                "pressure_data"
            ]
            pressure_data = np.ascontiguousarray(pressure_data)
            pressure_error = self.datasets[f"{dataset_name}"]["upstream_data"][
                "error_data"
            ]
            pressure_error = np.ascontiguousarray(pressure_error)

            colour = self.datasets[f"{dataset_name}"]["colour"]

            # Create scatter trace
            scatter_kwargs = {
                "mode": "lines+markers",
                "name": self.datasets[f"{dataset_name}"]["name"],
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
                for ds in self.datasets.values():
                    try:
                        vals = np.asarray(ds.get("time_data", []), dtype=float)
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
                for ds in self.datasets.values():
                    try:
                        v = ds.get("time_data", [])
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
                for ds in self.datasets.values():
                    try:
                        vals = np.asarray(
                            ds.get("upstream_data", {}).get("pressure_data", []),
                            dtype=float,
                        )
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
                for ds in self.datasets.values():
                    try:
                        v = ds.get("upstream_data", {}).get("pressure_data", [])
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

        # Iterate through datasets and obtain the upstream data
        for dataset_name in self.datasets.keys():
            time_data = self.datasets[f"{dataset_name}"]["time_data"]
            time_data = np.ascontiguousarray(time_data)
            pressure_data = self.datasets[f"{dataset_name}"]["downstream_data"][
                "pressure_data"
            ]
            pressure_data = np.ascontiguousarray(pressure_data)
            pressure_error = self.datasets[f"{dataset_name}"]["downstream_data"][
                "error_data"
            ]
            pressure_error = np.ascontiguousarray(pressure_error)

            colour = self.datasets[f"{dataset_name}"]["colour"]

            # Create scatter trace
            scatter_kwargs = {
                "mode": "lines+markers",
                "name": self.datasets[f"{dataset_name}"]["name"],
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
                    for dataset_name in self.datasets.keys():
                        try:
                            vals = self.datasets[f"{dataset_name}"]["time_data"]
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
                    for dataset_name in self.datasets.keys():
                        try:
                            vals = self.datasets[f"{dataset_name}"][
                                downstream_or_upstream
                            ]["pressure_data"]
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
