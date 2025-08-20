import json
import os
import threading
import webbrowser
from datetime import datetime

import dash
import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objects as go
from dash import ALL, dcc, html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from plotly_resampler import FigureResampler

from .pressure_gauge import (
    Baratron626D_Gauge,
    CVM211_Gauge,
    PressureGauge,
    WGM701_Gauge,
)


class DataPlotter:
    """
        DataPlotter is responsible for visualizing pressure gauge data using Dash.

    Args:
        dataset_paths: list of strings with paths to dataset folders
        dataset_names: List of strings to name the datasets (optional, must match data
            length if provided)
        port: Port for the Dash server, defaults to 8050

    Attributes:
        dataset_paths: list of strings with paths to dataset folders
        dataset_names: List of strings to name the datasets (optional, must match data
            length if provided)
        port: Port for the Dash server, defaults to 8050
        app: Dash app instance
        upstream_datasets: List of upstream datasets
        downstream_datasets: List of downstream datasets
        folder_datasets: List of folder-level datasets
    """

    dataset_paths: list[str]
    dataset_names: list[str]
    port: int

    app: dash.Dash
    upstream_datasets: list[dict]
    downstream_datasets: list[dict]
    folder_datasets: list[dict]

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

        # Store multiple datasets - separate lists for upstream and downstream
        self.upstream_datasets = []
        self.downstream_datasets = []

        # Store folder-level datasets for management
        # Each folder = 1 dataset with upstream/downstream gauges
        self.folder_datasets = []

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

    def load_data(self):
        """
        Load and process data from all specified data paths.
        """
        print(f"Loading data from {len(self.dataset_paths)} dataset(s)")

        for i, dataset_path in enumerate(self.dataset_paths):
            print(
                f"\n--- Processing dataset {i + 1}/{len(self.dataset_paths)}: "
                f"{dataset_path} ---"
            )

            # Read metadata file
            metadata_path = os.path.join(dataset_path, "run_metadata.json")
            with open(metadata_path) as f:
                metadata = json.load(f)

            # Process CSV data based on version
            self.process_csv_data(metadata, dataset_path)

    def process_csv_data(self, metadata: dict, data_folder: str):
        """
        Process CSV data based on metadata version.

        Args:
            metadata: Parsed JSON metadata dictionary
            data_folder: Path to folder containing CSV data files
        """
        version = metadata.get("version")

        if version == "0.0":
            self.process_csv_v0_0(metadata, data_folder)
        elif version == "1.0":
            self.process_csv_v1_0(metadata, data_folder)
        else:
            raise NotImplementedError(
                f"Unsupported metadata version: {version}. "
                f"Only versions '0.0' and '1.0' are supported."
            )

    def process_csv_v0_0(self, metadata: dict, data_folder: str):
        """
        Process CSV data for metadata version 0.0 (multiple CSV files).

        Args:
            metadata: Parsed JSON metadata dictionary
            data_folder: Path to folder containing CSV data files
        """

        # Create gauge instances from metadata
        self.gauge_instances = self.create_gauge_instances(metadata["gauges"])

        # Load data for each gauge
        for gauge_instance, gauge in zip(self.gauge_instances, metadata["gauges"]):
            csv_path = os.path.join(data_folder, gauge["filename"])
            data = np.genfromtxt(csv_path, delimiter=",", names=True)
            gauge_instance.time_data = data["RelativeTime"]
            gauge_instance.pressure_data = data["Pressure_Torr"]

            # Calculate error bars
            gauge_instance.pressure_error = gauge_instance.calculate_error(
                gauge_instance.pressure_data
            )

        # Create datasets for plotting
        self.create_datasets_from_gauges(self.gauge_instances, data_folder)

        # Log completion
        print("\nDatasets created:")
        print(f"  - Upstream: {len(self.upstream_datasets)} datasets")
        print(f"  - Downstream: {len(self.downstream_datasets)} datasets")

    def process_csv_v1_0(self, metadata: dict, data_folder: str):
        """
        Process CSV data for metadata version 1.0 (single CSV file).

        Args:
            metadata: Parsed JSON metadata dictionary
            data_folder: Path to folder containing CSV data file
        """

        # Create gauge instances from metadata
        self.gauge_instances = self.create_gauge_instances(metadata["gauges"])

        csv_path = os.path.join(data_folder, metadata["run_info"]["data_filename"])
        data = np.genfromtxt(csv_path, delimiter=",", names=True, dtype=None)

        # convert datetime strings to relative time in floats
        dt_objects = [
            datetime.strptime(t, "%Y-%m-%d %H:%M:%S.%f") for t in data["RealTimestamp"]
        ]
        relative_times = [(dt - dt_objects[0]).total_seconds() for dt in dt_objects]

        for gauge_instance in self.gauge_instances:
            gauge_instance.time_data = relative_times
            gauge_instance.pressure_data = gauge_instance.voltage_to_pressure(
                data[f"{gauge_instance.name}_Voltage_V"]
            )

            # Calculate error bars
            gauge_instance.pressure_error = gauge_instance.calculate_error(
                gauge_instance.pressure_data
            )

        # Create datasets for plotting
        self.create_datasets_from_gauges(self.gauge_instances, data_folder)

        # Log completion
        print("\nDatasets created:")
        print(f"  - Upstream: {len(self.upstream_datasets)} datasets")
        print(f"  - Downstream: {len(self.downstream_datasets)} datasets")

    def create_gauge_instances(self, gauges_metadata: dict) -> list[PressureGauge]:
        """Create gauge instances from metadata and load CSV data.

        Args:
            gauges_metadata: Metadata for gauges

        returns:
            List of PressureGauge instances
        """
        gauge_instances = []

        # Mapping of gauge types to classes
        gauge_type_map = {
            "WGM701_Gauge": WGM701_Gauge,
            "CVM211_Gauge": CVM211_Gauge,
            "Baratron626D_Gauge": Baratron626D_Gauge,
        }

        for gauge_data in gauges_metadata:
            gauge_type = gauge_data.get("type")

            if gauge_type not in gauge_type_map:
                raise ValueError(f"Unknown gauge type: {gauge_type}")

            gauge_class = gauge_type_map[gauge_type]

            # Extract common parameters
            name = gauge_data.get("name")
            ain_channel = gauge_data.get("ain_channel")
            gauge_location = gauge_data.get("gauge_location")

            # Create instance based on gauge type
            if gauge_type == "Baratron626D_Gauge":
                full_scale_torr = gauge_data.get("full_scale_torr")
                gauge_instance = gauge_class(
                    ain_channel=ain_channel,
                    name=name,
                    gauge_location=gauge_location,
                    full_scale_Torr=full_scale_torr,
                )
            else:
                # WGM701 and CVM211 use the same constructor parameters
                gauge_instance = gauge_class(
                    name=name, ain_channel=ain_channel, gauge_location=gauge_location
                )

            gauge_instances.append(gauge_instance)

        return gauge_instances

    def create_datasets_from_gauges(
        self,
        gauges: list[PressureGauge],
        data_folder: str,
    ):
        """
        Create dataset dictionaries from gauge instances for plotting.

        Args:
            upstream_gauges: List of gauge instances with upstream location
            downstream_gauges: List of gauge instances with downstream location
            data_folder: Path to folder containing the data
        """
        # Create a single folder-level dataset
        # Use custom dataset name if provided, otherwise use default naming
        dataset_index = len(self.folder_datasets)
        if self.dataset_names and dataset_index < len(self.dataset_names):
            dataset_name = self.dataset_names[dataset_index]
        else:
            dataset_name = f"Dataset_{dataset_index + 1}"
        dataset_color = self.get_next_color(len(self.folder_datasets))

        folder_dataset = {
            "name": dataset_name,
            "color": dataset_color,
            "folder": data_folder,
            "upstream_gauges": [],
            "downstream_gauges": [],
        }

        # Create upstream datasets with folder dataset reference
        for gauge in gauges:
            # Only Baratron626D_Gauge is visible by default
            is_visible = gauge.__class__.__name__ == "Baratron626D_Gauge"

            dataset = {
                "data": {
                    "RelativeTime": gauge.time_data,
                    "Pressure_Torr": gauge.pressure_data,
                    "Pressure_Error": gauge.pressure_error,
                },
                "name": gauge.name,
                "display_name": dataset_name,
                "color": dataset_color,
                "visible": is_visible,
                "gauge_type": gauge.__class__.__name__,
                "folder_dataset": dataset_name,
            }
            if gauge.gauge_location == "upstream":
                self.upstream_datasets.append(dataset)
                folder_dataset["upstream_gauges"].append(dataset)
            else:
                self.downstream_datasets.append(dataset)
                folder_dataset["downstream_gauges"].append(dataset)

            print(
                f"Added to {gauge.gauge_location} dataset: {dataset['display_name']}"
                f"(visible: {is_visible})"
            )

        # Add the folder dataset to our list
        self.folder_datasets.append(folder_dataset)

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
                                                                    dbc.Checkbox(
                                                                        id="show-gauge-names-upstream",
                                                                        label="Show gauge names",
                                                                        value=False,
                                                                        className="mb-2",
                                                                    ),
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
                                                                    dbc.Checkbox(
                                                                        id="show-gauge-names-downstream",
                                                                        label="Show gauge names",
                                                                        value=False,
                                                                        className="mb-2",
                                                                    ),
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
            ],
            fluid=True,
        )

    def create_dataset_table(self):
        """Create a table showing folder-level datasets with editable name and color"""
        if not self.folder_datasets:
            return html.Div("No datasets loaded", className="text-muted")

        # Create table rows
        rows = []

        # Header row
        header_row = html.Tr(
            [
                html.Th(
                    "Dataset Name",
                    style={
                        "text-align": "left",
                        "width": "40%",
                        "padding": "2px",
                        "font-weight": "normal",
                    },
                ),
                html.Th(
                    "Dataset Path",
                    style={
                        "text-align": "left",
                        "width": "40%",
                        "padding": "2px",
                        "font-weight": "normal",
                    },
                ),
                html.Th(
                    "Colour",
                    style={
                        "text-align": "center",
                        "width": "10%",
                        "padding": "2px",
                        "font-weight": "normal",
                    },
                ),
                html.Th(
                    "",
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
                        "width": "5%",
                        "padding": "2px",
                        "font-weight": "normal",
                    },
                ),
            ]
        )
        rows.append(header_row)

        # Add dataset rows
        for i, dataset in enumerate(self.folder_datasets):
            row = html.Tr(
                [
                    html.Td(
                        [
                            dcc.Input(
                                id={"type": "dataset-name", "index": i},
                                value=dataset["name"],
                                style={
                                    "width": "90%",
                                    "border": "1px solid #ccc",
                                    "padding": "4px",
                                    "border-radius": "4px",
                                    "transition": "all 0.2s ease",
                                },
                                className="dataset-name-input",
                            )
                        ],
                        style={"padding": "4px"},
                    ),
                    html.Td(
                        [
                            html.Div(
                                [
                                    html.Span(
                                        dataset["folder"],
                                        style={
                                            "font-family": "monospace",
                                            "font-size": "0.9em",
                                            "color": "#666",
                                            "word-break": "break-all",
                                        },
                                        title=dataset["folder"],  # Full path on hover
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
                        style={"padding": "4px"},
                    ),
                    html.Td(
                        [
                            dcc.Input(
                                id={"type": "dataset-color", "index": i},
                                type="color",
                                value=dataset["color"],
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
                        style={"text-align": "center", "padding": "4px"},
                    ),
                    html.Td(
                        [
                            html.Button(
                                html.Img(
                                    src="/assets/download-minimalistic-svgrepo-com.svg",
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
                                title=f"Download {dataset['name']}",
                            ),
                        ],
                        style={"text-align": "center", "padding": "4px"},
                    ),
                    html.Td(
                        [
                            html.Button(
                                html.Img(
                                    src="/assets/delete-svgrepo-com.svg",
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
                                title=f"Delete {dataset['name']}",
                            ),
                        ],
                        style={"text-align": "center", "padding": "4px"},
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
        # Callback for dataset name changes
        @self.app.callback(
            [
                Output("dataset-table-container", "children", allow_duplicate=True),
                Output("upstream-plot", "figure", allow_duplicate=True),
                Output("downstream-plot", "figure", allow_duplicate=True),
            ],
            [Input({"type": "dataset-name", "index": ALL}, "value")],
            [
                State("show-gauge-names-upstream", "value"),
                State("show-gauge-names-downstream", "value"),
                State("show-error-bars-upstream", "value"),
                State("show-error-bars-downstream", "value"),
            ],
            prevent_initial_call=True,
        )
        def update_dataset_names(
            names,
            show_gauge_names_upstream,
            show_gauge_names_downstream,
            show_error_bars_upstream,
            show_error_bars_downstream,
        ):
            # Update dataset names
            for i, name in enumerate(names):
                if i < len(self.folder_datasets) and name:
                    self.folder_datasets[i]["name"] = name

                    # Update upstream gauge display names based on checkbox state
                    for gauge_dataset in self.folder_datasets[i]["upstream_gauges"]:
                        if show_gauge_names_upstream:
                            gauge_dataset["display_name"] = (
                                f"{gauge_dataset['name']} - {name}"
                            )
                        else:
                            gauge_dataset["display_name"] = name
                        gauge_dataset["folder_dataset"] = name

                    # Update downstream gauge display names based on checkbox state
                    for gauge_dataset in self.folder_datasets[i]["downstream_gauges"]:
                        if show_gauge_names_downstream:
                            gauge_dataset["display_name"] = (
                                f"{gauge_dataset['name']} - {name}"
                            )
                        else:
                            gauge_dataset["display_name"] = name
                        gauge_dataset["folder_dataset"] = name

            # Return updated table and plots
            return [
                self.create_dataset_table(),
                self._generate_upstream_plot(
                    show_gauge_names_upstream, show_error_bars_upstream
                ),
                self._generate_downstream_plot(
                    show_gauge_names_downstream, show_error_bars_downstream
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
            # Update dataset colors
            for i, color in enumerate(colors):
                if i < len(self.folder_datasets) and color:
                    self.folder_datasets[i]["color"] = color

                    # Update colors for all gauges in this dataset
                    for gauge_dataset in self.folder_datasets[i]["upstream_gauges"]:
                        gauge_dataset["color"] = color

                    for gauge_dataset in self.folder_datasets[i]["downstream_gauges"]:
                        gauge_dataset["color"] = color

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
            [Output("upstream-plot", "figure", allow_duplicate=True)],
            [
                Input("upstream-x-scale", "value"),
                Input("upstream-y-scale", "value"),
                Input("upstream-x-min", "value"),
                Input("upstream-x-max", "value"),
                Input("upstream-y-min", "value"),
                Input("upstream-y-max", "value"),
            ],
            prevent_initial_call=True,
        )
        def update_upstream_plot_settings(x_scale, y_scale, x_min, x_max, y_min, y_max):
            # Generate updated upstream plot with new settings
            return [
                self._generate_upstream_plot(
                    x_scale, y_scale, x_min, x_max, y_min, y_max
                )
            ]

        # Callback for downstream plot settings changes
        @self.app.callback(
            [Output("downstream-plot", "figure", allow_duplicate=True)],
            [
                Input("downstream-x-scale", "value"),
                Input("downstream-y-scale", "value"),
                Input("downstream-x-min", "value"),
                Input("downstream-x-max", "value"),
                Input("downstream-y-min", "value"),
                Input("downstream-y-max", "value"),
            ],
            prevent_initial_call=True,
        )
        def update_downstream_plot_settings(
            x_scale, y_scale, x_min, x_max, y_min, y_max
        ):
            # Generate updated downstream plot with new settings
            return [
                self._generate_downstream_plot(
                    x_scale, y_scale, x_min, x_max, y_min, y_max
                )
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

        # Callback for upstream show gauge names checkbox
        @self.app.callback(
            [Output("upstream-plot", "figure", allow_duplicate=True)],
            [
                Input("show-gauge-names-upstream", "value"),
                Input("show-error-bars-upstream", "value"),
            ],
            prevent_initial_call=True,
        )
        def update_upstream_label_display(
            show_gauge_names_upstream, show_error_bars_upstream
        ):
            # Update display names for upstream gauges only
            for folder_dataset in self.folder_datasets:
                dataset_name = folder_dataset["name"]

                # Update upstream gauges
                for gauge_dataset in folder_dataset["upstream_gauges"]:
                    if show_gauge_names_upstream:
                        gauge_dataset["display_name"] = (
                            f"{gauge_dataset['name']} - {dataset_name}"
                        )
                    else:
                        gauge_dataset["display_name"] = dataset_name

            # Return updated upstream plot
            return [
                self._generate_upstream_plot(
                    show_gauge_names_upstream, show_error_bars_upstream
                )
            ]

        # Callback for downstream show gauge names checkbox
        @self.app.callback(
            [Output("downstream-plot", "figure", allow_duplicate=True)],
            [
                Input("show-gauge-names-downstream", "value"),
                Input("show-error-bars-downstream", "value"),
            ],
            prevent_initial_call=True,
        )
        def update_downstream_label_display(
            show_gauge_names_downstream, show_error_bars_downstream
        ):
            # Update display names for downstream gauges only
            for folder_dataset in self.folder_datasets:
                dataset_name = folder_dataset["name"]

                # Update downstream gauges
                for gauge_dataset in folder_dataset["downstream_gauges"]:
                    if show_gauge_names_downstream:
                        gauge_dataset["display_name"] = (
                            f"{gauge_dataset['name']} - {dataset_name}"
                        )
                    else:
                        gauge_dataset["display_name"] = dataset_name

            # Return updated downstream plot
            return [
                self._generate_downstream_plot(
                    show_gauge_names_downstream, show_error_bars_downstream
                )
            ]

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
            import os

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

            try:
                # Try to load the dataset
                self.load_data(new_path)

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

            except Exception as e:
                return [
                    self.create_dataset_table(),
                    self._generate_upstream_plot(),
                    self._generate_downstream_plot(),
                    new_path,
                    dbc.Alert(
                        f"Error loading dataset: {e!s}",
                        color="danger",
                        dismissable=True,
                        duration=5000,
                    ),
                ]

        # Callback for deleting datasets
        @self.app.callback(
            [
                Output("dataset-table", "children"),
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
                delete_index = button_data["index"]
            except (json.JSONDecodeError, KeyError, IndexError):
                raise PreventUpdate

            # Remove the dataset at the specified index
            if 0 <= delete_index < len(self.folder_datasets):
                deleted_dataset = self.folder_datasets.pop(delete_index)
                print(f"Deleted dataset: {deleted_dataset['name']}")

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
                download_index = button_data["index"]
            except (json.JSONDecodeError, KeyError, IndexError):
                raise PreventUpdate

            # Get the dataset to download
            if 0 <= download_index < len(self.folder_datasets):
                dataset = self.folder_datasets[download_index]

                # Create CSV data combining all gauges from this dataset
                import pandas as pd

                # Combine upstream and downstream gauges
                all_gauges = dataset["upstream_gauges"] + dataset["downstream_gauges"]

                # Collect all unique time points first
                all_times = set()
                gauge_data_dict = {}

                for gauge in all_gauges:
                    gauge_data = gauge["data"]
                    if len(gauge_data.get("RelativeTime", [])) > 0:
                        times = gauge_data["RelativeTime"]
                        pressures = gauge_data["Pressure_Torr"]

                        # Add times to our set
                        all_times.update(times)

                        # Store gauge data with unique column name
                        gauge_name = gauge.get(
                            "display_name", gauge.get("name", "Unknown")
                        )
                        gauge_data_dict[f"{gauge_name}_Pressure_Torr"] = dict(
                            zip(times, pressures)
                        )

                if all_times and gauge_data_dict:
                    # Create a sorted list of times
                    sorted_times = sorted(all_times)

                    # Build the final DataFrame
                    result_data = {"RelativeTime": sorted_times}

                    # Add each gauge's data
                    for gauge_column, time_pressure_map in gauge_data_dict.items():
                        result_data[gauge_column] = [
                            time_pressure_map.get(time, "") for time in sorted_times
                        ]

                    df = pd.DataFrame(result_data)
                    csv_data = df.to_csv(index=False)
                    return dict(
                        content=csv_data,
                        filename=f"{dataset['name']}_data.csv",
                        type="text/csv",
                    )

            raise PreventUpdate

        # Callback for exporting upstream plot
        @self.app.callback(
            Output("download-upstream-plot", "data", allow_duplicate=True),
            [Input("export-upstream-plot", "n_clicks")],
            [State("show-gauge-names-upstream", "value")],
            prevent_initial_call=True,
        )
        def export_upstream_plot(n_clicks, show_gauge_names):
            if not n_clicks:
                raise PreventUpdate

            # Generate the upstream plot with FULL DATA (no resampling)
            fig = self._generate_upstream_plot_full_data(show_gauge_names)

            # Convert to HTML
            html_str = fig.to_html(include_plotlyjs="inline")

            return dict(
                content=html_str,
                filename="upstream_plot_full_data.html",
                type="text/html",
            )

        # Callback for exporting downstream plot
        @self.app.callback(
            Output("download-downstream-plot", "data", allow_duplicate=True),
            [Input("export-downstream-plot", "n_clicks")],
            [State("show-gauge-names-downstream", "value")],
            prevent_initial_call=True,
        )
        def export_downstream_plot(n_clicks, show_gauge_names):
            if not n_clicks:
                raise PreventUpdate

            # Generate the downstream plot with FULL DATA (no resampling)
            fig = self._generate_downstream_plot_full_data(show_gauge_names)

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
            if not n_clicks:
                raise PreventUpdate

            # Toggle the collapse state
            new_is_open = not is_open

            # Change icon based on state
            icon_class = "fas fa-minus" if new_is_open else "fas fa-plus"

            return new_is_open, icon_class

    def _generate_plot(
        self,
        x_scale=None,
        y_scale=None,
        x_min=None,
        x_max=None,
        y_min=None,
        y_max=None,
    ):
        """Generate the plot based on current dataset state and settings"""
        # Use FigureResampler with parameters to hide resampling annotations
        fig = FigureResampler(
            go.Figure(),
            show_dash_kwargs={"mode": "disabled"},
            show_mean_aggregation_size=False,
            verbose=False,
        )

        # Iterate through folder datasets and all their gauges
        for folder_dataset in self.folder_datasets:
            all_gauges = (
                folder_dataset["upstream_gauges"] + folder_dataset["downstream_gauges"]
            )
            for dataset in all_gauges:
                # Skip invisible datasets
                if not dataset.get("visible", True):
                    continue

                data = dataset["data"]
                display_name = dataset.get(
                    "display_name", dataset.get("name", "Unknown Dataset")
                )
                color = dataset["color"]

                if len(data.get("RelativeTime", [])) > 0:
                    # Extract data directly from our structure
                    time_data = data["RelativeTime"]
                    pressure_data = data["Pressure_Torr"]

                    # Use plotly-resampler for automatic downsampling
                    fig.add_trace(
                        go.Scatter(
                            x=time_data,
                            y=pressure_data,
                            mode="lines+markers",
                            name=display_name,
                            line=dict(color=color, width=1.5),
                            marker=dict(size=3),
                        )
                    )

        # Configure the layout
        fig.update_layout(
            height=600,
            xaxis_title="Relative Time (s)",
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
            fig.update_xaxes(range=[x_min, x_max])

        if y_min is not None and y_max is not None:
            if y_axis_type == "log":
                # For log scale, use log10 of the values
                import math

                fig.update_yaxes(range=[math.log10(y_min), math.log10(y_max)])
            else:
                fig.update_yaxes(range=[y_min, y_max])

        # Clean up trace names to remove [R] annotations
        for trace in fig.data:
            if hasattr(trace, "name") and trace.name and "[R]" in trace.name:
                trace.name = trace.name.replace("[R] ", "").replace("[R]", "")

        return fig

    def _generate_upstream_plot(
        self,
        show_gauge_names=False,
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

        # Iterate through folder datasets and their upstream gauges
        for folder_dataset in self.folder_datasets:
            for dataset in folder_dataset["upstream_gauges"]:
                # Skip invisible datasets
                if not dataset.get("visible", True):
                    continue

                data = dataset["data"]
                display_name = dataset.get(
                    "display_name", dataset.get("name", "Unknown Dataset")
                )
                color = dataset["color"]

                if len(data.get("RelativeTime", [])) > 0:
                    # Extract data directly from our structure
                    time_data = data["RelativeTime"]
                    pressure_data = data["Pressure_Torr"]
                    pressure_error = data["Pressure_Error"]

                    # Create scatter trace
                    scatter_kwargs = {
                        "x": time_data,
                        "y": pressure_data,
                        "mode": "lines+markers",
                        "name": display_name,
                        "line": dict(color=color, width=1.5),
                        "marker": dict(size=3),
                    }

                    # Add error bars if enabled
                    if show_error_bars:
                        scatter_kwargs["error_y"] = dict(
                            type="data",
                            array=pressure_error,
                            visible=True,
                            color=color,
                            thickness=1.5,
                            width=3,
                        )

                    # Use plotly-resampler for automatic downsampling
                    fig.add_trace(go.Scatter(**scatter_kwargs))

        # Configure the layout
        fig.update_layout(
            height=500,
            xaxis_title="Relative Time (s)",
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
            fig.update_xaxes(range=[x_min, x_max])

        if y_min is not None and y_max is not None:
            if y_axis_type == "log":
                # For log scale, use log10 of the values
                import math

                fig.update_yaxes(range=[math.log10(y_min), math.log10(y_max)])
            else:
                fig.update_yaxes(range=[y_min, y_max])

        # Clean up trace names to remove [R] annotations
        for trace in fig.data:
            if hasattr(trace, "name") and trace.name and "[R]" in trace.name:
                trace.name = trace.name.replace("[R] ", "").replace("[R]", "")

        return fig

    def _generate_downstream_plot(
        self,
        show_gauge_names=False,
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

        # Iterate through folder datasets and their downstream gauges
        for folder_dataset in self.folder_datasets:
            for dataset in folder_dataset["downstream_gauges"]:
                # Skip invisible datasets
                if not dataset.get("visible", True):
                    continue

                data = dataset["data"]
                display_name = dataset.get(
                    "display_name", dataset.get("name", "Unknown Dataset")
                )
                color = dataset["color"]

                if len(data.get("RelativeTime", [])) > 0:
                    # Extract data directly from our structure
                    time_data = data["RelativeTime"]
                    pressure_data = data["Pressure_Torr"]
                    pressure_error = data["Pressure_Error"]

                    # Create scatter trace
                    scatter_kwargs = {
                        "x": time_data,
                        "y": pressure_data,
                        "mode": "lines+markers",
                        "name": display_name,
                        "line": dict(color=color, width=1.5),
                        "marker": dict(size=3),
                    }

                    # Add error bars if enabled
                    if show_error_bars:
                        scatter_kwargs["error_y"] = dict(
                            type="data",
                            array=pressure_error,
                            visible=True,
                            color=color,
                            thickness=1.5,
                            width=3,
                        )

                    # Use plotly-resampler for automatic downsampling
                    fig.add_trace(go.Scatter(**scatter_kwargs))

        # Configure the layout
        fig.update_layout(
            height=500,
            xaxis_title="Relative Time (s)",
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
            fig.update_xaxes(range=[x_min, x_max])

        if y_min is not None and y_max is not None:
            if y_axis_type == "log":
                # For log scale, use log10 of the values
                import math

                fig.update_yaxes(range=[math.log10(y_min), math.log10(y_max)])
            else:
                fig.update_yaxes(range=[y_min, y_max])

        # Clean up trace names to remove [R] annotations
        for trace in fig.data:
            if hasattr(trace, "name") and trace.name and "[R]" in trace.name:
                trace.name = trace.name.replace("[R] ", "").replace("[R]", "")

        return fig

    def _generate_upstream_plot_full_data(
        self, show_gauge_names=False, show_error_bars=True
    ):
        """Generate the upstream pressure plot with FULL DATA (no resampling)"""
        # Use regular plotly Figure WITHOUT FigureResampler
        fig = go.Figure()

        # Iterate through folder datasets and their upstream gauges
        for folder_dataset in self.folder_datasets:
            for dataset in folder_dataset["upstream_gauges"]:
                # Skip invisible datasets
                if not dataset.get("visible", True):
                    continue

                data = dataset["data"]
                # Determine display name based on checkbox state
                if show_gauge_names:
                    display_name = dataset.get(
                        "display_name", dataset.get("name", "Unknown Dataset")
                    )
                else:
                    display_name = folder_dataset["name"]
                color = dataset["color"]

                if len(data.get("RelativeTime", [])) > 0:
                    # Extract data directly from our structure
                    time_data = data["RelativeTime"]
                    pressure_data = data["Pressure_Torr"]
                    pressure_error = data["Pressure_Error"]

                    # Create scatter trace
                    scatter_kwargs = {
                        "x": time_data,
                        "y": pressure_data,
                        "mode": "lines+markers",
                        "name": display_name,
                        "line": dict(color=color, width=1.5),
                        "marker": dict(size=3),
                    }

                    # Add error bars if enabled
                    if show_error_bars:
                        scatter_kwargs["error_y"] = dict(
                            type="data",
                            array=pressure_error,
                            visible=True,
                            color=color,
                            thickness=1.5,
                            width=3,
                        )

                    # Add ALL data points (no resampling)
                    fig.add_trace(go.Scatter(**scatter_kwargs))

        # Configure the layout
        fig.update_layout(
            title="Upstream Pressure (Full Data)",
            height=500,
            xaxis_title="Relative Time (s)",
            yaxis_title="Pressure (Torr)",
            template="plotly_white",
            margin=dict(l=60, r=30, t=40, b=60),
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.05,
            ),
            hovermode="x unified",
        )

        return fig

    def _generate_downstream_plot_full_data(
        self, show_gauge_names=False, show_error_bars=True
    ):
        """Generate the downstream pressure plot with FULL DATA (no resampling)"""
        # Use regular plotly Figure WITHOUT FigureResampler
        fig = go.Figure()

        # Iterate through folder datasets and their downstream gauges
        for folder_dataset in self.folder_datasets:
            for dataset in folder_dataset["downstream_gauges"]:
                # Skip invisible datasets
                if not dataset.get("visible", True):
                    continue

                data = dataset["data"]
                # Determine display name based on checkbox state
                if show_gauge_names:
                    display_name = dataset.get(
                        "display_name", dataset.get("name", "Unknown Dataset")
                    )
                else:
                    display_name = folder_dataset["name"]
                color = dataset["color"]

                if len(data.get("RelativeTime", [])) > 0:
                    # Extract data directly from our structure
                    time_data = data["RelativeTime"]
                    pressure_data = data["Pressure_Torr"]
                    pressure_error = data["Pressure_Error"]

                    # Create scatter trace
                    scatter_kwargs = {
                        "x": time_data,
                        "y": pressure_data,
                        "mode": "lines+markers",
                        "name": display_name,
                        "line": dict(color=color, width=1.5),
                        "marker": dict(size=3),
                    }

                    # Add error bars if enabled
                    if show_error_bars:
                        scatter_kwargs["error_y"] = dict(
                            type="data",
                            array=pressure_error,
                            visible=True,
                            color=color,
                            thickness=1.5,
                            width=3,
                        )

                    # Add ALL data points (no resampling)
                    fig.add_trace(go.Scatter(**scatter_kwargs))

        # Configure the layout
        fig.update_layout(
            title="Downstream Pressure (Full Data)",
            height=500,
            xaxis_title="Relative Time (s)",
            yaxis_title="Pressure (Torr)",
            template="plotly_white",
            margin=dict(l=60, r=30, t=40, b=60),
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.05,
            ),
            hovermode="x unified",
        )

        return fig

    def start(self):
        """Process data and start the Dash web server"""

        # Process data
        self.load_data()

        # Setup the app layout
        self.app.layout = self.create_layout()

        # Add custom CSS for hover effects
        self.app.index_string = hover_css

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
