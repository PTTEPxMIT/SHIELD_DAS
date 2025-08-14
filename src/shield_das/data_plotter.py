import base64
import json
import os
import sys
import threading
import webbrowser

import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import ALL, MATCH, dcc, html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from plotly_resampler import FigureResampler

# Import gauge classes
from .pressure_gauge import Baratron626D_Gauge, CVM211_Gauge, WGM701_Gauge


class DataPlotter:
    def __init__(self, data_folder=None, port=8050):
        """
        Initialize the DataPlotter.

        Args:
            data_folder: Path to folder containing JSON metadata and CSV data files
            port: Port for the Dash server
        """

        self.data_folder = data_folder
        self.port = port
        self.app = dash.Dash(
            __name__,
            external_stylesheets=[
                dbc.themes.BOOTSTRAP,
                "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css",
            ],
        )
        self.app_running = False
        self.server_thread = None

        # Store multiple datasets - separate lists for upstream and downstream
        self.upstream_datasets = []
        self.downstream_datasets = []

        # Process data folder if provided
        if self.data_folder:
            self.load_data()
            print("\nData loading complete. Datasets ready for Dash app visualization.")

        # Setup the app layout
        self.app.layout = self.create_layout()

        # Register callbacks
        self.register_callbacks()

        # Flag to track if recording has been started
        self.recording_started = False

    def load_data(self):
        """
        Load and process data from the specified folder.
        Read JSON metadata first, then process CSV data based on version.
        """
        print(f"Loading data from folder: {self.data_folder}")

        if not os.path.exists(self.data_folder):
            print(f"ERROR: Data folder not found: {self.data_folder}")
            return

        try:
            # Process JSON metadata
            metadata = self.process_json_metadata()
            if metadata is None:
                return

            # Process CSV data based on version
            self.process_csv_data(metadata)

        except Exception as e:
            print(f"ERROR loading data from {self.data_folder}: {e}")
            import traceback

            traceback.print_exc()

    def process_json_metadata(self):
        """
        Find and process JSON metadata file.

        Returns:
            dict: Parsed metadata or None if error
        """
        # Find JSON files
        json_files = [
            f for f in os.listdir(self.data_folder) if f.lower().endswith(".json")
        ]

        if not json_files:
            print(f"ERROR: No JSON file found in {self.data_folder}")
            return None

        json_path = os.path.join(self.data_folder, json_files[0])
        print(f"Found JSON metadata: {json_path}")

        # Read JSON metadata
        with open(json_path) as f:
            metadata = json.load(f)

        return metadata

    def process_csv_data(self, metadata):
        """
        Process CSV data based on metadata version.

        Args:
            metadata: Parsed JSON metadata dictionary
        """
        version = metadata.get("version")

        if version == "0.0":
            self.process_csv_v0_0(metadata)
        elif version == "1.0":
            self.process_csv_v1_0(metadata)
        else:
            raise NotImplementedError(
                f"Unsupported metadata version: {version}. "
                f"Only versions '0.0' and '1.0' are supported."
            )

    def create_gauge_instances(self, gauges_metadata):
        """Create gauge instances from metadata and load CSV data."""
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
                # Baratron626D requires additional full_scale_Torr parameter
                full_scale_torr = gauge_data.get("full_scale_torr")
                if full_scale_torr is None:
                    raise ValueError(
                        f"Baratron626D gauge '{name}' missing required parameter",
                        "'full_scale_torr'",
                    )
                gauge_instance = gauge_class(
                    ain_channel, name, gauge_location, full_scale_torr
                )
            else:
                # WGM701 and CVM211 use the same constructor parameters
                gauge_instance = gauge_class(name, ain_channel, gauge_location)

            # Load CSV data for this gauge
            csv_filename = gauge_data.get("filename")
            if not csv_filename:
                raise ValueError(
                    f"Gauge '{name}' missing required 'filename' field in metadata"
                )

            csv_path = os.path.join(self.data_folder, csv_filename)
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"CSV file not found: {csv_path}")

            # Load CSV data using numpy
            data = np.genfromtxt(csv_path, delimiter=",", names=True)

            # Extract RelativeTime and Pressure columns (indices 1 and 2)
            gauge_instance.time_data = data["RelativeTime"]
            print(gauge_instance.time_data)
            gauge_instance.pressure_data = data["Pressure_Torr"]

            print(f"Loaded CSV data for {name}: {len(data)} rows")

            gauge_instances.append(gauge_instance)
            print(f"Created gauge instance: {gauge_type} - {name}")

        return gauge_instances

    def process_csv_v0_0(self, metadata):
        """
        Process CSV data for metadata version 0.0 (multiple CSV files).

        Args:
            metadata: Parsed JSON metadata dictionary
        """
        print("Processing data as version 0.0 (multiple CSV files)")

        # Create gauge instances from metadata
        self.gauge_instances = self.create_gauge_instances(metadata["gauges"])
        print(f"Created {len(self.gauge_instances)} gauge instances")

        # Separate gauges into upstream and downstream datasets based on gauge_location
        upstream_gauges = []
        downstream_gauges = []

        for gauge in self.gauge_instances:
            if gauge.gauge_location.lower() == "upstream":
                upstream_gauges.append(gauge)
            elif gauge.gauge_location.lower() == "downstream":
                downstream_gauges.append(gauge)
            else:
                print(
                    f"Warning: Unknown gauge location '{gauge.gauge_location}' for gauge '{gauge.name}'"
                )

        print(f"Upstream gauges: {len(upstream_gauges)}")
        print(f"Downstream gauges: {len(downstream_gauges)}")

        # Create datasets for plotting
        self.create_datasets_from_gauges(upstream_gauges, downstream_gauges)

        # Log completion
        print("\nDatasets created:")
        print(f"  - Upstream: {len(self.upstream_datasets)} datasets")
        print(f"  - Downstream: {len(self.downstream_datasets)} datasets")

    def create_datasets_from_gauges(self, upstream_gauges, downstream_gauges):
        """
        Create dataset dictionaries from gauge instances for plotting.

        Args:
            upstream_gauges: List of gauge instances with upstream location
            downstream_gauges: List of gauge instances with downstream location
        """
        # Clear existing datasets
        self.upstream_datasets = []
        self.downstream_datasets = []

        # Create upstream datasets
        for i, gauge in enumerate(upstream_gauges):
            if hasattr(gauge, "time_data") and hasattr(gauge, "pressure_data"):
                # Convert to relative time for performance
                if len(gauge.time_data) > 0:
                    relative_time = gauge.time_data - gauge.time_data[0]
                else:
                    relative_time = gauge.time_data

                # Only Baratron626D_Gauge is visible by default
                is_visible = gauge.__class__.__name__ == "Baratron626D_Gauge"

                dataset = {
                    "data": {
                        "RelativeTime": relative_time,
                        "Pressure_Torr": gauge.pressure_data,
                    },
                    "name": gauge.name,
                    "display_name": f"{gauge.name} ({gauge.gauge_location})",
                    "color": self.get_next_color(i),
                    "visible": is_visible,
                    "gauge_type": gauge.__class__.__name__,
                }
                self.upstream_datasets.append(dataset)
                print(
                    f"Added upstream dataset: {dataset['display_name']} (visible: {is_visible})"
                )

        # Create downstream datasets
        for i, gauge in enumerate(downstream_gauges):
            if hasattr(gauge, "time_data") and hasattr(gauge, "pressure_data"):
                # Convert to relative time for performance
                if len(gauge.time_data) > 0:
                    relative_time = gauge.time_data - gauge.time_data[0]
                else:
                    relative_time = gauge.time_data

                # Only Baratron626D_Gauge is visible by default
                is_visible = gauge.__class__.__name__ == "Baratron626D_Gauge"

                dataset = {
                    "data": {
                        "RelativeTime": relative_time,
                        "Pressure_Torr": gauge.pressure_data,
                    },
                    "name": gauge.name,
                    "display_name": f"{gauge.name} ({gauge.gauge_location})",
                    "color": self.get_next_color(len(upstream_gauges) + i),
                    "visible": is_visible,
                    "gauge_type": gauge.__class__.__name__,
                }
                self.downstream_datasets.append(dataset)
                print(
                    f"Added downstream dataset: {dataset['display_name']} (visible: {is_visible})"
                )

    def process_csv_v1_0(self, metadata):
        """
        Process CSV data for metadata version 1.0 (single CSV file).

        Args:
            metadata: Parsed JSON metadata dictionary
        """
        raise NotImplementedError("Version 1.0 processing not yet implemented")

    def parse_uploaded_file(self, contents, filename):
        """
        Parse uploaded JSON metadata file and load referenced data.

        Args:
            contents: Base64 encoded file content
            filename: Name of the uploaded file

        Returns:
            pandas.DataFrame: The parsed data or empty DataFrame on error
        """
        try:
            # Decode the base64 content
            content_type, content_string = contents.split(",")
            decoded = base64.b64decode(content_string)

            # Parse JSON metadata
            metadata = json.loads(decoded.decode("utf-8"))

            # Check version and get data filename
            if metadata.get("version") != "1.0":
                print(f"Unsupported version: {metadata.get('version')}")
                return pd.DataFrame()

            data_filename = metadata["run_info"]["data_filename"]
            print(f"Found data filename: {data_filename}")

            # For now, just demonstrate that we successfully parsed the JSON
            # In a real implementation, we'd need the user to upload both files
            # or have the CSV file accessible on the server
            print(f"Successfully extracted CSV filename: {data_filename}")
            print("JSON metadata parsing successful!")
            print("TEST: Metadata parsing complete - terminating program")
            sys.exit(0)

        except Exception as e:
            print(f"Error parsing uploaded file {filename}: {e}")
            return pd.DataFrame()

    def get_next_color(self, index):
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

    def is_valid_color(self, color):
        """
        Validate if a color string is a valid hex or RGB format.

        Args:
            color: Color string to validate

        Returns:
            bool: True if valid color format
        """
        import re

        if not color:
            return False

        color = color.strip()

        # Check hex format (#RGB or #RRGGBB)
        hex_pattern = r"^#([A-Fa-f0-9]{3}|[A-Fa-f0-9]{6})$"
        if re.match(hex_pattern, color):
            return True

        # Check RGB format rgb(r,g,b)
        rgb_pattern = r"^rgb\(\s*(\d{1,3})\s*,\s*(\d{1,3})\s*,\s*(\d{1,3})\s*\)$"
        rgb_match = re.match(rgb_pattern, color, re.IGNORECASE)
        if rgb_match:
            r, g, b = map(int, rgb_match.groups())
            return all(0 <= val <= 255 for val in [r, g, b])

        return False

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
                                        dbc.CardBody([dcc.Graph(id="upstream-plot")]),
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
                                        dbc.CardBody([dcc.Graph(id="downstream-plot")]),
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
                                                    )
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
                                                    )
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
            ],
            fluid=True,
        )

    def create_dataset_table(self):
        """Create a table showing all datasets with controls"""
        # Temporarily disabled to avoid KeyError issues
        return html.Div("Dataset table temporarily disabled", className="text-muted")

    def register_callbacks(self):
        # Callback for real-time dataset management changes
        @self.app.callback(
            [
                Output("datasets-store", "data", allow_duplicate=True),
                Output("dataset-table-container", "children", allow_duplicate=True),
            ],
            [
                Input({"type": "dataset-name", "index": ALL}, "value"),
                Input({"type": "show-dataset", "index": ALL}, "value"),
            ],
            prevent_initial_call=True,
        )
        def update_dataset_changes(display_names, visibility_values):
            # Update dataset properties based on form values
            all_datasets = self.upstream_datasets + self.downstream_datasets
            for i, dataset in enumerate(all_datasets):
                if i < len(display_names):
                    dataset["display_name"] = display_names[i] or dataset.get(
                        "name", "Unknown Dataset"
                    )
                if i < len(visibility_values):
                    dataset["visible"] = (
                        visibility_values[i] if visibility_values[i] else False
                    )

            # Return updated count and table
            return len(self.upstream_datasets) + len(
                self.downstream_datasets
            ), self.create_dataset_table()

        # Callback to toggle color picker popover
        @self.app.callback(
            Output({"type": "color-popover", "index": MATCH}, "is_open"),
            [Input({"type": "color-button", "index": MATCH}, "n_clicks")],
            [State({"type": "color-popover", "index": MATCH}, "is_open")],
            prevent_initial_call=True,
        )
        def toggle_color_popover(n_clicks, is_open):
            if n_clicks:
                return not is_open
            return is_open

        # Callback to handle color selection from popover
        @self.app.callback(
            [
                Output("datasets-store", "data", allow_duplicate=True),
                Output("dataset-table-container", "children", allow_duplicate=True),
            ],
            [Input({"type": "color-option", "index": ALL, "color": ALL}, "n_clicks")],
            [State({"type": "color-option", "index": ALL, "color": ALL}, "id")],
            prevent_initial_call=True,
        )
        def update_dataset_color(n_clicks_list, button_ids):
            if not any(n_clicks_list) or not n_clicks_list:
                raise PreventUpdate

            # Find which button was clicked
            ctx = dash.callback_context
            if not ctx.triggered:
                raise PreventUpdate

            # Get the triggered button's ID
            triggered_id = ctx.triggered[0]["prop_id"]
            # Extract the ID part (before the dot)
            import json

            id_str = triggered_id.split(".")[0]
            button_data = json.loads(id_str)

            dataset_index = button_data["index"]
            selected_color = button_data["color"]

            # Update the dataset color
            all_datasets = self.upstream_datasets + self.downstream_datasets
            for dataset in all_datasets:
                if dataset.get("id", 0) == dataset_index:
                    dataset["color"] = selected_color
                    break

            return len(self.upstream_datasets) + len(
                self.downstream_datasets
            ), self.create_dataset_table()

        # Callback for upstream plot settings changes
        @self.app.callback(
            [
                Output("upstream-settings-store", "data", allow_duplicate=True),
            ],
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
            # Store the upstream plot settings
            settings = {
                "x_scale": x_scale,
                "y_scale": y_scale,
                "x_min": x_min,
                "x_max": x_max,
                "y_min": y_min,
                "y_max": y_max,
            }
            return [settings]

        # Callback for downstream plot settings changes
        @self.app.callback(
            [
                Output("downstream-settings-store", "data", allow_duplicate=True),
            ],
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
            # Store the downstream plot settings
            settings = {
                "x_scale": x_scale,
                "y_scale": y_scale,
                "x_min": x_min,
                "x_max": x_max,
                "y_min": y_min,
                "y_max": y_max,
            }
            return [settings]

        # Callbacks for the upstream and downstream plots
        @self.app.callback(
            Output("upstream-plot", "figure"),
            [
                Input("datasets-store", "data"),
                Input("upstream-settings-store", "data"),
            ],
        )
        def update_upstream_plot(datasets_count, plot_settings):
            # Extract upstream plot settings or use defaults
            settings = plot_settings or {}
            x_scale = settings.get("x_scale")
            y_scale = settings.get("y_scale")
            x_min = settings.get("x_min")
            x_max = settings.get("x_max")
            y_min = settings.get("y_min")
            y_max = settings.get("y_max")

            print(f"Upstream plot callback with {len(self.upstream_datasets)} datasets")
            return self._generate_upstream_plot(
                x_scale, y_scale, x_min, x_max, y_min, y_max
            )

        @self.app.callback(
            Output("downstream-plot", "figure"),
            [
                Input("datasets-store", "data"),
                Input("downstream-settings-store", "data"),
            ],
        )
        def update_downstream_plot(datasets_count, plot_settings):
            # Extract downstream plot settings or use defaults
            settings = plot_settings or {}
            x_scale = settings.get("x_scale")
            y_scale = settings.get("y_scale")
            x_min = settings.get("x_min")
            x_max = settings.get("x_max")
            y_min = settings.get("y_min")
            y_max = settings.get("y_max")

            print(
                f"Downstream plot callback with {len(self.downstream_datasets)} datasets"
            )
            return self._generate_downstream_plot(
                x_scale, y_scale, x_min, x_max, y_min, y_max
            )

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
        fig = FigureResampler(go.Figure())

        for dataset in self.upstream_datasets + self.downstream_datasets:
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
        y_axis_type = y_scale if y_scale else "log"

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

        return fig

    def _generate_upstream_plot(
        self,
        x_scale=None,
        y_scale=None,
        x_min=None,
        x_max=None,
        y_min=None,
        y_max=None,
    ):
        """Generate the upstream pressure plot"""
        fig = FigureResampler(go.Figure())

        for dataset in self.upstream_datasets:
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
        y_axis_type = y_scale if y_scale else "log"

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

        return fig

    def _generate_downstream_plot(
        self,
        x_scale=None,
        y_scale=None,
        x_min=None,
        x_max=None,
        y_min=None,
        y_max=None,
    ):
        """Generate the downstream pressure plot"""
        fig = FigureResampler(go.Figure())

        for dataset in self.downstream_datasets:
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
        y_axis_type = y_scale if y_scale else "log"

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

        return fig

    def convert_timestamps_to_seconds(self, timestamp_strings):
        """Convert string timestamps to seconds since first timestamp"""
        if not timestamp_strings:
            return []

        # For simple numeric timestamps, just convert the strings to floats
        return [float(ts_str) for ts_str in timestamp_strings]

    def start(self):
        """Start the Dash web server"""
        print(f"Starting dashboard on http://localhost:{self.port}")

        # Open web browser after a short delay
        threading.Timer(
            1.0, lambda: webbrowser.open(f"http://127.0.0.1:{self.port}")
        ).start()

        # Run the server directly (blocking)
        self.app.run(debug=False, host="127.0.0.1", port=self.port)
