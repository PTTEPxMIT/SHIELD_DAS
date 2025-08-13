import base64
import io
import json
import threading
import webbrowser

import dash
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go
from dash import ALL, MATCH, dcc, html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate


class DataPlotter:
    def __init__(self, port=8050):
        """
        Initialize the DataPlotter.

        Args:
            port: Port for the Dash server
        """

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

        # Setup the app layout
        self.app.layout = self.create_layout()

        # Register callbacks
        self.register_callbacks()

        # Flag to track if recording has been started
        self.recording_started = False

    def parse_folder_upload(self, files_list, filenames_list):
        """
        Parse uploaded folder containing CSV and JSON files.

        Args:
            files_list: List of base64 encoded file contents
            filenames_list: List of corresponding filenames

        Returns:
            dict: Dictionary containing parsed data and metadata
        """
        result = {
            "pressure_data": None,
            "metadata": None,
            "success": False,
            "error": None,
        }

        try:
            csv_data = None
            json_data = None

            for contents, filename in zip(files_list, filenames_list):
                if filename.lower() == "pressure_gauge_data.csv":
                    csv_data = self.parse_csv_file(contents, filename)
                elif filename.lower() == "run_metadata.json":
                    json_data = self.parse_json_file(contents, filename)

            if csv_data is not None and not csv_data.empty:
                result["pressure_data"] = csv_data
                result["metadata"] = json_data  # Can be None if no JSON found
                result["success"] = True
                print(f"Successfully processed folder with {len(csv_data)} data points")
            else:
                result["error"] = "No valid pressure_gauge_data.csv file found"

        except Exception as e:
            result["error"] = f"Error processing folder upload: {str(e)}"
            print(result["error"])

        return result

    def parse_csv_file(self, contents, filename):
        """
        Parse CSV file content specifically for pressure gauge data.

        Args:
            contents: Base64 encoded file content
            filename: Name of the CSV file

        Returns:
            pandas.DataFrame: The parsed pressure data
        """
        try:
            # Decode the base64 content
            content_type, content_string = contents.split(",")
            decoded = base64.b64decode(content_string)

            # Read CSV from the decoded content
            df = pd.read_csv(io.StringIO(decoded.decode("utf-8")))

            # Validate expected columns (realtime + 4 voltage columns)
            if len(df.columns) < 5:
                print(
                    f"Warning: Expected 5 columns (realtime + 4 voltages), got {len(df.columns)}"
                )

            print(
                f"Successfully parsed {filename} with {len(df)} rows and {len(df.columns)} columns"
            )
            return df

        except Exception as e:
            print(f"Error parsing CSV file {filename}: {e}")
            return pd.DataFrame()

    def parse_json_file(self, contents, filename):
        """
        Parse JSON file content for run metadata.

        Args:
            contents: Base64 encoded file content
            filename: Name of the JSON file

        Returns:
            dict: The parsed metadata or None on error
        """
        try:
            # Decode the base64 content
            content_type, content_string = contents.split(",")
            decoded = base64.b64decode(content_string)

            # Parse JSON from the decoded content
            metadata = json.loads(decoded.decode("utf-8"))
            print(f"Successfully parsed metadata from {filename}")
            return metadata

        except Exception as e:
            print(f"Error parsing JSON file {filename}: {e}")
            return None

    def _determine_gauge_locations(self, voltage_columns, metadata):
        """
        Determine gauge locations for each voltage column based on metadata.

        Args:
            voltage_columns: List of column names from CSV (e.g., ['PG1_Voltage (V)', 'PG2_Voltage (V)'])
            metadata: Dictionary containing gauge information

        Returns:
            Dictionary mapping column names to locations ('upstream'/'downstream')
        """
        gauge_locations = {}

        # Get gauges from metadata
        gauges = metadata.get("gauges", [])
        if not gauges:
            print("Warning: No gauges found in metadata, defaulting all to upstream")
            return {col: "upstream" for col in voltage_columns}

        # Process each voltage column
        for col in voltage_columns:
            # Extract gauge name by removing "_Voltage (V)" suffix
            if col.endswith("_Voltage (V)"):
                gauge_name = col.replace("_Voltage (V)", "")

                # Find matching gauge in metadata
                matches = [g for g in gauges if g.get("name") == gauge_name]

                if len(matches) == 0:
                    print(
                        f"Warning: No gauge found for '{gauge_name}', defaulting to upstream"
                    )
                    gauge_locations[col] = "upstream"
                elif len(matches) > 1:
                    print(
                        f"Error: Multiple gauges found for '{gauge_name}', using first match"
                    )
                    gauge_locations[col] = matches[0].get("gauge_location", "upstream")
                else:
                    # Single match found
                    location = matches[0].get("gauge_location", "upstream")
                    gauge_locations[col] = location
            else:
                print(
                    f"Warning: Column '{col}' doesn't match expected format, defaulting to upstream"
                )
                gauge_locations[col] = "upstream"

        return gauge_locations

    def parse_csv_with_metadata(self, contents, filename):
        """
        Parse uploaded CSV file and automatically look for JSON metadata.
        JSON is parsed first to get version information for processing decisions.

        Args:
            contents: Base64 encoded file content
            filename: Name of the uploaded CSV file

        Returns:
            dict: Dictionary containing parsed data, metadata, and processing info
        """
        result = {
            "data": None,
            "metadata": None,
            "success": False,
            "error": None,
            "filename": filename,
            "version": None,
        }

        try:
            # First, try to find and parse JSON metadata in the same directory
            # Since we're in a web upload context, we can't directly access the file system
            # We'll need to implement a way to upload both files or read from a known location

            # For now, let's try to construct the expected JSON path and read it
            # This assumes the CSV is being uploaded from a specific directory structure
            expected_json_name = "run_metadata.json"

            # Try to read the JSON metadata from the same directory
            # Note: This is a simplified approach - in production you might want
            # to upload both files or have a different workflow
            import os

            metadata = None

            # Check if we can find the JSON file in common locations
            possible_paths = [
                f"results/08.12/test_run_4_11h05/run_metadata.json",  # Known test location
                f"run_metadata.json",  # Same directory
                expected_json_name,
            ]

            for json_path in possible_paths:
                try:
                    if os.path.exists(json_path):
                        with open(json_path) as f:
                            metadata = json.load(f)
                        break
                except Exception:
                    continue

            if metadata is None:
                # Fallback to basic metadata if JSON not found
                print("Warning: Could not find run_metadata.json, using basic metadata")
                metadata = {
                    "version": "1.0",
                    "run_info": {"test_mode": True},
                    "gauges": [],  # Empty gauges list
                }

            # Parse the version first (as requested)
            version = metadata.get("version", "1.0")
            result["version"] = version
            result["metadata"] = metadata

            # Now decode and parse the CSV content
            content_type, content_string = contents.split(",")
            decoded = base64.b64decode(content_string)

            # Read CSV from the decoded content
            df = pd.read_csv(io.StringIO(decoded.decode("utf-8")))

            if df.empty:
                result["error"] = "CSV file is empty"
                return result

            # Validate we have at least 2 columns (time + at least 1 voltage)
            if len(df.columns) < 2:
                result["error"] = f"Expected at least 2 columns, got {len(df.columns)}"
                return result

            # Process voltage columns and validate against metadata
            voltage_columns = list(df.columns[1:])  # All columns except first (time)
            gauge_locations = self._determine_gauge_locations(voltage_columns, metadata)

            result["data"] = df
            result["success"] = True

            # Update metadata with actual CSV information and gauge locations
            result["metadata"].update(
                {
                    "csv_columns": list(df.columns),
                    "rows": len(df),
                    "time_column": df.columns[0],
                    "voltage_columns": voltage_columns,
                    "gauge_locations": gauge_locations,
                }
            )

            print(
                f"Successfully parsed {filename} with {len(df)} rows and {len(df.columns)} columns"
            )
            print(f"Columns: {list(df.columns)}")
            print(f"Version-based processing: {version}")

        except Exception as e:
            result["error"] = f"Error parsing file {filename}: {e!s}"
            print(result["error"])

        return result

    def parse_uploaded_file(self, contents, filename):
        """
        Parse uploaded CSV file content.

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

            # Read CSV from the decoded content
            df = pd.read_csv(io.StringIO(decoded.decode("utf-8")))
            print(
                f"Successfully uploaded and parsed {filename} "
                f"with {len(df)} data points"
            )
            return df
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

    def calculate_error_bars(self, data, dataset_info=None, method="default"):
        """
        Calculate error bars for a dataset.

        This function can be extended to support different error calculation methods
        based on dataset type, measurement conditions, or other factors.

        Args:
            data: DataFrame containing the measurement data
            dataset_info: Dictionary with dataset metadata (for future extensions)
            method: Error calculation method ('default', 'percentage', 'stddev', etc.)

        Returns:
            numpy.array or None: Array of error values, or None if no error bars
        """

        if method == "default" or method == "percentage":
            if "Pressure (Torr)" in data.columns:
                pressure_values = data["Pressure (Torr)"].values
                # 10% error as default
                error_percentage = 0.10
                return pressure_values * error_percentage

        return None

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
                                                    # Upload and Clear buttons
                                                    dbc.Row(
                                                        [
                                                            dbc.Col(
                                                                [
                                                                    html.Div(
                                                                        [
                                                                            dcc.Upload(
                                                                                id="upload-data",
                                                                                children=dbc.Button(
                                                                                    [
                                                                                        html.I(
                                                                                            className="fa fa-upload me-2"
                                                                                        ),
                                                                                        "Upload CSV",
                                                                                    ],
                                                                                    color="primary",
                                                                                    size="sm",
                                                                                ),
                                                                                style={
                                                                                    "display": "inline-block",
                                                                                    "marginRight": "10px",
                                                                                },
                                                                                multiple=False,
                                                                                accept=".csv",
                                                                            ),
                                                                            dbc.Button(
                                                                                [
                                                                                    html.I(
                                                                                        className="fa fa-trash me-2"
                                                                                    ),
                                                                                    "Clear All",
                                                                                ],
                                                                                id="clear-data",
                                                                                color="danger",
                                                                                size="sm",
                                                                            ),
                                                                        ],
                                                                        style={
                                                                            "display": "flex",
                                                                            "gap": "10px",
                                                                            "alignItems": "center",
                                                                        },
                                                                    )
                                                                ],
                                                                width=12,
                                                                style={
                                                                    "textAlign": "left",
                                                                    "marginBottom": "15px",
                                                                },
                                                            ),
                                                        ]
                                                    ),
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
                                        dbc.CardHeader("Voltage Measurements"),
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
                                        dbc.CardHeader("Additional Data"),
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
                                                                    # Error bars checkbox
                                                                    dbc.Row(
                                                                        [
                                                                            dbc.Col(
                                                                                [
                                                                                    dbc.Checkbox(
                                                                                        id="upstream-error-bars",
                                                                                        label="Show Error Bars",
                                                                                        value=False,
                                                                                    ),
                                                                                ],
                                                                                width=12,
                                                                            ),
                                                                        ],
                                                                        className="mb-2",
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
                                                                    # Error bars checkbox
                                                                    dbc.Row(
                                                                        [
                                                                            dbc.Col(
                                                                                [
                                                                                    dbc.Checkbox(
                                                                                        id="downstream-error-bars",
                                                                                        label="Show Error Bars",
                                                                                        value=False,
                                                                                    ),
                                                                                ],
                                                                                width=12,
                                                                            ),
                                                                        ],
                                                                        className="mb-2",
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
        all_datasets = self.upstream_datasets + self.downstream_datasets
        if not all_datasets:
            return html.Div("No datasets loaded", className="text-muted")

        table_header = html.Thead(
            [
                html.Tr(
                    [
                        html.Th(
                            "Dataset Name",
                            style={"width": "50%", "fontWeight": "normal"},
                        ),
                        html.Th(
                            "Color", style={"width": "25%", "fontWeight": "normal"}
                        ),
                        html.Th(
                            "Display", style={"width": "25%", "fontWeight": "normal"}
                        ),
                    ]
                )
            ]
        )

        table_rows = []
        for i, dataset in enumerate(all_datasets):
            row = html.Tr(
                [
                    html.Td(
                        [
                            dbc.Input(
                                id={
                                    "type": "dataset-name",
                                    "index": dataset.get("id", i),
                                },
                                value=dataset.get("display_name", dataset["filename"]),
                                size="sm",
                                style={"fontSize": "12px"},
                            )
                        ]
                    ),
                    html.Td(
                        [
                            html.Div(
                                [
                                    # Color button that shows current color
                                    dbc.Button(
                                        "",
                                        id={
                                            "type": "color-button",
                                            "index": dataset.get("id", i),
                                        },
                                        style={
                                            "width": "30px",
                                            "height": "30px",
                                            "backgroundColor": dataset["color"],
                                            "border": "2px solid #ccc",
                                            "borderRadius": "5px",
                                            "padding": "0",
                                            "minWidth": "30px",
                                        },
                                        size="sm",
                                    ),
                                    # Color picker popup (hidden by default)
                                    dbc.Popover(
                                        [
                                            dbc.PopoverBody(
                                                [
                                                    # First row - 4 colors
                                                    html.Div(
                                                        [
                                                            dbc.Button(
                                                                "",
                                                                id={
                                                                    "type": "color-option",
                                                                    "index": dataset.get(
                                                                        "id", i
                                                                    ),
                                                                    "color": "#000000",
                                                                },
                                                                style={
                                                                    "width": "25px",
                                                                    "height": "25px",
                                                                    "backgroundColor": "#000000",
                                                                    "border": "1px solid #ccc",
                                                                    "borderRadius": "3px",
                                                                    "margin": "2px",
                                                                    "padding": "0",
                                                                    "minWidth": "25px",
                                                                },
                                                                size="sm",
                                                            ),
                                                            dbc.Button(
                                                                "",
                                                                id={
                                                                    "type": "color-option",
                                                                    "index": dataset.get(
                                                                        "id", i
                                                                    ),
                                                                    "color": "#DF1AD2",
                                                                },
                                                                style={
                                                                    "width": "25px",
                                                                    "height": "25px",
                                                                    "backgroundColor": "#DF1AD2",
                                                                    "border": "1px solid #ccc",
                                                                    "borderRadius": "3px",
                                                                    "margin": "2px",
                                                                    "padding": "0",
                                                                    "minWidth": "25px",
                                                                },
                                                                size="sm",
                                                            ),
                                                            dbc.Button(
                                                                "",
                                                                id={
                                                                    "type": "color-option",
                                                                    "index": dataset.get(
                                                                        "id", i
                                                                    ),
                                                                    "color": "#779BE7",
                                                                },
                                                                style={
                                                                    "width": "25px",
                                                                    "height": "25px",
                                                                    "backgroundColor": "#779BE7",
                                                                    "border": "1px solid #ccc",
                                                                    "borderRadius": "3px",
                                                                    "margin": "2px",
                                                                    "padding": "0",
                                                                    "minWidth": "25px",
                                                                },
                                                                size="sm",
                                                            ),
                                                            dbc.Button(
                                                                "",
                                                                id={
                                                                    "type": "color-option",
                                                                    "index": dataset.get(
                                                                        "id", i
                                                                    ),
                                                                    "color": "#49B6FF",
                                                                },
                                                                style={
                                                                    "width": "25px",
                                                                    "height": "25px",
                                                                    "backgroundColor": "#49B6FF",
                                                                    "border": "1px solid #ccc",
                                                                    "borderRadius": "3px",
                                                                    "margin": "2px",
                                                                    "padding": "0",
                                                                    "minWidth": "25px",
                                                                },
                                                                size="sm",
                                                            ),
                                                        ],
                                                        style={
                                                            "display": "flex",
                                                            "flexWrap": "nowrap",
                                                            "marginBottom": "2px",
                                                        },
                                                    ),
                                                    # Second row - 4 colors
                                                    html.Div(
                                                        [
                                                            dbc.Button(
                                                                "",
                                                                id={
                                                                    "type": "color-option",
                                                                    "index": dataset.get(
                                                                        "id", i
                                                                    ),
                                                                    "color": "#254E70",
                                                                },
                                                                style={
                                                                    "width": "25px",
                                                                    "height": "25px",
                                                                    "backgroundColor": "#254E70",
                                                                    "border": "1px solid #ccc",
                                                                    "borderRadius": "3px",
                                                                    "margin": "2px",
                                                                    "padding": "0",
                                                                    "minWidth": "25px",
                                                                },
                                                                size="sm",
                                                            ),
                                                            dbc.Button(
                                                                "",
                                                                id={
                                                                    "type": "color-option",
                                                                    "index": dataset.get(
                                                                        "id", i
                                                                    ),
                                                                    "color": "#0CCA4A",
                                                                },
                                                                style={
                                                                    "width": "25px",
                                                                    "height": "25px",
                                                                    "backgroundColor": "#0CCA4A",
                                                                    "border": "1px solid #ccc",
                                                                    "borderRadius": "3px",
                                                                    "margin": "2px",
                                                                    "padding": "0",
                                                                    "minWidth": "25px",
                                                                },
                                                                size="sm",
                                                            ),
                                                            dbc.Button(
                                                                "",
                                                                id={
                                                                    "type": "color-option",
                                                                    "index": dataset.get(
                                                                        "id", i
                                                                    ),
                                                                    "color": "#929487",
                                                                },
                                                                style={
                                                                    "width": "25px",
                                                                    "height": "25px",
                                                                    "backgroundColor": "#929487",
                                                                    "border": "1px solid #ccc",
                                                                    "borderRadius": "3px",
                                                                    "margin": "2px",
                                                                    "padding": "0",
                                                                    "minWidth": "25px",
                                                                },
                                                                size="sm",
                                                            ),
                                                            dbc.Button(
                                                                "",
                                                                id={
                                                                    "type": "color-option",
                                                                    "index": dataset.get(
                                                                        "id", i
                                                                    ),
                                                                    "color": "#A1B0AB",
                                                                },
                                                                style={
                                                                    "width": "25px",
                                                                    "height": "25px",
                                                                    "backgroundColor": "#A1B0AB",
                                                                    "border": "1px solid #ccc",
                                                                    "borderRadius": "3px",
                                                                    "margin": "2px",
                                                                    "padding": "0",
                                                                    "minWidth": "25px",
                                                                },
                                                                size="sm",
                                                            ),
                                                        ],
                                                        style={
                                                            "display": "flex",
                                                            "flexWrap": "nowrap",
                                                        },
                                                    ),
                                                ]
                                            )
                                        ],
                                        target={
                                            "type": "color-button",
                                            "index": dataset.get("id", i),
                                        },
                                        placement="top",
                                        is_open=False,
                                        id={
                                            "type": "color-popover",
                                            "index": dataset.get("id", i),
                                        },
                                    ),
                                ],
                                style={"position": "relative"},
                            )
                        ],
                        style={"textAlign": "center", "padding": "5px"},
                    ),
                    html.Td(
                        [
                            dbc.Checkbox(
                                id={
                                    "type": "show-dataset",
                                    "index": dataset.get("id", i),
                                },
                                value=dataset.get("visible", True),
                                style={"transform": "scale(1.2)"},
                            )
                        ],
                        style={
                            "textAlign": "center",
                            "verticalAlign": "middle",
                            "display": "flex",
                            "justifyContent": "center",
                            "alignItems": "center",
                        },
                    ),
                ]
            )
            table_rows.append(row)

        table_body = html.Tbody(table_rows)

        return dbc.Table(
            [table_header, table_body],
            striped=True,
            bordered=True,
            hover=True,
            responsive=True,
            size="sm",
        )

    def register_callbacks(self):
        # Callback to handle folder upload (multiple files)
        @self.app.callback(
            [
                Output("upload-status", "children"),
                Output("datasets-store", "data"),
                Output("dataset-table-container", "children"),
            ],
            [Input("upload-data", "contents")],
            [State("upload-data", "filename")],
        )
        def handle_csv_upload(contents, filename):
            if contents is None:
                return (
                    "",
                    len(self.upstream_datasets) + len(self.downstream_datasets),
                    self.create_dataset_table(),
                )

            # Handle single CSV file upload with automatic JSON detection
            if isinstance(contents, str) and isinstance(filename, str):
                # Parse CSV with metadata (JSON parsed first for version info)
                result = self.parse_csv_with_metadata(contents, filename)

                if result["success"] and result["data"] is not None:
                    df = result["data"]
                    metadata = result["metadata"]
                    version = result["version"]
                    gauge_locations = metadata.get("gauge_locations", {})

                    # Create dataset name from metadata
                    dataset_name = metadata.get(
                        "run_name", filename.replace(".csv", "")
                    )

                    # Now we'll create one dataset but track gauge locations for plotting
                    voltage_columns = metadata.get(
                        "voltage_columns", list(df.columns[1:])
                    )

                    # Separate columns by location for reference
                    upstream_columns = [
                        col
                        for col in voltage_columns
                        if gauge_locations.get(col, "upstream") == "upstream"
                    ]
                    downstream_columns = [
                        col
                        for col in voltage_columns
                        if gauge_locations.get(col, "upstream") == "downstream"
                    ]

                    # Create single dataset with all data but include location info
                    dataset_id = f"voltage_dataset_{len(self.upstream_datasets) + 1}"
                    color = self.get_next_color(len(self.upstream_datasets))

                    dataset = {
                        "data": df,
                        "metadata": metadata,
                        "filename": filename,
                        "display_name": dataset_name,
                        "color": color,
                        "visible": True,
                        "id": dataset_id,
                        "data_type": "voltage_measurements",
                        "version": version,
                        "gauge_locations": gauge_locations,
                        "upstream_columns": upstream_columns,
                        "downstream_columns": downstream_columns,
                    }

                    # Add to upstream datasets (we'll handle plotting logic elsewhere)
                    self.upstream_datasets.append(dataset)

                    return (
                        dbc.Alert(
                            [
                                html.I(className="fas fa-check-circle me-2"),
                                f"Successfully loaded {dataset_name} (v{version}) with {len(df)} data points. "
                                f"Upstream: {len(upstream_columns)} gauges, Downstream: {len(downstream_columns)} gauges",
                            ],
                            color="success",
                            dismissable=True,
                            duration=5000,
                            style={
                                "borderRadius": "8px",
                                "boxShadow": "0 4px 12px rgba(0,0,0,0.15)",
                                "border": "1px solid #d4edda",
                            },
                        ),
                        len(self.upstream_datasets) + len(self.downstream_datasets),
                        self.create_dataset_table(),
                    )
                else:
                    error_msg = result.get("error", "Unknown error processing CSV file")
                    return (
                        dbc.Alert(
                            [
                                html.I(className="fas fa-exclamation-triangle me-2"),
                                f"Error: {error_msg}",
                            ],
                            color="danger",
                            dismissable=True,
                            duration=5000,
                        ),
                        len(self.upstream_datasets) + len(self.downstream_datasets),
                        self.create_dataset_table(),
                    )
            else:
                return (
                    dbc.Alert(
                        [
                            html.I(className="fas fa-info-circle me-2"),
                            "Please upload a single CSV file.",
                        ],
                        color="info",
                        dismissable=True,
                        duration=4000,
                    ),
                    len(self.upstream_datasets) + len(self.downstream_datasets),
                    self.create_dataset_table(),
                )

        # Callback to handle clear button
        @self.app.callback(
            [
                Output("upload-status", "children", allow_duplicate=True),
                Output("datasets-store", "data", allow_duplicate=True),
                Output("dataset-table-container", "children", allow_duplicate=True),
            ],
            [Input("clear-data", "n_clicks")],
            prevent_initial_call=True,
        )
        def clear_all_data(n_clicks):
            if n_clicks:
                # Clear all datasets
                self.upstream_datasets = []
                self.downstream_datasets = []

                # Create empty figure
                fig = go.Figure()
                fig.update_yaxes(type="log")
                fig.update_layout(
                    height=600,
                    xaxis_title="Relative Time (s)",
                    yaxis_title="Pressure (Torr)",
                    template="plotly_white",
                    margin=dict(l=60, r=30, t=40, b=60),
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="center",
                        x=0.5,
                    ),
                )

                return (
                    dbc.Alert(
                        [
                            html.I(className="fas fa-info-circle me-2"),
                            "All data cleared",
                        ],
                        color="info",
                        dismissable=True,
                        duration=3000,
                        style={
                            "borderRadius": "8px",
                            "boxShadow": "0 4px 12px rgba(0,0,0,0.15)",
                            "border": "1px solid #bee5eb",
                        },
                    ),
                    0,
                    self.create_dataset_table(),
                )

            return (
                "",
                len(self.upstream_datasets) + len(self.downstream_datasets),
                self.create_dataset_table(),
            )

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
                    dataset["display_name"] = display_names[i] or dataset["filename"]
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
                Input("upstream-error-bars", "value"),
            ],
            prevent_initial_call=True,
        )
        def update_upstream_plot_settings(
            x_scale, y_scale, x_min, x_max, y_min, y_max, error_bars
        ):
            # Store the upstream plot settings
            settings = {
                "x_scale": x_scale,
                "y_scale": y_scale,
                "x_min": x_min,
                "x_max": x_max,
                "y_min": y_min,
                "y_max": y_max,
                "error_bars": error_bars,
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
                Input("downstream-error-bars", "value"),
            ],
            prevent_initial_call=True,
        )
        def update_downstream_plot_settings(
            x_scale, y_scale, x_min, x_max, y_min, y_max, error_bars
        ):
            # Store the downstream plot settings
            settings = {
                "x_scale": x_scale,
                "y_scale": y_scale,
                "x_min": x_min,
                "x_max": x_max,
                "y_min": y_min,
                "y_max": y_max,
                "error_bars": error_bars,
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
            error_bars = settings.get("error_bars")

            print(f"Upstream plot callback with {len(self.upstream_datasets)} datasets")
            return self._generate_upstream_plot(
                x_scale, y_scale, x_min, x_max, y_min, y_max, error_bars
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
            error_bars = settings.get("error_bars")

            print(
                f"Downstream plot callback with {len(self.downstream_datasets)} datasets"
            )
            return self._generate_downstream_plot(
                x_scale, y_scale, x_min, x_max, y_min, y_max, error_bars
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
        show_error_bars=None,
    ):
        """Generate the plot based on current dataset state and settings"""
        fig = go.Figure()

        for dataset in self.upstream_datasets + self.downstream_datasets:
            # Skip invisible datasets
            if not dataset.get("visible", True):
                continue

            data = dataset["data"]
            display_name = dataset.get("display_name", dataset["filename"])
            color = dataset["color"]

            if not data.empty:
                # Check dataset type and handle accordingly
                data_type = dataset.get("data_type", "pressure")

                if data_type == "voltage_measurements":
                    # Handle voltage data - plot all voltage columns
                    time_column = data.columns[0]  # First column is time
                    voltage_columns = data.columns[1:]  # Rest are voltage columns

                    if len(voltage_columns) > 0:
                        time_data = data[time_column].values

                        # Plot each voltage column as a separate trace
                        for i, voltage_col in enumerate(voltage_columns):
                            voltage_data = data[voltage_col].values

                            # Create unique trace name
                            trace_name = f"{display_name} - {voltage_col}"

                            fig.add_trace(
                                go.Scatter(
                                    x=time_data,
                                    y=voltage_data,
                                    mode="lines+markers",
                                    name=trace_name,
                                    line=dict(color=color, width=2),
                                    marker=dict(size=2),
                                )
                            )
                else:
                    # Handle pressure data (legacy)
                    required_cols = ["RelativeTime", "Pressure (Torr)"]
                    if all(col in data.columns for col in required_cols):
                        # Extract all data from CSV
                        time_data = data["RelativeTime"].values
                        pressure_data = data["Pressure (Torr)"].values

                        # Determine trace mode based on error bars setting
                        mode = "lines+markers"
                        error_y = None

                        # Add error bars if requested and data available
                        if show_error_bars and "Error" in data.columns:
                            error_data = data["Error"].values
                            error_y = dict(type="data", array=error_data, visible=True)

                        fig.add_trace(
                            go.Scatter(
                                x=time_data,
                                y=pressure_data,
                                mode=mode,
                                name=display_name,
                                line=dict(color=color, width=2),
                                marker=dict(size=2),
                                error_y=error_y,
                            )
                        )

        # Configure the layout - detect data type for appropriate labels
        has_voltage_data = any(
            dataset.get("data_type") == "voltage_measurements"
            for dataset in self.upstream_datasets + self.downstream_datasets
        )

        if has_voltage_data:
            y_title = "Voltage (V)"
            default_y_scale = "linear"
        else:
            y_title = "Pressure (Torr)"
            default_y_scale = "log"

        fig.update_layout(
            height=600,
            xaxis_title="Relative Time (s)",
            yaxis_title=y_title,
            template="plotly_white",
            margin=dict(l=60, r=30, t=40, b=60),
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5
            ),
        )

        # Apply axis scaling
        x_axis_type = x_scale if x_scale else "linear"
        y_axis_type = y_scale if y_scale else default_y_scale

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
        error_bars=None,
    ):
        """Generate the voltage measurements plot"""
        fig = go.Figure()

        for dataset in self.upstream_datasets:
            # Skip invisible datasets
            if not dataset.get("visible", True):
                continue

            data = dataset["data"]
            display_name = dataset.get("display_name", dataset["filename"])
            color = dataset["color"]
            data_type = dataset.get("data_type", "pressure")

            if not data.empty:
                if data_type == "voltage_measurements":
                    # Handle voltage data - plot only upstream columns for upstream plot
                    time_col = data.columns[0]  # First column should be realtime

                    # Get upstream columns for this dataset
                    upstream_columns = dataset.get("upstream_columns", [])
                    voltage_cols = upstream_columns  # Only plot upstream columns

                    # Define colors for voltage channels
                    channel_colors = [
                        "#FF6B6B",  # Red
                        "#4ECDC4",  # Teal
                        "#45B7D1",  # Blue
                        "#96CEB4",  # Green
                        "#FFD93D",  # Yellow
                        "#FF8A80",  # Light Red
                        "#A7FFEB",  # Light Teal
                        "#B3E5FC",  # Light Blue
                    ]

                    for i, voltage_col in enumerate(voltage_cols):
                        if voltage_col in data.columns:
                            time_data_raw = data[time_col].values
                            voltage_data = data[voltage_col].values

                            # Convert timestamp strings to numeric values
                            if len(time_data_raw) > 0 and isinstance(
                                time_data_raw[0], str
                            ):
                                # Parse timestamps and convert to seconds from start
                                import pandas as pd

                                time_series = pd.to_datetime(time_data_raw)
                                start_time = time_series[
                                    0
                                ]  # Use indexing instead of iloc
                                time_data = (
                                    (time_series - start_time).total_seconds().values
                                )
                            else:
                                time_data = time_data_raw

                            # Create trace for each voltage channel
                            trace_kwargs = {
                                "x": time_data,
                                "y": voltage_data,
                                "mode": "lines+markers",
                                "name": f"{display_name} - {voltage_col}",
                                "line": dict(
                                    color=channel_colors[i % len(channel_colors)],
                                    width=2,
                                ),
                                "marker": dict(size=2),
                            }

                            # Add error bars if enabled
                            if error_bars:
                                trace_kwargs["error_y"] = dict(
                                    type="constant",
                                    value=voltage_data.std() * 0.1,  # 10% of std dev
                                    visible=True,
                                )

                            fig.add_trace(go.Scatter(**trace_kwargs))

                else:
                    # Handle legacy pressure data format
                    required_cols = ["RelativeTime", "Pressure (Torr)"]
                    if all(col in data.columns for col in required_cols):
                        time_data = data["RelativeTime"].values
                        pressure_data = data["Pressure (Torr)"].values

                        trace_kwargs = {
                            "x": time_data,
                            "y": pressure_data,
                            "mode": "lines+markers",
                            "name": display_name,
                            "line": dict(color=color, width=2),
                            "marker": dict(size=2),
                        }

                        # Add error bars if enabled for pressure data
                        if error_bars:
                            error_values = self.calculate_error_bars(data, dataset)
                            if error_values is not None:
                                trace_kwargs["error_y"] = dict(
                                    type="data",
                                    array=error_values,
                                    visible=True,
                                    color=color,
                                    thickness=1.5,
                                    width=3,
                                )

                        fig.add_trace(go.Scatter(**trace_kwargs))

        # Configure the layout based on data type
        if any(
            ds.get("data_type") == "voltage_measurements"
            for ds in self.upstream_datasets
        ):
            # Voltage data layout
            fig.update_layout(
                height=500,
                xaxis_title="Time",
                yaxis_title="Voltage (V)",
                template="plotly_white",
                margin=dict(l=60, r=30, t=40, b=60),
                legend=dict(
                    orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5
                ),
            )
            # Default to linear scale for voltage data
            default_y_scale = "linear"
        else:
            # Pressure data layout
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
            # Default to log scale for pressure data
            default_y_scale = "log"

        # Apply axis scaling
        x_axis_type = x_scale if x_scale else "linear"
        y_axis_type = y_scale if y_scale else default_y_scale

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
        error_bars=None,
    ):
        """Generate the downstream voltage plot"""
        fig = go.Figure()

        # Look through all datasets for ones with downstream columns
        for dataset in self.upstream_datasets:
            # Skip invisible datasets
            if not dataset.get("visible", True):
                continue

            data = dataset["data"]
            display_name = dataset.get("display_name", dataset["filename"])
            color = dataset["color"]
            data_type = dataset.get("data_type", "pressure")

            if not data.empty:
                if data_type == "voltage_measurements":
                    # Handle voltage data - plot only downstream columns
                    time_col = data.columns[0]  # First column should be realtime

                    # Get downstream columns for this dataset
                    downstream_columns = dataset.get("downstream_columns", [])
                    voltage_cols = downstream_columns  # Only plot downstream columns

                    # Define colors for voltage channels
                    channel_colors = [
                        "#FF6B6B",  # Red
                        "#4ECDC4",  # Teal
                        "#45B7D1",  # Blue
                        "#96CEB4",  # Green
                        "#FFD93D",  # Yellow
                        "#FF8A80",  # Light Red
                        "#A7FFEB",  # Light Teal
                        "#B3E5FC",  # Light Blue
                    ]

                    for i, voltage_col in enumerate(voltage_cols):
                        if voltage_col in data.columns:
                            time_data_raw = data[time_col].values
                            voltage_data = data[voltage_col].values

                            # Convert timestamp strings to numeric values
                            if len(time_data_raw) > 0 and isinstance(
                                time_data_raw[0], str
                            ):
                                # Parse timestamps and convert to seconds from start
                                import pandas as pd

                                time_series = pd.to_datetime(time_data_raw)
                                start_time = time_series[0]
                                time_data = (
                                    (time_series - start_time).total_seconds().values
                                )
                            else:
                                time_data = time_data_raw

                            # Create trace for each voltage channel
                            trace_kwargs = {
                                "x": time_data,
                                "y": voltage_data,
                                "mode": "lines+markers",
                                "name": f"{display_name} - {voltage_col}",
                                "line": dict(
                                    color=channel_colors[i % len(channel_colors)],
                                    width=2,
                                ),
                                "marker": dict(size=2),
                            }

                            # Add error bars if enabled
                            if error_bars:
                                trace_kwargs["error_y"] = dict(
                                    type="constant",
                                    value=voltage_data.std() * 0.1,  # 10% of std dev
                                    visible=True,
                                    color=channel_colors[i % len(channel_colors)],
                                    thickness=1.5,
                                    width=3,
                                )

                            fig.add_trace(go.Scatter(**trace_kwargs))

                else:
                    # Handle pressure data (legacy code)
                    required_cols = ["RelativeTime", "Pressure (Torr)"]
                    if all(col in data.columns for col in required_cols):
                        # Extract all data from CSV
                        time_data = data["RelativeTime"].values
                        pressure_data = data["Pressure (Torr)"].values

                        # Create the trace based on error bars setting
                        trace_kwargs = {
                            "x": time_data,
                            "y": pressure_data,
                            "mode": "lines+markers",
                            "name": display_name,
                            "line": dict(color=color, width=2),
                            "marker": dict(size=2),
                        }

                        # Add error bars if enabled
                        if error_bars:
                            error_values = self.calculate_error_bars(data, dataset)
                            if error_values is not None:
                                trace_kwargs["error_y"] = dict(
                                    type="data",
                                    array=error_values,
                                    visible=True,
                                    color=color,
                                    thickness=1.5,
                                    width=3,
                                )

                        fig.add_trace(go.Scatter(**trace_kwargs))

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

    def stop(self):
        """Stop the data plotter (CSV mode - nothing to stop)"""
        pass


if __name__ == "__main__":
    plotter = DataPlotter()
    plotter.start()
