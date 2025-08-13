import base64
import io
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

    def create_layout(self):
        return dbc.Container(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            html.H1(
                                "SHIELD Data Visualization",
                                className="text-center mb-4",
                            ),
                            width=12,
                        ),
                    ]
                ),
                dbc.Row(
                    [
                        dbc.Col(width=8),  # Empty column for spacing
                        dbc.Col(
                            html.Div(
                                [
                                    dcc.Upload(
                                        id="upload-data",
                                        children=dbc.Button(
                                            [
                                                html.I(className="fa fa-upload me-2"),
                                                "Upload CSV",
                                            ],
                                            color="primary",
                                            size="sm",
                                            className="me-2",
                                        ),
                                        style={
                                            "display": "inline-block",
                                        },
                                        multiple=False,
                                        accept=".csv",
                                    ),
                                    dbc.Button(
                                        [
                                            html.I(className="fa fa-trash me-2"),
                                            "Clear All",
                                        ],
                                        id="clear-data",
                                        color="danger",
                                        size="sm",
                                        style={"display": "inline-block"},
                                    ),
                                ],
                                style={"textAlign": "right", "marginTop": "20px"},
                            ),
                            width=4,
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
                                                                                        value="log",
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
                                            is_open=True,
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
                                                                                        value="log",
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
                                            is_open=True,
                                        ),
                                    ]
                                ),
                            ],
                            width=6,
                        ),
                    ],
                    className="mt-3",
                ),
                # Dataset Management Card
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
        # Callback to handle file upload
        @self.app.callback(
            [
                Output("upload-status", "children"),
                Output("datasets-store", "data"),
                Output("dataset-table-container", "children"),
            ],
            [Input("upload-data", "contents")],
            [State("upload-data", "filename")],
        )
        def handle_file_upload(contents, filename):
            if contents is None:
                return (
                    "",
                    len(self.upstream_datasets) + len(self.downstream_datasets),
                    self.create_dataset_table(),
                )

            # Parse the uploaded file
            new_data = self.parse_uploaded_file(contents, filename)

            if not new_data.empty:
                # Determine which plot to add to: first upload goes to upstream, second to downstream
                total_datasets = len(self.upstream_datasets) + len(
                    self.downstream_datasets
                )

                if total_datasets % 2 == 0:  # Even number = upstream (0, 2, 4, ...)
                    target_datasets = self.upstream_datasets
                    plot_type = "Upstream"
                    dataset_id = f"upstream_dataset_{len(self.upstream_datasets) + 1}"
                    color = self.get_next_color(len(self.upstream_datasets))
                else:  # Odd number = downstream (1, 3, 5, ...)
                    target_datasets = self.downstream_datasets
                    plot_type = "Downstream"
                    dataset_id = (
                        f"downstream_dataset_{len(self.downstream_datasets) + 1}"
                    )
                    color = self.get_next_color(len(self.downstream_datasets))

                dataset = {
                    "data": new_data,
                    "filename": filename,
                    "display_name": f"{plot_type} Dataset {len(target_datasets) + 1}",
                    "color": color,
                    "visible": True,
                    "id": dataset_id,
                }
                target_datasets.append(dataset)

                return (
                    dbc.Alert(
                        [
                            html.I(className="fas fa-check-circle me-2"),
                            f"Successfully loaded {filename} to {plot_type} plot with {len(new_data)} data points. "
                            f"Total datasets: {len(self.upstream_datasets) + len(self.downstream_datasets)}",
                        ],
                        color="success",
                        dismissable=True,
                        duration=4000,
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
                return (
                    dbc.Alert(
                        [
                            html.I(className="fas fa-exclamation-circle me-2"),
                            f"Failed to load {filename}. Please check the file format.",
                        ],
                        color="danger",
                        dismissable=True,
                        duration=4000,
                        style={
                            "borderRadius": "8px",
                            "boxShadow": "0 4px 12px rgba(0,0,0,0.15)",
                            "border": "1px solid #f5c6cb",
                        },
                    ),
                    len(self.datasets),
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
                # Check if the required columns exist
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
        fig = go.Figure()

        for dataset in self.upstream_datasets:
            # Skip invisible datasets
            if not dataset.get("visible", True):
                continue

            data = dataset["data"]
            display_name = dataset.get("display_name", dataset["filename"])
            color = dataset["color"]

            if not data.empty:
                # Check if the required columns exist for upstream
                required_cols = ["RelativeTime", "Pressure (Torr)"]
                if all(col in data.columns for col in required_cols):
                    # Extract all data from CSV
                    time_data = data["RelativeTime"].values
                    pressure_data = data["Pressure (Torr)"].values

                    fig.add_trace(
                        go.Scatter(
                            x=time_data,
                            y=pressure_data,
                            mode="lines+markers",
                            name=display_name,
                            line=dict(color=color, width=2),
                            marker=dict(size=2),
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
        fig = go.Figure()

        for dataset in self.downstream_datasets:
            # Skip invisible datasets
            if not dataset.get("visible", True):
                continue

            data = dataset["data"]
            display_name = dataset.get("display_name", dataset["filename"])
            color = dataset["color"]

            if not data.empty:
                # Check if the required columns exist for downstream
                # For now, using same data but could be different columns
                required_cols = ["RelativeTime", "Pressure (Torr)"]
                if all(col in data.columns for col in required_cols):
                    # Extract all data from CSV
                    time_data = data["RelativeTime"].values
                    pressure_data = data["Pressure (Torr)"].values

                    fig.add_trace(
                        go.Scatter(
                            x=time_data,
                            y=pressure_data,
                            mode="lines+markers",
                            name=display_name,
                            line=dict(color=color, width=2),
                            marker=dict(size=2),
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

    def stop(self):
        """Stop the data plotter (CSV mode - nothing to stop)"""
        pass
