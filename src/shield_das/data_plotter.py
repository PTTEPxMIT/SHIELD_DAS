import base64
import io
import math
import threading
import webbrowser

import dash
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go
from dash import dcc, html
from dash.dependencies import Input, Output, State


class DataPlotter:
    def __init__(self, port=8050):
        """
        Initialize the DataPlotter.

        Args:
            port: Port for the Dash server
        """

        self.port = port
        self.app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        self.app_running = False
        self.server_thread = None

        # Store multiple datasets - list of dictionaries with data and metadata
        self.datasets = []

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
            "#0066cc",  # Blue
            "#ff6600",  # Orange
            "#00cc66",  # Green
            "#cc0066",  # Pink
            "#6600cc",  # Purple
            "#cc6600",  # Brown
            "#0066ff",  # Light Blue
            "#ff0066",  # Red
            "#66cc00",  # Lime
            "#6600ff",  # Indigo
        ]
        return colors[index % len(colors)]

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
                # Status message for upload feedback
                dbc.Row(
                    [
                        dbc.Col(
                            html.Div(id="upload-status", className="text-center"),
                            width=12,
                        )
                    ],
                    className="mb-2",
                ),
                # Single plot for all data
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Card(
                                    [
                                        dbc.CardHeader("Pressure Data"),
                                        dbc.CardBody([dcc.Graph(id="main-plot")]),
                                    ]
                                ),
                            ],
                            width=12,
                        ),
                    ]
                ),
            ],
            fluid=True,
        )

    def register_callbacks(self):
        # Callback to handle file upload
        @self.app.callback(
            [Output("upload-status", "children"), Output("datasets-store", "data")],
            [Input("upload-data", "contents")],
            [State("upload-data", "filename")],
        )
        def handle_file_upload(contents, filename):
            if contents is None:
                return "", len(self.datasets)

            # Parse the uploaded file
            new_data = self.parse_uploaded_file(contents, filename)

            if not new_data.empty:
                # Add the new dataset to our collection
                dataset = {
                    "data": new_data,
                    "filename": filename,
                    "color": self.get_next_color(len(self.datasets)),
                }
                self.datasets.append(dataset)

                return (
                    dbc.Alert(
                        f"Successfully loaded {filename} with {len(new_data)} data points. "
                        f"Total datasets: {len(self.datasets)}",
                        color="success",
                        dismissable=True,
                        duration=4000,
                    ),
                    len(self.datasets),
                )
            else:
                return (
                    dbc.Alert(
                        f"Failed to load {filename}. Please check the file format.",
                        color="danger",
                        dismissable=True,
                        duration=4000,
                    ),
                    len(self.datasets),
                )

        # Callback to handle clear button
        @self.app.callback(
            [
                Output("upload-status", "children", allow_duplicate=True),
                Output("main-plot", "figure", allow_duplicate=True),
                Output("datasets-store", "data", allow_duplicate=True),
            ],
            [Input("clear-data", "n_clicks")],
            prevent_initial_call=True,
        )
        def clear_all_data(n_clicks):
            if n_clicks:
                # Clear all datasets
                self.datasets = []

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
                        orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
                    ),
                    title="Pressure vs Time",
                )

                return (
                    dbc.Alert(
                        "All data cleared",
                        color="info",
                        dismissable=True,
                        duration=3000,
                    ),
                    fig,
                    0,
                )

            return "", go.Figure(), len(self.datasets)

        # Single callback for the main plot - updates when data changes
        @self.app.callback(
            Output("main-plot", "figure"),
            [Input("datasets-store", "data"), Input("main-plot", "id")],
        )
        def update_main_plot(datasets_count, plot_id):
            # Create figure
            fig = go.Figure()

            # Add traces for each dataset
            for i, dataset in enumerate(self.datasets):
                data = dataset["data"]
                filename = dataset["filename"]
                color = dataset["color"]

                if not data.empty:
                    # Check if the required columns exist
                    required_cols = ["RelativeTime", "Pressure (Torr)"]
                    if all(col in data.columns for col in required_cols):
                        # Extract all data from CSV
                        time_data = data["RelativeTime"].values
                        pressure_data = data["Pressure (Torr)"].values

                        # Add trace to plot with all data
                        fig.add_trace(
                            go.Scatter(
                                x=time_data,
                                y=pressure_data,
                                mode="lines+markers",
                                name=filename,
                                line=dict(color=color, width=2),
                                marker=dict(size=2),
                            )
                        )
                        print(
                            f"Added trace for {filename} with {len(time_data)} points"
                        )
                    else:
                        print(
                            f"Warning: {filename} missing required columns. "
                            f"Available: {data.columns.tolist()}"
                        )

            # Set up the plot layout
            if self.datasets:
                # If we have data, set y-axis to log scale
                all_pressure_data = []
                for dataset in self.datasets:
                    if not dataset["data"].empty:
                        pressure_values = dataset["data"]["Pressure (Torr)"].values
                        all_pressure_data.extend(pressure_values)

                if all_pressure_data and min(all_pressure_data) > 0:
                    y_min = min(all_pressure_data) * 0.5
                    y_max = max(all_pressure_data) * 2
                    fig.update_yaxes(
                        type="log", range=[math.log10(y_min), math.log10(y_max)]
                    )
                else:
                    fig.update_yaxes(type="log")
            else:
                # Empty plot - just set log scale
                fig.update_yaxes(type="log")

            # Update layout for the plot
            fig.update_layout(
                height=600,
                xaxis_title="Relative Time (s)",
                yaxis_title="Pressure (Torr)",
                template="plotly_white",
                margin=dict(l=60, r=30, t=40, b=60),
                legend=dict(
                    orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
                ),
                title="Pressure vs Time",
            )

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
