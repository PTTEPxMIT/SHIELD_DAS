import math
import threading
import webbrowser

import dash
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go
from dash import dcc, html
from dash.dependencies import Input, Output


class DataPlotter:
    def __init__(self, csv_file_path=None, port=8050):
        """
        Initialize the DataPlotter with a CSV file path.

        Args:
            csv_file_path: Path to the CSV file containing pressure data
            port: Port for the Dash server
        """
        # Read the CSV data
        self.csv_file_path = (
            csv_file_path
            or "results/07.15/run_5_13h56/Baratron626D_1T_downstream_pressure_data.csv"
        )
        self.data = self.load_csv_data()

        self.port = port
        self.app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        self.app_running = False
        self.server_thread = None

        # Setup the app layout
        self.app.layout = self.create_layout()

        # Register callbacks
        self.register_callbacks()

        # Flag to track if recording has been started
        self.recording_started = False

    def load_csv_data(self):
        """
        Load data from the CSV file.

        Returns:
            pandas.DataFrame: The loaded data
        """
        try:
            df = pd.read_csv(self.csv_file_path)
            print(f"Loaded {len(df)} data points from {self.csv_file_path}")
            return df
        except Exception as e:
            print(f"Error loading CSV data: {e}")
            return pd.DataFrame()  # Return empty DataFrame on error

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
                        )
                    ]
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
        # Single callback for the main plot - loads once when app starts
        @self.app.callback(
            Output("main-plot", "figure"),
            [Input("main-plot", "id")],  # Triggers once when component is loaded
        )
        def update_main_plot(plot_id):
            # Create figure
            fig = go.Figure()

            # Check if we have data
            if not self.data.empty:
                # Extract all data from CSV
                time_data = self.data["RelativeTime"].values
                pressure_data = self.data["Pressure (Torr)"].values

                # Add trace to plot with all data
                fig.add_trace(
                    go.Scatter(
                        x=time_data,
                        y=pressure_data,
                        mode="lines+markers",
                        name="Pressure Data",
                        line=dict(color="#0066cc", width=2),
                        marker=dict(size=2),
                    )
                )

                # Set y-axis to log scale if we have positive data
                if len(pressure_data) > 0 and min(pressure_data) > 0:
                    y_min = min(pressure_data) * 0.5
                    y_max = max(pressure_data) * 2
                    fig.update_yaxes(
                        type="log", range=[math.log10(y_min), math.log10(y_max)]
                    )
                else:
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
