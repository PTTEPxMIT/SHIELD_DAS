from .data_recorder import DataRecorder
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash_bootstrap_components as dbc

import threading
import webbrowser


class DataPlotter:
    def __init__(self, recorder: DataRecorder, port=8050):
        self.recorder = recorder
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

    def create_layout(self):
        return dbc.Container([
            dbc.Row([
                dbc.Col(
                    html.H1("SHIELD Data Acquisition System", className="text-center mb-4"),
                    width=12
                )
            ]),
            dbc.Row([
                dbc.Col([
                    dbc.Button("Start Recording", id="start-button", color="success", className="me-2"),
                    dbc.Button("Stop Recording", id="stop-button", color="danger")
                ], width=12, className="text-center mb-4")
            ]),
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Pressure Readings"),
                        dbc.CardBody([
                            dcc.Graph(id="pressure-plots"),
                            dcc.Interval(
                                id="interval-component",
                                interval=500,  # in milliseconds
                                n_intervals=0
                            )
                        ])
                    ])
                ], width=12)
            ]),
            # New row for the toggle, centered below the graphs
            dbc.Row([
                dbc.Col([
                    dbc.Switch(
                        id="error-bars-toggle",
                        label="Show Error Bars",
                        value=True,
                        className="d-inline-block mt-3"  # Add margin top
                    )
                ], width=12, className="text-center mb-3")  # Center the content
            ])
        ], fluid=True)

    def register_callbacks(self):
        @self.app.callback(
            Output("pressure-plots", "figure"),
            [Input("interval-component", "n_intervals"),
            Input("error-bars-toggle", "value")],
        )
        def update_plots(n_intervals, show_errors):  # No 'self' parameter here
            # show_errors will be True/False instead of a list with dbc.Switch
            show_error_bars = show_errors

            # Create figure with two subplots side by side
            fig = make_subplots(
                rows=1,
                cols=2,
                subplot_titles=("Upstream Pressures", "Downstream Pressures"),
                shared_yaxes=False,
            )

            # Track if we have data to display
            has_data = False
            all_times = []
            
            # Track gauge counts for each location (to assign colors)
            upstream_count = 0
            downstream_count = 0
            
            # Define colors for each location
            upstream_colors = ["#0066cc", "#003366"]  # Blue, Dark Blue
            downstream_colors = ["#ff9900", "#cc0000"]  # Orange, Red

            # Create figure
            for gauge in self.recorder.gauges:
                # Create a copy of the data to prevent changes during plotting
                timestamp_copy = gauge.timestamp_data.copy()
                pressure_copy = gauge.pressure_data.copy()

                # Only proceed if we have data
                if len(timestamp_copy) > 0:
                    has_data = True
                    # Convert string timestamps to seconds since start
                    time_seconds = self.convert_timestamps_to_seconds(timestamp_copy)

                    # Only show the last 20 data points if we have more
                    if len(time_seconds) > 20:
                        time_seconds = time_seconds[-20:]
                        pressure_copy = pressure_copy[-20:]

                    # Keep track of all time values for setting axis limits later
                    all_times.extend(time_seconds)
                    
                    # Calculate error bars for each pressure value
                    error_values = [gauge.calculate_error(p) for p in pressure_copy]
                    
                    # Add trace to appropriate subplot
                    if gauge.gauge_location == "upstream":
                        color = upstream_colors[upstream_count % len(upstream_colors)]
                        upstream_count += 1
                        
                        fig.add_trace(
                            go.Scatter(
                                x=time_seconds,
                                y=pressure_copy,
                                mode="lines+markers",
                                name=gauge.name,
                                line=dict(color=color),
                                error_y=dict(
                                    type='data',
                                    array=error_values,
                                    visible=show_error_bars,
                                    color=color,
                                    thickness=1.5,
                                    width=3
                                )
                            ),
                            row=1,
                            col=1,
                        )
                    elif gauge.gauge_location == "downstream":
                        color = downstream_colors[downstream_count % len(downstream_colors)]
                        downstream_count += 1
                        
                        fig.add_trace(
                            go.Scatter(
                                x=time_seconds,
                                y=pressure_copy,
                                mode="lines+markers",
                                name=gauge.name,
                                line=dict(color=color),
                                error_y=dict(
                                    type='data',
                                    array=error_values,
                                    visible=show_error_bars,
                                    color=color,
                                    thickness=1.5,
                                    width=3
                                )
                            ),
                            row=1,
                            col=2,
                        )

            # Set x-axis limits based on data
            if all_times:
                x_min = min(all_times)
                x_max = max(all_times)
                margin = (x_max - x_min) * 0.05 if x_max > x_min else 0.1

                fig.update_xaxes(range=[x_min - margin, x_max + margin], row=1, col=1)
                fig.update_xaxes(range=[x_min - margin, x_max + margin], row=1, col=2)

            # Update layout
            fig.update_layout(
                height=500,
                xaxis_title="Time (s)",
                yaxis_title="Pressure (Torr)",
                xaxis2_title="Time (s)",
                yaxis2_title="Pressure (Torr)",
                legend=dict(
                    orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5
                ),
                template="plotly_white",
            )

            return fig

        @self.app.callback(
            [Output("start-button", "disabled"),
             Output("stop-button", "disabled")],
            [Input("start-button", "n_clicks"),
             Input("stop-button", "n_clicks")],
            prevent_initial_call=True,
        )
        def handle_buttons(start_clicks, stop_clicks):
            ctx = dash.callback_context

            if not ctx.triggered:
                return False, True

            button_id = ctx.triggered[0]["prop_id"].split(".")[0]

            if button_id == "start-button":
                # Start a fresh recording session
                self.recorder.reset()  # Reset everything first
                self.recorder.start()  # Then start recording
                self.recording_started = True
                return True, False
            elif button_id == "stop-button":
                # Stop recording and reset for next run
                self.recorder.stop()
                self.recording_started = False
                return False, True

            # Default state
            return False, True

    def convert_timestamps_to_seconds(self, timestamp_strings):
        """Convert string timestamps to seconds since first timestamp"""
        if not timestamp_strings:
            return []

        # For simple numeric timestamps, just convert the strings to floats
        return [float(ts_str) for ts_str in timestamp_strings]

    def start(self):
        """Start the Dash web server in a separate thread"""
        if not self.app_running:
            self.app_running = True

            # Open web browser after a short delay
            threading.Timer(
                1.0, lambda: webbrowser.open(f"http://127.0.0.1:{self.port}")
            ).start()

            # Start the server in a separate thread
            self.server_thread = threading.Thread(
                target=lambda: self.app.run(debug=False, port=self.port)
            )
            self.server_thread.daemon = True
            self.server_thread.start()

    def stop(self):
        """Stop the data recorder"""
        if self.recording_started:
            self.recorder.stop()