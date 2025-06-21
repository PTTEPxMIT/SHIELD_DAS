import math
import threading
import webbrowser

import dash
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import dcc, html
from dash.dependencies import Input, Output

from .data_recorder import DataRecorder


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
        return dbc.Container(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            html.H1(
                                "SHIELD Data Acquisition System",
                                className="text-center mb-4",
                            ),
                            width=12,
                        )
                    ]
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Button(
                                    "Start Recording",
                                    id="start-button",
                                    color="success",
                                    className="me-2",
                                ),
                                dbc.Button(
                                    "Stop Recording", id="stop-button", color="danger"
                                ),
                            ],
                            width=12,
                            className="text-center mb-4",
                        )
                    ]
                ),
                # Add a row for the time window control
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            dbc.Label(
                                                "Time Window (seconds):",
                                                className="me-2 align-middle",
                                            ),
                                            width="auto",
                                        ),
                                        dbc.Col(
                                            dbc.Input(
                                                id="time-window-input",
                                                type="number",
                                                min=1,
                                                max=86400,
                                                step=1,
                                                value=10,
                                                style={"width": "80px"},
                                            ),
                                            width="auto",
                                        ),
                                    ],
                                    className="d-flex justify-content-center align-items-center mb-3",
                                )
                            ],
                            width=12,
                        )
                    ]
                ),
                # Three columns layout: Upstream (left), Temperature (centre)
                # and Downstream (right)
                dbc.Row(
                    [
                        # Upstream Column (Left)
                        dbc.Col(
                            [
                                html.H3(
                                    "Upstream",
                                    className="text-center text-primary mb-3",
                                ),
                                # Upstream Gauge Cards (will be filled by callback)
                                html.Div(id="upstream-gauges", className="mb-3"),
                                # Upstream Plot
                                dbc.Card(
                                    [
                                        dbc.CardHeader("Upstream Pressure"),
                                        dbc.CardBody([dcc.Graph(id="upstream-plot")]),
                                    ]
                                ),
                            ],
                            width=4,
                        ),
                        # Temperature Column (Center)
                        dbc.Col(
                            [
                                html.H3(
                                    "Temperature",
                                    className="text-center text-success mb-3",
                                ),
                                # Temperature Card
                                dbc.Card(
                                    [
                                        dbc.CardHeader("Temperature"),
                                        dbc.CardBody(
                                            [
                                                html.H3(
                                                    id="current-temperature",
                                                    className="text-center",
                                                ),
                                                html.P(
                                                    "°C", className="text-center mb-0"
                                                ),
                                            ]
                                        ),
                                    ],
                                    className="border-success mb-3",
                                ),
                                # Temperature Plot
                                dbc.Card(
                                    [
                                        dbc.CardHeader("Temperature History"),
                                        dbc.CardBody(
                                            [dcc.Graph(id="temperature-plot")]
                                        ),
                                    ]
                                ),
                            ],
                            width=4,
                        ),
                        # Downstream Column (Right)
                        dbc.Col(
                            [
                                html.H3(
                                    "Downstream",
                                    className="text-center text-danger mb-3",
                                ),
                                # Downstream Gauge Cards (will be filled by callback)
                                html.Div(id="downstream-gauges", className="mb-3"),
                                # Downstream Plot
                                dbc.Card(
                                    [
                                        dbc.CardHeader("Downstream Pressure"),
                                        dbc.CardBody([dcc.Graph(id="downstream-plot")]),
                                    ]
                                ),
                            ],
                            width=4,
                        ),
                    ]
                ),
                # Row for the toggle and interval
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Switch(
                                    id="error-bars-toggle",
                                    label="Show Error Bars",
                                    value=True,
                                    className="d-inline-block mt-3",
                                ),
                                dcc.Interval(
                                    id="interval-component",
                                    interval=500,  # in milliseconds
                                    n_intervals=0,
                                ),
                            ],
                            width=12,
                            className="text-center mb-3",
                        )
                    ]
                ),
            ],
            fluid=True,
        )

    def register_callbacks(self):
        # Update the gauge cards based on location
        @self.app.callback(
            [
                Output("downstream-gauges", "children"),
                Output("upstream-gauges", "children"),
            ],
            [Input("interval-component", "n_intervals")],
        )
        def update_gauge_cards(n_intervals):
            # Separate gauges by location
            downstream_gauges = []
            upstream_gauges = []

            # Count gauges by location to determine card width
            downstream_count = sum(
                1
                for gauge in self.recorder.gauges
                if gauge.gauge_location == "downstream"
            )
            upstream_count = sum(
                1
                for gauge in self.recorder.gauges
                if gauge.gauge_location == "upstream"
            )

            # Calculate card width (ensure it fits in a row)
            downstream_width = 12 // downstream_count if downstream_count > 0 else 12
            upstream_width = 12 // upstream_count if upstream_count > 0 else 12

            # Create cards for each gauge
            for gauge in self.recorder.gauges:
                # Prepare gauge card
                header = gauge.name
                value = "--"  # Default when no data

                # Get latest pressure if available
                if gauge.pressure_data:
                    latest_pressure = gauge.pressure_data[-1]
                    value = f"{latest_pressure:.2e}"

                # Create the gauge card content
                card_content = [
                    dbc.CardHeader(header),
                    dbc.CardBody(
                        [
                            html.H3(value, className="text-center"),
                            html.P("Torr", className="text-center mb-0"),
                        ]
                    ),
                ]

                # Add to appropriate list with the correct width
                if gauge.gauge_location == "downstream":
                    downstream_gauges.append(
                        dbc.Col(
                            dbc.Card(card_content, className="border-danger h-100"),
                            width=downstream_width,
                        )
                    )
                else:
                    upstream_gauges.append(
                        dbc.Col(
                            dbc.Card(card_content, className="border-primary h-100"),
                            width=upstream_width,
                        )
                    )

            # Wrap cards in rows if there are any
            downstream_content = (
                dbc.Row(downstream_gauges)
                if downstream_gauges
                else html.Div("No downstream gauges")
            )
            upstream_content = (
                dbc.Row(upstream_gauges)
                if upstream_gauges
                else html.Div("No upstream gauges")
            )

            return downstream_content, upstream_content

        # Create separate plots for upstream and downstream
        @self.app.callback(
            [Output("downstream-plot", "figure"), Output("upstream-plot", "figure")],
            [
                Input("interval-component", "n_intervals"),
                Input("error-bars-toggle", "value"),
                Input("time-window-input", "value"),
            ],
        )
        def update_plots(n_intervals, show_errors, time_window):
            # Default to 10 seconds if invalid value
            if time_window is None or time_window < 1:
                time_window = 10

            # Create figures for each location
            downstream_fig = go.Figure()
            upstream_fig = go.Figure()

            # Track data statistics
            has_data = False
            all_times = []

            # Track min/max values for y-axis scaling
            upstream_min = float("inf")
            upstream_max = float("-inf")
            downstream_min = float("inf")
            downstream_max = float("-inf")

            # Define colors for each location
            upstream_colors = ["#0066cc", "#003366"]  # Blue, Dark Blue
            downstream_colors = ["#ff9900", "#cc0000"]  # Orange, Red

            # Track gauge counts for each location
            upstream_count = 0
            downstream_count = 0

            # Get current time (latest reading)
            current_time = 0
            for gauge in self.recorder.gauges:
                if (
                    gauge.timestamp_data
                    and float(gauge.timestamp_data[-1]) > current_time
                ):
                    current_time = float(gauge.timestamp_data[-1])

            # Process each gauge
            for gauge in self.recorder.gauges:
                # Create a copy of the data to prevent changes during plotting
                timestamp_copy = gauge.timestamp_data.copy()
                pressure_copy = gauge.pressure_data.copy()

                # Only proceed if we have data
                if len(timestamp_copy) > 0:
                    has_data = True
                    # Convert string timestamps to seconds since start
                    time_seconds = self.convert_timestamps_to_seconds(timestamp_copy)

                    # Filter data based on time window
                    if len(time_seconds) > 0:
                        # Find data points within the time window
                        time_window_data = []
                        for i, t in enumerate(time_seconds):
                            if current_time - t <= time_window:
                                time_window_data.append(i)

                        if time_window_data:
                            time_seconds = [time_seconds[i] for i in time_window_data]
                            pressure_copy = [pressure_copy[i] for i in time_window_data]
                        else:
                            # If no data in window, show the most recent points
                            time_seconds = (
                                time_seconds[-5:]
                                if len(time_seconds) > 5
                                else time_seconds
                            )
                            pressure_copy = (
                                pressure_copy[-5:]
                                if len(pressure_copy) > 5
                                else pressure_copy
                            )

                    # Keep track of all time values for setting axis limits later
                    all_times.extend(time_seconds)

                    # Calculate error bars for each pressure value
                    error_values = [gauge.calculate_error(p) for p in pressure_copy]

                    # Add trace to appropriate plot
                    if gauge.gauge_location == "upstream":
                        # Update min/max for upstream
                        if pressure_copy:
                            upstream_min = min(upstream_min, min(pressure_copy))
                            upstream_max = max(upstream_max, max(pressure_copy))

                        color = upstream_colors[upstream_count % len(upstream_colors)]
                        upstream_count += 1

                        upstream_fig.add_trace(
                            go.Scatter(
                                x=time_seconds,
                                y=pressure_copy,
                                mode="lines+markers",
                                name=gauge.name,
                                line=dict(color=color),
                                error_y=dict(
                                    type="data",
                                    array=error_values,
                                    visible=show_errors,
                                    color=color,
                                    thickness=1.5,
                                    width=3,
                                ),
                            )
                        )
                    elif gauge.gauge_location == "downstream":
                        # Update min/max for downstream
                        if pressure_copy:
                            downstream_min = min(downstream_min, min(pressure_copy))
                            downstream_max = max(downstream_max, max(pressure_copy))

                        color = downstream_colors[
                            downstream_count % len(downstream_colors)
                        ]
                        downstream_count += 1

                        downstream_fig.add_trace(
                            go.Scatter(
                                x=time_seconds,
                                y=pressure_copy,
                                mode="lines+markers",
                                name=gauge.name,
                                line=dict(color=color),
                                error_y=dict(
                                    type="data",
                                    array=error_values,
                                    visible=show_errors,
                                    color=color,
                                    thickness=1.5,
                                    width=3,
                                ),
                            )
                        )

            # Set x-axis limits based on time window
            if has_data:
                # Synchronize x-axis based on current time and window
                x_min = max(0, current_time - time_window)
                x_max = current_time

                # Use same x range for both plots
                for fig in [upstream_fig, downstream_fig]:
                    fig.update_xaxes(range=[x_min, x_max])

            # Set y-axis limits and log scale for upstream plot
            if upstream_min != float("inf") and upstream_max != float("-inf"):
                # For log scale, we need to ensure values are positive
                if upstream_min <= 0:
                    upstream_min = 1e-10  # Small positive value

                # Set the limits with factor of 10 above and below
                y_min = upstream_min * 0.1
                y_max = upstream_max * 10

                upstream_fig.update_yaxes(
                    type="log", range=[math.log10(y_min), math.log10(y_max)]
                )
            else:
                # Default log scale if no data
                upstream_fig.update_yaxes(type="log")

            # Set y-axis limits and log scale for downstream plot
            if downstream_min != float("inf") and downstream_max != float("-inf"):
                # For log scale, we need to ensure values are positive
                if downstream_min <= 0:
                    downstream_min = 1e-10  # Small positive value

                # Set the limits with factor of 10 above and below
                y_min = downstream_min * 0.1
                y_max = downstream_max * 10

                downstream_fig.update_yaxes(
                    type="log", range=[math.log10(y_min), math.log10(y_max)]
                )
            else:
                # Default log scale if no data
                downstream_fig.update_yaxes(type="log")

            # Update layout for upstream plot
            upstream_fig.update_layout(
                height=400,
                xaxis_title="Relative Time (s)",
                yaxis_title="Pressure (Torr)",
                template="plotly_white",
                margin=dict(l=50, r=20, t=30, b=50),
                legend=dict(
                    orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
                ),
            )
            # Update layout for downstream plot
            downstream_fig.update_layout(
                height=400,
                xaxis_title="Relative Time (s)",
                yaxis_title="Pressure (Torr)",
                template="plotly_white",
                margin=dict(l=50, r=20, t=30, b=50),
                legend=dict(
                    orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
                ),
            )

            return downstream_fig, upstream_fig

        # Create temperature plot and update temperature value
        @self.app.callback(
            [
                Output("temperature-plot", "figure"),
                Output("current-temperature", "children"),
            ],
            [
                Input("interval-component", "n_intervals"),
                Input("time-window-input", "value"),
                Input("error-bars-toggle", "value"),
            ],
        )
        def update_temperature_plot(n_intervals, time_window, show_errors):
            # Default to 10 seconds if invalid value
            if time_window is None or time_window < 1:
                time_window = 10

            # Create temperature figure
            temp_fig = go.Figure()

            # Track if we have data
            has_data = False
            current_temp = "--"

            # Get temperature data from recorder
            temp_data = self.recorder.temperature_data.copy()
            temp_timestamps = self.recorder.temperature_timestamps.copy()

            # Get global time range from all data for synchronized x-axis
            current_time = 0

            # First find the latest time point from all sources (gauges and temperature)
            for gauge in self.recorder.gauges:
                if (
                    gauge.timestamp_data
                    and float(gauge.timestamp_data[-1]) > current_time
                ):
                    current_time = float(gauge.timestamp_data[-1])

            if temp_timestamps and len(temp_timestamps) > 0:
                latest_temp_time = max(temp_timestamps)
                if latest_temp_time > current_time:
                    current_time = latest_temp_time

                has_data = True

                # Filter data based on time window
                time_window_data = []
                for i, t in enumerate(temp_timestamps):
                    if current_time - t <= time_window:
                        time_window_data.append(i)

                if time_window_data:
                    filtered_timestamps = [temp_timestamps[i] for i in time_window_data]
                    filtered_temps = [temp_data[i] for i in time_window_data]
                else:
                    # If no data in window, show the most recent points
                    filtered_timestamps = (
                        temp_timestamps[-5:]
                        if len(temp_timestamps) > 5
                        else temp_timestamps
                    )
                    filtered_temps = temp_data[-5:] if len(temp_data) > 5 else temp_data

                # Add trace for temperature
                temp_fig.add_trace(
                    go.Scatter(
                        x=filtered_timestamps,
                        y=filtered_temps,
                        mode="lines+markers",
                        name="Temperature",
                        line=dict(color="#28a745"),  # Bootstrap success color
                        error_y=dict(
                            type="data",
                            # Calculate error using the specified formula
                            array=[
                                math.sqrt(2.2**2 + 2.0**2 + 0.5**2)
                                for _ in filtered_temps
                            ],
                            visible=show_errors,
                            color="#28a745",
                            thickness=1.5,
                            width=3,
                        ),
                    )
                )

                # Update current temperature display
                if temp_data:
                    current_temp = f"{temp_data[-1]:.1f}"

            # Set x-axis limits based on time window
            if has_data:
                # Synchronize x-axis with pressure plots
                x_min = max(0, current_time - time_window)
                x_max = current_time
                temp_fig.update_xaxes(range=[x_min, x_max])

            # Update layout for temperature plot
            temp_fig.update_layout(
                height=400,
                xaxis_title="Relative Time (s)",
                yaxis_title="Temperature (°C)",
                template="plotly_white",
                margin=dict(l=50, r=20, t=30, b=50),
                legend=dict(
                    orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
                ),
            )

            return temp_fig, current_temp

        @self.app.callback(
            [Output("start-button", "disabled"), Output("stop-button", "disabled")],
            [Input("start-button", "n_clicks"), Input("stop-button", "n_clicks")],
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
