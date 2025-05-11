import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.animation as animation
from matplotlib.figure import Figure
from .data_recorder import DataRecorder
from datetime import datetime


class DataPlotter:
    def __init__(self, root, recorder: DataRecorder):
        self.root = root
        self.recorder = recorder
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        # Layout frames
        self.control_frame = tk.Frame(root)
        self.control_frame.pack(side=tk.TOP, fill=tk.X)

        self.plot_frame = tk.Frame(root)
        self.plot_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Create the figure with two subplots
        self.fig = Figure(figsize=(10, 4), dpi=100)
        self.ax_upstream = self.fig.add_subplot(121)
        self.ax_downstream = self.fig.add_subplot(122)

        self.ax_upstream.set_title("Upstream Pressures")
        self.ax_downstream.set_title("Downstream Pressures")

        self.ax_upstream.set_xlabel("Time (s)")
        self.ax_upstream.set_ylabel("Pressure (Torr)")

        self.ax_downstream.set_xlabel("Time (s)")
        self.ax_downstream.set_ylabel("Pressure (Torr)")

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Animation
        self.ani = animation.FuncAnimation(
            self.fig, self.update_plot, interval=500, cache_frame_data=False
        )

    def start(self):
        self.recorder.start()

    def stop(self):
        self.recorder.stop()

    def convert_timestamps_to_seconds(self, timestamp_strings):
        """Convert string timestamps to seconds since first timestamp"""
        if not timestamp_strings:
            return []

        times = []
        for ts_str in timestamp_strings:
            dt = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S.%f")
            times.append(dt.timestamp())

        # Calculate seconds since first timestamp
        start_time = times[0]
        return [t - start_time for t in times]

    def update_plot(self, frame):

        # Clear the plots first
        self.ax_downstream.clear()
        self.ax_upstream.clear()

        # Reset titles and labels
        self.ax_upstream.set_title("Upstream Pressures")
        self.ax_downstream.set_title("Downstream Pressures")

        self.ax_upstream.set_xlabel("Time (s)")
        self.ax_upstream.set_ylabel("Pressure (Torr)")

        self.ax_downstream.set_xlabel("Time (s)")
        self.ax_downstream.set_ylabel("Pressure (Torr)")

        # Variables to track the overall min/max x values across all gauges
        all_times = []
        has_data = False  # Flag to check if any data was plotted

        for gauge in self.recorder.gauges:
            # Create a copy of the data to prevent changes during plotting
            timestamp_copy = gauge.timestamp_data.copy()
            pressure_copy = gauge.pressure_data.copy()

            # Only proceed if we have data
            if len(timestamp_copy) > 0:
                has_data = True  # We have data to plot
                # Convert string timestamps to seconds since start
                time_seconds = self.convert_timestamps_to_seconds(timestamp_copy)

                # Only show the last 20 data points if we have more
                if len(time_seconds) > 20:
                    time_seconds = time_seconds[-20:]
                    pressure_copy = pressure_copy[-20:]

                # Keep track of all time values for setting axis limits later
                all_times.extend(time_seconds)

                if gauge.gauge_location == "upstream":
                    self.ax_upstream.plot(
                        time_seconds,
                        pressure_copy,
                        label=gauge.name,
                        color="blue",
                    )
                elif gauge.gauge_location == "downstream":
                    self.ax_downstream.plot(
                        time_seconds,
                        pressure_copy,
                        label=gauge.name,
                        color="orange",
                    )

        # Only add a legend if we have data to plot
        if has_data:
            self.ax_downstream.legend()
            self.ax_downstream.spines["top"].set_visible(False)
            self.ax_downstream.spines["right"].set_visible(False)
            self.ax_upstream.spines["top"].set_visible(False)
            self.ax_upstream.spines["right"].set_visible(False)

        # Set limits based on all plotted data, outside the gauge loop
        if all_times:
            x_min = min(all_times)
            x_max = max(all_times)

            # Add a small margin (5%) for better visualization
            margin = (x_max - x_min) * 0.05 if x_max > x_min else 0.1
            self.ax_downstream.set_xlim(x_min - margin, x_max + margin)
        else:
            # Set default limits if no data
            self.ax_downstream.set_xlim(0, 10)

    def on_close(self):
        self.stop()
        self.root.destroy()
