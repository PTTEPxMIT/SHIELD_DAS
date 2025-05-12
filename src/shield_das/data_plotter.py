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

        # Default window size in seconds
        self.time_window = 10

        # Layout frames
        self.control_frame = tk.Frame(root)
        self.control_frame.pack(side=tk.TOP, fill=tk.X)

        # Add window size control
        self.window_frame = tk.Frame(self.control_frame)
        self.window_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

        # Label for the entry
        self.window_label = tk.Label(self.window_frame, text="Time Window (seconds):")
        self.window_label.pack(side=tk.LEFT, padx=(0, 5))

        # Entry for time window
        self.window_entry = tk.Entry(self.window_frame, width=10)
        self.window_entry.insert(0, str(self.time_window))
        self.window_entry.pack(side=tk.LEFT)

        # Button to apply the changes
        self.apply_button = tk.Button(
            self.window_frame, text="Apply", command=self.update_time_window
        )
        self.apply_button.pack(side=tk.LEFT, padx=5)

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

    def update_time_window(self):
        """Update the time window value from the entry field"""
        try:
            new_window = float(self.window_entry.get())
            if new_window > 0:
                self.time_window = new_window
                print(f"Time window updated to {self.time_window} seconds")
            else:
                print("Time window must be a positive number")
                # Reset to previous value
                self.window_entry.delete(0, tk.END)
                self.window_entry.insert(0, str(self.time_window))
        except ValueError:
            print("Invalid input. Please enter a number.")
            # Reset to previous value
            self.window_entry.delete(0, tk.END)
            self.window_entry.insert(0, str(self.time_window))

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

                # Calculate number of points to show based on time_window
                target_points = int(
                    self.time_window / 0.5
                )  # Assuming 0.5s per data point

                # Only show the last N data points if we have more
                if len(time_seconds) > target_points:
                    time_seconds = time_seconds[-target_points:]
                    pressure_copy = pressure_copy[-target_points:]

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
            self.ax_upstream.legend()
            self.ax_downstream.legend()

            # Remove top and right spines for cleaner look
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

            # Apply the same limits to both plots
            self.ax_upstream.set_xlim(x_min - margin, x_max + margin)
            self.ax_downstream.set_xlim(x_min - margin, x_max + margin)
        else:
            # Set default limits if no data
            self.ax_upstream.set_xlim(0, self.time_window)
            self.ax_downstream.set_xlim(0, self.time_window)

        self.canvas.draw()

    def on_close(self):
        self.stop()
        self.root.destroy()
