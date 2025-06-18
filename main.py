from shield_das import (
    DataRecorder,
    WGM701_Gauge,
    DataPlotter,
    CVM211_Gauge,
    Baratron626D_Gauge
)
import time
import sys

# Define gauges
gauge_1 = WGM701_Gauge(
    gauge_location="downstream",
    export_filename="WGM701_pressure_data.csv",
)
gauge_2 = CVM211_Gauge(
    gauge_location="upstream",
    export_filename="CVM211_pressure_data.csv",
)
gauge_3 = Baratron626D_Gauge(
    name="Baratron626D_1KT",
    gauge_location="upstream",
    export_filename="Baratron626D_1000T_upstream_pressure_data.csv",
)
gauge_4 = Baratron626D_Gauge(
    name="Baratron626D_1T",
    gauge_location="downstream",
    export_filename="Baratron626D_1T_downstream_pressure_data.csv",
)

# Create recorder
my_recorder = DataRecorder(
    gauges=[gauge_1, gauge_2, gauge_3, gauge_4],
    test_mode=True,
)

if __name__ == "__main__":
    # Check if we're running in headless mode
    headless = "--headless" in sys.argv
    
    if headless:
        # Start recorder directly in headless mode
        my_recorder.start()
    else:
        # Create and start the plotter
        plotter = DataPlotter(my_recorder)
        plotter.start()
    
    # Keep the main thread running (same for both modes)
    try:
        while True:
            time.sleep(1)
            # Print status every 10 seconds in headless mode
            if headless and int(time.time()) % 10 == 0:
                import datetime
                print(f"Current time: {datetime.datetime.now()} - Recording in progress... Elapsed time: {my_recorder.elapsed_time:.1f}s")
    except KeyboardInterrupt:
        my_recorder.stop()
        print("Recorder stopped")