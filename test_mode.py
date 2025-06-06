from shield_das import (
    DataRecorder,
    WGM701_Gauge,
    DataPlotter,
    CVM211_Gauge,
)
import time


gauge_1 = WGM701_Gauge(
    gauge_location="downstream",
    export_filename="WGM701_pressure_data.csv",

)
gauge_2 = CVM211_Gauge(
    gauge_location="upstream",
    export_filename="CVM211_pressure_data.csv",
)

my_recorder = DataRecorder(
    gauges=[gauge_1, gauge_2],
    test_mode=True
)


if __name__ == "__main__":
    # Create and start the plotter
    plotter = DataPlotter(my_recorder)
    plotter.start()

    # Keep the main thread running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        # If user presses Ctrl+C, stop the recorder
        my_recorder.stop()
