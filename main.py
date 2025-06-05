# main.py
from shield_das import (
    DataRecorder,
    WGM701_Gauge,
    DataPlotter,
    CVM211_Gauge,
    Baratron626D_Gauge
)
import time
import u6

# Create gauges
gauge_1 = WGM701_Gauge(
    gauge_location="downstream",
    export_filename="WGM701_pressure_data.csv",
)
gauge_2 = CVM211_Gauge(
    gauge_location="upstream",
    export_filename="CVM211_pressure_data.csv",
)
gauge_3 = Baratron626D_Gauge(
    gauge_location="upstream",
    export_filename="Baratron626D_1000T_upstream_pressure_data.csv",
)
gauge_4 = Baratron626D_Gauge(
    gauge_location="downstream",
    export_filename="Baratron626D_1T_downstream_pressure_data.csv",
)

labjack = u6.U6(firstFound=True)

# Create recorder
my_recorder = DataRecorder(
    gauges=[gauge_1, gauge_2, gauge_3, gauge_4],
    labjack=labjack,
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
        my_recorder.stop()