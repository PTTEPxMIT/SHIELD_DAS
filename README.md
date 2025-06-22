# SHIELD permeation rig Data Aquisistion System

This is a tool to be used with the SHIELD hydrogen permeation rig, providing a way to both record data from the rig and have a live UI displaying plots of the pressure values in the gauges connected to the rig and the temperature of the connected thermocouple.

<img width="1435" alt="Image" src="https://github.com/user-attachments/assets/c88b2da4-6051-4302-baa7-43a56a5254d2" />

## Example script

This is an example of a script that can be used to activate the DAS.

```python
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
    export_filename="Baratron626D_1KT_upstream_pressure_data.csv",
    full_scale_Torr=1000,
)
gauge_4 = Baratron626D_Gauge(
    name="Baratron626D_1T",
    gauge_location="downstream",
    export_filename="Baratron626D_1T_downstream_pressure_data.csv",
    full_scale_Torr=1,
)

# Create recorder
my_recorder = DataRecorder(
    gauges=[gauge_1, gauge_2, gauge_3, gauge_4],
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
```

# Test mode
If the labjack is not connected, the program can be run in `test_mode`, where dummy data is generated which can then be recorded and veiwed for testing the program works.

It can be activate with the argument `test_mode` in the `DataRecorder` class:

```python
# Create recorder
my_recorder = DataRecorder(
    gauges=[gauge_1, gauge_2, gauge_3, gauge_4],
    test_mode=True
)
```