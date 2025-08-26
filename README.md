# SHIELD permeation rig Data Aquisistion System

This is a tool to be used with the SHIELD hydrogen permeation rig, providing a way to both record data from the rig and have a live UI displaying plots of the pressure values in the gauges connected to the rig and the temperature of the connected thermocouple.

<img width="1901" height="900" alt="Image" src="https://github.com/user-attachments/assets/4cbdcaeb-0226-4381-a8f3-61f411e6f0aa" />

## Installation

The shield DAS package can be downloaded with `pip`

```python
pip install SHIELD-DAS
```

However, in order to interact with the Labjack, additional drivers are required from the [manufacturers site](https://support.labjack.com/docs/windows-setup-basic-driver-only).


## Example data recording script

This is an example of a script that can be used to activate the DAS.

```python
from shield_das import (
    DataRecorder,
    WGM701_Gauge,
    CVM211_Gauge,
    Baratron626D_Gauge
)

# Define gauges
gauge_1 = WGM701_Gauge(
    gauge_location="downstream",
    ain_channel=10,
)
gauge_2 = CVM211_Gauge(
    gauge_location="upstream",
    ain_channel=8,
)
gauge_3 = Baratron626D_Gauge(
    name="Baratron626D_1KT",
    gauge_location="upstream",
    full_scale_torr=1000,
    ain_channel=6,
)
gauge_4 = Baratron626D_Gauge(
    name="Baratron626D_1T",
    gauge_location="downstream",
    full_scale_torr=1,
    ain_channel=4,
)

# Create recorder
my_recorder = DataRecorder(
    gauges=[gauge_1, gauge_2, gauge_3, gauge_4],
    thermocouples=[thermocouple_1],
    run_type="test_mode",
    recording_interval=0.5,
    backup_interval=5,
    furnace_setpoint=500,
)

# Start recording
my_recorder.run()

```

## Example data visulisation script

```python

from shield_das import DataPlotter

data_500C_run1 = "results/08.12/run_2_11h45/"
data_500C_run2 = "results/08.18/run_2_09h47/"
data_500C_run3 = "results/08.19/run_2_09h21/"
data_500C_run4 = "results/08.25/run_1_09h07/"

my_plotter = DataPlotter(
    dataset_paths=[data_500C_run1, data_500C_run2, data_500C_run3, data_500C_run4],
    dataset_names=["500C_run1", "500C_run2", "500C_run3", "500C_run4"],
)
my_plotter.start()

```