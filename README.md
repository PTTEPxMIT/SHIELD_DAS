# SHIELD_DAS
SHIELD data acquisition system


## example usage in test mode

Test mode meaning random pressure data is generated rather than reading and converting voltage data from a LabJack

'''python
from shield_das import (
    DataRecorder,
    WGM701_Gauge,
    DataPlotter,
    CVM211_Gauge,
    Baratron626D_Gauge,
)
import tkinter as tk

gauge_1 = WGM701_Gauge(
    test_mode=True,
    gauge_location="downstream",
    export_filename="WGM701_pressure_data.csv",
)
gauge_2 = CVM211_Gauge(
    test_mode=True,
    gauge_location="upstream",
    export_filename="CVM211_pressure_data.csv",
)
gauge_3 = Baratron626D_Gauge(
    name="Baratron626D_1000T",
    test_mode=True,
    gauge_location="upstream",
    export_filename="Baratron626D_1000T_pressure_data.csv",
)
gauge_4 = Baratron626D_Gauge(
    name="Baratron626D_0.1T",
    test_mode=True,
    gauge_location="downstream",
    export_filename="Baratron626D_0.1T_pressure_data.csv",
)

my_recorder = DataRecorder(
    gauges=[gauge_1, gauge_2, gauge_3, gauge_4],
    labjack=None,
)

root = tk.Tk()
root.title("SHILED DAS")
root.geometry("800x600")

start_button = tk.Button(root, text="Start", command=my_recorder.start)
start_button.pack(padx=20, pady=10)

stop_button = tk.Button(root, text="Stop", command=my_recorder.stop)
stop_button.pack(padx=20, pady=10)

plotter = DataPlotter(root, my_recorder)


def on_close():
    try:
        my_recorder.stop()  # Stop the data recorder safely
        # If needed, explicitly stop the animation:
        plotter.ani.event_source.stop()
    except Exception as e:
        print(f"Error during shutdown: {e}")
    root.destroy()  # Close the GUI window


root.protocol("WM_DELETE_WINDOW", on_close)
root.mainloop()
```