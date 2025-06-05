from threading import Thread, Event
import time
from datetime import datetime
from .pressure_gauge import PressureGauge


class DataRecorder:

    def __init__(
        self,
        gauges: list[PressureGauge],
        labjack=None,
    ):

        self.gauges = gauges
        self.labjack = labjack

        self.stop_event = Event()
        self.thread = None

        if self.labjack is not None:
            # Get the calibration constants from the U6, otherwise default nominal values
            # will be be used for binary to decimal (analog) conversions.
            self.labjack.getCalibrationData()

        for gauge in self.gauges:
            gauge.initialise_export()

    def start(self):
        self.stop_event.clear()
        self.thread = Thread(target=self.record_data)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join()

        # Close the U6.
        if self.labjack is not None:
            self.labjack.close()

    def record_data(self):
        """
        Record data from all gauges at a fixed interval of 0.5 seconds.
        """
        # Start with elapsed time of 0
        self.elapsed_time = 0.0

        while not self.stop_event.is_set():
            # Format the elapsed time with 1 decimal place
            timestamp = f"{self.elapsed_time:.1f}"

            # Get data from each gauge
            for gauge in self.gauges:
                gauge.get_data(labjack=self.labjack, timestamp=timestamp)
                gauge.export_write()

            # Sleep for 0.5 seconds before next reading
            time.sleep(0.5)

            # Increment the elapsed time
            self.elapsed_time += 0.5
