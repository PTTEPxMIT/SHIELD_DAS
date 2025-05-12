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
        # Overwrite the output file with header

        last_recorded = -1

        while not self.stop_event.is_set():
            now = time.time()
            fractional = now % 1

            if abs(fractional - 0.0) < 0.05 or abs(fractional - 0.5) < 0.05:
                rounded_time = round(now * 2) / 2
                if rounded_time != last_recorded:
                    dt = datetime.fromtimestamp(rounded_time)
                    timestamp = (
                        dt.strftime("%Y-%m-%d %H:%M:%S")
                        + f".{int(dt.microsecond / 100000)}"
                    )

                    for gauge in self.gauges:
                        gauge.get_data(labjack=self.labjack, timestamp=timestamp)
                        gauge.export_write()

                    last_recorded = rounded_time

            time.sleep(0.01)
