import glob
import os
import threading
import time
from datetime import datetime

import numpy as np
import pandas as pd
import u6

from .pressure_gauge import PressureGauge
from .thermocouple import Thermocouple


class DataRecorder:
    """
    Class to manage data recording from multiple pressure gauges. This class handles the
    setup, start, stop, and reset of data recording, as well as the management of
    results directories and gauge exports.

    Arguements:
        gauges: List of PressureGauge instances to record data from
        thermocouples: List of Thermocouple instances to record temperature data from
        results_dir: Directory where results will be stored, defaults to "results"
        test_mode: If True, runs in test mode without actual hardware interaction,
            defaults to False
        recording_interval: Time interval (seconds) between recordings, defaults to 0.5s
        backup_interval: How often to backup dara (seconds)

    Attributes:
        gauges: List of PressureGauge instances to record data from
        thermocouples: List of Thermocouple instances to record temperature data from
        results_dir: Directory where results will be stored, defaults to "results"
        test_mode: If True, runs in test mode without actual hardware interaction,
            defaults to False
        recording_interval: Time interval (in seconds) between recordings, defaults to
            0.5 seconds
        backup_interval: How often to rotate backup CSV files (seconds)
        stop_event: Event to control the recording thread
        thread: Thread for recording data
        run_dir: Directory for the current run's results
        backup_dir: Directory for backup files
        elapsed_time: Time elapsed since the start of recording
    """

    gauges: list[PressureGauge]
    thermocouples: list[Thermocouple]
    results_dir: str
    test_mode: bool
    recording_interval: float
    backup_interval: float

    stop_event: threading.Event
    thread: threading.Thread
    run_dir: str
    backup_dir: str
    elapsed_time: float

    def __init__(
        self,
        gauges: list[PressureGauge],
        thermocouples: list[Thermocouple],
        results_dir: str = "results",
        test_mode=False,
        recording_interval: float = 0.5,
        backup_interval: float = 5.0,
    ):
        self.gauges = gauges
        self.thermocouples = thermocouples
        self.results_dir = results_dir
        self.test_mode = test_mode
        self.recording_interval = recording_interval
        self.backup_interval = backup_interval

        # Thread control
        self.stop_event = threading.Event()
        self.thread = None

    def _create_results_directory(self):
        """Creates a new directory for results based on date and run number and if
        test_mode is enabled, it will not create directories."""

        # Create main results directory
        os.makedirs(self.results_dir, exist_ok=True)

        # Get current date and time
        now = datetime.now()
        current_date = now.strftime("%m.%d")
        current_time = now.strftime("%Hh%M")  # Format as HHhMM

        # Create date directory
        date_dir = os.path.join(self.results_dir, current_date)
        os.makedirs(date_dir, exist_ok=True)

        # Use test_run for test mode, otherwise increment run number
        if self.test_mode:
            # Include time in test run directory
            run_dir = os.path.join(date_dir, f"test_run_{current_time}")
            # Remove existing directory if it exists
            if os.path.exists(run_dir):
                import shutil

                shutil.rmtree(run_dir)
            os.makedirs(run_dir)
            print(f"Created test results directory: {run_dir}")
        else:
            # Find highest run number
            run_dirs = glob.glob(os.path.join(date_dir, "run_*"))
            run_numbers = [
                int(os.path.basename(d).split("_")[1])  # Extract just the number part
                for d in run_dirs
                if os.path.basename(d).split("_")[1].isdigit()
            ]

            # Set next run number
            next_run = 1 if not run_numbers else max(run_numbers) + 1

            # Create run directory with time included
            run_dir = os.path.join(date_dir, f"run_{next_run}_{current_time}")
            os.makedirs(run_dir)
            print(f"Created results directory: {run_dir}")

        return run_dir

    def start(self):
        """Start recording data"""
        # Create directories and setup files only when recording starts
        self.run_dir = self._create_results_directory()
        self.backup_dir = os.path.join(self.run_dir, "backup")
        os.makedirs(self.backup_dir, exist_ok=True)

        self.stop_event.clear()
        self.thread = threading.Thread(target=self.record_data)
        self.thread.daemon = True
        self.thread.start()

    def stop(self):
        """Stop recording data"""
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=2.0)

    def record_data(self):
        """Record data from all gauges passed to recorder"""

        # If test mode, do not connect to LabJack
        if self.test_mode:
            labjack = None
        else:
            try:
                labjack = u6.U6(firstFound=True)
                labjack.getCalibrationData()
                print("LabJack connected")
            except Exception as e:
                print(f"LabJack connection error: {e}")

        # Start with elapsed time of 0 and record start time
        self.elapsed_time = 0.0
        self.start_time = datetime.now()
        print(f"Recording started at {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")

        time_stamp_data = []

        # Main data collection loop
        n = 0
        m = 1
        while not self.stop_event.is_set():
            # Get real timestamp for this measurement
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            time_stamp_data.append(timestamp)

            # Collect voltages from all gauges
            for gauge in self.gauges:
                gauge.record_ain_channel_voltage(labjack=labjack)

            # Prepare data row for CSV
            pressure_gauge_data_row = {
                "RealTimestamp": timestamp,
                **{f"{g.name}_Voltage (V)": g.voltage_data[-1] for g in self.gauges},
            }

            # Append to main CSV file
            pd.DataFrame([pressure_gauge_data_row]).to_csv(
                f"{self.run_dir}/pressure_gauge_data.csv",
                mode="a",
                header=(n == 0),
                index=False,
            )

            n += 1

            # write backup data every index_for_backup iterations
            index_for_backup = int(self.backup_interval / self.recording_interval)
            if n % index_for_backup == 0:
                backup_pressure_gauge_data = {
                    "RealTimestamp": time_stamp_data[-index_for_backup:],
                    **{
                        f"{g.name}_Voltage (V)": g.voltage_data[-index_for_backup:]
                        for g in self.gauges
                    },
                }
                pd.DataFrame(backup_pressure_gauge_data).to_csv(
                    f"{self.backup_dir}/pressure_gauge_backup_data_{m}.csv",
                    index=False,
                )
                m += 1

            # Sleep and increment time
            time.sleep(self.recording_interval)
            self.elapsed_time += self.recording_interval

    def run(self):
        """Start the recorder and keep it running"""
        self.start()

        # Keep the main thread running
        try:
            while True:
                time.sleep(1)
                # Print status every 10 seconds
                if int(time.time()) % 10 == 0:
                    print(
                        f"Current time: {datetime.now()} - Recording in progress... "
                        f"Elapsed time: {self.elapsed_time:.1f}s"
                    )
        except KeyboardInterrupt:
            self.stop()
            print("Recorder stopped")
