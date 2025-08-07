import glob
import os
import threading
import time
from datetime import datetime

import numpy as np
import u6

from .pressure_gauge import PressureGauge
from .thermocouple import Thermocouple


class DataRecorder:
    """
    Class to manage data recording from multiple pressure gauges.
    This class handles the setup, start, stop, and reset of data recording,
    as well as the management of results directories and gauge exports.

    Arguements:
        gauges: List of PressureGauge instances to record data from
        thermocouples: List of Thermocouple instances to record temperature data from
        results_dir: Directory where results will be stored, defaults to "results"
        test_mode: If True, runs in test mode without actual hardware interaction,
            defaults to False

    Attributes:
        gauges: List of PressureGauge instances to record data from
        thermocouples: List of Thermocouple instances to record temperature data from
        results_dir: Directory where results will be stored, defaults to "results"
        test_mode: If True, runs in test mode without actual hardware interaction,
            defaults to False
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

    stop_event: threading.Event
    thread: threading.Thread
    run_dir: str
    backup_dir: str
    elapsed_time: float
    temperature_data: list
    temperature_timestamps: list

    def __init__(
        self,
        gauges: list[PressureGauge],
        thermocouples: list[Thermocouple],
        results_dir: str = "results",
        test_mode=False,
    ):
        self.gauges = gauges
        self.thermocouples = thermocouples
        self.results_dir = results_dir
        self.test_mode = test_mode

        # Thread control
        self.stop_event = threading.Event()
        self.thread = None

        # Initialize directory paths but don't create them yet
        self.run_dir = None
        self.backup_dir = None

        # Single CSV file for all data
        self.main_csv_filename = None

        # Initialize time tracking
        self.elapsed_time = 0.0
        self.start_time = None

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

    def _initialise_main_csv(self):
        """Initialize the main CSV file with all gauge data"""
        self.main_csv_filename = os.path.join(self.run_dir, "pressure_gauge_data.csv")

        # Create header with RealTimestamp and voltage columns for each gauge
        header = "RealTimestamp"
        for gauge in self.gauges:
            header += f",{gauge.name}_Voltage (V)"
        header += "\n"

        # Write header to file
        with open(self.main_csv_filename, "w") as f:
            f.write(header)

    def start(self):
        """Start recording data"""
        # Create directories and setup files only when recording starts
        if self.run_dir is None:
            self.run_dir = self._create_results_directory()
            self.backup_dir = os.path.join(self.run_dir, "backup")
            os.makedirs(self.backup_dir, exist_ok=True)

            # Initialize the main CSV file
            self._initialise_main_csv()

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
        """Record data from all gauges"""

        if not self.test_mode:
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

        # Main data collection loop
        while not self.stop_event.is_set():
            # Get real timestamp for this measurement
            real_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

            # Collect voltages from all gauges
            voltages = []
            for gauge in self.gauges:
                # Get voltage based on mode
                if self.test_mode:
                    # Generate random voltage for test mode
                    rng = np.random.default_rng()
                    voltage = rng.uniform(0, 10)
                else:
                    voltage = gauge.get_ain_channel_voltage(labjack=labjack)

                voltages.append(voltage)

            # Write all data to single CSV
            self._write_to_csv(real_timestamp, voltages)

            # Sleep and increment time
            time.sleep(0.5)
            self.elapsed_time += 0.5

    def _write_to_csv(self, real_timestamp, voltages):
        """Write timestamp and all voltages to the main CSV file"""
        # Create row with timestamp and all voltages
        row = real_timestamp
        for voltage in voltages:
            row += f",{voltage}"
        row += "\n"

        # Write to file
        with open(self.main_csv_filename, "a") as f:
            f.write(row)

    def run(self):
        """Start the recorder and keep it running"""
        self.start()

        # Keep the main thread running
        try:
            while True:
                time.sleep(1)
                # Print status every 10 seconds in headless mode
                if int(time.time()) % 10 == 0:
                    print(
                        f"Current time: {datetime.now()} - Recording in progress... "
                        f"Elapsed time: {self.elapsed_time:.1f}s"
                    )
        except KeyboardInterrupt:
            self.stop()
            print("Recorder stopped")
