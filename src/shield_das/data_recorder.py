import glob
import os
import threading
import time
from datetime import datetime

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
        results_dir: Directory where results will be stored, defaults to "results"
        test_mode: If True, runs in test mode without actual hardware interaction,
            defaults to False
        record_temperature: If True, records temperature data from a thermocouple,
            defaults to True

    Attributes:
        gauges: List of PressureGauge instances to record data from
        results_dir: Directory where results will be stored
        test_mode: If True, runs in test mode without actual hardware interaction
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
    record_temperature: bool = True
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
        record_temperature=True,
    ):
        self.gauges = gauges
        self.thermocouples = thermocouples
        self.results_dir = results_dir
        self.test_mode = test_mode
        self.record_temperature = record_temperature

        # Thread control
        self.stop_event = threading.Event()
        self.thread = None

        # Create results directories and setup files
        self.run_dir = self._create_results_directory()
        self.backup_dir = os.path.join(self.run_dir, "backup")
        os.makedirs(self.backup_dir, exist_ok=True)

        # Initialise exports
        self._initialise_gauge_exports()
        self._initialise_thermocouple_exports()

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

    def _initialise_gauge_exports(self):
        """Initialise export files for all gauges"""
        for gauge in self.gauges:
            # Update path and initialize export
            original_filename = os.path.basename(gauge.export_filename)
            gauge.export_filename = os.path.join(self.run_dir, original_filename)
            gauge.initialise_export()

    def _initialise_thermocouple_exports(self):
        """Initialise export files for all gauges"""
        for thermocouple in self.thermocouples:
            # Update path and initialize export
            original_filename = os.path.basename(thermocouple.export_filename)
            thermocouple.export_filename = os.path.join(self.run_dir, original_filename)
            thermocouple.initialise_export()

    def start(self):
        """Start recording data"""
        self.stop_event.clear()
        self.thread = threading.Thread(target=self.record_data)
        self.thread.daemon = True
        self.thread.start()

    def stop(self):
        """Stop recording data"""
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=2.0)

    def reset(self):
        """Reset for a new run"""
        # Stop current recording
        self.stop()

        # Clear data
        for gauge in self.gauges:
            gauge.timestamp_data = []
            gauge.pressure_data = []
            gauge.voltage_data = []
            if hasattr(gauge, "backup_counter"):
                gauge.backup_counter = 0

        # Clear temperature data
        self.temperature_data = []
        self.temperature_timestamps = []

        # Create new directories
        self.run_dir = self._create_results_directory()
        self.backup_dir = os.path.join(self.run_dir, "backup")
        os.makedirs(self.backup_dir, exist_ok=True)

        # Initialise exports again
        self._initialise_gauge_exports()
        self._initialise_thermocouple_exports()

        # Reset time
        self.elapsed_time = 0.0

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
            timestamp = f"{self.elapsed_time:.1f}"

            for gauge in self.gauges:
                try:
                    # Get data based on mode
                    if self.test_mode:
                        gauge.get_data(labjack=None, timestamp=timestamp)
                    else:
                        gauge.get_data(labjack=labjack, timestamp=timestamp)
                except Exception as e:
                    print(f"Error reading from {gauge.name}: {e}")

                # Always write to file
                gauge.export_write()

            for thermocouple in self.thermocouples:
                try:
                    # Get temperature data based on mode
                    if self.test_mode:
                        thermocouple.get_temperature(labjack=None, timestamp=timestamp)
                    else:
                        thermocouple.get_temperature(
                            labjack=labjack, timestamp=timestamp
                        )
                except Exception as e:
                    print(f"Error reading from thermocouple: {e}")

                # Always write to file
                thermocouple.export_write()

            # Sleep and increment time
            time.sleep(0.5)
            self.elapsed_time += 0.5
