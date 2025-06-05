from threading import Thread, Event
import time
from datetime import datetime
from .pressure_gauge import PressureGauge
import os
import glob


class DataRecorder:

    def __init__(
        self,
        gauges: list[PressureGauge],
        labjack=None,
        results_dir="results"
    ):

        self.gauges = gauges
        self.labjack = labjack
        self.results_dir = results_dir

        self.stop_event = Event()
        self.thread = None

        # Create the results directory path
        self.run_dir = self._create_results_directory()

        # Create backup directory
        self.backup_dir = os.path.join(self.run_dir, "backup")
        os.makedirs(self.backup_dir, exist_ok=True)

        if self.labjack is not None:
            # Get the calibration constants from the U6, otherwise default nominal values
            # will be be used for binary to decimal (analog) conversions.
            self.labjack.getCalibrationData()

        # Initialize gauge exports with the new paths
        for gauge in self.gauges:
            # Update export filename to include the results directory
            original_filename = os.path.basename(gauge.export_filename)
            gauge.export_filename = os.path.join(self.run_dir, original_filename)
            gauge.initialise_export()

            # Initialize backup for this gauge
            gauge.initialise_backup(self.backup_dir)

        # Initialize elapsed time
        self.elapsed_time = 0.0

    
    def _create_results_directory(self):
        """
        Creates a new directory for the results based on current date and run number.
        Returns the path to the created directory.
        """
        # Create the main results directory if it doesn't exist
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
        
        # Get current date in MM.DD format
        current_date = datetime.now().strftime("%m.%d")
        date_dir = os.path.join(self.results_dir, current_date)
        
        # Create the date directory if it doesn't exist
        if not os.path.exists(date_dir):
            os.makedirs(date_dir)
        
        # Find the highest run number for today
        run_dirs = glob.glob(os.path.join(date_dir, "run_*"))
        run_numbers = []
        
        for dir_path in run_dirs:
            dir_name = os.path.basename(dir_path)
            try:
                # Extract the number after "run_"
                run_number = int(dir_name.split("_")[1])
                run_numbers.append(run_number)
            except (IndexError, ValueError):
                # Skip directories that don't match the pattern
                continue
        
        # Set next run number (start with 1 if none exist)
        next_run = 1
        if run_numbers:
            next_run = max(run_numbers) + 1
        
        # Create the new run directory
        run_dir = os.path.join(date_dir, f"run_{next_run}")
        os.makedirs(run_dir)
        
        print(f"Created results directory: {run_dir}")
        return run_dir

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
    
    def reset(self):
        """
        Reset the recorder to its initial state.
        This clears all data and prepares for a fresh start.
        """
        # Stop recording if it's running
        self.stop()
        
        # Clear data from all gauges
        for gauge in self.gauges:
            gauge.timestamp_data = []
            gauge.pressure_data = []
            gauge.voltage_data = []
            gauge.backup_counter = 0
            gauge.measurements_since_backup = 0
        
        # Create a new results directory
        self.run_dir = self._create_results_directory()
        
        # Create new backup directory
        self.backup_dir = os.path.join(self.run_dir, "backup")
        os.makedirs(self.backup_dir, exist_ok=True)
        
        # Reinitialize gauge exports with the new paths
        for gauge in self.gauges:
            # Update export filename to include the new results directory
            original_filename = os.path.basename(gauge.export_filename)
            gauge.export_filename = os.path.join(self.run_dir, original_filename)
            gauge.initialise_export()
            
            # Initialize backup for this gauge
            gauge.initialise_backup(self.backup_dir)
        
        # Reset elapsed time
        self.elapsed_time = 0.0
        
        print(f"DataRecorder reset. New run directory: {self.run_dir}")

    def record_data(self):
        """
        Record data from all gauges at a fixed interval of 0.5 seconds.
        """

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
