import glob
import json
import os
import threading
import time
from datetime import datetime

import keyboard
import pandas as pd
import u6

from .pressure_gauge import PressureGauge
from .thermocouple import Thermocouple


class DataRecorder:
    """
    Class to manage data recording from multiple pressure gauges. This class handles the
    setup, start, stop, and reset of data recording, as well as the management of
    results directories and gauge exports.

    Arguments:
        gauges: List of PressureGauge instances to record data from
        thermocouples: List of Thermocouple instances to record temperature data from
        furnace_setpoint: Setpoint temperature for the furnace
        results_dir: Directory where results will be stored, defaults to "results"
        run_type: Permeation exp, leak test or test mode, defaults to "permeation_exp",
            if in test mode, runs without actual hardware interaction
        recording_interval: Time interval (seconds) between recordings, defaults to 0.5s
        backup_interval: How often to backup data (seconds)

    Attributes:
        gauges: List of PressureGauge instances to record data from
        thermocouples: List of Thermocouple instances to record temperature data from
        furnace_setpoint: Setpoint temperature for the furnace
        results_dir: Directory where results will be stored, defaults to "results"
        run_type: Permeation exp, leak test or test mode, defaults to "permeation_exp",
            if in test mode, runs without actual hardware interaction
        recording_interval: Time interval (in seconds) between recordings, defaults to
            0.5 seconds
        backup_interval: How often to rotate backup CSV files (seconds)
        stop_event: Event to control the recording thread
        thread: Thread for recording data
        run_dir: Directory for the current run's results
        backup_dir: Directory for backup files
        elapsed_time: Time elapsed since the start of recording
        v5_close_time: Timestamp when V5 valve was closed (spacebar press 1)
        v6_close_time: Timestamp when V6 valve was closed (spacebar press 2)
        v7_close_time: Timestamp when V7 valve was closed (spacebar press 3)
        v3_open_time: Timestamp when V3 valve was opened (spacebar press 4)
        valve_event_sequence: Ordered list of valve events to track
        current_valve_index: Current position in the valve event sequence
    """

    gauges: list[PressureGauge]
    thermocouples: list[Thermocouple]
    furnace_setpoint: float | None
    results_dir: str
    run_type: str
    recording_interval: float
    backup_interval: float

    stop_event: threading.Event
    thread: threading.Thread
    run_dir: str
    backup_dir: str
    elapsed_time: float
    v5_close_time: str | None
    v6_close_time: str | None
    v7_close_time: str | None
    v3_open_time: str | None
    start_time: datetime
    valve_event_sequence: list[str]
    current_valve_index: int

    def __init__(
        self,
        gauges: list[PressureGauge],
        thermocouples: list[Thermocouple],
        furnace_setpoint: float | None = None,
        results_dir: str = "results",
        run_type="permeation_exp",
        recording_interval: float = 0.5,
        backup_interval: float = 5.0,
    ):
        self.gauges = gauges
        self.thermocouples = thermocouples
        self.furnace_setpoint = furnace_setpoint
        self.results_dir = results_dir
        self.run_type = run_type
        self.recording_interval = recording_interval
        self.backup_interval = backup_interval

        # Thread control
        self.stop_event = threading.Event()
        self.thread = None

        self.elapsed_time = 0.0
        self.v5_close_time = None
        self.v6_close_time = None
        self.v7_close_time = None
        self.v3_open_time = None
        self.start_time = None

        # Valve event sequence tracking
        self.valve_event_sequence = [
            "v5_close_time",
            "v6_close_time",
            "v7_close_time",
            "v3_open_time",
        ]
        self.current_valve_index = 0

    @property
    def gauges(self) -> list[PressureGauge]:
        return self._gauges

    @gauges.setter
    def gauges(self, value: list[PressureGauge]):
        if not isinstance(value, list) or not all(
            isinstance(g, PressureGauge) for g in value
        ):
            raise ValueError("gauges must be a list of PressureGauge instances")
        self._gauges = value

    @property
    def thermocouples(self) -> list[Thermocouple]:
        return self._thermocouples

    @thermocouples.setter
    def thermocouples(self, value: list[Thermocouple]):
        if not isinstance(value, list) or not all(
            isinstance(t, Thermocouple) for t in value
        ):
            raise ValueError("thermocouples must be a list of Thermocouple instances")
        self._thermocouples = value

    @property
    def results_dir(self) -> str:
        return self._results_dir

    @results_dir.setter
    def results_dir(self, value: str):
        if not isinstance(value, str):
            raise ValueError("results_dir must be a string")

        self._results_dir = value

    @property
    def run_type(self) -> str:
        return self._run_type

    @run_type.setter
    def run_type(self, value: str):
        if value not in ["permeation_exp", "leak_test", "test_mode"]:
            raise ValueError(
                "run_type must be one of 'permeation_exp', 'leak_test', or 'test_mode'"
            )
        self._run_type = value

    @property
    def test_mode(self) -> bool:
        """Check if the recorder is in test mode."""
        return self.run_type == "test_mode"

    def _create_results_directory(self):
        """Creates a new directory for results based on date and run number."""
        # Create main results directory
        os.makedirs(self.results_dir, exist_ok=True)

        # Get current date and time
        now = datetime.now()
        current_date = now.strftime("%m.%d")
        current_time = now.strftime("%Hh%M")

        # Create date directory
        date_dir = os.path.join(self.results_dir, current_date)
        os.makedirs(date_dir, exist_ok=True)

        # Determine directory type and message based on test mode
        if self.test_mode:
            prefix = "test_run"
            message = "Created test results directory"
        else:
            prefix = "run"
            message = "Created results directory"

        return self._create_numbered_directory(date_dir, prefix, current_time, message)

    def _create_numbered_directory(
        self, parent_dir: str, prefix: str, timestamp: str, success_message: str
    ) -> str:
        """Create a numbered directory with the given prefix and timestamp.

        Args:
            parent_dir: Parent directory where the new directory will be created
            prefix: Directory prefix ('run' or 'test_run')
            timestamp: Timestamp to append to directory name
            success_message: Message to print on successful creation

        Returns:
            Path to the created directory
        """
        next_number = self._get_next_directory_number(parent_dir, prefix)
        dir_path = os.path.join(parent_dir, f"{prefix}_{next_number}_{timestamp}")

        os.makedirs(dir_path)
        print(f"{success_message}: {dir_path}")
        return dir_path

    def _get_next_directory_number(self, parent_dir: str, prefix: str) -> int:
        """Find the next available directory number for the given prefix.

        Args:
            parent_dir: The parent directory to search in
            prefix: Directory prefix ('run' or 'test_run')

        Returns:
            Next available number for the directory
        """
        pattern = os.path.join(parent_dir, f"{prefix}_*")
        dirs = glob.glob(pattern)
        numbers = []

        for dir_path in dirs:
            basename = os.path.basename(dir_path)
            parts = basename.split("_")

            # For 'run_X_time' format, number is at index 1
            # For 'test_run_X_time' format, number is at index 2
            number_index = 2 if prefix == "test_run" else 1

            if len(parts) > number_index and parts[number_index].isdigit():
                numbers.append(int(parts[number_index]))

        return 1 if not numbers else max(numbers) + 1

    def _create_metadata_file(self):
        """Create a JSON metadata file with run information."""
        metadata = {
            "version": "1.0",
            "run_info": {
                "date": datetime.now().strftime("%Y-%m-%d"),
                "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "run_type": self.run_type,
                "furnace_setpoint": self.furnace_setpoint,
                "recording_interval_seconds": self.recording_interval,
                "backup_interval_seconds": self.backup_interval,
                "data_filename": "pressure_gauge_data.csv",
            },
            "gauges": [
                {
                    "name": gauge.name,
                    "type": type(gauge).__name__,
                    "ain_channel": gauge.ain_channel,
                    "gauge_location": gauge.gauge_location,
                    **(
                        {"full_scale_torr": gauge.full_scale_Torr}
                        if hasattr(gauge, "full_scale_Torr")
                        else {}
                    ),
                }
                for gauge in self.gauges
            ],
            "thermocouples": [
                {
                    "name": (
                        thermocouple.name
                        if hasattr(thermocouple, "name")
                        else f"Thermocouple_{i}"
                    )
                }
                for i, thermocouple in enumerate(self.thermocouples)
            ],
        }

        metadata_path = os.path.join(self.run_dir, "run_metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"Created metadata file: {metadata_path}")
        return metadata_path

    def _monitor_keyboard(self):
        """Monitor for spacebar press to record valve events in sequence."""

        # Detect if we're in a CI environment (not local test mode)
        is_ci = self._is_ci_environment()
        if is_ci:
            print("CI environment detected. Keyboard monitoring disabled.")
            return

        if not (0 <= self.current_valve_index < len(self.valve_event_sequence)):
            print(
                "Warning: current_valve_index is out of bounds. "
                "Keyboard monitoring aborted."
            )
            return

        current_event = self.valve_event_sequence[self.current_valve_index]
        print(f"Press SPACEBAR to record {current_event}...")

        def on_spacebar():
            if self.current_valve_index < len(self.valve_event_sequence):
                current_event = self.valve_event_sequence[self.current_valve_index]
                # Include milliseconds for precise timing
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

                # Set the appropriate attribute
                setattr(self, current_event, timestamp)

                print(f"{current_event} recorded: {timestamp}")
                self._update_metadata_with_valve_time(current_event, timestamp)

                # Move to next event
                self.current_valve_index += 1

                # Show next event or completion message
                if self.current_valve_index < len(self.valve_event_sequence):
                    next_event = self.valve_event_sequence[self.current_valve_index]
                    print(f"Next: Press SPACEBAR to record {next_event}...")
                else:
                    print("All valve events recorded!")

        # Set up keyboard listener for spacebar
        keyboard.on_press_key("space", lambda _: on_spacebar())

    def _is_ci_environment(self) -> bool:
        """Detect if we're running in a CI environment."""
        # Common CI environment variables
        ci_indicators = [
            "CI",  # GitHub Actions, GitLab CI, etc.
            "GITHUB_ACTIONS",
            "GITLAB_CI",
            "TRAVIS",
            "CIRCLECI",
            "JENKINS_URL",
            "BUILDKITE",
            "TF_BUILD",  # Azure DevOps
        ]

        return any(os.getenv(var) for var in ci_indicators)

    def _update_metadata_with_valve_time(self, event_name: str, timestamp: str):
        """Update the metadata file with the valve event time.

        Args:
            event_name: Name of the valve event (e.g., 'v5_close_time')
            timestamp: Timestamp when the event occurred
        """
        metadata_path = os.path.join(self.run_dir, "run_metadata.json")

        try:
            with open(metadata_path) as f:
                metadata = json.load(f)

            metadata["run_info"][event_name] = timestamp

            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)

            print(f"Updated metadata with {event_name}: {timestamp}")
        except Exception as e:
            print(f"Error updating metadata with {event_name}: {e}")

    def start(self):
        """Start recording data"""
        # Initialize LabJack in main thread before starting recording thread
        labjack = self._initialize_labjack()

        # check pressure gauges have unique AIN channels
        ain_channels = [g.ain_channel for g in self.gauges]
        if len(ain_channels) != len(set(ain_channels)):
            raise ValueError("Error: Duplicate AIN channels detected among gauges")

        # Record start time for valve event time tracking
        self.start_time = datetime.now()

        # Reset all valve events for new run
        self.v5_close_time = None
        self.v6_close_time = None
        self.v7_close_time = None
        self.v3_open_time = None
        self.current_valve_index = 0

        # Create directories and setup files only when recording starts
        self.run_dir = self._create_results_directory()
        self.backup_dir = os.path.join(self.run_dir, "backup")
        os.makedirs(self.backup_dir, exist_ok=True)

        # Create metadata file with run information
        self._create_metadata_file()

        # Start keyboard monitoring for valve events
        self._monitor_keyboard()

        self.stop_event.clear()
        # Pass the initialized LabJack to the recording thread
        self.thread = threading.Thread(target=self.record_data, args=(labjack,))
        self.thread.daemon = True
        self.thread.start()

    def stop(self):
        """Stop recording data"""
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=1.0)

        # Clean up keyboard listeners
        if not self._is_ci_environment():
            keyboard.unhook_all()

    def record_data(self, labjack=None):
        """Record data from all gauges passed to recorder

        Args:
            labjack: Pre-initialized LabJack device instance, or None for test mode
        """
        self._initialize_recording_session()

        # Calculate backup parameters once
        backup_frequency = max(1, int(self.backup_interval / self.recording_interval))

        # Data buffers
        data_buffer = []
        measurement_count = 0
        backup_count = 1

        while not self.stop_event.is_set():
            timestamp = self._get_current_timestamp()
            data_row = self._collect_measurement_data(labjack, timestamp)
            data_buffer.append(data_row)
            measurement_count += 1

            # Write individual measurement to main CSV
            self._write_single_measurement(data_row, is_first=(measurement_count == 1))

            # Handle backup if needed
            if measurement_count % backup_frequency == 0:
                self._write_backup_data(data_buffer[-backup_frequency:], backup_count)
                backup_count += 1

            # Control timing
            time.sleep(self.recording_interval)
            self.elapsed_time += self.recording_interval

    def _initialize_labjack(self):
        """Initialize LabJack connection for data recording."""
        if self.test_mode:
            return None

        labjack = u6.U6(firstFound=True)
        labjack.getCalibrationData()
        print("LabJack connected")
        return labjack

    def _initialize_recording_session(self):
        """Initialize recording session parameters."""
        self.elapsed_time = 0.0
        self.start_time = datetime.now()
        print(f"Recording started at {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    def _get_current_timestamp(self) -> str:
        """Get formatted timestamp for current measurement."""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

    def _collect_measurement_data(self, labjack, timestamp: str) -> dict:
        """Collect data from all gauges for a single measurement.

        Args:
            labjack: LabJack device instance
            timestamp: Formatted timestamp string

        Returns:
            Dictionary containing measurement data
        """
        # Collect voltages from all gauges
        for gauge in self.gauges:
            gauge.record_ain_channel_voltage(labjack=labjack)

        # Prepare data row for CSV
        return {
            "RealTimestamp": timestamp,
            **{f"{g.name}_Voltage (V)": g.voltage_data[-1] for g in self.gauges},
        }

    def _write_single_measurement(self, data_row: dict, is_first: bool):
        """Write a single measurement to the main CSV file.

        Args:
            data_row: Dictionary containing measurement data
            is_first: Whether this is the first measurement (determines header)
        """
        csv_path = os.path.join(self.run_dir, "pressure_gauge_data.csv")
        pd.DataFrame([data_row]).to_csv(
            csv_path,
            mode="a",
            header=is_first,
            index=False,
        )

    def _write_backup_data(self, recent_data: list[dict], backup_number: int):
        """Write backup data to a separate CSV file.

        Args:
            recent_data: List of recent measurement dictionaries
            backup_number: Sequential backup file number
        """
        if not recent_data:
            return

        backup_path = os.path.join(
            self.backup_dir, f"pressure_gauge_backup_data_{backup_number}.csv"
        )
        pd.DataFrame(recent_data).to_csv(backup_path, index=False)

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
