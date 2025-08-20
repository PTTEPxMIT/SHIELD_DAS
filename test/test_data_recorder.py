import json
import os
import shutil
import tempfile
import threading
import time
from unittest.mock import Mock, patch

import pytest

from shield_das import DataRecorder, PressureGauge


class TestDataRecorder:
    """Test suite for DataRecorder class"""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        # Create temporary directory for test results
        self.temp_dir = tempfile.mkdtemp()

        # Create mock gauges
        self.mock_gauge1 = Mock(spec=PressureGauge)
        self.mock_gauge1.name = "TestGauge1"
        self.mock_gauge1.voltage_data = [5.0]  # Mock voltage data list
        self.mock_gauge1.record_ain_channel_voltage.return_value = None
        self.mock_gauge1.ain_channel = 10
        self.mock_gauge1.gauge_location = "downstream"

        self.mock_gauge2 = Mock(spec=PressureGauge)
        self.mock_gauge2.name = "TestGauge2"
        self.mock_gauge2.voltage_data = [3.5]  # Mock voltage data list
        self.mock_gauge2.record_ain_channel_voltage.return_value = None
        self.mock_gauge2.ain_channel = 6
        self.mock_gauge2.gauge_location = "upstream"

        # Create mock thermocouples (empty list for now)
        self.mock_thermocouples = []

        # Create DataRecorder instance
        self.recorder = DataRecorder(
            gauges=[self.mock_gauge1, self.mock_gauge2],
            thermocouples=self.mock_thermocouples,
            results_dir=self.temp_dir,
            run_type="test_mode",
            recording_interval=0.1,  # Fast interval for testing
        )

    def teardown_method(self):
        """Clean up after each test method."""
        # Stop recorder if running
        if self.recorder.thread and self.recorder.thread.is_alive():
            self.recorder.stop()

        # Remove temporary directory
        shutil.rmtree(self.temp_dir)

    def test_init_basic_attributes(self):
        """Test that DataRecorder initializes with correct basic attributes."""
        assert self.recorder.gauges == [self.mock_gauge1, self.mock_gauge2]
        assert self.recorder.thermocouples == []
        assert self.recorder.results_dir == self.temp_dir
        assert self.recorder.test_mode is True
        assert isinstance(self.recorder.stop_event, threading.Event)
        assert self.recorder.thread is None
        assert self.recorder.elapsed_time == 0.0

    def test_init_default_values(self):
        """Test DataRecorder initialization with default values."""
        default_recorder = DataRecorder(gauges=[self.mock_gauge1], thermocouples=[])
        assert default_recorder.results_dir == "results"
        assert default_recorder.test_mode is False

    def test_create_results_directory_test_mode(self):
        """Test directory creation in test mode."""
        # Test directory creation (actual dates will be used)
        run_dir = self.recorder._create_results_directory()

        # Check that directory was created and follows expected pattern
        assert os.path.exists(run_dir)
        assert "test_run_" in os.path.basename(run_dir)

        # Check that it's a subdirectory of temp_dir
        assert run_dir.startswith(self.temp_dir)

    def test_create_results_directory_normal_mode(self):
        """Test directory creation in normal mode."""
        # Setup recorder in normal mode
        normal_recorder = DataRecorder(
            gauges=[self.mock_gauge1],
            thermocouples=[],
            results_dir=self.temp_dir,
            run_type="permeation_exp",
        )

        # Test directory creation (actual dates will be used)
        run_dir = normal_recorder._create_results_directory()

        # Check that directory was created and follows expected pattern
        assert os.path.exists(run_dir)
        assert "run_" in os.path.basename(run_dir)

        # Check that it's a subdirectory of temp_dir
        assert run_dir.startswith(self.temp_dir)

        # Create a second run directory to test incrementing
        run_dir2 = normal_recorder._create_results_directory()
        assert run_dir2 != run_dir  # Should be different directories

    def test_csv_file_creation_on_start(self):
        """Test that CSV file is created when recording starts."""
        # Start recorder
        self.recorder.start()
        time.sleep(0.1)  # Give it time to initialize

        # Check that CSV file was created
        expected_csv_path = os.path.join(
            self.recorder.run_dir, "pressure_gauge_data.csv"
        )
        assert os.path.exists(expected_csv_path)

        # Stop recorder
        self.recorder.stop()

    def test_data_recording_functionality(self):
        """Test that data recording works with the new structure."""
        # Start recorder
        self.recorder.start()
        time.sleep(0.2)  # Let it record some data

        # Check that gauges were called to record data
        assert self.mock_gauge1.record_ain_channel_voltage.called
        assert self.mock_gauge2.record_ain_channel_voltage.called

        # Check CSV file has data
        csv_path = os.path.join(self.recorder.run_dir, "pressure_gauge_data.csv")
        assert os.path.exists(csv_path)

        with open(csv_path) as f:
            content = f.read()
            # Should have header and at least one data row
            lines = content.strip().split("\n")
            assert len(lines) >= 2
            # Check header contains gauge names
            assert "TestGauge1_Voltage (V)" in lines[0]
            assert "TestGauge2_Voltage (V)" in lines[0]

        # Stop recorder
        self.recorder.stop()

    def test_start_creates_directories_and_files(self):
        """Test that start() creates necessary directories and files."""
        # Start recorder
        self.recorder.start()

        # Give it a moment to initialize
        time.sleep(0.1)

        # Check directories were created
        assert self.recorder.run_dir is not None
        assert os.path.exists(self.recorder.run_dir)
        assert self.recorder.backup_dir is not None
        assert os.path.exists(self.recorder.backup_dir)

        # Check CSV file gets created when recording starts
        csv_path = os.path.join(self.recorder.run_dir, "pressure_gauge_data.csv")
        assert os.path.exists(csv_path)

        # Check thread is running
        assert self.recorder.thread is not None
        assert self.recorder.thread.is_alive()

        # Stop recorder
        self.recorder.stop()

    def test_stop_recorder(self):
        """Test stopping the recorder."""
        # Start recorder
        self.recorder.start()
        time.sleep(0.1)
        assert self.recorder.thread.is_alive()

        # Stop recorder
        self.recorder.stop()
        time.sleep(0.1)

        # Check thread is stopped
        assert not self.recorder.thread.is_alive()

    def test_record_data_with_gauge_calls(self):
        """Test that recording calls gauge methods correctly."""
        # Start recording in test mode (no LabJack needed)
        self.recorder.start()

        # Let it record just one data point
        time.sleep(0.15)  # Slightly more than one interval (0.1s)

        # Stop recording
        self.recorder.stop()

        # Check that gauges were called to record voltage data
        assert self.mock_gauge1.record_ain_channel_voltage.called
        assert self.mock_gauge2.record_ain_channel_voltage.called

        # Verify the CSV file has the expected structure
        csv_path = os.path.join(self.recorder.run_dir, "pressure_gauge_data.csv")
        with open(csv_path) as f:
            lines = f.readlines()

        # Should have header + at least 1 data line
        assert len(lines) >= 2

        # Check that each line has the right number of columns
        for line in lines[1:]:  # Skip header
            data_parts = line.strip().split(",")
            assert len(data_parts) == 3  # timestamp + 2 voltages

    def test_normal_mode_initialization(self):
        """Test that normal mode recorder can be created and initialized."""
        # Create a normal mode recorder (but don't start it to avoid LabJack issues)
        normal_recorder = DataRecorder(
            gauges=[self.mock_gauge1, self.mock_gauge2],
            thermocouples=[],
            results_dir=self.temp_dir,
            run_type="permeation_exp",
        )

        # Check basic attributes
        assert normal_recorder.test_mode is False
        assert normal_recorder.gauges == [self.mock_gauge1, self.mock_gauge2]

        # Test that directories can be created
        run_dir = normal_recorder._create_results_directory()
        assert os.path.exists(run_dir)
        assert "run_" in os.path.basename(run_dir)

    def test_record_data_test_mode(self):
        """Test data recording in test mode."""
        # Start recording
        self.recorder.start()

        # Let it record just one data point
        time.sleep(0.15)  # Slightly more than one interval (0.1s)

        # Stop recording
        self.recorder.stop()

        # Check CSV file has data
        csv_path = os.path.join(self.recorder.run_dir, "pressure_gauge_data.csv")
        with open(csv_path) as f:
            lines = f.readlines()

        # Should have header + at least 1 data line
        assert len(lines) >= 2

        # Check data format
        data_line = lines[1].strip().split(",")
        assert len(data_line) == 3  # timestamp + 2 voltages

        # Check timestamp format (should be datetime string)
        timestamp = data_line[0]
        assert len(timestamp) == 23  # YYYY-MM-DD HH:MM:SS.mmm format

        # Check voltages are numeric (using mock data)
        voltage1 = float(data_line[1])
        voltage2 = float(data_line[2])
        assert voltage1 == 5.0  # From mock_gauge1.voltage_data
        assert voltage2 == 3.5  # From mock_gauge2.voltage_data

    def test_run_method(self):
        """Test the run() method."""
        # Start run in a separate thread (since it blocks)
        run_thread = threading.Thread(target=self.recorder.run)
        run_thread.daemon = True
        run_thread.start()

        # Let it run briefly - just enough to verify it starts
        time.sleep(0.2)

        # Check that recorder is running
        assert self.recorder.thread is not None
        assert self.recorder.thread.is_alive()

        # Simulate KeyboardInterrupt by stopping manually
        self.recorder.stop()

        # Wait for run thread to finish
        run_thread.join(timeout=2.0)

    def test_elapsed_time_tracking(self):
        """Test that elapsed time is tracked correctly."""
        # Start recording
        self.recorder.start()

        initial_time = self.recorder.elapsed_time
        assert initial_time == 0.0

        # Let it run for just over one interval
        time.sleep(0.15)

        # Check elapsed time increased
        assert self.recorder.elapsed_time > initial_time
        assert self.recorder.elapsed_time >= 0.1  # Should be at least 0.1 second

        # Stop recording
        self.recorder.stop()

    @pytest.mark.filterwarnings(
        "ignore:Exception in thread:pytest.PytestUnhandledThreadExceptionWarning"
    )
    def test_error_handling_in_recording(self):
        """Test error handling during data recording. #TODO need to replace this"""
        # Make one gauge raise an exception
        self.mock_gauge1.record_ain_channel_voltage.side_effect = Exception(
            "Test error"
        )

        # Start recording - should handle the exception gracefully
        self.recorder.start()

        # Let it run briefly
        time.sleep(0.15)

        # Stop recording
        self.recorder.stop()

        # Check that recording attempted to call the gauges
        assert self.mock_gauge1.record_ain_channel_voltage.called

    def test_multiple_gauges_different_names(self):
        """Test CSV header generation with different gauge names."""
        # Create gauges with specific names
        gauge_a = Mock(spec=PressureGauge)
        gauge_a.name = "WGM701"
        gauge_a.voltage_data = [1.0]
        gauge_a.record_ain_channel_voltage.return_value = None
        gauge_a.ain_channel = 10
        gauge_a.gauge_location = "downstream"

        gauge_b = Mock(spec=PressureGauge)
        gauge_b.name = "Baratron626D"
        gauge_b.voltage_data = [2.0]
        gauge_b.record_ain_channel_voltage.return_value = None
        gauge_b.ain_channel = 6
        gauge_b.gauge_location = "downstream"

        gauge_c = Mock(spec=PressureGauge)
        gauge_c.name = "CVM211"
        gauge_c.voltage_data = [3.0]
        gauge_c.record_ain_channel_voltage.return_value = None
        gauge_c.ain_channel = 4
        gauge_c.gauge_location = "upstream"

        recorder = DataRecorder(
            gauges=[gauge_a, gauge_b, gauge_c],
            thermocouples=[],
            results_dir=self.temp_dir,
            run_type="test_mode",
        )

        # Start recording to create CSV file
        recorder.start()
        time.sleep(0.1)
        recorder.stop()

        # Check header
        csv_path = os.path.join(recorder.run_dir, "pressure_gauge_data.csv")
        with open(csv_path) as f:
            header = f.readline().strip()

        expected = (
            "RealTimestamp,WGM701_Voltage (V),"
            "Baratron626D_Voltage (V),CVM211_Voltage (V)"
        )
        assert header == expected

    def test_csv_file_naming(self):
        """Test that CSV file is named correctly."""
        self.recorder.start()
        time.sleep(0.1)

        expected_filename = os.path.join(
            self.recorder.run_dir, "pressure_gauge_data.csv"
        )
        assert os.path.exists(expected_filename)

        self.recorder.stop()

    def test_metadata_file_creation(self):
        """Test that metadata JSON file is created with correct information."""
        # Start recorder to trigger metadata creation
        self.recorder.start()
        time.sleep(0.1)

        # Check that metadata file was created
        metadata_path = os.path.join(self.recorder.run_dir, "run_metadata.json")
        assert os.path.exists(metadata_path)

        # Read and verify metadata content
        with open(metadata_path) as f:
            metadata = json.load(f)

        # Verify required top-level keys
        assert "version" in metadata
        assert "run_info" in metadata
        assert "gauges" in metadata
        assert "thermocouples" in metadata

        # Verify version info
        assert isinstance(metadata["version"], str)

        # Verify run_info content
        run_info = metadata["run_info"]
        assert "date" in run_info
        assert "start_time" in run_info
        assert run_info["run_type"] == "test_mode"
        assert run_info["recording_interval_seconds"] == 0.1
        assert run_info["backup_interval_seconds"] == 5.0

        # Verify gauges information
        gauges_info = metadata["gauges"]
        assert len(gauges_info) == 2
        assert gauges_info[0]["name"] == "TestGauge1"
        assert gauges_info[1]["name"] == "TestGauge2"

        self.recorder.stop()

    def test_duplicate_ain_channels_raises_value_error(self):
        """Test that duplicate AIN channels raise ValueError when starting recorder."""
        # Create two mock gauges with the same AIN channel
        duplicate_gauge1 = Mock(spec=PressureGauge)
        duplicate_gauge1.name = "DuplicateGauge1"
        duplicate_gauge1.ain_channel = 5  # Same AIN channel
        duplicate_gauge1.gauge_location = "downstream"
        duplicate_gauge1.voltage_data = [2.0]
        duplicate_gauge1.record_ain_channel_voltage.return_value = None

        duplicate_gauge2 = Mock(spec=PressureGauge)
        duplicate_gauge2.name = "DuplicateGauge2"
        duplicate_gauge2.ain_channel = 5  # Same AIN channel as gauge1
        duplicate_gauge2.gauge_location = "upstream"
        duplicate_gauge2.voltage_data = [3.0]
        duplicate_gauge2.record_ain_channel_voltage.return_value = None

        # Create recorder with duplicate AIN channels
        recorder_with_duplicates = DataRecorder(
            gauges=[duplicate_gauge1, duplicate_gauge2],
            thermocouples=[],
            results_dir=self.temp_dir,
            run_type="test_mode",  # Use test mode to avoid LabJack initialization
        )

        # Starting the recorder should raise ValueError due to duplicate AIN channels
        with pytest.raises(ValueError) as exc_info:
            recorder_with_duplicates.start()

        # Verify the error message
        assert "Duplicate AIN channels detected among gauges" in str(exc_info.value)

    def test_unique_ain_channels_no_error(self):
        """Test that unique AIN channels don't raise any error."""
        # Create two mock gauges with different AIN channels
        unique_gauge1 = Mock(spec=PressureGauge)
        unique_gauge1.name = "UniqueGauge1"
        unique_gauge1.ain_channel = 7
        unique_gauge1.gauge_location = "downstream"
        unique_gauge1.voltage_data = [2.5]
        unique_gauge1.record_ain_channel_voltage.return_value = None

        unique_gauge2 = Mock(spec=PressureGauge)
        unique_gauge2.name = "UniqueGauge2"
        unique_gauge2.ain_channel = 9
        unique_gauge2.gauge_location = "upstream"
        unique_gauge2.voltage_data = [4.2]
        unique_gauge2.record_ain_channel_voltage.return_value = None

        # Create recorder with unique AIN channels
        recorder_with_unique = DataRecorder(
            gauges=[unique_gauge1, unique_gauge2],
            thermocouples=[],
            results_dir=self.temp_dir,
            run_type="test_mode",
        )

        # Starting the recorder should not raise any error
        try:
            recorder_with_unique.start()
            # Give it a moment to start
            time.sleep(0.1)
            recorder_with_unique.stop()
        except ValueError:
            pytest.fail("ValueError should not be raised for unique AIN channels")
