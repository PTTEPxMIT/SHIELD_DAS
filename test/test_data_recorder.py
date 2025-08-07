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
        self.mock_gauge1.get_ain_channel_voltage.return_value = 5.0

        self.mock_gauge2 = Mock(spec=PressureGauge)
        self.mock_gauge2.name = "TestGauge2"
        self.mock_gauge2.get_ain_channel_voltage.return_value = 3.5

        # Create mock thermocouples (empty list for now)
        self.mock_thermocouples = []

        # Create DataRecorder instance
        self.recorder = DataRecorder(
            gauges=[self.mock_gauge1, self.mock_gauge2],
            thermocouples=self.mock_thermocouples,
            results_dir=self.temp_dir,
            test_mode=True,
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
        assert self.recorder.run_dir is None
        assert self.recorder.backup_dir is None
        assert self.recorder.main_csv_filename is None
        assert self.recorder.elapsed_time == 0.0
        assert self.recorder.start_time is None

    def test_init_default_values(self):
        """Test DataRecorder initialization with default values."""
        default_recorder = DataRecorder(gauges=[self.mock_gauge1], thermocouples=[])
        assert default_recorder.results_dir == "results"
        assert default_recorder.test_mode is False

    @patch("src.shield_das.data_recorder.datetime")
    def test_create_results_directory_test_mode(self, mock_datetime):
        """Test directory creation in test mode."""
        # Mock datetime.now()
        mock_now = Mock()
        mock_now.strftime.side_effect = lambda fmt: {
            "%m.%d": "08.07",
            "%Hh%M": "14h30",
        }[fmt]
        mock_datetime.now.return_value = mock_now

        # Test directory creation
        run_dir = self.recorder._create_results_directory()

        expected_path = os.path.join(self.temp_dir, "08.07", "test_run_14h30")
        assert run_dir == expected_path
        assert os.path.exists(run_dir)

    @patch("src.shield_das.data_recorder.datetime")
    @patch("src.shield_das.data_recorder.glob.glob")
    def test_create_results_directory_normal_mode(self, mock_glob, mock_datetime):
        """Test directory creation in normal mode."""
        # Setup recorder in normal mode
        normal_recorder = DataRecorder(
            gauges=[self.mock_gauge1],
            thermocouples=[],
            results_dir=self.temp_dir,
            test_mode=False,
        )

        # Mock datetime.now()
        mock_now = Mock()
        mock_now.strftime.side_effect = lambda fmt: {
            "%m.%d": "08.07",
            "%Hh%M": "14h30",
        }[fmt]
        mock_datetime.now.return_value = mock_now

        # Mock existing run directories
        mock_glob.return_value = [
            os.path.join(self.temp_dir, "08.07", "run_1_13h00"),
            os.path.join(self.temp_dir, "08.07", "run_2_13h30"),
        ]

        # Test directory creation
        run_dir = normal_recorder._create_results_directory()

        expected_path = os.path.join(self.temp_dir, "08.07", "run_3_14h30")
        assert run_dir == expected_path
        assert os.path.exists(run_dir)

    def test_initialise_main_csv(self):
        """Test CSV file initialization."""
        # Set up run directory
        self.recorder.run_dir = self.temp_dir

        # Initialize CSV
        self.recorder._initialise_main_csv()

        # Check file was created
        expected_filename = os.path.join(self.temp_dir, "pressure_gauge_data.csv")
        assert self.recorder.main_csv_filename == expected_filename
        assert os.path.exists(expected_filename)

        # Check header content
        with open(expected_filename) as f:
            header = f.read().strip()

        expected_header = "RealTimestamp,TestGauge1_Voltage (V),TestGauge2_Voltage (V)"
        assert header == expected_header

    def test_write_to_csv(self):
        """Test writing data to CSV file."""
        # Set up CSV file
        self.recorder.run_dir = self.temp_dir
        self.recorder._initialise_main_csv()

        # Write test data
        test_timestamp = "2025-08-07 14:30:00.123"
        test_voltages = [5.0, 3.5]
        self.recorder._write_to_csv(test_timestamp, test_voltages)

        # Read and verify content
        with open(self.recorder.main_csv_filename) as f:
            lines = f.readlines()

        assert len(lines) == 2  # Header + data
        assert lines[1].strip() == "2025-08-07 14:30:00.123,5.0,3.5"

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

        # Check CSV file was created
        assert self.recorder.main_csv_filename is not None
        assert os.path.exists(self.recorder.main_csv_filename)

        # Check thread is running
        assert self.recorder.thread is not None
        assert self.recorder.thread.is_alive()

        # Stop recorder
        self.recorder.stop()

    def test_start_only_initializes_once(self):
        """Test that start() only initializes directories once."""
        # Start recorder first time
        self.recorder.start()
        time.sleep(0.1)
        first_run_dir = self.recorder.run_dir

        # Stop and start again
        self.recorder.stop()
        self.recorder.start()
        time.sleep(0.1)

        # Should be same directory
        assert self.recorder.run_dir == first_run_dir

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

    @patch("src.shield_das.data_recorder.u6.U6")
    def test_record_data_normal_mode(self, mock_u6):
        """Test data recording in normal mode."""
        # Set up normal mode recorder
        normal_recorder = DataRecorder(
            gauges=[self.mock_gauge1, self.mock_gauge2],
            thermocouples=[],
            results_dir=self.temp_dir,
            test_mode=False,
        )

        # Mock LabJack
        mock_labjack = Mock()
        mock_u6.return_value = mock_labjack

        # Start recording
        normal_recorder.start()

        # Let it record a few data points
        time.sleep(1.5)  # Should get ~3 data points (0.5s interval)

        # Stop recording
        normal_recorder.stop()

        # Verify LabJack was initialized
        mock_u6.assert_called_once_with(firstFound=True)
        mock_labjack.getCalibrationData.assert_called_once()

        # Verify gauges were called
        assert self.mock_gauge1.get_ain_channel_voltage.call_count >= 2
        assert self.mock_gauge2.get_ain_channel_voltage.call_count >= 2

    def test_record_data_test_mode(self):
        """Test data recording in test mode."""
        # Start recording
        self.recorder.start()

        # Let it record a few data points
        time.sleep(1.5)  # Should get ~3 data points (0.5s interval)

        # Stop recording
        self.recorder.stop()

        # Check CSV file has data
        with open(self.recorder.main_csv_filename) as f:
            lines = f.readlines()

        # Should have header + at least 2 data lines
        assert len(lines) >= 3

        # Check data format
        data_line = lines[1].strip().split(",")
        assert len(data_line) == 3  # timestamp + 2 voltages

        # Check timestamp format (should be datetime string)
        timestamp = data_line[0]
        assert len(timestamp) == 23  # YYYY-MM-DD HH:MM:SS.mmm format

        # Check voltages are numeric
        voltage1 = float(data_line[1])
        voltage2 = float(data_line[2])
        assert 0 <= voltage1 <= 10  # Random range we set
        assert 0 <= voltage2 <= 10

    @patch("src.shield_das.data_recorder.datetime")
    def test_run_method(self, mock_datetime):
        """Test the run() method."""
        # Mock datetime.now() - note that _create_results_directory calls it twice
        mock_now = Mock()
        mock_now.strftime.side_effect = lambda fmt: {
            "%m.%d": "08.07",
            "%Hh%M": "14h30",
            "%Y-%m-%d %H:%M:%S": "2025-08-07 14:30:00",
        }.get(fmt, "2025-08-07 14:30:00")
        mock_datetime.now.return_value = mock_now

        # Start run in a separate thread (since it blocks)
        run_thread = threading.Thread(target=self.recorder.run)
        run_thread.daemon = True
        run_thread.start()

        # Let it run briefly
        time.sleep(1.0)

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

        # Let it run for a bit
        time.sleep(1.5)

        # Check elapsed time increased
        assert self.recorder.elapsed_time > initial_time
        assert self.recorder.elapsed_time >= 1.0  # Should be at least 1 second

        # Stop recording
        self.recorder.stop()

    def test_error_handling_in_recording(self):
        """Test error handling during data recording."""
        # Make one gauge raise an exception
        self.mock_gauge1.get_ain_channel_voltage.side_effect = Exception("Test error")

        # Start recording - this should fail due to unhandled exception
        self.recorder.start()

        # Let it run briefly - the thread should die due to the exception
        time.sleep(1.0)

        # Stop recording
        self.recorder.stop()

        # Check that the thread died due to exception
        # In current implementation, unhandled exceptions will terminate the thread
        assert not self.recorder.thread.is_alive()

        # The CSV file might be created but may have incomplete data
        if os.path.exists(self.recorder.main_csv_filename):
            with open(self.recorder.main_csv_filename) as f:
                lines = f.readlines()
            # Should at least have the header
            assert len(lines) >= 1

    def test_multiple_gauges_different_names(self):
        """Test CSV header generation with different gauge names."""
        # Create gauges with specific names
        gauge_a = Mock(spec=PressureGauge)
        gauge_a.name = "WGM701"
        gauge_b = Mock(spec=PressureGauge)
        gauge_b.name = "Baratron626D"
        gauge_c = Mock(spec=PressureGauge)
        gauge_c.name = "CVM211"

        recorder = DataRecorder(
            gauges=[gauge_a, gauge_b, gauge_c],
            thermocouples=[],
            results_dir=self.temp_dir,
            test_mode=True,
        )

        # Initialize CSV
        recorder.run_dir = self.temp_dir
        recorder._initialise_main_csv()

        # Check header
        with open(recorder.main_csv_filename) as f:
            header = f.read().strip()

        expected = (
            "RealTimestamp,WGM701_Voltage (V),"
            "Baratron626D_Voltage (V),CVM211_Voltage (V)"
        )
        assert header == expected

    def test_csv_file_naming(self):
        """Test that CSV file is named correctly."""
        self.recorder.run_dir = self.temp_dir
        self.recorder._initialise_main_csv()

        expected_filename = os.path.join(self.temp_dir, "pressure_gauge_data.csv")
        assert self.recorder.main_csv_filename == expected_filename


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v"])
