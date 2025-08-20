import json
import os
import tempfile
import time
from unittest.mock import Mock

import pytest

from shield_das import DataRecorder, PressureGauge, Thermocouple


class TestMetadata:
    """Test suite for metadata file creation functionality"""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        # Create temporary directory for test results
        self.temp_dir = tempfile.mkdtemp()

        # Create mock gauges
        self.mock_gauge1 = Mock(spec=PressureGauge)
        self.mock_gauge1.name = "WGM701_Test"
        self.mock_gauge1.ain_channel = 10
        self.mock_gauge1.gauge_location = "downstream"
        self.mock_gauge1.voltage_data = [5.0]
        self.mock_gauge1.record_ain_channel_voltage.return_value = None

        self.mock_gauge2 = Mock(spec=PressureGauge)
        self.mock_gauge2.name = "CVM211_Test"
        self.mock_gauge2.ain_channel = 8
        self.mock_gauge2.gauge_location = "upstream"
        self.mock_gauge2.voltage_data = [3.5]
        self.mock_gauge2.record_ain_channel_voltage.return_value = None

        # Create mock thermocouples (empty list for now)
        self.mock_thermocouples = []

    def teardown_method(self):
        """Clean up after each test method."""
        # Remove temporary directory
        import shutil

        shutil.rmtree(self.temp_dir)

    def test_metadata_file_creation(self):
        """Test that metadata JSON file is created correctly."""
        # Create DataRecorder instance
        recorder = DataRecorder(
            gauges=[self.mock_gauge1, self.mock_gauge2],
            thermocouples=self.mock_thermocouples,
            results_dir=self.temp_dir,
            run_type="test_mode",
            recording_interval=0.1,
            backup_interval=1.0,
        )

        # Start recording (this should create the metadata file)
        recorder.start()
        time.sleep(0.1)  # Give it time to initialize
        recorder.stop()

        # Check if metadata file was created
        metadata_path = os.path.join(recorder.run_dir, "run_metadata.json")
        assert os.path.exists(metadata_path), (
            f"Metadata file not found at {metadata_path}"
        )

        # Read and verify metadata content
        with open(metadata_path) as f:
            metadata = json.load(f)

        # Verify expected top-level keys
        assert "version" in metadata
        assert "run_info" in metadata
        assert "gauges" in metadata
        assert "thermocouples" in metadata

        # Verify version is present and correctly formatted
        assert isinstance(metadata["version"], str)
        assert "." in metadata["version"]  # Should be in format like "1.0.0"

    def test_metadata_run_info_content(self):
        """Test that run_info section contains correct data."""
        recorder = DataRecorder(
            gauges=[self.mock_gauge1],
            thermocouples=[],
            results_dir=self.temp_dir,
            run_type="test_mode",
            recording_interval=0.5,
            backup_interval=2.0,
        )

        recorder.start()
        time.sleep(0.1)
        recorder.stop()

        metadata_path = os.path.join(recorder.run_dir, "run_metadata.json")
        with open(metadata_path) as f:
            metadata = json.load(f)

        # Check run_info
        run_info = metadata["run_info"]
        assert "date" in run_info
        assert "start_time" in run_info
        assert run_info["run_type"] == "test_mode"
        assert "furnace_setpoint" in run_info  # Should always be present
        assert run_info["recording_interval_seconds"] == 0.5
        assert run_info["backup_interval_seconds"] == 2.0

    def test_metadata_gauges_information(self):
        """Test that gauges information is correctly captured."""
        recorder = DataRecorder(
            gauges=[self.mock_gauge1, self.mock_gauge2],
            thermocouples=[],
            results_dir=self.temp_dir,
            run_type="test_mode",
        )

        recorder.start()
        time.sleep(0.1)
        recorder.stop()

        metadata_path = os.path.join(recorder.run_dir, "run_metadata.json")
        with open(metadata_path) as f:
            metadata = json.load(f)

        # Check gauges information
        gauges = metadata["gauges"]
        assert len(gauges) == 2

        # Check first gauge
        assert gauges[0]["name"] == "WGM701_Test"
        assert gauges[0]["ain_channel"] == 10
        assert gauges[0]["gauge_location"] == "downstream"

        # Check second gauge
        assert gauges[1]["name"] == "CVM211_Test"
        assert gauges[1]["ain_channel"] == 8
        assert gauges[1]["gauge_location"] == "upstream"

    def test_metadata_version_info(self):
        """Test that version information is correctly included."""
        recorder = DataRecorder(
            gauges=[self.mock_gauge1],
            thermocouples=[],
            results_dir=self.temp_dir,
            run_type="test_mode",
        )

        recorder.start()
        time.sleep(0.1)
        recorder.stop()

        metadata_path = os.path.join(recorder.run_dir, "run_metadata.json")
        with open(metadata_path) as f:
            metadata = json.load(f)

        # Check version info
        assert "version" in metadata
        version = metadata["version"]
        assert isinstance(version, str)
        assert len(version.split(".")) == 2  # Should be semantic versioning (x.y)

    def test_metadata_with_thermocouples(self):
        """Test metadata creation when thermocouples are present."""
        # Create mock thermocouple
        mock_thermocouple = Mock(spec=Thermocouple)
        mock_thermocouple.name = "TestThermocouple"

        recorder = DataRecorder(
            gauges=[self.mock_gauge1],
            thermocouples=[mock_thermocouple],
            results_dir=self.temp_dir,
            run_type="test_mode",
        )

        recorder.start()
        time.sleep(0.1)
        recorder.stop()

        metadata_path = os.path.join(recorder.run_dir, "run_metadata.json")
        with open(metadata_path) as f:
            metadata = json.load(f)

        # Check thermocouples information
        thermocouples = metadata["thermocouples"]
        assert len(thermocouples) == 1
        assert thermocouples[0]["name"] == "TestThermocouple"

    def test_metadata_file_format(self):
        """Test that metadata file is valid JSON and properly formatted."""
        recorder = DataRecorder(
            gauges=[self.mock_gauge1],
            thermocouples=[],
            results_dir=self.temp_dir,
            run_type="test_mode",
        )

        recorder.start()
        time.sleep(0.1)
        recorder.stop()

        metadata_path = os.path.join(recorder.run_dir, "run_metadata.json")

        # Test that file can be read as valid JSON
        with open(metadata_path) as f:
            content = f.read()

        # Should not raise an exception when parsing JSON
        json.loads(content)

        # Check that it's properly indented (contains newlines and spaces)
        assert "\n" in content
        assert "  " in content  # Should have indentation

    def test_metadata_baratron_full_scale_torr(self):
        """Test that Baratron626D_Gauge includes full_scale_torr in metadata."""
        from unittest.mock import Mock

        # Create mock Baratron gauge
        mock_baratron = Mock(spec=PressureGauge)
        mock_baratron.name = "Baratron_Test"
        mock_baratron.ain_channel = 6
        mock_baratron.gauge_location = "downstream"
        mock_baratron.voltage_data = [2.5]
        mock_baratron.record_ain_channel_voltage.return_value = None
        mock_baratron.full_scale_Torr = 1000.0

        # Mock the type to return Baratron626D_Gauge
        type(mock_baratron).__name__ = "Baratron626D_Gauge"

        recorder = DataRecorder(
            gauges=[mock_baratron],
            thermocouples=[],
            results_dir=self.temp_dir,
            run_type="test_mode",
        )

        recorder.start()
        time.sleep(0.1)
        recorder.stop()

        metadata_path = os.path.join(recorder.run_dir, "run_metadata.json")
        with open(metadata_path) as f:
            metadata = json.load(f)

        # Check that the Baratron gauge has full_scale_torr
        gauges = metadata["gauges"]
        assert len(gauges) == 1
        baratron_gauge = gauges[0]
        assert baratron_gauge["type"] == "Baratron626D_Gauge"
        assert "full_scale_torr" in baratron_gauge
        assert baratron_gauge["full_scale_torr"] == 1000.0

    def test_metadata_non_baratron_no_full_scale_torr(self):
        """Test that non-Baratron gauges don't include full_scale_torr."""
        recorder = DataRecorder(
            gauges=[self.mock_gauge1],  # This is not a Baratron
            thermocouples=[],
            results_dir=self.temp_dir,
            run_type="test_mode",
        )

        recorder.start()
        time.sleep(0.1)
        recorder.stop()

        metadata_path = os.path.join(recorder.run_dir, "run_metadata.json")
        with open(metadata_path) as f:
            metadata = json.load(f)

        # Check that non-Baratron gauge doesn't have full_scale_torr
        gauges = metadata["gauges"]
        assert len(gauges) == 1
        gauge = gauges[0]
        assert "full_scale_torr" not in gauge

    def test_metadata_furnace_setpoint_provided(self):
        """Test that furnace setpoint is correctly included when provided."""
        furnace_temp = 850.0
        recorder = DataRecorder(
            gauges=[self.mock_gauge1],
            thermocouples=[],
            results_dir=self.temp_dir,
            furnace_setpoint=furnace_temp,
            run_type="test_mode",
        )

        recorder.start()
        time.sleep(0.1)
        recorder.stop()

        metadata_path = os.path.join(recorder.run_dir, "run_metadata.json")
        with open(metadata_path) as f:
            metadata = json.load(f)

        # Check that furnace_setpoint is present and correct
        run_info = metadata["run_info"]
        assert "furnace_setpoint" in run_info
        assert run_info["furnace_setpoint"] == furnace_temp
        assert isinstance(run_info["furnace_setpoint"], int | float)

    def test_metadata_furnace_setpoint_none(self):
        """Test that furnace setpoint is None when not provided."""
        recorder = DataRecorder(
            gauges=[self.mock_gauge1],
            thermocouples=[],
            results_dir=self.temp_dir,
            # furnace_setpoint not provided, should default to None
            run_type="test_mode",
        )

        recorder.start()
        time.sleep(0.1)
        recorder.stop()

        metadata_path = os.path.join(recorder.run_dir, "run_metadata.json")
        with open(metadata_path) as f:
            metadata = json.load(f)

        # Check that furnace_setpoint is present but None
        run_info = metadata["run_info"]
        assert "furnace_setpoint" in run_info
        assert run_info["furnace_setpoint"] is None

    def test_metadata_furnace_setpoint_edge_cases(self):
        """Test furnace setpoint with various valid values."""
        test_cases = [
            0.0,  # Zero temperature
            25.5,  # Room temperature
            1200.0,  # High temperature
            -273.15,  # Absolute zero (unlikely but valid for setpoint)
        ]

        for setpoint in test_cases:
            recorder = DataRecorder(
                gauges=[self.mock_gauge1],
                thermocouples=[],
                results_dir=self.temp_dir,
                furnace_setpoint=setpoint,
                run_type="test_mode",
            )

            recorder.start()
            time.sleep(0.1)
            recorder.stop()

            metadata_path = os.path.join(recorder.run_dir, "run_metadata.json")
            with open(metadata_path) as f:
                metadata = json.load(f)

            # Verify the setpoint is correctly stored
            run_info = metadata["run_info"]
            assert "furnace_setpoint" in run_info
            assert run_info["furnace_setpoint"] == setpoint
            assert isinstance(run_info["furnace_setpoint"], int | float)
