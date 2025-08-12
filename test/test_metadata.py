import json
import os
import tempfile
import time
from unittest.mock import Mock

import pytest

from shield_das import DataRecorder, PressureGauge


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
            test_mode=True,
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
        assert "run_info" in metadata
        assert "gauges" in metadata
        assert "thermocouples" in metadata
        assert "system_info" in metadata

    def test_metadata_run_info_content(self):
        """Test that run_info section contains correct data."""
        recorder = DataRecorder(
            gauges=[self.mock_gauge1],
            thermocouples=[],
            results_dir=self.temp_dir,
            test_mode=True,
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
        assert run_info["test_mode"] is True
        assert run_info["recording_interval_seconds"] == 0.5
        assert run_info["backup_interval_seconds"] == 2.0

    def test_metadata_gauges_information(self):
        """Test that gauges information is correctly captured."""
        recorder = DataRecorder(
            gauges=[self.mock_gauge1, self.mock_gauge2],
            thermocouples=[],
            results_dir=self.temp_dir,
            test_mode=True,
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

    def test_metadata_system_info(self):
        """Test that system_info section contains correct data."""
        recorder = DataRecorder(
            gauges=[self.mock_gauge1],
            thermocouples=[],
            results_dir=self.temp_dir,
            test_mode=False,  # Test non-test mode too
        )

        recorder.start()
        time.sleep(0.1)
        recorder.stop()

        metadata_path = os.path.join(recorder.run_dir, "run_metadata.json")
        with open(metadata_path) as f:
            metadata = json.load(f)

        # Check system_info
        system_info = metadata["system_info"]
        assert "results_directory" in system_info
        assert "run_directory" in system_info
        assert "backup_directory" in system_info
        assert system_info["results_directory"] == self.temp_dir

    def test_metadata_with_thermocouples(self):
        """Test metadata creation when thermocouples are present."""
        # Create mock thermocouple
        mock_thermocouple = Mock()
        mock_thermocouple.name = "TestThermocouple"

        recorder = DataRecorder(
            gauges=[self.mock_gauge1],
            thermocouples=[mock_thermocouple],
            results_dir=self.temp_dir,
            test_mode=True,
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
            test_mode=True,
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


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v"])
