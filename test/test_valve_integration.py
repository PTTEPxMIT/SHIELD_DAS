"""
Integration test for valve event functionality.
This test can be run with: python -m pytest test/test_valve_integration.py -v
"""

import json
import os
import tempfile
import time
from unittest.mock import Mock, patch

from shield_das import DataRecorder, PressureGauge


class TestValveEventIntegration:
    """Integration tests for the complete valve event workflow"""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        # Create temporary directory for test results
        self.temp_dir = tempfile.mkdtemp()

        # Create mock gauge
        self.mock_gauge = Mock(spec=PressureGauge)
        self.mock_gauge.name = "Test_Gauge"
        self.mock_gauge.ain_channel = 10
        self.mock_gauge.gauge_location = "test"
        self.mock_gauge.voltage_data = [5.0]
        self.mock_gauge.record_ain_channel_voltage.return_value = None

    def test_complete_valve_workflow(self):
        """Test the complete valve event recording workflow."""
        # Create recorder
        recorder = DataRecorder(
            gauges=[self.mock_gauge],
            thermocouples=[],
            results_dir=self.temp_dir,
            run_type="test_mode",
            recording_interval=0.01,  # Much faster for testing
            backup_interval=0.05,  # Much faster for testing
        )

        # Mock keyboard to avoid actual keyboard interaction
        with patch("shield_das.data_recorder.keyboard") as mock_keyboard:
            # Start recording
            recorder.start()

            # Verify initial state
            assert recorder.v5_close_time is None
            assert recorder.v6_close_time is None
            assert recorder.v7_close_time is None
            assert recorder.v3_open_time is None
            assert recorder.current_valve_index == 0

            # Simulate valve events being recorded
            valve_times = {
                "v5_close_time": "2025-08-12 15:30:00.123",
                "v6_close_time": "2025-08-12 15:31:00.456",
                "v7_close_time": "2025-08-12 15:32:00.789",
                "v3_open_time": "2025-08-12 15:33:00.012",
            }  # Record each valve event
            for i, (event_name, timestamp) in enumerate(valve_times.items()):
                # Set the attribute on the recorder
                setattr(recorder, event_name, timestamp)
                # Update metadata
                recorder._update_metadata_with_valve_time(event_name, timestamp)
                # Update index
                recorder.current_valve_index = i + 1

            # Let recording run briefly
            time.sleep(0.03)  # Very brief for testing

            # Stop recording
            recorder.stop()

            # Verify metadata file contains all valve events
            metadata_path = os.path.join(recorder.run_dir, "run_metadata.json")
            assert os.path.exists(metadata_path)

            with open(metadata_path) as f:
                metadata = json.load(f)

            # Check that all valve events are in metadata
            for event_name, expected_time in valve_times.items():
                assert event_name in metadata["run_info"]
                assert metadata["run_info"][event_name] == expected_time

            # Verify other metadata fields are present
            assert "version" in metadata
            assert "date" in metadata["run_info"]
            assert "start_time" in metadata["run_info"]
            assert "run_type" in metadata["run_info"]
            assert metadata["run_info"]["run_type"] == "test_mode"

    def test_partial_valve_recording(self):
        """Test recording only some valve events."""
        recorder = DataRecorder(
            gauges=[self.mock_gauge],
            thermocouples=[],
            results_dir=self.temp_dir,
            run_type="test_mode",
            recording_interval=0.01,  # Much faster for testing
            backup_interval=0.05,  # Much faster for testing
        )

        with patch("shield_das.data_recorder.keyboard"):
            recorder.start()

            # Record only first two valve events
            recorder._update_metadata_with_valve_time(
                "v5_close_time", "2025-08-12 15:30:00.123"
            )
            recorder._update_metadata_with_valve_time(
                "v6_close_time", "2025-08-12 15:31:00.456"
            )

            time.sleep(0.02)  # Very brief for testing
            recorder.stop()

            # Check metadata
            metadata_path = os.path.join(recorder.run_dir, "run_metadata.json")
            with open(metadata_path) as f:
                metadata = json.load(f)

            # Should have first two events
            assert "v5_close_time" in metadata["run_info"]
            assert "v6_close_time" in metadata["run_info"]

            # Should not have last two events
            assert "v7_close_time" not in metadata["run_info"]
            assert "v3_open_time" not in metadata["run_info"]

    def test_valve_event_sequence_reset_between_runs(self):
        """Test that valve events are properly reset between recording sessions."""
        recorder = DataRecorder(
            gauges=[self.mock_gauge],
            thermocouples=[],
            results_dir=self.temp_dir,
            run_type="test_mode",
            recording_interval=0.01,  # Much faster for testing
            backup_interval=0.05,  # Much faster for testing
        )

        with patch("shield_das.data_recorder.keyboard"):
            # First recording session
            recorder.start()
            recorder._update_metadata_with_valve_time(
                "v5_close_time", "2025-08-12 15:30:00.111"
            )
            time.sleep(0.02)  # Very brief for testing
            recorder.stop()

            # Verify first session had the valve event
            first_run_dir = recorder.run_dir

            # Second recording session
            recorder.start()

            # Should be reset
            assert recorder.v5_close_time is None
            assert recorder.current_valve_index == 0

            recorder._update_metadata_with_valve_time(
                "v5_close_time", "2025-08-12 16:30:00.222"
            )
            time.sleep(0.02)  # Very brief for testing
            recorder.stop()

            # Check both metadata files exist and have correct data
            first_metadata_path = os.path.join(first_run_dir, "run_metadata.json")
            second_metadata_path = os.path.join(recorder.run_dir, "run_metadata.json")

            with open(first_metadata_path) as f:
                first_metadata = json.load(f)
            with open(second_metadata_path) as f:
                second_metadata = json.load(f)

            assert (
                first_metadata["run_info"]["v5_close_time"] == "2025-08-12 15:30:00.111"
            )
            assert (
                second_metadata["run_info"]["v5_close_time"]
                == "2025-08-12 16:30:00.222"
            )
