import json
import os
import tempfile
import time
from unittest.mock import Mock, patch

from shield_das import DataRecorder, PressureGauge


class TestValveEvents:
    """Test suite for valve event timing functionality"""

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

        # Create DataRecorder instance
        self.recorder = DataRecorder(
            gauges=[self.mock_gauge],
            thermocouples=[],
            results_dir=self.temp_dir,
            run_type="test_mode",
            recording_interval=0.01,  # Much faster for testing
            backup_interval=0.1,  # Much faster for testing
        )

    def teardown_method(self):
        """Clean up after each test method."""
        if hasattr(self.recorder, "run_dir") and self.recorder.run_dir:
            # Stop recording if running
            if (
                hasattr(self.recorder, "stop_event")
                and not self.recorder.stop_event.is_set()
            ):
                self.recorder.stop()

    def test_valve_event_initialization(self):
        """Test that valve event attributes are properly initialized."""
        assert self.recorder.v5_close_time is None
        assert self.recorder.v6_close_time is None
        assert self.recorder.v7_close_time is None
        assert self.recorder.v3_open_time is None
        assert self.recorder.current_valve_index == 0
        assert self.recorder.valve_event_sequence == [
            "v5_close_time",
            "v6_close_time",
            "v7_close_time",
            "v3_open_time",
        ]

    def test_valve_event_reset_on_start(self):
        """Test that valve events are reset when starting a new recording."""
        # Set some initial values
        self.recorder.v5_close_time = "2025-01-01 12:00:00"
        self.recorder.v6_close_time = "2025-01-01 12:01:00"
        self.recorder.current_valve_index = 2

        with patch("shield_das.data_recorder.keyboard"):
            self.recorder.start()
            time.sleep(0.02)  # Very brief sleep for testing
            self.recorder.stop()

        # Check that values were reset
        assert self.recorder.v5_close_time is None
        assert self.recorder.v6_close_time is None
        assert self.recorder.v7_close_time is None
        assert self.recorder.v3_open_time is None
        assert self.recorder.current_valve_index == 0

    def test_valve_event_sequence_progression(self):
        """Test that valve events progress through the sequence correctly."""
        test_timestamp = "2025-08-12 15:30:00.123"  # Include milliseconds

        # Start the recorder first to create directories
        with patch("shield_das.data_recorder.keyboard"):
            self.recorder.start()

            # Now simulate valve events being recorded
            for i, event_name in enumerate(self.recorder.valve_event_sequence):
                # Manually trigger the valve event
                self.recorder.current_valve_index = i
                setattr(self.recorder, event_name, test_timestamp)
                self.recorder._update_metadata_with_valve_time(
                    event_name, test_timestamp
                )

                # Check that the event was recorded
                assert getattr(self.recorder, event_name) == test_timestamp

                # Check metadata was updated
                metadata_path = os.path.join(self.recorder.run_dir, "run_metadata.json")
                with open(metadata_path) as f:
                    metadata = json.load(f)
                assert metadata["run_info"][event_name] == test_timestamp

            self.recorder.stop()

    def test_update_metadata_with_valve_time(self):
        """Test that metadata is correctly updated with valve event times."""
        with patch("shield_das.data_recorder.keyboard"):
            self.recorder.start()

            test_timestamp = "2025-08-12 15:30:00.456"  # Include milliseconds
            event_name = "v5_close_time"

            # Update metadata with valve time
            self.recorder._update_metadata_with_valve_time(event_name, test_timestamp)

            # Verify metadata was updated
            metadata_path = os.path.join(self.recorder.run_dir, "run_metadata.json")
            with open(metadata_path) as f:
                metadata = json.load(f)

            assert event_name in metadata["run_info"]
            assert metadata["run_info"][event_name] == test_timestamp

            self.recorder.stop()

    def test_multiple_valve_events_in_metadata(self):
        """Test that multiple valve events can be recorded in metadata."""
        with patch("shield_das.data_recorder.keyboard"):
            self.recorder.start()

            # Record multiple valve events
            valve_times = {
                "v5_close_time": "2025-08-12 15:30:00.123",
                "v6_close_time": "2025-08-12 15:31:00.456",
                "v7_close_time": "2025-08-12 15:32:00.789",
                "v3_open_time": "2025-08-12 15:33:00.012",
            }

            for event_name, timestamp in valve_times.items():
                self.recorder._update_metadata_with_valve_time(event_name, timestamp)

            # Verify all events are in metadata
            metadata_path = os.path.join(self.recorder.run_dir, "run_metadata.json")
            with open(metadata_path) as f:
                metadata = json.load(f)

            for event_name, expected_time in valve_times.items():
                assert event_name in metadata["run_info"]
                assert metadata["run_info"][event_name] == expected_time

            self.recorder.stop()

    @patch("shield_das.data_recorder.keyboard")
    def test_keyboard_monitoring_with_keyboard_module(self, mock_keyboard):
        """Test that keyboard monitoring works when keyboard module is available."""
        # Mock the CI detection to return False (simulate local environment)
        with patch.object(self.recorder, "_is_ci_environment", return_value=False):
            # Should not raise an exception and should set up listener
            self.recorder._monitor_keyboard()

            # Verify that keyboard.on_press_key was called
            mock_keyboard.on_press_key.assert_called_once()
            call_args = mock_keyboard.on_press_key.call_args
            assert call_args[0][0] == "space"  # First argument should be "space"

    @patch("shield_das.data_recorder.keyboard")
    def test_keyboard_listener_setup(self, mock_keyboard):
        """Test that keyboard listener is properly set up when not in CI."""
        # Mock the CI detection to return False (simulate local environment)
        with patch.object(self.recorder, "_is_ci_environment", return_value=False):
            self.recorder.start()

            # Verify that keyboard.on_press_key was called
            mock_keyboard.on_press_key.assert_called_once()
            call_args = mock_keyboard.on_press_key.call_args
            assert call_args[0][0] == "space"  # First argument should be "space"

            self.recorder.stop()

            # Verify keyboard cleanup
            mock_keyboard.unhook_all.assert_called_once()

    @patch("shield_das.data_recorder.keyboard")
    def test_keyboard_listener_disabled_in_ci(self, mock_keyboard):
        """Test that keyboard listener is disabled in CI environment."""
        # Mock the CI detection to return True (simulate CI environment)
        with patch.object(self.recorder, "_is_ci_environment", return_value=True):
            self.recorder.start()

            # Verify that keyboard.on_press_key was NOT called
            mock_keyboard.on_press_key.assert_not_called()

            self.recorder.stop()

            # Verify keyboard cleanup was NOT called in CI environment
            mock_keyboard.unhook_all.assert_not_called()

    def test_valve_event_attributes_in_class(self):
        """Test that all valve event attributes are properly defined in the class."""
        # Check that all expected attributes exist
        assert hasattr(self.recorder, "v5_close_time")
        assert hasattr(self.recorder, "v6_close_time")
        assert hasattr(self.recorder, "v7_close_time")
        assert hasattr(self.recorder, "v3_open_time")
        assert hasattr(self.recorder, "valve_event_sequence")
        assert hasattr(self.recorder, "current_valve_index")

    def test_valve_sequence_order(self):
        """Test that the valve event sequence is in the correct order."""
        expected_sequence = [
            "v5_close_time",
            "v6_close_time",
            "v7_close_time",
            "v3_open_time",
        ]
        assert self.recorder.valve_event_sequence == expected_sequence

    def test_metadata_error_handling(self):
        """Test error handling when metadata file cannot be updated."""
        # Create a scenario where metadata update would fail
        with patch("shield_das.data_recorder.keyboard"):
            self.recorder.start()

            # Remove the run directory to cause an error
            metadata_path = os.path.join(self.recorder.run_dir, "run_metadata.json")
            os.remove(metadata_path)

            # This should not raise an exception, but should print an error message
            self.recorder._update_metadata_with_valve_time(
                "v5_close_time", "2025-08-12 15:30:00.999"
            )

            self.recorder.stop()
