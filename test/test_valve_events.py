import json
import os
import shutil
import tempfile
import time
from unittest.mock import Mock, patch

import pytest

from shield_das import DataRecorder, PressureGauge

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test results."""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    if os.path.exists(temp_path):
        shutil.rmtree(temp_path)


@pytest.fixture
def mock_gauge():
    """Create a mock PressureGauge for testing."""
    gauge = Mock(spec=PressureGauge)
    gauge.name = "Test_Gauge"
    gauge.ain_channel = 10
    gauge.gauge_location = "test"
    gauge.voltage_data = [5.0]
    gauge.record_ain_channel_voltage.return_value = None
    return gauge


@pytest.fixture
def recorder(temp_dir, mock_gauge):
    """Create a DataRecorder instance for valve event testing."""
    rec = DataRecorder(
        gauges=[mock_gauge],
        thermocouples=[],
        results_dir=temp_dir,
        run_type="test_mode",
        recording_interval=0.01,
        backup_interval=0.1,
    )
    yield rec
    if rec.thread and rec.thread.is_alive():
        rec.stop()


# =============================================================================
# Tests for Valve Event Initialization
# =============================================================================


def test_valve_event_v4_close_time_initializes_to_none(recorder):
    """
    Test DataRecorder to verify v4_close_time attribute is initialized to None.
    """
    assert recorder.v4_close_time is None


def test_valve_event_v5_close_time_initializes_to_none(recorder):
    """
    Test DataRecorder to verify v5_close_time attribute is initialized to None.
    """
    assert recorder.v5_close_time is None


def test_valve_event_v6_close_time_initializes_to_none(recorder):
    """
    Test DataRecorder to verify v6_close_time attribute is initialized to None.
    """
    assert recorder.v6_close_time is None


def test_valve_event_v3_open_time_initializes_to_none(recorder):
    """
    Test DataRecorder to verify v3_open_time attribute is initialized to None.
    """
    assert recorder.v3_open_time is None


def test_valve_event_current_valve_index_initializes_to_zero(recorder):
    """
    Test DataRecorder to verify current_valve_index is initialized to 0.
    """
    assert recorder.current_valve_index == 0


def test_valve_event_sequence_contains_four_events(recorder):
    """
    Test DataRecorder to verify valve_event_sequence contains exactly 4 events.
    """
    assert len(recorder.valve_event_sequence) == 4


def test_valve_event_sequence_first_event_is_v4_close_time(recorder):
    """
    Test DataRecorder to verify first event in sequence is v4_close_time.
    """
    assert recorder.valve_event_sequence[0] == "v4_close_time"


def test_valve_event_sequence_second_event_is_v5_close_time(recorder):
    """
    Test DataRecorder to verify second event in sequence is v5_close_time.
    """
    assert recorder.valve_event_sequence[1] == "v5_close_time"


def test_valve_event_sequence_third_event_is_v6_close_time(recorder):
    """
    Test DataRecorder to verify third event in sequence is v6_close_time.
    """
    assert recorder.valve_event_sequence[2] == "v6_close_time"


def test_valve_event_sequence_fourth_event_is_v3_open_time(recorder):
    """
    Test DataRecorder to verify fourth event in sequence is v3_open_time.
    """
    assert recorder.valve_event_sequence[3] == "v3_open_time"


# =============================================================================
# Tests for Valve Event Reset
# =============================================================================


def test_valve_event_v5_close_time_resets_on_start(recorder):
    """
    Test DataRecorder to verify v5_close_time is reset to None when start() called.
    """
    recorder.v5_close_time = "2025-01-01 12:00:00"
    with patch("shield_das.data_recorder.keyboard"):
        recorder.start()
        time.sleep(0.02)
        recorder.stop()
    assert recorder.v5_close_time is None


def test_valve_event_v6_close_time_resets_on_start(recorder):
    """
    Test DataRecorder to verify v6_close_time is reset to None when start() called.
    """
    recorder.v6_close_time = "2025-01-01 12:01:00"
    with patch("shield_das.data_recorder.keyboard"):
        recorder.start()
        time.sleep(0.02)
        recorder.stop()
    assert recorder.v6_close_time is None


def test_valve_event_v4_close_time_resets_on_start(recorder):
    """
    Test DataRecorder to verify v4_close_time is reset to None when start() called.
    """
    recorder.v4_close_time = "2025-01-01 11:59:00"
    with patch("shield_das.data_recorder.keyboard"):
        recorder.start()
        time.sleep(0.02)
        recorder.stop()
    assert recorder.v4_close_time is None


def test_valve_event_v3_open_time_resets_on_start(recorder):
    """
    Test DataRecorder to verify v3_open_time is reset to None when start() called.
    """
    recorder.v3_open_time = "2025-01-01 12:02:00"
    with patch("shield_das.data_recorder.keyboard"):
        recorder.start()
        time.sleep(0.02)
        recorder.stop()
    assert recorder.v3_open_time is None


def test_valve_event_current_valve_index_resets_to_zero_on_start(recorder):
    """
    Test DataRecorder to verify current_valve_index resets to 0 when start() called.
    """
    recorder.current_valve_index = 2
    with patch("shield_das.data_recorder.keyboard"):
        recorder.start()
        time.sleep(0.02)
        recorder.stop()
    assert recorder.current_valve_index == 0


# =============================================================================
# Tests for Valve Event Metadata Updates
# =============================================================================


def test_update_metadata_with_valve_time_adds_event_to_metadata(recorder):
    """
    Test DataRecorder _update_metadata_with_valve_time to verify it adds
    event to metadata file.
    """
    with patch("shield_das.data_recorder.keyboard"):
        recorder.start()
        test_timestamp = "2025-08-12 15:30:00.456"
        event_name = "v5_close_time"
        recorder._update_metadata_with_valve_time(event_name, test_timestamp)

        metadata_path = os.path.join(recorder.run_dir, "run_metadata.json")
        with open(metadata_path) as f:
            metadata = json.load(f)

        assert event_name in metadata["run_info"]
        recorder.stop()


def test_update_metadata_with_valve_time_stores_correct_timestamp(recorder):
    """
    Test DataRecorder _update_metadata_with_valve_time to verify timestamp
    is correctly stored.
    """
    with patch("shield_das.data_recorder.keyboard"):
        recorder.start()
        test_timestamp = "2025-08-12 15:30:00.456"
        event_name = "v5_close_time"
        recorder._update_metadata_with_valve_time(event_name, test_timestamp)

        metadata_path = os.path.join(recorder.run_dir, "run_metadata.json")
        with open(metadata_path) as f:
            metadata = json.load(f)

        assert metadata["run_info"][event_name] == test_timestamp
        recorder.stop()


def test_update_metadata_with_valve_time_handles_multiple_events(recorder):
    """
    Test DataRecorder _update_metadata_with_valve_time to verify multiple
    valve events can be stored.
    """
    with patch("shield_das.data_recorder.keyboard"):
        recorder.start()

        valve_times = {
            "v5_close_time": "2025-08-12 15:30:00.123",
            "v6_close_time": "2025-08-12 15:31:00.456",
        }

        for event_name, timestamp in valve_times.items():
            recorder._update_metadata_with_valve_time(event_name, timestamp)

        metadata_path = os.path.join(recorder.run_dir, "run_metadata.json")
        with open(metadata_path) as f:
            metadata = json.load(f)

        for event_name in valve_times:
            assert event_name in metadata["run_info"]

        recorder.stop()


def test_update_metadata_with_valve_time_raises_error_when_file_missing(recorder):
    """
    Test DataRecorder _update_metadata_with_valve_time to verify it raises
    FileNotFoundError when metadata file is missing.
    """
    with patch("shield_das.data_recorder.keyboard"):
        recorder.start()

        metadata_path = os.path.join(recorder.run_dir, "run_metadata.json")
        os.remove(metadata_path)

        with pytest.raises(FileNotFoundError):
            recorder._update_metadata_with_valve_time(
                "v5_close_time", "2025-08-12 15:30:00.999"
            )

        recorder.stop()


# =============================================================================
# Tests for Valve Event Sequence Progression
# =============================================================================


@pytest.mark.parametrize(
    "event_index,event_name",
    [
        (0, "v4_close_time"),
        (1, "v5_close_time"),
        (2, "v6_close_time"),
        (3, "v3_open_time"),
    ],
)
def test_valve_event_can_be_set_and_retrieved(recorder, event_index, event_name):
    """
    Test DataRecorder to verify each valve event can be set and retrieved.
    """
    with patch("shield_das.data_recorder.keyboard"):
        recorder.start()
        test_timestamp = "2025-08-12 15:30:00.123"

        recorder.current_valve_index = event_index
        setattr(recorder, event_name, test_timestamp)
        recorder._update_metadata_with_valve_time(event_name, test_timestamp)

        assert getattr(recorder, event_name) == test_timestamp
        recorder.stop()


@pytest.mark.parametrize(
    "event_index,event_name",
    [
        (0, "v4_close_time"),
        (1, "v5_close_time"),
        (2, "v6_close_time"),
        (3, "v3_open_time"),
    ],
)
def test_valve_event_updates_metadata_correctly(recorder, event_index, event_name):
    """
    Test DataRecorder to verify valve event updates are reflected in metadata.
    """
    with patch("shield_das.data_recorder.keyboard"):
        recorder.start()
        test_timestamp = "2025-08-12 15:30:00.123"

        recorder.current_valve_index = event_index
        setattr(recorder, event_name, test_timestamp)
        recorder._update_metadata_with_valve_time(event_name, test_timestamp)

        metadata_path = os.path.join(recorder.run_dir, "run_metadata.json")
        with open(metadata_path) as f:
            metadata = json.load(f)

        assert metadata["run_info"][event_name] == test_timestamp
        recorder.stop()


# =============================================================================
# Tests for Keyboard Monitoring
# =============================================================================


@patch("shield_das.data_recorder.keyboard")
def test_keyboard_monitoring_sets_up_listener_when_not_in_ci(mock_keyboard, recorder):
    """
    Test DataRecorder _monitor_keyboard to verify keyboard listener is set up
    when not in CI environment.
    """
    with patch.object(recorder, "_is_ci_environment", return_value=False):
        recorder._monitor_keyboard()
        mock_keyboard.on_press_key.assert_called_once()


@patch("shield_das.data_recorder.keyboard")
def test_keyboard_monitoring_uses_space_key(mock_keyboard, recorder):
    """
    Test DataRecorder _monitor_keyboard to verify it listens for spacebar key.
    """
    with patch.object(recorder, "_is_ci_environment", return_value=False):
        recorder._monitor_keyboard()
        call_args = mock_keyboard.on_press_key.call_args
        assert call_args[0][0] == "space"


@patch("shield_das.data_recorder.keyboard")
def test_keyboard_listener_setup_on_start(mock_keyboard, recorder):
    """
    Test DataRecorder start() to verify keyboard listener is set up.
    """
    with patch.object(recorder, "_is_ci_environment", return_value=False):
        recorder.start()
        mock_keyboard.on_press_key.assert_called_once()
        recorder.stop()


@patch("shield_das.data_recorder.keyboard")
def test_keyboard_listener_cleanup_on_stop(mock_keyboard, recorder):
    """
    Test DataRecorder stop() to verify keyboard listener is cleaned up.
    """
    with patch.object(recorder, "_is_ci_environment", return_value=False):
        recorder.start()
        recorder.stop()
        mock_keyboard.unhook_all.assert_called_once()


@patch("shield_das.data_recorder.keyboard")
def test_keyboard_listener_disabled_in_ci_environment(mock_keyboard, recorder):
    """
    Test DataRecorder to verify keyboard listener is not set up in CI.
    """
    with patch.object(recorder, "_is_ci_environment", return_value=True):
        recorder.start()
        mock_keyboard.on_press_key.assert_not_called()
        recorder.stop()


@patch("shield_das.data_recorder.keyboard")
def test_keyboard_cleanup_not_called_in_ci_environment(mock_keyboard, recorder):
    """
    Test DataRecorder to verify keyboard cleanup is not called in CI.
    """
    with patch.object(recorder, "_is_ci_environment", return_value=True):
        recorder.start()
        recorder.stop()
        mock_keyboard.unhook_all.assert_not_called()


# =============================================================================
# Tests for Valve Event Attributes
# =============================================================================


def test_valve_event_v4_close_time_attribute_exists(recorder):
    """
    Test DataRecorder to verify v4_close_time attribute exists.
    """
    assert hasattr(recorder, "v4_close_time")


def test_valve_event_v5_close_time_attribute_exists(recorder):
    """
    Test DataRecorder to verify v5_close_time attribute exists.
    """
    assert hasattr(recorder, "v5_close_time")


def test_valve_event_v6_close_time_attribute_exists(recorder):
    """
    Test DataRecorder to verify v6_close_time attribute exists.
    """
    assert hasattr(recorder, "v6_close_time")


def test_valve_event_v3_open_time_attribute_exists(recorder):
    """
    Test DataRecorder to verify v3_open_time attribute exists.
    """
    assert hasattr(recorder, "v3_open_time")


def test_valve_event_sequence_attribute_exists(recorder):
    """
    Test DataRecorder to verify valve_event_sequence attribute exists.
    """
    assert hasattr(recorder, "valve_event_sequence")


def test_valve_event_current_valve_index_attribute_exists(recorder):
    """
    Test DataRecorder to verify current_valve_index attribute exists.
    """
    assert hasattr(recorder, "current_valve_index")
