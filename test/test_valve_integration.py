"""Integration tests for valve event functionality."""

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
    """Create a DataRecorder instance for integration testing."""
    rec = DataRecorder(
        gauges=[mock_gauge],
        thermocouples=[],
        results_dir=temp_dir,
        run_type="test_mode",
        recording_interval=0.01,
        backup_interval=0.05,
    )
    yield rec
    if rec.thread and rec.thread.is_alive():
        rec.stop()


# =============================================================================
# Integration Tests for Complete Valve Workflow
# =============================================================================


def test_complete_valve_workflow_initializes_all_events_to_none(recorder):
    """
    Test DataRecorder complete valve workflow to verify all valve events
    initialize to None.
    """
    with patch("shield_das.data_recorder.keyboard"):
        recorder.start()
        assert recorder.v4_close_time is None
        assert recorder.v5_close_time is None
        assert recorder.v6_close_time is None
        assert recorder.v3_open_time is None
        recorder.stop()


def test_complete_valve_workflow_initializes_valve_index_to_zero(recorder):
    """
    Test DataRecorder complete valve workflow to verify current_valve_index
    initializes to 0.
    """
    with patch("shield_das.data_recorder.keyboard"):
        recorder.start()
        assert recorder.current_valve_index == 0
        recorder.stop()


def test_complete_valve_workflow_records_v4_close_time_in_metadata(recorder):
    """
    Test DataRecorder complete valve workflow to verify v4_close_time is
    recorded in metadata.
    """
    with patch("shield_das.data_recorder.keyboard"):
        recorder.start()
        timestamp = "2025-08-12 15:30:00.123"
        setattr(recorder, "v4_close_time", timestamp)
        recorder._update_metadata_with_valve_time("v4_close_time", timestamp)

        metadata_path = os.path.join(recorder.run_dir, "run_metadata.json")
        with open(metadata_path) as f:
            metadata = json.load(f)

        assert metadata["run_info"]["v4_close_time"] == timestamp
        recorder.stop()


def test_complete_valve_workflow_records_v5_close_time_in_metadata(recorder):
    """
    Test DataRecorder complete valve workflow to verify v5_close_time is
    recorded in metadata.
    """
    with patch("shield_das.data_recorder.keyboard"):
        recorder.start()
        timestamp = "2025-08-12 15:31:00.456"
        setattr(recorder, "v5_close_time", timestamp)
        recorder._update_metadata_with_valve_time("v5_close_time", timestamp)

        metadata_path = os.path.join(recorder.run_dir, "run_metadata.json")
        with open(metadata_path) as f:
            metadata = json.load(f)

        assert metadata["run_info"]["v5_close_time"] == timestamp
        recorder.stop()


def test_complete_valve_workflow_records_v6_close_time_in_metadata(recorder):
    """
    Test DataRecorder complete valve workflow to verify v6_close_time is
    recorded in metadata.
    """
    with patch("shield_das.data_recorder.keyboard"):
        recorder.start()
        timestamp = "2025-08-12 15:32:00.789"
        setattr(recorder, "v6_close_time", timestamp)
        recorder._update_metadata_with_valve_time("v6_close_time", timestamp)

        metadata_path = os.path.join(recorder.run_dir, "run_metadata.json")
        with open(metadata_path) as f:
            metadata = json.load(f)

        assert metadata["run_info"]["v6_close_time"] == timestamp
        recorder.stop()


def test_complete_valve_workflow_records_v3_open_time_in_metadata(recorder):
    """
    Test DataRecorder complete valve workflow to verify v3_open_time is
    recorded in metadata.
    """
    with patch("shield_das.data_recorder.keyboard"):
        recorder.start()
        timestamp = "2025-08-12 15:33:00.012"
        setattr(recorder, "v3_open_time", timestamp)
        recorder._update_metadata_with_valve_time("v3_open_time", timestamp)

        metadata_path = os.path.join(recorder.run_dir, "run_metadata.json")
        with open(metadata_path) as f:
            metadata = json.load(f)

        assert metadata["run_info"]["v3_open_time"] == timestamp
        recorder.stop()


def test_complete_valve_workflow_metadata_contains_version(recorder):
    """
    Test DataRecorder complete valve workflow to verify metadata contains
    version field.
    """
    with patch("shield_das.data_recorder.keyboard"):
        recorder.start()

        metadata_path = os.path.join(recorder.run_dir, "run_metadata.json")
        with open(metadata_path) as f:
            metadata = json.load(f)

        assert "version" in metadata
        recorder.stop()


def test_complete_valve_workflow_metadata_contains_date(recorder):
    """
    Test DataRecorder complete valve workflow to verify metadata contains
    date field.
    """
    with patch("shield_das.data_recorder.keyboard"):
        recorder.start()

        metadata_path = os.path.join(recorder.run_dir, "run_metadata.json")
        with open(metadata_path) as f:
            metadata = json.load(f)

        assert "date" in metadata["run_info"]
        recorder.stop()


def test_complete_valve_workflow_metadata_contains_start_time(recorder):
    """
    Test DataRecorder complete valve workflow to verify metadata contains
    start_time field.
    """
    with patch("shield_das.data_recorder.keyboard"):
        recorder.start()

        metadata_path = os.path.join(recorder.run_dir, "run_metadata.json")
        with open(metadata_path) as f:
            metadata = json.load(f)

        assert "start_time" in metadata["run_info"]
        recorder.stop()


def test_complete_valve_workflow_metadata_contains_run_type(recorder):
    """
    Test DataRecorder complete valve workflow to verify metadata contains
    run_type field with correct value.
    """
    with patch("shield_das.data_recorder.keyboard"):
        recorder.start()

        metadata_path = os.path.join(recorder.run_dir, "run_metadata.json")
        with open(metadata_path) as f:
            metadata = json.load(f)

        assert metadata["run_info"]["run_type"] == "test_mode"
        recorder.stop()


# =============================================================================
# Tests for Partial Valve Recording
# =============================================================================


def test_partial_valve_recording_stores_v5_close_time(recorder):
    """
    Test DataRecorder partial valve recording to verify v5_close_time is
    stored in metadata.
    """
    with patch("shield_das.data_recorder.keyboard"):
        recorder.start()
        recorder._update_metadata_with_valve_time(
            "v5_close_time", "2025-08-12 15:30:00.123"
        )

        metadata_path = os.path.join(recorder.run_dir, "run_metadata.json")
        with open(metadata_path) as f:
            metadata = json.load(f)

        assert "v5_close_time" in metadata["run_info"]
        recorder.stop()


def test_partial_valve_recording_stores_v6_close_time(recorder):
    """
    Test DataRecorder partial valve recording to verify v6_close_time is
    stored in metadata.
    """
    with patch("shield_das.data_recorder.keyboard"):
        recorder.start()
        recorder._update_metadata_with_valve_time(
            "v6_close_time", "2025-08-12 15:31:00.456"
        )

        metadata_path = os.path.join(recorder.run_dir, "run_metadata.json")
        with open(metadata_path) as f:
            metadata = json.load(f)

        assert "v6_close_time" in metadata["run_info"]
        recorder.stop()


def test_partial_valve_recording_does_not_store_unrecorded_events(recorder):
    """
    Test DataRecorder partial valve recording to verify unrecorded events
    are not in metadata.
    """
    with patch("shield_das.data_recorder.keyboard"):
        recorder.start()
        recorder._update_metadata_with_valve_time(
            "v5_close_time", "2025-08-12 15:30:00.123"
        )

        metadata_path = os.path.join(recorder.run_dir, "run_metadata.json")
        with open(metadata_path) as f:
            metadata = json.load(f)

        assert "v7_close_time" not in metadata["run_info"]
        assert "v3_open_time" not in metadata["run_info"]
        recorder.stop()


# =============================================================================
# Tests for Valve Event Reset Between Runs
# =============================================================================


def test_valve_event_v5_close_time_resets_between_runs(temp_dir, mock_gauge):
    """
    Test DataRecorder to verify v5_close_time resets to None between runs.
    """
    recorder = DataRecorder(
        gauges=[mock_gauge],
        thermocouples=[],
        results_dir=temp_dir,
        run_type="test_mode",
        recording_interval=0.01,
        backup_interval=0.05,
    )

    with patch("shield_das.data_recorder.keyboard"):
        recorder.start()
        recorder._update_metadata_with_valve_time(
            "v5_close_time", "2025-08-12 15:30:00.111"
        )
        time.sleep(0.02)
        recorder.stop()

        recorder.start()
        assert recorder.v5_close_time is None
        recorder.stop()


def test_valve_event_current_valve_index_resets_between_runs(temp_dir, mock_gauge):
    """
    Test DataRecorder to verify current_valve_index resets to 0 between runs.
    """
    recorder = DataRecorder(
        gauges=[mock_gauge],
        thermocouples=[],
        results_dir=temp_dir,
        run_type="test_mode",
        recording_interval=0.01,
        backup_interval=0.05,
    )

    with patch("shield_das.data_recorder.keyboard"):
        recorder.start()
        recorder.current_valve_index = 2
        time.sleep(0.02)
        recorder.stop()

        recorder.start()
        assert recorder.current_valve_index == 0
        recorder.stop()


def test_valve_event_different_metadata_files_for_each_run(temp_dir, mock_gauge):
    """
    Test DataRecorder to verify separate metadata files are created for
    each recording session.
    """
    recorder = DataRecorder(
        gauges=[mock_gauge],
        thermocouples=[],
        results_dir=temp_dir,
        run_type="test_mode",
        recording_interval=0.01,
        backup_interval=0.05,
    )

    with patch("shield_das.data_recorder.keyboard"):
        recorder.start()
        recorder._update_metadata_with_valve_time(
            "v5_close_time", "2025-08-12 15:30:00.111"
        )
        time.sleep(0.02)
        first_run_dir = recorder.run_dir
        recorder.stop()

        recorder.start()
        recorder._update_metadata_with_valve_time(
            "v5_close_time", "2025-08-12 16:30:00.222"
        )
        time.sleep(0.02)
        second_run_dir = recorder.run_dir
        recorder.stop()

        first_metadata_path = os.path.join(first_run_dir, "run_metadata.json")
        second_metadata_path = os.path.join(second_run_dir, "run_metadata.json")

        assert os.path.exists(first_metadata_path)
        assert os.path.exists(second_metadata_path)


def test_valve_event_metadata_has_correct_timestamps_for_each_run(temp_dir, mock_gauge):
    """
    Test DataRecorder to verify each run's metadata contains its specific
    valve event timestamps.
    """
    recorder = DataRecorder(
        gauges=[mock_gauge],
        thermocouples=[],
        results_dir=temp_dir,
        run_type="test_mode",
        recording_interval=0.01,
        backup_interval=0.05,
    )

    with patch("shield_das.data_recorder.keyboard"):
        recorder.start()
        recorder._update_metadata_with_valve_time(
            "v5_close_time", "2025-08-12 15:30:00.111"
        )
        time.sleep(0.02)
        first_run_dir = recorder.run_dir
        recorder.stop()

        recorder.start()
        recorder._update_metadata_with_valve_time(
            "v5_close_time", "2025-08-12 16:30:00.222"
        )
        time.sleep(0.02)
        recorder.stop()

        first_metadata_path = os.path.join(first_run_dir, "run_metadata.json")
        second_metadata_path = os.path.join(recorder.run_dir, "run_metadata.json")

        with open(first_metadata_path) as f:
            first_metadata = json.load(f)
        with open(second_metadata_path) as f:
            second_metadata = json.load(f)

        assert first_metadata["run_info"]["v5_close_time"] == "2025-08-12 15:30:00.111"
        assert second_metadata["run_info"]["v5_close_time"] == "2025-08-12 16:30:00.222"
