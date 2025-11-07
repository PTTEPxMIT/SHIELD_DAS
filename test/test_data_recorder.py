import json
import os
import shutil
import tempfile
import threading
import time
from datetime import datetime
from unittest.mock import Mock, patch

import pytest

from shield_das import DataRecorder, PressureGauge, Thermocouple

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test results."""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    # Cleanup after test
    if os.path.exists(temp_path):
        shutil.rmtree(temp_path)


@pytest.fixture
def mock_gauge():
    """Create a mock PressureGauge for testing."""
    gauge = Mock(spec=PressureGauge)
    gauge.name = "TestGauge"
    gauge.voltage_data = [5.0]
    gauge.record_ain_channel_voltage.return_value = None
    gauge.ain_channel = 10
    gauge.gauge_location = "downstream"
    return gauge


@pytest.fixture
def mock_gauge2():
    """Create a second mock PressureGauge for testing."""
    gauge = Mock(spec=PressureGauge)
    gauge.name = "TestGauge2"
    gauge.voltage_data = [3.5]
    gauge.record_ain_channel_voltage.return_value = None
    gauge.ain_channel = 6
    gauge.gauge_location = "upstream"
    return gauge


@pytest.fixture
def mock_thermocouple():
    """Create a mock Thermocouple for testing."""
    tc = Mock(spec=Thermocouple)
    tc.name = "TestThermocouple"
    tc.voltage_data = [1.2]
    tc.local_temperature_data = [25.0]
    tc.record_ain_channel_voltage.return_value = None
    return tc


@pytest.fixture
def recorder(temp_dir, mock_gauge, mock_gauge2):
    """Create a DataRecorder instance for testing."""
    rec = DataRecorder(
        gauges=[mock_gauge, mock_gauge2],
        thermocouples=[],
        results_dir=temp_dir,
        run_type="test_mode",
        recording_interval=0.1,
        backup_interval=0.3,
    )
    yield rec
    # Cleanup: stop recorder if running
    if rec.thread and rec.thread.is_alive():
        rec.stop()


# =============================================================================
# Tests for Initialization
# =============================================================================


def test_data_recorder_initializes_with_gauges(mock_gauge, mock_gauge2):
    """
    Test DataRecorder to verify it correctly stores the list of gauges
    provided during initialization.
    """
    recorder = DataRecorder(
        gauges=[mock_gauge, mock_gauge2],
        thermocouples=[],
    )
    assert recorder.gauges == [mock_gauge, mock_gauge2]


def test_data_recorder_initializes_with_thermocouples(mock_gauge, mock_thermocouple):
    """
    Test DataRecorder to verify it correctly stores the list of thermocouples
    provided during initialization.
    """
    recorder = DataRecorder(
        gauges=[mock_gauge],
        thermocouples=[mock_thermocouple],
    )
    assert recorder.thermocouples == [mock_thermocouple]


def test_data_recorder_initializes_with_default_results_dir(mock_gauge):
    """
    Test DataRecorder to confirm it uses "results" as the default results
    directory when none is specified.
    """
    recorder = DataRecorder(gauges=[mock_gauge], thermocouples=[])
    assert recorder.results_dir == "results"


def test_data_recorder_initializes_with_custom_results_dir(mock_gauge):
    """
    Test DataRecorder to verify it correctly stores a custom results directory
    path when provided during initialization.
    """
    custom_dir = "/custom/path"
    recorder = DataRecorder(
        gauges=[mock_gauge],
        thermocouples=[],
        results_dir=custom_dir,
    )
    assert recorder.results_dir == custom_dir


@pytest.mark.parametrize(
    "run_type",
    ["permeation_exp", "leak_test", "test_mode"],
)
def test_data_recorder_initializes_with_valid_run_type(mock_gauge, run_type):
    """
    Test DataRecorder to ensure it accepts and stores valid run_type values:
    'permeation_exp', 'leak_test', or 'test_mode'.
    """
    recorder = DataRecorder(
        gauges=[mock_gauge],
        thermocouples=[],
        run_type=run_type,
    )
    assert recorder.run_type == run_type


def test_data_recorder_initializes_with_default_recording_interval(mock_gauge):
    """
    Test DataRecorder to confirm it uses 0.5 seconds as the default recording
    interval when none is specified.
    """
    recorder = DataRecorder(gauges=[mock_gauge], thermocouples=[])
    assert recorder.recording_interval == 0.5


def test_data_recorder_initializes_with_custom_recording_interval(mock_gauge):
    """
    Test DataRecorder to verify it correctly stores a custom recording interval
    when provided during initialization.
    """
    recorder = DataRecorder(
        gauges=[mock_gauge],
        thermocouples=[],
        recording_interval=0.2,
    )
    assert recorder.recording_interval == 0.2


def test_data_recorder_initializes_with_default_backup_interval(mock_gauge):
    """
    Test DataRecorder to confirm it uses 5.0 seconds as the default backup
    interval when none is specified.
    """
    recorder = DataRecorder(gauges=[mock_gauge], thermocouples=[])
    assert recorder.backup_interval == 5.0


def test_data_recorder_initializes_with_custom_backup_interval(mock_gauge):
    """
    Test DataRecorder to verify it correctly stores a custom backup interval
    when provided during initialization.
    """
    recorder = DataRecorder(
        gauges=[mock_gauge],
        thermocouples=[],
        backup_interval=10.0,
    )
    assert recorder.backup_interval == 10.0


def test_data_recorder_initializes_stop_event_as_threading_event(mock_gauge):
    """
    Test DataRecorder to ensure the stop_event attribute is initialized as a
    threading.Event instance for thread control.
    """
    recorder = DataRecorder(gauges=[mock_gauge], thermocouples=[])
    assert isinstance(recorder.stop_event, threading.Event)


def test_data_recorder_initializes_thread_as_none(mock_gauge):
    """
    Test DataRecorder to verify the thread attribute is None before recording
    starts, indicating no active recording thread.
    """
    recorder = DataRecorder(gauges=[mock_gauge], thermocouples=[])
    assert recorder.thread is None


def test_data_recorder_initializes_elapsed_time_to_zero(mock_gauge):
    """
    Test DataRecorder to confirm elapsed_time is initialized to 0.0 seconds
    before recording begins.
    """
    recorder = DataRecorder(gauges=[mock_gauge], thermocouples=[])
    assert recorder.elapsed_time == 0.0


def test_data_recorder_initializes_valve_times_to_none(mock_gauge):
    """
    Test DataRecorder to verify all valve event times (v4_close, v5_close,
    v6_close, v3_open) are initialized to None.
    """
    recorder = DataRecorder(gauges=[mock_gauge], thermocouples=[])
    assert recorder.v4_close_time is None
    assert recorder.v5_close_time is None
    assert recorder.v6_close_time is None
    assert recorder.v3_open_time is None


def test_data_recorder_initializes_current_valve_index_to_zero(mock_gauge):
    """
    Test DataRecorder to confirm current_valve_index starts at 0, pointing to
    the first valve event in the sequence.
    """
    recorder = DataRecorder(gauges=[mock_gauge], thermocouples=[])
    assert recorder.current_valve_index == 0


def test_data_recorder_initializes_valve_event_sequence(mock_gauge):
    """
    Test DataRecorder to verify valve_event_sequence is initialized with the
    correct ordered list of valve events.
    """
    recorder = DataRecorder(gauges=[mock_gauge], thermocouples=[])
    expected = ["v4_close_time", "v5_close_time", "v6_close_time", "v3_open_time"]
    assert recorder.valve_event_sequence == expected


# =============================================================================
# Tests for Property Validation
# =============================================================================


def test_data_recorder_raises_error_for_invalid_gauges_type(mock_gauge):
    """
    Test DataRecorder to verify it raises ValueError when gauges parameter
    is not a list.
    """
    with pytest.raises(ValueError, match="gauges must be a list"):
        DataRecorder(gauges=mock_gauge, thermocouples=[])


def test_data_recorder_raises_error_for_non_pressure_gauge_in_list():
    """
    Test DataRecorder to verify it raises ValueError when gauges list contains
    objects that are not PressureGauge instances.
    """
    with pytest.raises(ValueError, match="gauges must be a list of PressureGauge"):
        DataRecorder(gauges=["not a gauge"], thermocouples=[])


def test_data_recorder_raises_error_for_invalid_thermocouples_type(mock_gauge):
    """
    Test DataRecorder to verify it raises ValueError when thermocouples parameter
    is not a list.
    """
    with pytest.raises(ValueError, match="thermocouples must be a list"):
        DataRecorder(gauges=[mock_gauge], thermocouples="not a list")


def test_data_recorder_raises_error_for_non_thermocouple_in_list(mock_gauge):
    """
    Test DataRecorder to verify it raises ValueError when thermocouples list
    contains objects that are not Thermocouple instances.
    """
    with pytest.raises(
        ValueError, match="thermocouples must be a list of Thermocouple"
    ):
        DataRecorder(gauges=[mock_gauge], thermocouples=["not a thermocouple"])


def test_data_recorder_raises_error_for_non_string_results_dir(mock_gauge):
    """
    Test DataRecorder to verify it raises ValueError when results_dir is not
    a string.
    """
    with pytest.raises(ValueError, match="results_dir must be a string"):
        DataRecorder(gauges=[mock_gauge], thermocouples=[], results_dir=123)


@pytest.mark.parametrize("invalid_run_type", ["invalid", "PERMEATION_EXP", ""])
def test_data_recorder_raises_error_for_invalid_run_type(mock_gauge, invalid_run_type):
    """
    Test DataRecorder to verify it raises ValueError when run_type is not one
    of the valid options: 'permeation_exp', 'leak_test', or 'test_mode'.
    """
    with pytest.raises(ValueError, match="run_type must be one of"):
        DataRecorder(
            gauges=[mock_gauge],
            thermocouples=[],
            run_type=invalid_run_type,
        )


@pytest.mark.parametrize(
    "invalid_material",
    ["304", "stainless steel", "AISI1018", ""],
)
def test_data_recorder_raises_error_for_invalid_sample_material(
    mock_gauge, invalid_material
):
    """
    Test DataRecorder to verify it raises ValueError when sample_material is
    not one of the valid options: '316' or 'AISI 1018'.
    """
    with pytest.raises(ValueError, match="sample_material must be one of"):
        DataRecorder(
            gauges=[mock_gauge],
            thermocouples=[],
            sample_material=invalid_material,
        )


def test_data_recorder_accepts_none_for_sample_material(mock_gauge):
    """
    Test DataRecorder to verify it accepts None as a valid value for
    sample_material parameter.
    """
    recorder = DataRecorder(
        gauges=[mock_gauge],
        thermocouples=[],
        sample_material=None,
    )
    assert recorder.sample_material is None


# =============================================================================
# Tests for test_mode Property
# =============================================================================


def test_data_recorder_test_mode_is_true_when_run_type_is_test_mode(mock_gauge):
    """
    Test DataRecorder test_mode property to verify it returns True when
    run_type is set to 'test_mode'.
    """
    recorder = DataRecorder(
        gauges=[mock_gauge],
        thermocouples=[],
        run_type="test_mode",
    )
    assert recorder.test_mode is True


def test_data_recorder_test_mode_is_false_when_run_type_is_permeation_exp(mock_gauge):
    """
    Test DataRecorder test_mode property to verify it returns False when
    run_type is set to 'permeation_exp'.
    """
    recorder = DataRecorder(
        gauges=[mock_gauge],
        thermocouples=[],
        run_type="permeation_exp",
    )
    assert recorder.test_mode is False


def test_data_recorder_test_mode_is_false_when_run_type_is_leak_test(mock_gauge):
    """
    Test DataRecorder test_mode property to verify it returns False when
    run_type is set to 'leak_test'.
    """
    recorder = DataRecorder(
        gauges=[mock_gauge],
        thermocouples=[],
        run_type="leak_test",
    )
    assert recorder.test_mode is False


# =============================================================================
# Tests for Directory Creation
# =============================================================================


def test_data_recorder_creates_results_directory_structure(recorder):
    """
    Test DataRecorder _create_results_directory to verify it creates a
    directory structure with date and run number subdirectories.
    """
    run_dir = recorder._create_results_directory()
    assert os.path.exists(run_dir)


def test_data_recorder_creates_test_run_directory_in_test_mode(recorder):
    """
    Test DataRecorder _create_results_directory to confirm it creates
    directories with 'test_run_' prefix when in test mode.
    """
    run_dir = recorder._create_results_directory()
    assert "test_run_" in os.path.basename(run_dir)


def test_data_recorder_creates_run_directory_in_normal_mode(temp_dir, mock_gauge):
    """
    Test DataRecorder _create_results_directory to confirm it creates
    directories with 'run_' prefix when not in test mode.
    """
    recorder = DataRecorder(
        gauges=[mock_gauge],
        thermocouples=[],
        results_dir=temp_dir,
        run_type="permeation_exp",
    )
    run_dir = recorder._create_results_directory()
    assert "run_" in os.path.basename(run_dir)


def test_data_recorder_creates_date_subdirectory(recorder):
    """
    Test DataRecorder _create_results_directory to verify it creates a
    date-based subdirectory (MM.DD format) under the results directory.
    """
    run_dir = recorder._create_results_directory()
    # Check that run_dir contains a date directory in the path
    path_parts = run_dir.split(os.sep)
    # Should have at least results_dir / date_dir / run_dir
    assert len(path_parts) >= 3


def test_data_recorder_increments_run_numbers(recorder):
    """
    Test DataRecorder _create_results_directory to confirm it increments run
    numbers when multiple directories are created on the same date.
    """
    run_dir1 = recorder._create_results_directory()
    run_dir2 = recorder._create_results_directory()
    assert run_dir1 != run_dir2


def test_data_recorder_get_next_directory_number_returns_one_for_empty_dir(
    temp_dir, recorder
):
    """
    Test DataRecorder _get_next_directory_number to verify it returns 1 when
    no existing run directories are found.
    """
    next_num = recorder._get_next_directory_number(temp_dir, "test_run")
    assert next_num == 1


def test_data_recorder_get_next_directory_number_increments_existing(
    temp_dir, recorder
):
    """
    Test DataRecorder _get_next_directory_number to verify it returns the
    next sequential number after existing directories.
    """
    # Create some existing directories
    os.makedirs(os.path.join(temp_dir, "test_run_1_10h00"))
    os.makedirs(os.path.join(temp_dir, "test_run_2_11h00"))

    next_num = recorder._get_next_directory_number(temp_dir, "test_run")
    assert next_num == 3


# =============================================================================
# Tests for Metadata File Creation
# =============================================================================


def test_data_recorder_creates_metadata_file(recorder):
    """
    Test DataRecorder _create_metadata_file to verify it creates a JSON
    metadata file in the run directory.
    """
    recorder.run_dir = recorder._create_results_directory()
    metadata_path = recorder._create_metadata_file()
    assert os.path.exists(metadata_path)


def test_data_recorder_metadata_file_is_valid_json(recorder):
    """
    Test DataRecorder _create_metadata_file to ensure the created file
    contains valid JSON data.
    """
    recorder.run_dir = recorder._create_results_directory()
    metadata_path = recorder._create_metadata_file()

    with open(metadata_path) as f:
        metadata = json.load(f)

    assert isinstance(metadata, dict)


def test_data_recorder_metadata_contains_run_info(recorder):
    """
    Test DataRecorder _create_metadata_file to verify the metadata includes
    a 'run_info' section with recording parameters.
    """
    recorder.run_dir = recorder._create_results_directory()
    metadata_path = recorder._create_metadata_file()

    with open(metadata_path) as f:
        metadata = json.load(f)

    assert "run_info" in metadata


def test_data_recorder_metadata_includes_run_type(recorder):
    """
    Test DataRecorder _create_metadata_file to confirm the metadata contains
    the run_type value.
    """
    recorder.run_dir = recorder._create_results_directory()
    metadata_path = recorder._create_metadata_file()

    with open(metadata_path) as f:
        metadata = json.load(f)

    assert metadata["run_info"]["run_type"] == "test_mode"


def test_data_recorder_metadata_includes_recording_interval(recorder):
    """
    Test DataRecorder _create_metadata_file to verify the metadata contains
    the recording_interval value.
    """
    recorder.run_dir = recorder._create_results_directory()
    metadata_path = recorder._create_metadata_file()

    with open(metadata_path) as f:
        metadata = json.load(f)

    assert metadata["run_info"]["recording_interval_seconds"] == 0.1


def test_data_recorder_metadata_includes_gauges_info(recorder):
    """
    Test DataRecorder _create_metadata_file to verify the metadata contains
    gauge configuration information.
    """
    recorder.run_dir = recorder._create_results_directory()
    metadata_path = recorder._create_metadata_file()

    with open(metadata_path) as f:
        metadata = json.load(f)

    assert "gauges" in metadata
    assert len(metadata["gauges"]) == 2


def test_data_recorder_metadata_gauge_contains_name(recorder):
    """
    Test DataRecorder _create_metadata_file to confirm each gauge entry in
    metadata includes the gauge name.
    """
    recorder.run_dir = recorder._create_results_directory()
    metadata_path = recorder._create_metadata_file()

    with open(metadata_path) as f:
        metadata = json.load(f)

    assert metadata["gauges"][0]["name"] == "TestGauge"


def test_data_recorder_metadata_includes_thermocouples_when_present(
    temp_dir, mock_gauge, mock_thermocouple
):
    """
    Test DataRecorder _create_metadata_file to verify the metadata includes
    thermocouple information when thermocouples are configured.
    """
    recorder = DataRecorder(
        gauges=[mock_gauge],
        thermocouples=[mock_thermocouple],
        results_dir=temp_dir,
        run_type="test_mode",
    )
    recorder.run_dir = recorder._create_results_directory()
    metadata_path = recorder._create_metadata_file()

    with open(metadata_path) as f:
        metadata = json.load(f)

    assert "thermocouples" in metadata
    assert len(metadata["thermocouples"]) == 1


# =============================================================================
# Tests for CI Environment Detection
# =============================================================================


def test_data_recorder_detects_ci_environment_with_ci_variable(recorder):
    """
    Test DataRecorder _is_ci_environment to verify it returns True when
    CI environment variable is set.
    """
    with patch.dict(os.environ, {"CI": "true"}):
        assert recorder._is_ci_environment() is True


def test_data_recorder_detects_ci_environment_with_github_actions(recorder):
    """
    Test DataRecorder _is_ci_environment to verify it returns True when
    GITHUB_ACTIONS environment variable is set.
    """
    with patch.dict(os.environ, {"GITHUB_ACTIONS": "true"}):
        assert recorder._is_ci_environment() is True


def test_data_recorder_does_not_detect_ci_in_local_environment(recorder):
    """
    Test DataRecorder _is_ci_environment to verify it returns False when
    no CI environment variables are set.
    """
    # Clear any CI variables that might be set
    ci_vars = [
        "CI",
        "GITHUB_ACTIONS",
        "GITLAB_CI",
        "TRAVIS",
        "CIRCLECI",
        "JENKINS_URL",
        "BUILDKITE",
        "TF_BUILD",
    ]
    clean_env = {k: v for k, v in os.environ.items() if k not in ci_vars}

    with patch.dict(os.environ, clean_env, clear=True):
        assert recorder._is_ci_environment() is False


# =============================================================================
# Tests for Start/Stop Recording
# =============================================================================


def test_data_recorder_start_creates_run_directory(recorder):
    """
    Test DataRecorder start method to verify it creates the run directory
    before starting to record.
    """
    recorder.start()
    time.sleep(0.05)

    assert recorder.run_dir is not None
    assert os.path.exists(recorder.run_dir)

    recorder.stop()


def test_data_recorder_start_creates_backup_directory(recorder):
    """
    Test DataRecorder start method to verify it creates a backup subdirectory
    within the run directory.
    """
    recorder.start()
    time.sleep(0.05)

    assert recorder.backup_dir is not None
    assert os.path.exists(recorder.backup_dir)

    recorder.stop()


def test_data_recorder_start_creates_metadata_file(recorder):
    """
    Test DataRecorder start method to verify it creates a metadata JSON file
    when recording begins.
    """
    recorder.start()
    time.sleep(0.05)

    metadata_path = os.path.join(recorder.run_dir, "run_metadata.json")
    assert os.path.exists(metadata_path)

    recorder.stop()


def test_data_recorder_start_initializes_thread(recorder):
    """
    Test DataRecorder start method to verify it creates and starts a
    recording thread.
    """
    recorder.start()
    time.sleep(0.05)

    assert recorder.thread is not None
    assert isinstance(recorder.thread, threading.Thread)

    recorder.stop()


def test_data_recorder_start_thread_is_alive(recorder):
    """
    Test DataRecorder start method to confirm the recording thread is
    active after starting.
    """
    recorder.start()
    time.sleep(0.05)

    assert recorder.thread.is_alive()

    recorder.stop()


def test_data_recorder_start_resets_valve_times(recorder):
    """
    Test DataRecorder start method to verify it resets all valve event times
    to None at the beginning of a new recording session.
    """
    # Set some valve times
    recorder.v4_close_time = "2025-01-01 12:00:00"
    recorder.v5_close_time = "2025-01-01 12:01:00"

    recorder.start()
    time.sleep(0.05)

    assert recorder.v4_close_time is None
    assert recorder.v5_close_time is None
    assert recorder.v6_close_time is None
    assert recorder.v3_open_time is None

    recorder.stop()


def test_data_recorder_start_resets_current_valve_index(recorder):
    """
    Test DataRecorder start method to verify it resets current_valve_index
    to 0 for a new recording session.
    """
    recorder.current_valve_index = 2

    recorder.start()
    time.sleep(0.05)

    assert recorder.current_valve_index == 0

    recorder.stop()


def test_data_recorder_stop_terminates_thread(recorder):
    """
    Test DataRecorder stop method to verify it terminates the recording
    thread.
    """
    recorder.start()
    time.sleep(0.05)
    assert recorder.thread.is_alive()

    recorder.stop()
    time.sleep(0.15)

    assert not recorder.thread.is_alive()


def test_data_recorder_stop_sets_stop_event(recorder):
    """
    Test DataRecorder stop method to confirm it sets the stop_event to
    signal the recording thread to terminate.
    """
    recorder.start()
    time.sleep(0.05)

    recorder.stop()

    assert recorder.stop_event.is_set()


def test_data_recorder_raises_error_for_duplicate_ain_channels(
    temp_dir, mock_gauge, mock_gauge2
):
    """
    Test DataRecorder start method to verify it raises ValueError when
    multiple gauges are configured with the same AIN channel.
    """
    # Set both gauges to same AIN channel
    mock_gauge.ain_channel = 10
    mock_gauge2.ain_channel = 10

    recorder = DataRecorder(
        gauges=[mock_gauge, mock_gauge2],
        thermocouples=[],
        results_dir=temp_dir,
        run_type="test_mode",
    )

    with pytest.raises(ValueError, match="Duplicate AIN channels"):
        recorder.start()


# =============================================================================
# Tests for Data Recording
# =============================================================================


def test_data_recorder_creates_csv_file_on_start(recorder):
    """
    Test DataRecorder recording to verify it creates a CSV data file when
    recording starts.
    """
    recorder.start()
    time.sleep(0.15)

    csv_path = os.path.join(recorder.run_dir, "shield_data.csv")
    assert os.path.exists(csv_path)

    recorder.stop()


def test_data_recorder_csv_has_header_row(recorder):
    """
    Test DataRecorder recording to verify the CSV file contains a header
    row with column names.
    """
    recorder.start()
    time.sleep(0.15)

    csv_path = os.path.join(recorder.run_dir, "shield_data.csv")
    with open(csv_path) as f:
        lines = f.readlines()

    assert len(lines) >= 1
    assert "RealTimestamp" in lines[0]

    recorder.stop()


def test_data_recorder_csv_includes_gauge_voltage_columns(recorder):
    """
    Test DataRecorder recording to verify the CSV header includes columns
    for each gauge's voltage readings.
    """
    recorder.start()
    time.sleep(0.15)

    csv_path = os.path.join(recorder.run_dir, "shield_data.csv")
    with open(csv_path) as f:
        header = f.readline()

    assert "TestGauge_Voltage (V)" in header
    assert "TestGauge2_Voltage (V)" in header

    recorder.stop()


def test_data_recorder_records_data_rows(recorder):
    """
    Test DataRecorder recording to verify it writes data rows to the CSV
    file during recording.
    """
    recorder.start()
    time.sleep(0.25)  # Allow time for multiple recordings

    csv_path = os.path.join(recorder.run_dir, "shield_data.csv")
    with open(csv_path) as f:
        lines = f.readlines()

    # Should have header + at least 1 data row
    assert len(lines) >= 2

    recorder.stop()


def test_data_recorder_calls_gauge_record_method(recorder, mock_gauge, mock_gauge2):
    """
    Test DataRecorder recording to verify it calls each gauge's
    record_ain_channel_voltage method during data collection.
    """
    recorder.start()
    time.sleep(0.15)

    assert mock_gauge.record_ain_channel_voltage.called
    assert mock_gauge2.record_ain_channel_voltage.called

    recorder.stop()


def test_data_recorder_increments_elapsed_time(recorder):
    """
    Test DataRecorder recording to verify elapsed_time increases during
    the recording session.
    """
    recorder.start()
    time.sleep(0.25)

    elapsed = recorder.elapsed_time
    assert elapsed > 0

    recorder.stop()


def test_data_recorder_records_thermocouple_data(
    temp_dir, mock_gauge, mock_thermocouple
):
    """
    Test DataRecorder recording to verify it records thermocouple voltage
    data when thermocouples are configured.
    """
    recorder = DataRecorder(
        gauges=[mock_gauge],
        thermocouples=[mock_thermocouple],
        results_dir=temp_dir,
        run_type="test_mode",
        recording_interval=0.1,
    )

    recorder.start()
    time.sleep(0.15)

    csv_path = os.path.join(recorder.run_dir, "shield_data.csv")
    with open(csv_path) as f:
        header = f.readline()

    assert "TestThermocouple_Voltage (mV)" in header
    assert "Local_temperature (C)" in header

    recorder.stop()


def test_data_recorder_calls_thermocouple_record_method(
    temp_dir, mock_gauge, mock_thermocouple
):
    """
    Test DataRecorder recording to verify it calls each thermocouple's
    record_ain_channel_voltage method during data collection.
    """
    recorder = DataRecorder(
        gauges=[mock_gauge],
        thermocouples=[mock_thermocouple],
        results_dir=temp_dir,
        run_type="test_mode",
        recording_interval=0.1,
    )

    recorder.start()
    time.sleep(0.15)

    assert mock_thermocouple.record_ain_channel_voltage.called

    recorder.stop()


# =============================================================================
# Tests for Backup Data Creation
# =============================================================================


def test_data_recorder_creates_backup_files(recorder):
    """
    Test DataRecorder backup functionality to verify it creates backup
    CSV files at regular intervals.
    """
    recorder.start()
    time.sleep(0.5)  # Allow time for backup to occur (backup_interval=0.3)

    backup_files = os.listdir(recorder.backup_dir)
    assert len(backup_files) > 0

    recorder.stop()


def test_data_recorder_backup_files_are_csv(recorder):
    """
    Test DataRecorder backup functionality to verify backup files have
    .csv extension.
    """
    recorder.start()
    time.sleep(0.5)

    backup_files = os.listdir(recorder.backup_dir)
    csv_files = [f for f in backup_files if f.endswith(".csv")]
    assert len(csv_files) > 0

    recorder.stop()


def test_data_recorder_backup_files_are_numbered(recorder):
    """
    Test DataRecorder backup functionality to verify backup files include
    sequential numbers in their names.
    """
    recorder.start()
    time.sleep(0.5)

    backup_files = os.listdir(recorder.backup_dir)
    assert any("backup_data_1.csv" in f for f in backup_files)

    recorder.stop()


# =============================================================================
# Tests for Metadata Updates
# =============================================================================


def test_data_recorder_updates_metadata_with_valve_time(recorder):
    """
    Test DataRecorder _update_metadata_with_valve_time to verify it adds
    valve event timestamps to the metadata file.
    """
    recorder.run_dir = recorder._create_results_directory()
    recorder._create_metadata_file()

    test_timestamp = "2025-01-01 12:00:00.123"
    recorder._update_metadata_with_valve_time("v4_close_time", test_timestamp)

    metadata_path = os.path.join(recorder.run_dir, "run_metadata.json")
    with open(metadata_path) as f:
        metadata = json.load(f)

    assert metadata["run_info"]["v4_close_time"] == test_timestamp


def test_data_recorder_metadata_update_preserves_existing_data(recorder):
    """
    Test DataRecorder _update_metadata_with_valve_time to confirm it preserves
    existing metadata when adding new valve event times.
    """
    recorder.run_dir = recorder._create_results_directory()
    recorder._create_metadata_file()

    # Get original run_type
    metadata_path = os.path.join(recorder.run_dir, "run_metadata.json")
    with open(metadata_path) as f:
        original_metadata = json.load(f)
    original_run_type = original_metadata["run_info"]["run_type"]

    # Update with valve time
    recorder._update_metadata_with_valve_time("v4_close_time", "2025-01-01 12:00:00")

    # Check run_type still exists
    with open(metadata_path) as f:
        updated_metadata = json.load(f)

    assert updated_metadata["run_info"]["run_type"] == original_run_type


# =============================================================================
# Tests for Helper Methods
# =============================================================================


def test_data_recorder_get_current_timestamp_returns_string(recorder):
    """
    Test DataRecorder _get_current_timestamp to verify it returns a string
    formatted timestamp.
    """
    timestamp = recorder._get_current_timestamp()
    assert isinstance(timestamp, str)


def test_data_recorder_timestamp_includes_milliseconds(recorder):
    """
    Test DataRecorder _get_current_timestamp to verify the timestamp string
    includes millisecond precision.
    """
    timestamp = recorder._get_current_timestamp()
    # Format: "YYYY-MM-DD HH:MM:SS.mmm"
    assert "." in timestamp
    parts = timestamp.split(".")
    assert len(parts) == 2
    assert len(parts[1]) == 3  # 3 digits for milliseconds


def test_data_recorder_collect_measurement_data_returns_dict(recorder):
    """
    Test DataRecorder _collect_measurement_data to verify it returns a
    dictionary containing measurement data.
    """
    timestamp = "2025-01-01 12:00:00.000"
    data = recorder._collect_measurement_data(labjack=None, timestamp=timestamp)
    assert isinstance(data, dict)


def test_data_recorder_measurement_data_includes_timestamp(recorder):
    """
    Test DataRecorder _collect_measurement_data to verify the returned
    dictionary includes the RealTimestamp field.
    """
    timestamp = "2025-01-01 12:00:00.000"
    data = recorder._collect_measurement_data(labjack=None, timestamp=timestamp)
    assert "RealTimestamp" in data
    assert data["RealTimestamp"] == timestamp


def test_data_recorder_measurement_data_includes_gauge_voltages(recorder):
    """
    Test DataRecorder _collect_measurement_data to verify the returned
    dictionary includes voltage readings for all configured gauges.
    """
    timestamp = "2025-01-01 12:00:00.000"
    data = recorder._collect_measurement_data(labjack=None, timestamp=timestamp)

    assert "TestGauge_Voltage (V)" in data
    assert "TestGauge2_Voltage (V)" in data


# =============================================================================
# Tests for Keyboard Monitoring
# =============================================================================


def test_data_recorder_monitor_keyboard_returns_early_in_ci(recorder):
    """
    Test DataRecorder _monitor_keyboard to verify it returns immediately
    without setting up keyboard listeners when in CI environment.
    """
    with patch.dict(os.environ, {"CI": "true"}):
        # Should not raise any errors and return immediately
        result = recorder._monitor_keyboard()
        assert result is None


def test_data_recorder_monitor_keyboard_handles_invalid_valve_index(recorder):
    """
    Test DataRecorder _monitor_keyboard returns early with warning when
    current_valve_index is out of bounds.
    """
    recorder.current_valve_index = 10  # Out of bounds
    result = recorder._monitor_keyboard()
    assert result is None


def test_data_recorder_monitor_keyboard_handles_negative_valve_index(recorder):
    """
    Test DataRecorder _monitor_keyboard returns early when
    current_valve_index is negative.
    """
    # Negative index would access last element without bounds check
    recorder.current_valve_index = -1
    result = recorder._monitor_keyboard()
    assert result is None


# =============================================================================
# Tests for Backup Data Writing
# =============================================================================


def test_data_recorder_write_backup_data_skips_empty_data(recorder):
    """
    Test DataRecorder _write_backup_data to verify it returns early without
    creating a file when given empty data list.
    """
    recorder.run_dir = recorder._create_results_directory()
    recorder.backup_dir = os.path.join(recorder.run_dir, "backup")
    os.makedirs(recorder.backup_dir, exist_ok=True)

    # Call with empty list
    recorder._write_backup_data(
        filename="test_data",
        recent_data=[],
        backup_number=1,
    )

    # No backup file should be created
    backup_files = os.listdir(recorder.backup_dir)
    assert len(backup_files) == 0


def test_data_recorder_write_backup_data_creates_file_with_data(recorder):
    """
    Test DataRecorder _write_backup_data to verify it creates a backup CSV
    file when given non-empty data list.
    """
    recorder.run_dir = recorder._create_results_directory()
    recorder.backup_dir = os.path.join(recorder.run_dir, "backup")
    os.makedirs(recorder.backup_dir, exist_ok=True)

    test_data = [
        {"timestamp": "2025-01-01 12:00:00", "voltage": 5.0},
        {"timestamp": "2025-01-01 12:00:01", "voltage": 5.1},
    ]

    recorder._write_backup_data(
        filename="test_data",
        recent_data=test_data,
        backup_number=1,
    )

    # Backup file should be created
    backup_files = os.listdir(recorder.backup_dir)
    assert len(backup_files) == 1
    assert "test_data_backup_data_1.csv" in backup_files[0]


def test_data_recorder_write_backup_data_includes_all_rows(recorder):
    """
    Test DataRecorder _write_backup_data to verify the created backup file
    contains all provided data rows.
    """
    recorder.run_dir = recorder._create_results_directory()
    recorder.backup_dir = os.path.join(recorder.run_dir, "backup")
    os.makedirs(recorder.backup_dir, exist_ok=True)

    test_data = [
        {"timestamp": "2025-01-01 12:00:00", "voltage": 5.0},
        {"timestamp": "2025-01-01 12:00:01", "voltage": 5.1},
        {"timestamp": "2025-01-01 12:00:02", "voltage": 5.2},
    ]

    recorder._write_backup_data(
        filename="test_data",
        recent_data=test_data,
        backup_number=1,
    )

    backup_path = os.path.join(recorder.backup_dir, "test_data_backup_data_1.csv")
    with open(backup_path) as f:
        lines = f.readlines()

    # Should have header + 3 data rows
    assert len(lines) == 4


# =============================================================================
# Tests for Single Measurement Writing
# =============================================================================


def test_data_recorder_write_single_measurement_creates_file_on_first_write(recorder):
    """
    Test DataRecorder _write_single_measurement to verify it creates a new
    CSV file with header when is_first=True.
    """
    recorder.run_dir = recorder._create_results_directory()

    test_data = {"timestamp": "2025-01-01 12:00:00", "voltage": 5.0}

    recorder._write_single_measurement(
        filename="test_data",
        data_row=test_data,
        is_first=True,
    )

    csv_path = os.path.join(recorder.run_dir, "test_data.csv")
    assert os.path.exists(csv_path)


def test_data_recorder_write_single_measurement_includes_header_on_first_write(
    recorder,
):
    """
    Test DataRecorder _write_single_measurement to verify the CSV file
    includes a header row when is_first=True.
    """
    recorder.run_dir = recorder._create_results_directory()

    test_data = {"timestamp": "2025-01-01 12:00:00", "voltage": 5.0}

    recorder._write_single_measurement(
        filename="test_data",
        data_row=test_data,
        is_first=True,
    )

    csv_path = os.path.join(recorder.run_dir, "test_data.csv")
    with open(csv_path) as f:
        first_line = f.readline()

    assert "timestamp" in first_line
    assert "voltage" in first_line


def test_data_recorder_write_single_measurement_appends_without_header(recorder):
    """
    Test DataRecorder _write_single_measurement to verify it appends data
    without adding another header when is_first=False.
    """
    recorder.run_dir = recorder._create_results_directory()

    test_data1 = {"timestamp": "2025-01-01 12:00:00", "voltage": 5.0}
    test_data2 = {"timestamp": "2025-01-01 12:00:01", "voltage": 5.1}

    # Write first row with header
    recorder._write_single_measurement(
        filename="test_data",
        data_row=test_data1,
        is_first=True,
    )

    # Append second row without header
    recorder._write_single_measurement(
        filename="test_data",
        data_row=test_data2,
        is_first=False,
    )

    csv_path = os.path.join(recorder.run_dir, "test_data.csv")
    with open(csv_path) as f:
        lines = f.readlines()

    # Should have: 1 header + 2 data rows = 3 lines total
    assert len(lines) == 3


# =============================================================================
# Tests for Recording Session Initialization
# =============================================================================


def test_data_recorder_initialize_recording_session_resets_elapsed_time(recorder):
    """
    Test DataRecorder _initialize_recording_session to verify it resets
    elapsed_time to 0.0 at the start of recording.
    """
    recorder.elapsed_time = 10.5  # Set to non-zero value

    recorder._initialize_recording_session()

    assert recorder.elapsed_time == 0.0


def test_data_recorder_initialize_recording_session_sets_start_time(recorder):
    """
    Test DataRecorder _initialize_recording_session to verify it sets
    start_time to the current datetime.
    """
    recorder._initialize_recording_session()

    assert recorder.start_time is not None
    assert isinstance(recorder.start_time, datetime)


# =============================================================================
# Tests for LabJack Initialization in Test Mode
# =============================================================================


def test_data_recorder_initialize_labjack_returns_none_in_test_mode(recorder):
    """
    Test DataRecorder _initialize_labjack to verify it returns None when
    recorder is in test mode, avoiding hardware initialization.
    """
    result = recorder._initialize_labjack()
    assert result is None


def test_data_recorder_initialize_labjack_test_mode_property_check():
    """
    Test DataRecorder _initialize_labjack to confirm test_mode property
    correctly determines whether to initialize hardware.
    """
    mock_gauge = Mock(spec=PressureGauge)
    mock_gauge.name = "TestGauge"
    mock_gauge.voltage_data = [5.0]
    mock_gauge.ain_channel = 10

    # Test mode should return None
    test_recorder = DataRecorder(
        gauges=[mock_gauge],
        thermocouples=[],
        run_type="test_mode",
    )
    assert test_recorder._initialize_labjack() is None

    # Note: We don't test non-test mode as it requires actual hardware
