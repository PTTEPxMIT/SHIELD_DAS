"""Tests for Dataset class in SHIELD DAS.

This module tests the Dataset class functionality including initialization,
properties, metadata loading, CSV reading, and data processing.
"""

import json
import os
from unittest.mock import patch

import numpy as np
import pytest

from shield_das.dataset import Dataset

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def temp_dataset_dir(tmp_path):
    """Create a temporary dataset directory with required files.

    Returns:
        Path to temporary dataset directory
    """
    dataset_dir = tmp_path / "test_dataset"
    dataset_dir.mkdir()
    return dataset_dir


@pytest.fixture
def sample_metadata_v1_3():
    """Create sample metadata dictionary for version 1.3.

    Returns:
        Dictionary with sample metadata
    """
    return {
        "version": "1.3",
        "run_info": {
            "date": "2025-08-20",
            "start_time": "2025-08-20 10:58:01.123456",
            "run_type": "permeation_exp",
            "furnace_setpoint": 600.0,
            "sample_material": "316",
            "sample_thickness": 0.00088,
            "v5_close_time": "2025-08-20 10:58:05.428000",
            "v6_close_time": "2025-08-20 10:58:05.844000",
            "v7_close_time": "2025-08-20 10:58:07.128000",
            "v3_open_time": "2025-08-20 10:58:08.044000",
        },
        "gauges": [
            {
                "name": "Baratron626D_1KT",
                "type": "Baratron626D_Gauge",
                "ain_channel": 4,
                "gauge_location": "upstream",
                "full_scale_torr": 1000.0,
            },
            {
                "name": "Baratron626D_1T",
                "type": "Baratron626D_Gauge",
                "ain_channel": 6,
                "gauge_location": "downstream",
                "full_scale_torr": 1.0,
            },
        ],
        "thermocouples": [{"name": "Thermocouple_furnace"}],
    }


@pytest.fixture
def sample_csv_data():
    """Create sample CSV data string.

    Returns:
        String representing CSV content with headers and data
    """
    return (
        "RealTimestamp,Baratron626D_1KT_Voltage_V,Baratron626D_1T_Voltage_V,"
        "Local_temperature_C,Thermocouple_furnace_Voltage_mV\n"
        "2025-08-20 10:58:01.000000,0.5,0.2,25.0,2.5\n"
        "2025-08-20 10:58:01.500000,1.0,0.4,25.5,3.0\n"
        "2025-08-20 10:58:02.000000,1.5,0.6,26.0,3.5\n"
        "2025-08-20 10:58:02.500000,2.0,0.8,26.5,4.0\n"
        "2025-08-20 10:58:03.000000,2.5,1.0,27.0,4.5\n"
    )


@pytest.fixture
def dataset_with_files(temp_dataset_dir, sample_metadata_v1_3, sample_csv_data):
    """Create a dataset directory with metadata and CSV files.

    Args:
        temp_dataset_dir: Temporary directory fixture
        sample_metadata_v1_3: Sample metadata fixture
        sample_csv_data: Sample CSV data fixture

    Returns:
        Path to dataset directory with files
    """
    # Write metadata file
    metadata_path = temp_dataset_dir / "run_metadata.json"
    metadata_path.write_text(json.dumps(sample_metadata_v1_3, indent=2))

    # Write CSV file
    csv_path = temp_dataset_dir / "shield_data.csv"
    csv_path.write_text(sample_csv_data)

    return temp_dataset_dir


# =============================================================================
# Tests for Dataset Initialization
# =============================================================================


def test_dataset_initializes_with_path_and_name():
    """
    Test Dataset initialization to verify it correctly stores path and name
    parameters.
    """
    path = "/test/path"
    name = "Test Dataset"
    dataset = Dataset(path=path, name=name)

    assert dataset.path == path
    assert dataset.name == name


def test_dataset_initializes_all_data_attributes_to_none():
    """
    Test Dataset initialization to confirm all data arrays are set to None
    before data processing.
    """
    dataset = Dataset(path="/test/path", name="Test")

    assert dataset.time_data is None
    assert dataset.upstream_pressure is None
    assert dataset.upstream_error is None
    assert dataset.downstream_pressure is None
    assert dataset.downstream_error is None
    assert dataset.valve_times is None


def test_dataset_initializes_metadata_attributes_to_none():
    """
    Test Dataset initialization to verify all metadata attributes are set
    to None before processing.
    """
    dataset = Dataset(path="/test/path", name="Test")

    assert dataset.colour is None
    assert dataset.sample_material is None
    assert dataset.sample_thickness is None
    assert dataset.furnace_setpoint is None


def test_dataset_initializes_temperature_attributes_to_none():
    """
    Test Dataset initialization to confirm temperature-related attributes
    are set to None before processing.
    """
    dataset = Dataset(path="/test/path", name="Test")

    assert dataset.local_temperature_data is None
    assert dataset.thermocouple_data is None
    assert dataset.thermocouple_name is None


def test_dataset_initializes_live_data_to_false():
    """
    Test Dataset initialization to verify live_data is set to False by default.
    """
    dataset = Dataset(path="/test/path", name="Test")
    assert dataset.live_data is False


# =============================================================================
# Tests for live_data Property
# =============================================================================


def test_dataset_live_data_getter_returns_value():
    """
    Test Dataset live_data property getter to verify it returns the current
    live_data status.
    """
    dataset = Dataset(path="/test/path", name="Test")
    assert dataset.live_data is False


def test_dataset_live_data_setter_sets_value():
    """
    Test Dataset live_data property setter to verify it correctly updates
    the live_data status.
    """
    dataset = Dataset(path="/test/path", name="Test")
    dataset.live_data = True
    assert dataset.live_data is True


@pytest.mark.parametrize("value", [True, False])
def test_dataset_live_data_accepts_boolean_values(value):
    """
    Test Dataset live_data property to confirm it accepts both True and False
    boolean values.
    """
    dataset = Dataset(path="/test/path", name="Test")
    dataset.live_data = value
    assert dataset.live_data == value


# =============================================================================
# Tests for dataset_file Property
# =============================================================================


def test_dataset_file_property_returns_csv_path():
    """
    Test Dataset dataset_file property to verify it returns the correct path
    to shield_data.csv.
    """
    path = "/test/path"
    dataset = Dataset(path=path, name="Test")
    expected = os.path.join(path, "shield_data.csv")
    assert dataset.dataset_file == expected


def test_dataset_file_property_uses_os_path_join():
    """
    Test Dataset dataset_file property to confirm it uses os.path.join for
    proper cross-platform path construction.
    """
    path = "/test/path"
    dataset = Dataset(path=path, name="Test")
    assert os.sep in dataset.dataset_file or dataset.dataset_file.endswith(
        "shield_data.csv"
    )


# =============================================================================
# Tests for metadata Property
# =============================================================================


def test_dataset_metadata_property_loads_json_file(dataset_with_files):
    """
    Test Dataset metadata property to verify it successfully loads the
    run_metadata.json file.
    """
    dataset = Dataset(path=str(dataset_with_files), name="Test")
    metadata = dataset.metadata

    assert isinstance(metadata, dict)
    assert "version" in metadata


def test_dataset_metadata_property_validates_version_1_3(dataset_with_files):
    """
    Test Dataset metadata property to confirm it accepts version 1.3
    metadata without raising errors.
    """
    dataset = Dataset(path=str(dataset_with_files), name="Test")
    metadata = dataset.metadata
    assert metadata["version"] == "1.3"


def test_dataset_metadata_property_raises_error_for_wrong_version(temp_dataset_dir):
    """
    Test Dataset metadata property to verify it raises ValueError when
    metadata version is not 1.3.
    """
    # Create metadata with wrong version
    wrong_metadata = {"version": "1.0", "run_info": {}}
    metadata_path = temp_dataset_dir / "run_metadata.json"
    metadata_path.write_text(json.dumps(wrong_metadata))

    dataset = Dataset(path=str(temp_dataset_dir), name="Test")

    with pytest.raises(ValueError, match="Unsupported metadata version: 1.0"):
        _ = dataset.metadata


@pytest.mark.parametrize("wrong_version", ["1.0", "1.1", "1.2", "2.0", "0.9"])
def test_dataset_metadata_rejects_unsupported_versions(temp_dataset_dir, wrong_version):
    """
    Test Dataset metadata property to confirm it rejects various unsupported
    version numbers (1.0, 1.1, 1.2, 2.0, 0.9).
    """
    metadata = {"version": wrong_version, "run_info": {}}
    metadata_path = temp_dataset_dir / "run_metadata.json"
    metadata_path.write_text(json.dumps(metadata))

    dataset = Dataset(path=str(temp_dataset_dir), name="Test")

    expected_msg = f"Unsupported metadata version: {wrong_version}"
    with pytest.raises(ValueError, match=expected_msg):
        _ = dataset.metadata


def test_dataset_metadata_error_message_includes_version(temp_dataset_dir):
    """
    Test Dataset metadata property to verify the error message includes the
    unsupported version number and suggests using version 1.3.
    """
    metadata = {"version": "2.5", "run_info": {}}
    metadata_path = temp_dataset_dir / "run_metadata.json"
    metadata_path.write_text(json.dumps(metadata))

    dataset = Dataset(path=str(temp_dataset_dir), name="Test")

    with pytest.raises(ValueError, match="Only version 1.3 is supported"):
        _ = dataset.metadata


def test_dataset_metadata_property_returns_full_metadata_dict(dataset_with_files):
    """
    Test Dataset metadata property to confirm it returns the complete metadata
    dictionary with all sections.
    """
    dataset = Dataset(path=str(dataset_with_files), name="Test")
    metadata = dataset.metadata

    assert "version" in metadata
    assert "run_info" in metadata
    assert "gauges" in metadata
    assert "thermocouples" in metadata


# =============================================================================
# Tests for read_csv_file Method
# =============================================================================


def test_dataset_read_csv_file_returns_numpy_array(dataset_with_files):
    """
    Test Dataset read_csv_file to verify it returns a numpy structured array
    from the CSV file.
    """
    dataset = Dataset(path=str(dataset_with_files), name="Test")
    csv_path = dataset.dataset_file
    data = dataset.read_csv_file(csv_path)

    assert isinstance(data, np.ndarray)


def test_dataset_read_csv_file_has_named_columns(dataset_with_files):
    """
    Test Dataset read_csv_file to confirm the returned array has named
    fields from the CSV header.
    """
    dataset = Dataset(path=str(dataset_with_files), name="Test")
    csv_path = dataset.dataset_file
    data = dataset.read_csv_file(csv_path)

    assert "RealTimestamp" in data.dtype.names
    assert "Baratron626D_1KT_Voltage_V" in data.dtype.names


def test_dataset_read_csv_file_reads_correct_number_of_rows(dataset_with_files):
    """
    Test Dataset read_csv_file to verify it reads all data rows from the
    CSV file (5 rows in sample data).
    """
    dataset = Dataset(path=str(dataset_with_files), name="Test")
    csv_path = dataset.dataset_file
    data = dataset.read_csv_file(csv_path)

    assert len(data) == 5


def test_dataset_read_csv_file_uses_comma_delimiter(dataset_with_files):
    """
    Test Dataset read_csv_file to confirm it correctly parses comma-delimited
    CSV data.
    """
    dataset = Dataset(path=str(dataset_with_files), name="Test")
    csv_path = dataset.dataset_file
    data = dataset.read_csv_file(csv_path)

    # Should successfully parse multiple columns
    assert len(data.dtype.names) >= 3


def test_dataset_read_csv_file_uses_utf8_encoding(dataset_with_files):
    """
    Test Dataset read_csv_file to verify it uses UTF-8 encoding when reading
    the CSV file.
    """
    dataset = Dataset(path=str(dataset_with_files), name="Test")
    csv_path = dataset.dataset_file

    # Should not raise encoding errors
    data = dataset.read_csv_file(csv_path)
    assert data is not None


# =============================================================================
# Tests for process_data Method - Time Conversion
# =============================================================================


def test_dataset_process_data_converts_timestamps_to_relative_time(dataset_with_files):
    """
    Test Dataset process_data to verify it converts absolute timestamps to
    relative time in seconds from start.
    """
    dataset = Dataset(path=str(dataset_with_files), name="Test")
    dataset.process_data()

    assert dataset.time_data is not None
    assert len(dataset.time_data) == 5
    assert dataset.time_data[0] == pytest.approx(0.0, abs=0.01)


def test_dataset_process_data_time_is_monotonically_increasing(dataset_with_files):
    """
    Test Dataset process_data to confirm time_data is monotonically increasing
    with proper 0.5 second intervals.
    """
    dataset = Dataset(path=str(dataset_with_files), name="Test")
    dataset.process_data()

    # Check differences between consecutive times
    time_diffs = np.diff(dataset.time_data)
    assert np.all(time_diffs >= 0)
    assert time_diffs[0] == pytest.approx(0.5, abs=0.01)


def test_dataset_process_data_time_starts_at_zero(dataset_with_files):
    """
    Test Dataset process_data to verify the first time value is approximately
    zero seconds.
    """
    dataset = Dataset(path=str(dataset_with_files), name="Test")
    dataset.process_data()

    assert dataset.time_data[0] == pytest.approx(0.0, abs=1e-6)


def test_dataset_process_data_time_is_numpy_array(dataset_with_files):
    """
    Test Dataset process_data to confirm time_data is stored as a numpy array.
    """
    dataset = Dataset(path=str(dataset_with_files), name="Test")
    dataset.process_data()

    assert isinstance(dataset.time_data, np.ndarray)


# =============================================================================
# Tests for process_data Method - Gauge Processing
# =============================================================================


@patch("shield_das.dataset.voltage_to_pressure")
def test_dataset_process_data_calls_voltage_to_pressure(mock_v2p, dataset_with_files):
    """
    Test Dataset process_data to verify it calls voltage_to_pressure for
    gauge voltage conversion.
    """
    mock_v2p.return_value = np.array([100.0, 200.0, 300.0, 400.0, 500.0])

    dataset = Dataset(path=str(dataset_with_files), name="Test")
    dataset.process_data()

    assert mock_v2p.called


@patch("shield_das.dataset.voltage_to_pressure")
def test_dataset_process_data_passes_correct_full_scale_to_converter(
    mock_v2p, dataset_with_files
):
    """
    Test Dataset process_data to confirm it passes the correct full_scale_torr
    parameter from metadata to voltage_to_pressure.
    """
    mock_v2p.return_value = np.array([100.0, 200.0, 300.0, 400.0, 500.0])

    dataset = Dataset(path=str(dataset_with_files), name="Test")
    dataset.process_data()

    # Check if called with full_scale_torr keyword argument
    calls = mock_v2p.call_args_list
    assert any("full_scale_torr" in str(call) for call in calls), (
        "full_scale_torr not passed to voltage_to_pressure"
    )


@patch("shield_das.dataset.voltage_to_pressure")
@patch("shield_das.dataset.calculate_error_on_pressure_reading")
def test_dataset_process_data_assigns_upstream_pressure(
    mock_error, mock_v2p, dataset_with_files
):
    """
    Test Dataset process_data to verify it correctly assigns upstream pressure
    data from gauges marked as 'upstream'.
    """
    mock_v2p.return_value = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
    mock_error.return_value = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    dataset = Dataset(path=str(dataset_with_files), name="Test")
    dataset.process_data()

    assert dataset.upstream_pressure is not None
    assert len(dataset.upstream_pressure) == 5


@patch("shield_das.dataset.voltage_to_pressure")
@patch("shield_das.dataset.calculate_error_on_pressure_reading")
def test_dataset_process_data_assigns_downstream_pressure(
    mock_error, mock_v2p, dataset_with_files
):
    """
    Test Dataset process_data to verify it correctly assigns downstream pressure
    data from gauges marked as 'downstream'.
    """
    mock_v2p.return_value = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    mock_error.return_value = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

    dataset = Dataset(path=str(dataset_with_files), name="Test")
    dataset.process_data()

    assert dataset.downstream_pressure is not None
    assert len(dataset.downstream_pressure) == 5


@patch("shield_das.dataset.voltage_to_pressure")
@patch("shield_das.dataset.calculate_error_on_pressure_reading")
def test_dataset_process_data_calculates_pressure_errors(
    mock_error, mock_v2p, dataset_with_files
):
    """
    Test Dataset process_data to confirm it calculates pressure errors for
    both upstream and downstream gauges.
    """
    mock_v2p.return_value = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
    mock_error.return_value = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    dataset = Dataset(path=str(dataset_with_files), name="Test")
    dataset.process_data()

    assert dataset.upstream_error is not None
    assert dataset.downstream_error is not None
    assert mock_error.call_count == 2


# =============================================================================
# Tests for process_data Method - Temperature Processing
# =============================================================================


@patch("shield_das.dataset.voltage_to_temperature")
@patch("shield_das.dataset.voltage_to_pressure")
@patch("shield_das.dataset.calculate_error_on_pressure_reading")
def test_dataset_process_data_loads_local_temperature(
    mock_error, mock_v2p, mock_v2t, dataset_with_files
):
    """
    Test Dataset process_data to verify it loads local temperature data from
    the CSV file when available.
    """
    mock_v2p.return_value = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
    mock_error.return_value = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    mock_v2t.return_value = np.array([600.0, 610.0, 620.0, 630.0, 640.0])

    dataset = Dataset(path=str(dataset_with_files), name="Test")
    dataset.process_data()

    assert dataset.local_temperature_data is not None
    assert len(dataset.local_temperature_data) == 5


@patch("shield_das.dataset.voltage_to_temperature")
@patch("shield_das.dataset.voltage_to_pressure")
@patch("shield_das.dataset.calculate_error_on_pressure_reading")
def test_dataset_process_data_loads_thermocouple_data(
    mock_error, mock_v2p, mock_v2t, dataset_with_files
):
    """
    Test Dataset process_data to confirm it loads and converts thermocouple
    voltage data to temperature.
    """
    mock_v2p.return_value = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
    mock_error.return_value = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    mock_v2t.return_value = np.array([600.0, 610.0, 620.0, 630.0, 640.0])

    dataset = Dataset(path=str(dataset_with_files), name="Test")
    dataset.process_data()

    assert dataset.thermocouple_data is not None
    assert len(dataset.thermocouple_data) == 5


@patch("shield_das.dataset.voltage_to_temperature")
@patch("shield_das.dataset.voltage_to_pressure")
@patch("shield_das.dataset.calculate_error_on_pressure_reading")
def test_dataset_process_data_sets_thermocouple_name(
    mock_error, mock_v2p, mock_v2t, dataset_with_files
):
    """
    Test Dataset process_data to verify it stores the thermocouple name from
    metadata.
    """
    mock_v2p.return_value = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
    mock_error.return_value = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    mock_v2t.return_value = np.array([600.0, 610.0, 620.0, 630.0, 640.0])

    dataset = Dataset(path=str(dataset_with_files), name="Test")
    dataset.process_data()

    assert dataset.thermocouple_name == "Thermocouple_furnace"


@patch("shield_das.dataset.voltage_to_pressure")
@patch("shield_das.dataset.calculate_error_on_pressure_reading")
def test_dataset_process_data_handles_missing_temperature_data(
    mock_error, mock_v2p, temp_dataset_dir, sample_metadata_v1_3
):
    """
    Test Dataset process_data to confirm it gracefully handles missing
    temperature data by setting temperature attributes to None.
    """
    mock_v2p.return_value = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
    mock_error.return_value = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    # Create CSV without temperature columns
    csv_data = (
        "RealTimestamp,Baratron626D_1KT_Voltage_V,Baratron626D_1T_Voltage_V\n"
        "2025-08-20 10:58:01.000000,0.5,0.2\n"
        "2025-08-20 10:58:01.500000,1.0,0.4\n"
        "2025-08-20 10:58:02.000000,1.5,0.6\n"
        "2025-08-20 10:58:02.500000,2.0,0.8\n"
        "2025-08-20 10:58:03.000000,2.5,1.0\n"
    )
    metadata_path = temp_dataset_dir / "run_metadata.json"
    metadata_path.write_text(json.dumps(sample_metadata_v1_3, indent=2))
    csv_path = temp_dataset_dir / "shield_data.csv"
    csv_path.write_text(csv_data)

    dataset = Dataset(path=str(temp_dataset_dir), name="Test")
    dataset.process_data()

    assert dataset.local_temperature_data is None
    assert dataset.thermocouple_data is None
    assert dataset.thermocouple_name is None


# =============================================================================
# Tests for process_data Method - Valve Time Extraction
# =============================================================================


def test_dataset_process_data_extracts_valve_times(dataset_with_files):
    """
    Test Dataset process_data to verify it extracts valve event times from
    metadata and converts them to relative seconds.
    """
    dataset = Dataset(path=str(dataset_with_files), name="Test")
    dataset.process_data()

    assert dataset.valve_times is not None
    assert isinstance(dataset.valve_times, dict)


def test_dataset_process_data_valve_times_are_relative(dataset_with_files):
    """
    Test Dataset process_data to confirm valve times are calculated relative
    to the start time in seconds.
    """
    dataset = Dataset(path=str(dataset_with_files), name="Test")
    dataset.process_data()

    # All valve times should be positive and reasonable
    for key, value in dataset.valve_times.items():
        assert value >= 0
        assert value < 100  # Should be within reasonable time


def test_dataset_process_data_extracts_correct_valve_keys(dataset_with_files):
    """
    Test Dataset process_data to verify it extracts the correct valve event
    keys (v5_close_time, v6_close_time, v7_close_time, v3_open_time).
    """
    dataset = Dataset(path=str(dataset_with_files), name="Test")
    dataset.process_data()

    expected_keys = ["v5_close_time", "v6_close_time", "v7_close_time", "v3_open_time"]
    for key in expected_keys:
        assert key in dataset.valve_times


def test_dataset_process_data_valve_times_match_metadata(dataset_with_files):
    """
    Test Dataset process_data to confirm valve times are correctly calculated
    from the start_time baseline.
    """
    dataset = Dataset(path=str(dataset_with_files), name="Test")
    dataset.process_data()

    # v5_close_time is at 10:58:05.428, start is 10:58:01.123456
    # Difference should be approximately 4.3 seconds
    assert dataset.valve_times["v5_close_time"] == pytest.approx(4.3, abs=0.1)


def test_dataset_process_data_handles_start_time_without_microseconds(
    temp_dataset_dir, sample_csv_data
):
    """
    Test Dataset process_data to verify it handles start_time strings without
    microseconds (YYYY-MM-DD HH:MM:SS format).
    """
    # Create metadata with start time lacking microseconds
    metadata = {
        "version": "1.3",
        "run_info": {
            "start_time": "2025-08-20 10:58:01",
            "furnace_setpoint": 600.0,
            "sample_material": "316",
            "sample_thickness": 0.00088,
            "v5_close_time": "2025-08-20 10:58:05.428000",
        },
        "gauges": [
            {
                "name": "Baratron626D_1KT",
                "type": "Baratron626D_Gauge",
                "gauge_location": "upstream",
                "full_scale_torr": 1000.0,
            }
        ],
    }

    metadata_path = temp_dataset_dir / "run_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))
    csv_path = temp_dataset_dir / "shield_data.csv"
    csv_path.write_text(sample_csv_data)

    dataset = Dataset(path=str(temp_dataset_dir), name="Test")
    dataset.process_data()

    # Should successfully parse and calculate valve times
    assert "v5_close_time" in dataset.valve_times


# =============================================================================
# Tests for process_data Method - Metadata Extraction
# =============================================================================


def test_dataset_process_data_extracts_furnace_setpoint(dataset_with_files):
    """
    Test Dataset process_data to verify it extracts furnace setpoint from
    metadata and converts Celsius to Kelvin.
    """
    dataset = Dataset(path=str(dataset_with_files), name="Test")
    dataset.process_data()

    # 600°C + 273.15 = 873.15K
    assert dataset.furnace_setpoint == pytest.approx(873.15, abs=0.01)


def test_dataset_process_data_converts_celsius_to_kelvin(dataset_with_files):
    """
    Test Dataset process_data to confirm furnace_setpoint is stored in Kelvin
    by adding 273.15 to Celsius value.
    """
    dataset = Dataset(path=str(dataset_with_files), name="Test")
    dataset.process_data()

    # Original is 600°C, should be 873.15K
    assert dataset.furnace_setpoint > 273.15


def test_dataset_process_data_uses_default_furnace_setpoint_when_missing(
    temp_dataset_dir, sample_csv_data
):
    """
    Test Dataset process_data to verify it uses default furnace setpoint of
    25°C (298.15K) when not specified in metadata.
    """
    # Create metadata without furnace_setpoint
    metadata = {
        "version": "1.3",
        "run_info": {
            "start_time": "2025-08-20 10:58:01",
            "sample_material": "316",
            "sample_thickness": 0.00088,
        },
        "gauges": [
            {
                "name": "Baratron626D_1KT",
                "type": "Baratron626D_Gauge",
                "gauge_location": "upstream",
                "full_scale_torr": 1000.0,
            }
        ],
    }

    metadata_path = temp_dataset_dir / "run_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))
    csv_path = temp_dataset_dir / "shield_data.csv"
    csv_path.write_text(sample_csv_data)

    dataset = Dataset(path=str(temp_dataset_dir), name="Test")
    dataset.process_data()

    # Default 25°C + 273.15 = 298.15K
    assert dataset.furnace_setpoint == pytest.approx(298.15, abs=0.01)


def test_dataset_process_data_extracts_sample_material(dataset_with_files):
    """
    Test Dataset process_data to verify it extracts sample_material from
    metadata.
    """
    dataset = Dataset(path=str(dataset_with_files), name="Test")
    dataset.process_data()

    assert dataset.sample_material == "316"


def test_dataset_process_data_uses_default_sample_material_when_missing(
    temp_dataset_dir, sample_csv_data
):
    """
    Test Dataset process_data to confirm it uses 'Unknown' as default
    sample_material when not specified.
    """
    # Create metadata without sample_material
    metadata = {
        "version": "1.3",
        "run_info": {
            "start_time": "2025-08-20 10:58:01",
            "furnace_setpoint": 600.0,
            "sample_thickness": 0.00088,
        },
        "gauges": [
            {
                "name": "Baratron626D_1KT",
                "type": "Baratron626D_Gauge",
                "gauge_location": "upstream",
                "full_scale_torr": 1000.0,
            }
        ],
    }

    metadata_path = temp_dataset_dir / "run_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))
    csv_path = temp_dataset_dir / "shield_data.csv"
    csv_path.write_text(sample_csv_data)

    dataset = Dataset(path=str(temp_dataset_dir), name="Test")
    dataset.process_data()

    assert dataset.sample_material == "Unknown"


def test_dataset_process_data_extracts_sample_thickness(dataset_with_files):
    """
    Test Dataset process_data to verify it extracts sample_thickness from
    metadata (in meters).
    """
    dataset = Dataset(path=str(dataset_with_files), name="Test")
    dataset.process_data()

    assert dataset.sample_thickness == pytest.approx(0.00088, abs=1e-6)


def test_dataset_process_data_uses_default_sample_thickness_when_missing(
    temp_dataset_dir, sample_csv_data
):
    """
    Test Dataset process_data to confirm it uses 0.00088 m as default
    sample_thickness when not specified.
    """
    # Create metadata without sample_thickness
    metadata = {
        "version": "1.3",
        "run_info": {
            "start_time": "2025-08-20 10:58:01",
            "furnace_setpoint": 600.0,
            "sample_material": "316",
        },
        "gauges": [
            {
                "name": "Baratron626D_1KT",
                "type": "Baratron626D_Gauge",
                "gauge_location": "upstream",
                "full_scale_torr": 1000.0,
            }
        ],
    }

    metadata_path = temp_dataset_dir / "run_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))
    csv_path = temp_dataset_dir / "shield_data.csv"
    csv_path.write_text(sample_csv_data)

    dataset = Dataset(path=str(temp_dataset_dir), name="Test")
    dataset.process_data()

    assert dataset.sample_thickness == pytest.approx(0.00088, abs=1e-6)


# =============================================================================
# Tests for process_data Method - Integration
# =============================================================================


@patch("shield_das.dataset.voltage_to_temperature")
@patch("shield_das.dataset.voltage_to_pressure")
@patch("shield_das.dataset.calculate_error_on_pressure_reading")
def test_dataset_process_data_completes_full_processing(
    mock_error, mock_v2p, mock_v2t, dataset_with_files
):
    """
    Test Dataset process_data to verify it successfully completes all
    processing steps without errors.
    """
    mock_v2p.return_value = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
    mock_error.return_value = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    mock_v2t.return_value = np.array([600.0, 610.0, 620.0, 630.0, 640.0])

    dataset = Dataset(path=str(dataset_with_files), name="Test")
    dataset.process_data()

    # Verify all major attributes are populated
    assert dataset.time_data is not None
    assert dataset.upstream_pressure is not None
    assert dataset.downstream_pressure is not None
    assert dataset.upstream_error is not None
    assert dataset.downstream_error is not None
    assert dataset.valve_times is not None
    assert dataset.furnace_setpoint is not None
    assert dataset.sample_material is not None
    assert dataset.sample_thickness is not None


@patch("shield_das.dataset.voltage_to_temperature")
@patch("shield_das.dataset.voltage_to_pressure")
@patch("shield_das.dataset.calculate_error_on_pressure_reading")
def test_dataset_process_data_arrays_have_consistent_length(
    mock_error, mock_v2p, mock_v2t, dataset_with_files
):
    """
    Test Dataset process_data to confirm all data arrays (time, pressure,
    temperature) have the same length.
    """
    mock_v2p.return_value = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
    mock_error.return_value = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    mock_v2t.return_value = np.array([600.0, 610.0, 620.0, 630.0, 640.0])

    dataset = Dataset(path=str(dataset_with_files), name="Test")
    dataset.process_data()

    expected_length = 5
    assert len(dataset.time_data) == expected_length
    assert len(dataset.upstream_pressure) == expected_length
    assert len(dataset.downstream_pressure) == expected_length
    assert len(dataset.local_temperature_data) == expected_length
    assert len(dataset.thermocouple_data) == expected_length


def test_dataset_can_be_created_and_processed_end_to_end(dataset_with_files):
    """
    Test Dataset complete workflow to verify a dataset can be created,
    initialized, and processed from real files without errors.
    """
    dataset = Dataset(path=str(dataset_with_files), name="End-to-End Test")

    # Should complete without raising exceptions
    dataset.process_data()

    # Basic sanity checks
    assert dataset.name == "End-to-End Test"
    assert dataset.path == str(dataset_with_files)
    assert dataset.time_data is not None
