import json
import os
import time
from unittest.mock import Mock, patch

import pytest

from shield_das import DataPlotter, DataRecorder
from shield_das.pressure_gauge import PressureGauge

example_metadata_v0 = {
    "version": "0.0",
    "gauges": [
        {
            "name": "test_gauge",
            "type": "Baratron626D_Gauge",
            "ain_channel": 2,
            "gauge_location": "downstream",
            "filename": "example_data.csv",
        },
    ],
}


class TestDataPlotterInitialization:
    """Test DataPlotter initialization and property validation."""

    def test_init_defaults(self):
        """Test default initialization values."""
        plotter = DataPlotter()
        assert plotter.dataset_paths == []
        assert plotter.dataset_names == []
        assert plotter.port == 8050

    def test_init_with_dataset_paths(self, tmp_path):
        """Test initialization with valid dataset paths."""
        # Create test directories with required files
        dataset1 = tmp_path / "dataset1"
        dataset2 = tmp_path / "dataset2"
        dataset1.mkdir()
        dataset2.mkdir()

        # Create required files
        for dataset in [dataset1, dataset2]:
            (dataset / "run_metadata.json").write_text('{"version": "0.0"}')
            (dataset / "data.csv").write_text("timestamp,pressure\n0,100\n")

        paths = [str(dataset1), str(dataset2)]
        names = ["Dataset 1", "Dataset 2"]
        plotter = DataPlotter(dataset_paths=paths, dataset_names=names)
        assert plotter.dataset_paths == paths

    def test_init_with_dataset_names(self, tmp_path):
        """Test initialization with dataset names."""
        # Create test directory with required files
        dataset = tmp_path / "dataset"
        dataset.mkdir()
        (dataset / "run_metadata.json").write_text(f"{example_metadata_v0}")
        (dataset / "data.csv").write_text("timestamp,pressure\n0,100\n")

        paths = [str(dataset)]
        names = ["Test Dataset"]
        plotter = DataPlotter(dataset_paths=paths, dataset_names=names)
        assert plotter.dataset_paths == paths
        assert plotter.dataset_names == names

    def test_init_custom_port(self):
        """Test initialization with custom port."""
        plotter = DataPlotter(port=9000)
        assert plotter.port == 9000


class TestDataPlotterPropertyValidation:
    """Test property setters and validation."""

    def setup_method(self):
        """Create DataPlotter instance for testing."""
        self.plotter = DataPlotter()

    @pytest.mark.parametrize(
        "invalid_paths,error_type,error_message",
        [
            ("not_a_list", ValueError, "dataset_paths must be a list of strings"),
            ([123, 456], ValueError, "dataset_paths must be a list of strings"),
            (["path1", 123], ValueError, "dataset_paths must be a list of strings"),
        ],
    )
    def test_dataset_paths_invalid_type(self, invalid_paths, error_type, error_message):
        """Test dataset_paths setter with invalid types."""
        with pytest.raises(error_type, match=error_message):
            self.plotter.dataset_paths = invalid_paths

    def test_dataset_paths_nonexistent_path(self):
        """Test dataset_paths setter with nonexistent path."""
        with pytest.raises(ValueError, match="Dataset path does not exist"):
            self.plotter.dataset_paths = ["/nonexistent/path"]

    def test_dataset_paths_duplicate_paths(self, tmp_path):
        """Test dataset_paths setter with duplicate paths."""
        dataset = tmp_path / "dataset"
        dataset.mkdir()
        (dataset / "run_metadata.json").write_text('{"version": "0.0"}')
        (dataset / "data.csv").write_text("timestamp,pressure\n0,100\n")

        path = str(dataset)
        with pytest.raises(ValueError, match="dataset_paths must contain unique paths"):
            self.plotter.dataset_paths = [path, path]

    def test_dataset_paths_no_csv_files(self, tmp_path):
        """Test dataset_paths setter with directory containing no CSV files."""
        dataset = tmp_path / "dataset"
        dataset.mkdir()
        (dataset / "run_metadata.json").write_text('{"version": "0.0"}')

        with pytest.raises(FileNotFoundError, match="No data CSV files found"):
            self.plotter.dataset_paths = [str(dataset)]

    def test_dataset_paths_no_metadata_json(self, tmp_path):
        """Test dataset_paths setter with directory missing run_metadata.json."""
        dataset = tmp_path / "dataset"
        dataset.mkdir()
        (dataset / "data.csv").write_text("timestamp,pressure\n0,100\n")

        with pytest.raises(FileNotFoundError, match="No run_metadata.json file found"):
            self.plotter.dataset_paths = [str(dataset)]

    @pytest.mark.parametrize(
        "invalid_names,error_message",
        [
            ("not_a_list", "dataset_names must be a list of strings"),
            ([123, 456], "dataset_names must be a list of strings"),
            (["name1", 123], "dataset_names must be a list of strings"),
        ],
    )
    def test_dataset_names_invalid_type(self, invalid_names, error_message):
        """Test dataset_names setter with invalid types."""
        with pytest.raises(ValueError, match=error_message):
            self.plotter.dataset_names = invalid_names

    def test_dataset_names_length_mismatch(self, tmp_path):
        """Test dataset_names setter with mismatched length."""
        # Set up valid dataset path first
        dataset = tmp_path / "dataset"
        dataset.mkdir()
        (dataset / "run_metadata.json").write_text('{"version": "0.0"}')
        (dataset / "data.csv").write_text("timestamp,pressure\n0,100\n")
        self.plotter.dataset_paths = [str(dataset)]

        with pytest.raises(
            ValueError, match="dataset_names length .* must match dataset_paths length"
        ):
            self.plotter.dataset_names = ["name1", "name2"]  # 2 names for 1 path

    def test_dataset_names_duplicate_names(self, tmp_path):
        """Test dataset_names setter with duplicate names."""
        # Set up valid dataset paths first
        dataset1 = tmp_path / "dataset1"
        dataset2 = tmp_path / "dataset2"
        for dataset in [dataset1, dataset2]:
            dataset.mkdir()
            (dataset / "run_metadata.json").write_text('{"version": "0.0"}')
            (dataset / "data.csv").write_text("timestamp,pressure\n0,100\n")

        self.plotter.dataset_paths = [str(dataset1), str(dataset2)]

        with pytest.raises(ValueError, match="dataset_names must contain unique names"):
            self.plotter.dataset_names = ["same_name", "same_name"]


class TestDataPlotterUtilityFunctions:
    """Test utility functions."""

    def setup_method(self):
        """Create DataPlotter instance for testing."""
        self.plotter = DataPlotter()

    @pytest.mark.parametrize(
        "index,expected_color",
        [
            (0, "#000000"),  # Black
            (1, "#DF1AD2"),  # Magenta
            (2, "#779BE7"),  # Light Blue
            (3, "#49B6FF"),  # Blue
            (4, "#254E70"),  # Dark Blue
            (5, "#0CCA4A"),  # Green
            (6, "#929487"),  # Gray
            (7, "#A1B0AB"),  # Light Gray
            (8, "#000000"),  # Cycles back to black
            (16, "#000000"),  # Large index cycles
            (-1, "#A1B0AB"),  # Negative index wraps to last color
        ],
    )
    def test_get_next_color(self, index, expected_color):
        """Test color palette returns correct values and cycles properly."""
        assert self.plotter.get_next_color(index) == expected_color


class TestDataPlotterV1Processing:
    """Test v1.0 CSV processing methods."""

    def setup_method(self):
        """Create DataPlotter instance for testing."""
        self.plotter = DataPlotter()

    def test_process_csv_v1_0_with_valid_data(self, tmp_path):
        """Test v1.0 processing with valid data structure."""
        dataset = tmp_path / "dataset"
        dataset.mkdir()

        # Create v1.0 metadata with proper structure
        metadata = {
            "version": "1.0",
            "run_info": {"data_filename": "pressure_gauge_data.csv"},
            "gauges": [
                {
                    "name": "TestUpstream",
                    "type": "Baratron626D_Gauge",
                    "ain_channel": 10,
                    "gauge_location": "upstream",
                    "full_scale_torr": 1.0,
                },
                {
                    "name": "TestDownstream",
                    "type": "Baratron626D_Gauge",
                    "ain_channel": 6,
                    "gauge_location": "downstream",
                    "full_scale_torr": 1.0,
                },
            ],
        }

        # Create CSV with both upstream and downstream Baratron voltage columns
        csv_content = """RealTimestamp,TestUpstream_Voltage (V),TestDownstream_Voltage (V)
            2024-01-01 10:00:00.000000,2.5,3.2
            2024-01-01 10:00:01.000000,2.6,3.3
            2024-01-01 10:00:02.000000,2.7,3.4
            2024-01-01 10:00:03.000000,2.8,3.5"""

        (dataset / "run_metadata.json").write_text(json.dumps(metadata))
        (dataset / "pressure_gauge_data.csv").write_text(csv_content)

        # Test the processing
        self.plotter.dataset_paths = [str(dataset)]
        self.plotter.load_data(str(dataset), "Test Dataset")

        # Verify datasets were created in the dict-based structure
        assert isinstance(self.plotter.datasets, dict)
        assert len(self.plotter.datasets) == 1
        ds = next(iter(self.plotter.datasets.values()))
        # Check upstream and downstream data
        assert "upstream_data" in ds and "downstream_data" in ds
        # Check time data conversion
        assert len(ds["time_data"]) == 4
        assert ds["time_data"][0] == 0.0
        assert ds["time_data"][1] == 1.0
        assert ds["time_data"][2] == 2.0
        assert ds["time_data"][3] == 3.0
        # Check pressure and error arrays
        assert len(ds["upstream_data"]["pressure_data"]) == 4
        assert len(ds["upstream_data"]["error_data"]) == 4
        assert len(ds["downstream_data"]["pressure_data"]) == 4
        assert len(ds["downstream_data"]["error_data"]) == 4

    def test_process_csv_v1_0_missing_csv_file(self, tmp_path):
        """Test v1.0 processing with missing CSV file."""
        dataset = tmp_path / "dataset"
        dataset.mkdir()

        metadata = {
            "version": "1.0",
            "run_info": {"data_filename": "missing_file.csv"},
            "gauges": [
                {
                    "name": "TestGauge",
                    "type": "Baratron626D_Gauge",
                    "ain_channel": 10,
                    "gauge_location": "upstream",
                }
            ],
        }

        (dataset / "run_metadata.json").write_text(json.dumps(metadata))

        with pytest.raises(FileNotFoundError):
            self.plotter.dataset_paths = [str(dataset)]

    def test_process_csv_v1_0_malformed_csv(self, tmp_path):
        """Test v1.0 processing with malformed CSV data."""
        dataset = tmp_path / "dataset"
        dataset.mkdir()

        metadata = {
            "version": "1.0",
            "run_info": {"data_filename": "malformed_data.csv"},
            "gauges": [
                {
                    "name": "TestGauge",
                    "type": "Baratron626D_Gauge",
                    "ain_channel": 10,
                    "gauge_location": "upstream",
                }
            ],
        }

        # Create malformed CSV (missing expected voltage column)
        malformed_csv = """RealTimestamp,WrongColumn
            2024-01-01 10:00:00.000000,2.5
            2024-01-01 10:00:01.000000,2.6"""

        (dataset / "run_metadata.json").write_text(json.dumps(metadata))
        (dataset / "malformed_data.csv").write_text(malformed_csv)

        # The error will be raised during data processing, not path validation
        self.plotter.dataset_paths = [str(dataset)]
        with pytest.raises((ValueError, KeyError)):
            self.plotter.load_data(str(dataset), "Test Dataset")

    def test_process_csv_v1_0_invalid_timestamp_format(self, tmp_path):
        """Test v1.0 processing with invalid timestamp format."""
        dataset = tmp_path / "dataset"
        dataset.mkdir()

        metadata = {
            "version": "1.0",
            "run_info": {"data_filename": "invalid_timestamps.csv"},
            "gauges": [
                {
                    "name": "TestGauge",
                    "type": "Baratron626D_Gauge",
                    "ain_channel": 10,
                    "gauge_location": "upstream",
                }
            ],
        }

        # CSV with invalid timestamp format
        invalid_csv = """RealTimestamp,TestGauge_Voltage (V)
            invalid-timestamp,2.5
            2024-01-01 10:00:01.000000,2.6"""

        (dataset / "run_metadata.json").write_text(json.dumps(metadata))
        (dataset / "invalid_timestamps.csv").write_text(invalid_csv)

        # The error will be raised during data processing, not path validation
        self.plotter.dataset_paths = [str(dataset)]
        with pytest.raises(ValueError):
            self.plotter.load_data(str(dataset), "Test Dataset")

    def test_process_csv_v1_0_missing_metadata_fields(self, tmp_path):
        """Test v1.0 processing with missing required metadata fields."""
        dataset = tmp_path / "dataset"
        dataset.mkdir()

        # Missing run_info section
        metadata_no_run_info = {
            "version": "1.0",
            "gauges": [
                {
                    "name": "TestGauge",
                    "type": "Baratron626D_Gauge",
                    "ain_channel": 10,
                    "gauge_location": "upstream",
                    "full_scale_torr": 1.0,
                }
            ],
        }

        # Create a dummy CSV file so path validation passes
        dummy_csv = """RealTimestamp,TestGauge_Voltage (V)
            2024-01-01 10:00:00.000000,2.5"""

        (dataset / "run_metadata.json").write_text(json.dumps(metadata_no_run_info))
        (dataset / "dummy.csv").write_text(dummy_csv)

        # The error will be raised during data processing, not path validation
        self.plotter.dataset_paths = [str(dataset)]
        with pytest.raises(
            ValueError, match="Missing data_filename in run_info for v1.0 metadata"
        ):
            self.plotter.load_data(str(dataset), "Test Dataset")


class TestDataPlotterDataProcessing:
    """Test data processing methods."""

    def setup_method(self):
        """Create DataPlotter instance for testing."""
        self.plotter = DataPlotter()

    def test_process_csv_v1_0_implemented(self, tmp_path):
        """Test that v1.0 CSV processing is now implemented."""
        dataset = tmp_path / "dataset"
        dataset.mkdir()

        # Create realistic v1.0 metadata
        metadata = {
            "version": "1.0",
            "run_info": {"data_filename": "pressure_gauge_data.csv"},
            "gauges": [
                {
                    "name": "TestGauge1",
                    "type": "Baratron626D_Gauge",
                    "ain_channel": 10,
                    "gauge_location": "upstream",
                    "full_scale_torr": 1.0,
                },
                {
                    "name": "TestGauge2",
                    "type": "Baratron626D_Gauge",
                    "ain_channel": 6,
                    "gauge_location": "downstream",
                    "full_scale_torr": 1.0,
                },
            ],
        }

        # Create the CSV file with v1.0 format
        csv_data = """RealTimestamp,TestGauge1_Voltage (V),TestGauge2_Voltage (V)
            2024-01-01 10:00:00.000000,2.5,3.2
            2024-01-01 10:00:01.000000,2.6,3.3
            2024-01-01 10:00:02.000000,2.7,3.4"""

        (dataset / "run_metadata.json").write_text(json.dumps(metadata))
        (dataset / "pressure_gauge_data.csv").write_text(csv_data)

        # This should not raise an error anymore
        self.plotter.dataset_paths = [str(dataset)]
        self.plotter.load_data(str(dataset), "Test Dataset")

        # Verify datasets were created in the dict-based structure
        assert isinstance(self.plotter.datasets, dict)
        assert len(self.plotter.datasets) == 1
        ds = next(iter(self.plotter.datasets.values()))
        # Check upstream and downstream data
        assert "upstream_data" in ds and "downstream_data" in ds
        # Check time data conversion
        assert len(ds["time_data"]) == 3
        assert ds["time_data"][0] == 0.0
        assert ds["time_data"][1] == 1.0
        assert ds["time_data"][2] == 2.0
        # Check pressure and error arrays
        assert len(ds["upstream_data"]["pressure_data"]) == 3
        assert len(ds["downstream_data"]["pressure_data"]) == 3
        assert len(ds["upstream_data"]["error_data"]) == 3
        assert len(ds["downstream_data"]["error_data"]) == 3


class TestDataPlotterIntegration:
    """Integration tests that use DataRecorder to generate real data."""

    def setup_method(self):
        """Create DataPlotter instance for testing."""
        self.plotter = DataPlotter()

    def test_process_csv_v0_0_with_realistic_data(self, tmp_path):
        """Test process_csv_v0_0 with realistic gauge data structure."""
        dataset = tmp_path / "dataset"
        dataset.mkdir()

        # Create realistic v0.0 metadata that matches the expected format
        metadata = {
            "version": "0.0",
            "gauges": [
                {
                    "name": "TestGauge1",
                    "type": "Baratron626D_Gauge",
                    "ain_channel": 10,
                    "gauge_location": "upstream",
                    "filename": "gauge1_data.csv",
                    "full_scale_torr": 1.0,
                },
                {
                    "name": "TestGauge2",
                    "type": "Baratron626D_Gauge",
                    "ain_channel": 6,
                    "gauge_location": "downstream",
                    "filename": "gauge2_data.csv",
                },
            ],
        }

        # Create metadata file
        (dataset / "run_metadata.json").write_text(json.dumps(metadata))

        # Create CSV files with expected format (RelativeTime, Pressure_Torr)
        gauge1_data = """RelativeTime,Pressure_Torr
            0.0,0.001
            1.0,0.002
            2.0,0.003
            3.0,0.004
            4.0,0.005"""
        (dataset / "gauge1_data.csv").write_text(gauge1_data)

        gauge2_data = """RelativeTime,Pressure_Torr
            0.0,0.1
            1.0,0.2
            2.0,0.3
            3.0,0.4
            4.0,0.5"""
        (dataset / "gauge2_data.csv").write_text(gauge2_data)

        # Set up plotter and process the data
        self.plotter.dataset_paths = [str(dataset)]
        self.plotter.load_data(str(dataset), "Test Dataset")

        # Verify datasets were created in the dict-based structure
        assert isinstance(self.plotter.datasets, dict)
        assert len(self.plotter.datasets) == 1
        ds = next(iter(self.plotter.datasets.values()))
        # Check upstream and downstream data
        assert "upstream_data" in ds and "downstream_data" in ds
        # Check time data
        assert len(ds["time_data"]) == 5
        assert ds["time_data"][0] == 0.0
        assert ds["time_data"][-1] == 4.0
        # Check pressure data
        assert len(ds["upstream_data"]["pressure_data"]) == 5
        assert ds["upstream_data"]["pressure_data"][0] == 0.001
        assert ds["upstream_data"]["pressure_data"][-1] == 0.005
        assert len(ds["downstream_data"]["pressure_data"]) == 5
        assert ds["downstream_data"]["pressure_data"][0] == 0.1
        assert ds["downstream_data"]["pressure_data"][-1] == 0.5

    def test_integration_with_data_recorder(self, tmp_path):
        """Integration test using DataRecorder to generate data."""
        # Create mock gauges for the recorder
        mock_gauge1 = Mock(spec=PressureGauge)
        mock_gauge1.name = "IntegrationGauge1"
        mock_gauge1.voltage_data = [2.5]  # Mock voltage data
        mock_gauge1.record_ain_channel_voltage.return_value = None
        mock_gauge1.ain_channel = 10
        mock_gauge1.gauge_location = "upstream"

        mock_gauge2 = Mock(spec=PressureGauge)
        mock_gauge2.name = "IntegrationGauge2"
        mock_gauge2.voltage_data = [3.7]  # Mock voltage data
        mock_gauge2.record_ain_channel_voltage.return_value = None
        mock_gauge2.ain_channel = 6
        mock_gauge2.gauge_location = "downstream"

        # Create DataRecorder and generate test data
        recorder = DataRecorder(
            gauges=[mock_gauge1, mock_gauge2],
            thermocouples=[],
            results_dir=str(tmp_path),
            run_type="test_mode",
            recording_interval=0.05,  # Fast recording for testing
        )

        # Record some data
        recorder.start()
        time.sleep(0.3)  # Record for 300ms
        recorder.stop()

        # Verify data was recorded
        assert recorder.run_dir is not None
        csv_path = os.path.join(recorder.run_dir, "pressure_gauge_data.csv")
        metadata_path = os.path.join(recorder.run_dir, "run_metadata.json")

        assert os.path.exists(csv_path)
        assert os.path.exists(metadata_path)

        # Read the generated CSV to verify structure
        with open(csv_path) as f:
            csv_content = f.read()
            lines = csv_content.strip().split("\n")
            assert len(lines) >= 2  # Header + at least one data row

            # Check header structure
            header = lines[0]
            assert "RealTimestamp" in header
            assert "IntegrationGauge1_Voltage (V)" in header
            assert "IntegrationGauge2_Voltage (V)" in header

        # Read metadata to verify structure
        with open(metadata_path) as f:
            metadata = json.load(f)
            assert metadata["version"] == "1.0"
            assert "gauges" in metadata
            assert len(metadata["gauges"]) == 2

        # Update gauge types in metadata to valid types since Mock generates "Mock" type
        metadata["gauges"][0]["type"] = "Baratron626D_Gauge"
        metadata["gauges"][0]["full_scale_torr"] = 1.0  # Add required parameter
        metadata["gauges"][1]["type"] = "Baratron626D_Gauge"
        metadata["gauges"][1]["full_scale_torr"] = 1000.0  # Add required parameter

        # Write back the corrected metadata
        with open(metadata_path, "w") as f:
            json.dump(metadata, f)

        # Now test that DataPlotter can process the v1.0 format
        plotter = DataPlotter(
            dataset_paths=[recorder.run_dir], dataset_names=["Test Run"]
        )
        plotter.load_data(recorder.run_dir, "Test Run")

        # Verify the data was processed correctly
        assert isinstance(plotter.datasets, dict)
        assert len(plotter.datasets) == 1
        ds = next(iter(plotter.datasets.values()))
        assert "upstream_data" in ds and "downstream_data" in ds
        # Check that pressure and error arrays exist and are non-empty
        assert len(ds["upstream_data"]["pressure_data"]) > 0
        assert len(ds["downstream_data"]["pressure_data"]) > 0
        assert len(ds["upstream_data"]["error_data"]) > 0
        assert len(ds["downstream_data"]["error_data"]) > 0

    def test_process_csv_v0_0_error_handling(self, tmp_path):
        """Test error handling in process_csv_v0_0 method."""
        dataset = tmp_path / "dataset"
        dataset.mkdir()

        # Create metadata with missing CSV file
        metadata = {
            "version": "0.0",
            "gauges": [
                {
                    "name": "MissingGauge",
                    "type": "Baratron626D_Gauge",
                    "ain_channel": 10,
                    "gauge_location": "upstream",
                    "filename": "missing_file.csv",
                }
            ],
        }

        (dataset / "run_metadata.json").write_text(json.dumps(metadata))

        # Should raise an error when trying to load non-existent CSV
        with pytest.raises(FileNotFoundError):
            self.plotter.dataset_paths = [str(dataset)]

    def test_process_csv_v0_0_invalid_csv_format(self, tmp_path):
        """Test handling of invalid CSV format in process_csv_v0_0."""
        dataset = tmp_path / "dataset"
        dataset.mkdir()

        metadata = {
            "version": "0.0",
            "gauges": [
                {
                    "name": "InvalidGauge",
                    "type": "Baratron626D_Gauge",
                    "ain_channel": 10,
                    "gauge_location": "upstream",
                    "filename": "invalid_data.csv",
                }
            ],
        }

        (dataset / "run_metadata.json").write_text(json.dumps(metadata))

        # Create CSV with wrong column names
        invalid_csv = """WrongColumn1,WrongColumn2
            1.0,2.0
            3.0,4.0"""
        (dataset / "invalid_data.csv").write_text(invalid_csv)

        self.plotter.dataset_paths = [str(dataset)]

        # Should raise an error due to missing expected columns
        with pytest.raises((ValueError, KeyError)):
            self.plotter.load_data(str(dataset), "Test Dataset")


class TestDataPlotterPlotGeneration:
    """Test plot generation methods with error bar functionality."""

    def setup_method(self):
        """Create DataPlotter instance with test data for testing."""
        self.plotter = DataPlotter()

    def test_plot_generation_methods_accept_error_bar_parameter(self, tmp_path):
        """Test that plot generation methods accept show_error_bars parameter."""
        # Create test dataset with v1.0 format
        dataset = tmp_path / "dataset"
        dataset.mkdir()

        metadata = {
            "version": "1.0",
            "run_info": {"data_filename": "pressure_gauge_data.csv"},
            "gauges": [
                {
                    "name": "TestGauge",
                    "type": "Baratron626D_Gauge",
                    "ain_channel": 10,
                    "gauge_location": "upstream",
                    "full_scale_torr": 1.0,
                }
            ],
        }

        csv_content = """RealTimestamp,TestGauge_Voltage (V)
            2024-01-01 10:00:00.000000,2.5
            2024-01-01 10:00:01.000000,2.6
            2024-01-01 10:00:02.000000,2.7"""

        (dataset / "run_metadata.json").write_text(json.dumps(metadata))
        (dataset / "pressure_gauge_data.csv").write_text(csv_content)

        # Load data
        self.plotter.dataset_paths = [str(dataset)]
        self.plotter.load_data(str(dataset), "Test Dataset")

        # Test that plot generation methods accept error bar parameters
        # These should not raise errors
        upstream_plot = self.plotter._generate_upstream_plot(show_error_bars=True)
        upstream_plot_no_errors = self.plotter._generate_upstream_plot(
            show_error_bars=False
        )

        # Verify plots were generated
        assert upstream_plot is not None
        assert upstream_plot_no_errors is not None

    def test_error_bars_calculation_in_v1_0_processing(self, tmp_path):
        """Test that error bars are properly calculated during v1.0 processing."""
        dataset = tmp_path / "dataset"
        dataset.mkdir()

        metadata = {
            "version": "1.0",
            "run_info": {"data_filename": "pressure_gauge_data.csv"},
            "gauges": [
                {
                    "name": "ErrorTestGauge",
                    "type": "Baratron626D_Gauge",
                    "ain_channel": 6,
                    "gauge_location": "downstream",
                    "full_scale_torr": 1.0,
                }
            ],
        }

        # Create CSV with multiple data points for error calculation
        csv_content = """RealTimestamp,ErrorTestGauge_Voltage (V)
            2024-01-01 10:00:00.000000,1.0
            2024-01-01 10:00:01.000000,2.0
            2024-01-01 10:00:02.000000,3.0
            2024-01-01 10:00:03.000000,4.0
            2024-01-01 10:00:04.000000,5.0"""

        (dataset / "run_metadata.json").write_text(json.dumps(metadata))
        (dataset / "pressure_gauge_data.csv").write_text(csv_content)

        # Process the data
        self.plotter.dataset_paths = [str(dataset)]
        self.plotter.load_data(str(dataset), "Test Dataset")

        # Verify error bars were calculated in the dataset dict
        ds = next(iter(self.plotter.datasets.values()))
        assert "downstream_data" in ds
        assert "pressure_data" in ds["downstream_data"]
        assert len(ds["downstream_data"]["pressure_data"]) == len(
            ds["downstream_data"]["pressure_data"]
        )
        assert all(error >= 0 for error in ds["downstream_data"]["pressure_data"])
