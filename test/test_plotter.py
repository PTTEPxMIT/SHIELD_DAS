import json
from unittest.mock import patch

import pytest

from shield_das import DataPlotter
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


class TestDataPlotterDataProcessing:
    """Test data processing methods."""

    def setup_method(self):
        """Create DataPlotter instance for testing."""
        self.plotter = DataPlotter()

    def test_process_csv_v1_0_not_implemented(self):
        """Test that v1.0 CSV processing raises NotImplementedError."""
        with pytest.raises(
            NotImplementedError, match="Version 1.0 processing not yet implemented"
        ):
            self.plotter.process_csv_v1_0()

    @pytest.mark.parametrize(
        "version,should_raise",
        [
            ("0.0", False),
            ("1.0", True),
            ("2.0", True),
            ("unknown", True),
        ],
    )
    def test_process_csv_data_version_handling(self, version, should_raise):
        """Test process_csv_data handles different versions correctly."""
        metadata = {"version": version}

        if should_raise:
            if version == "1.0":
                with pytest.raises(
                    NotImplementedError,
                    match="Version 1.0 processing not yet implemented",
                ):
                    self.plotter.process_csv_data(metadata, "/fake/path")
            else:
                with pytest.raises(
                    NotImplementedError,
                    match=f"Unsupported metadata version: {version}",
                ):
                    self.plotter.process_csv_data(metadata, "/fake/path")
        else:
            # For version 0.0, we need to mock the process_csv_v0_0 method
            with patch.object(self.plotter, "process_csv_v0_0") as mock_process:
                self.plotter.process_csv_data(metadata, "/fake/path")
                mock_process.assert_called_once_with(metadata, "/fake/path")

    def test_create_gauge_instances(self):
        """Test creation of gauge instances from metadata."""
        gauges_metadata = [
            {
                "name": "CVM211",
                "type": "CVM211_Gauge",
                "ain_channel": 8,
                "gauge_location": "upstream",
            },
            {
                "name": "Baratron626D_1T",
                "type": "Baratron626D_Gauge",
                "ain_channel": 10,
                "full_scale_torr": 1.0,
                "gauge_location": "downstream",
            },
        ]

        gauges = self.plotter.create_gauge_instances(gauges_metadata)

        assert len(gauges) == 2
        assert all(isinstance(gauge, PressureGauge) for gauge in gauges)
        assert gauges[0].gauge_location == "upstream"
        assert gauges[1].gauge_location == "downstream"


class TestDataPlotterFileOperations:
    """Test file operation methods."""

    def setup_method(self):
        """Create DataPlotter instance for testing."""
        self.plotter = DataPlotter()

    def test_load_data_with_valid_datasets(self, tmp_path):
        """Test loading data from valid dataset paths."""
        # Create test dataset
        dataset = tmp_path / "dataset"
        dataset.mkdir()

        metadata = {
            "version": "0.0",
            "gauges": {
                "gauge1": {
                    "gauge_id": "G001",
                    "gauge_location": "Test Location",
                    "gauge_brand": "Test Brand",
                    "gauge_model": "Test Model",
                    "gauge_range": [0, 100],
                    "gauge_units": "psi",
                }
            },
        }
        (dataset / "run_metadata.json").write_text(json.dumps(metadata))
        (dataset / "data.csv").write_text("timestamp,pressure\n0,100\n1,200\n")

        # Set dataset paths and load data
        self.plotter.dataset_paths = [str(dataset)]

        # Mock the process_csv_v0_0 method to avoid complex setup
        with patch.object(self.plotter, "process_csv_v0_0") as mock_process:
            self.plotter.load_data()
            mock_process.assert_called_once()

    def test_load_data_prints_progress(self, tmp_path, capsys):
        """Test that load_data prints progress information."""
        # Create test dataset
        dataset = tmp_path / "dataset"
        dataset.mkdir()

        metadata = {"version": "0.0", "gauges": {}}
        (dataset / "run_metadata.json").write_text(json.dumps(metadata))
        (dataset / "data.csv").write_text("timestamp,pressure\n0,100\n")

        self.plotter.dataset_paths = [str(dataset)]

        with patch.object(self.plotter, "process_csv_v0_0"):
            self.plotter.load_data()

        captured = capsys.readouterr()
        assert "Loading data from 1 dataset(s)" in captured.out
        assert "Processing dataset 1/1" in captured.out
