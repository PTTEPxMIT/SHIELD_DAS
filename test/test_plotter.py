import base64
import json
from unittest.mock import patch

import pandas as pd
import pytest

from shield_das import DataPlotter


class TestDataPlotter:
    """Test DataPlotter pure Python utility functions."""

    def setup_method(self):
        """Create DataPlotter instance for testing."""
        self.plotter = DataPlotter()

    @pytest.mark.parametrize(
        "index,expected_color",
        [
            (0, "#000000"),
            (1, "#DF1AD2"),
            (7, "#A1B0AB"),
            (8, "#000000"),  # Cycles back
            (-1, "#A1B0AB"),  # Last color
            (16, "#000000"),  # Large index cycles
        ],
    )
    def test_get_next_color(self, index, expected_color):
        """Test color palette returns correct values and cycles properly."""
        assert self.plotter.get_next_color(index) == expected_color

    @pytest.mark.parametrize(
        "color,expected",
        [
            ("#000000", True),
            ("#fff", True),
            ("#ABC123", True),
            ("  #000000  ", True),  # Whitespace stripped
            ("#GGG", False),  # Invalid chars
            ("123456", False),  # Missing #
            ("", False),
            (None, False),
            ("   ", False),
        ],
    )
    def test_is_valid_color_hex(self, color, expected):
        """Test hex color validation."""
        assert self.plotter.is_valid_color(color) is expected

    @pytest.mark.parametrize(
        "color,expected",
        [
            ("rgb(0,0,0)", True),
            ("rgb(255,255,255)", True),
            ("RGB(100,200,50)", True),  # Case check
            ("rgb(256,0,0)", False),  # Out of range
            ("rgb(-1,0,0)", False),  # Negative
        ],
    )
    def test_is_valid_color_rgb(self, color, expected):
        """Test RGB color validation."""
        assert self.plotter.is_valid_color(color) is expected

    @pytest.mark.parametrize(
        "input_timestamps,expected",
        [
            (["0", "1.5", "3.0"], [0.0, 1.5, 3.0]),
            (["123.45"], [123.45]),
            ([], []),
            (None, []),
        ],
    )
    def test_convert_timestamps_to_seconds(self, input_timestamps, expected):
        """Test timestamp string to float conversion."""
        result = self.plotter.convert_timestamps_to_seconds(input_timestamps)
        assert result == expected

    def test_convert_timestamps_invalid(self):
        """Test timestamp conversion with invalid input raises ValueError."""
        with pytest.raises(ValueError):
            self.plotter.convert_timestamps_to_seconds(["not_a_number"])


class TestDataPlotterInitialization:
    """Test DataPlotter initialization and parameter handling."""

    def test_init_defaults(self):
        """Test default initialization values."""
        plotter = DataPlotter()
        assert plotter.data_paths == []
        assert plotter.dataset_names is None
        assert plotter.port == 8050

    @pytest.mark.parametrize(
        "data,expected_paths",
        [
            ("/path/to/data", ["/path/to/data"]),
            (["/path1", "/path2"], ["/path1", "/path2"]),
        ],
    )
    def test_init_with_data(self, data, expected_paths):
        """Test initialization with data paths."""
        plotter = DataPlotter(data=data)
        assert plotter.data_paths == expected_paths

    def test_init_with_dataset_names(self):
        """Test initialization with dataset names."""
        paths = ["/path1", "/path2"]
        names = ["Dataset 1", "Dataset 2"]
        plotter = DataPlotter(data=paths, dataset_names=names)
        assert plotter.data_paths == paths
        assert plotter.dataset_names == names

    def test_init_custom_port(self):
        """Test initialization with custom port."""
        plotter = DataPlotter(port=9000)
        assert plotter.port == 9000

    @pytest.mark.parametrize(
        "invalid_data,error_match",
        [
            (123, "data parameter must be"),
        ],
    )
    def test_init_invalid_data_type(self, invalid_data, error_match):
        """Test initialization with invalid data type raises ValueError."""
        with pytest.raises(ValueError, match=error_match):
            DataPlotter(data=invalid_data)

    def test_init_mismatched_names_length(self):
        """Test initialization with mismatched dataset names length."""
        with pytest.raises(ValueError, match="dataset_names length"):
            DataPlotter(data=["/path1", "/path2"], dataset_names=["name1"])


class TestDataPlotterFileProcessing:
    """Test file processing methods."""

    def setup_method(self):
        """Create DataPlotter instance for testing."""
        self.plotter = DataPlotter()

    def test_parse_uploaded_file_valid_json(self):
        """Test parsing valid JSON file with version 1.0."""
        metadata = {"version": "1.0", "run_info": {"data_filename": "test.csv"}}
        json_content = json.dumps(metadata)
        encoded_content = base64.b64encode(json_content.encode()).decode()
        contents = f"data:application/json;base64,{encoded_content}"

        with patch("sys.exit") as mock_exit:
            self.plotter.parse_uploaded_file(contents, "test.json")
            mock_exit.assert_called_once_with(0)

    @pytest.mark.parametrize(
        "test_data,description",
        [
            ({"version": "2.0"}, "unsupported version"),
            ("not valid json", "invalid JSON content"),
            ("invalid_content", "malformed base64 content"),
        ],
    )
    def test_parse_uploaded_file_returns_empty_dataframe(self, test_data, description):
        """Test parsing files that should return empty DataFrame."""
        if isinstance(test_data, dict):
            # Valid JSON but unsupported version
            json_content = json.dumps(test_data)
            encoded_content = base64.b64encode(json_content.encode()).decode()
            contents = f"data:application/json;base64,{encoded_content}"
        elif test_data == "not valid json":
            # Invalid JSON
            encoded_content = base64.b64encode(test_data.encode()).decode()
            contents = f"data:application/json;base64,{encoded_content}"
        else:
            # Malformed content
            contents = test_data

        result = self.plotter.parse_uploaded_file(contents, "test.json")
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_process_json_metadata_no_file(self, tmp_path):
        """Test processing directory with no JSON file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="No metadata JSON file found"):
            self.plotter.process_json_metadata(str(tmp_path))

    def test_process_json_metadata_success(self, tmp_path):
        """Test successful JSON metadata processing."""
        metadata = {"version": "1.0", "test": "data"}
        json_file = tmp_path / "metadata.json"
        json_file.write_text(json.dumps(metadata))

        result = self.plotter.process_json_metadata(str(tmp_path))
        assert result == metadata

    def test_process_json_metadata_invalid_json(self, tmp_path):
        """Test processing directory with invalid JSON file."""
        json_file = tmp_path / "invalid.json"
        json_file.write_text("invalid json content")

        with pytest.raises(json.JSONDecodeError):
            self.plotter.process_json_metadata(str(tmp_path))

    def test_process_csv_v1_0_not_implemented(self):
        """Test that v1.0 CSV processing raises NotImplementedError."""
        with pytest.raises(
            NotImplementedError, match="Version 1.0 processing not yet implemented"
        ):
            self.plotter.process_csv_v1_0()
