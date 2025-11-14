"""Unit tests for data_plotter module.

This module contains comprehensive unit tests for the DataPlotter class,
verifying initialization, data loading, plot generation, and layout creation.
"""

import io
import os
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, PropertyMock, patch

import dash
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pytest

from shield_das.data_plotter import DataPlotter
from shield_das.dataset import Dataset


@pytest.fixture
def mock_dataset_path(tmp_path):
    """Create a temporary dataset directory with required files."""
    dataset_dir = tmp_path / "test_dataset"
    dataset_dir.mkdir()

    # Create dummy CSV file
    csv_file = dataset_dir / "data.csv"
    csv_file.write_text("time,pressure\n0,100\n1,101\n")

    # Create dummy metadata file
    metadata_file = dataset_dir / "run_metadata.json"
    metadata_file.write_text('{"version": "1.3", "description": "test"}')

    return str(dataset_dir)


@pytest.fixture
def mock_dataset_paths(tmp_path):
    """Create multiple temporary dataset directories."""
    paths = []
    for i in range(2):
        dataset_dir = tmp_path / f"test_dataset_{i}"
        dataset_dir.mkdir()

        csv_file = dataset_dir / f"data_{i}.csv"
        csv_file.write_text("time,pressure\n0,100\n1,101\n")

        metadata_file = dataset_dir / "run_metadata.json"
        metadata_file.write_text('{"version": "1.3", "description": "test"}')

        paths.append(str(dataset_dir))

    return paths


class TestDataPlotterInitialization:
    """Test DataPlotter initialization and configuration."""

    def test_init_with_defaults(self):
        """Test initialization with default parameters."""
        plotter = DataPlotter()

        assert plotter.dataset_paths == []
        assert plotter.dataset_names == []
        assert plotter.port == 8050
        assert isinstance(plotter.app, dash.Dash)
        assert plotter.datasets == []
        assert plotter.figure_resamplers == {}

    def test_init_with_custom_port(self):
        """Test initialization with custom port."""
        plotter = DataPlotter(port=9000)

        assert plotter.port == 9000

    def test_init_with_dataset_paths(self, mock_dataset_paths):
        """Test initialization with dataset paths."""
        plotter = DataPlotter(
            dataset_paths=mock_dataset_paths, dataset_names=["Dataset 1", "Dataset 2"]
        )

        assert plotter.dataset_paths == mock_dataset_paths
        assert plotter.dataset_names == ["Dataset 1", "Dataset 2"]

    def test_app_title_set(self):
        """Test that Dash app title is set correctly."""
        plotter = DataPlotter()

        assert plotter.app.title == "SHIELD Data Visualisation"

    def test_app_has_bootstrap_theme(self):
        """Test that app includes Bootstrap theme."""
        plotter = DataPlotter()

        external_stylesheets = plotter.app.config.external_stylesheets
        assert any("bootstrap" in str(s).lower() for s in external_stylesheets)

    def test_app_has_fontawesome_icons(self):
        """Test that app includes Font Awesome icons."""
        plotter = DataPlotter()

        external_stylesheets = plotter.app.config.external_stylesheets
        assert any("font-awesome" in str(s).lower() for s in external_stylesheets)

    def test_plot_generators_initialized_as_none(self):
        """Test that plot generators are initially None."""
        plotter = DataPlotter()

        assert plotter.upstream_plot is None
        assert plotter.downstream_plot is None
        assert plotter.temperature_plot is None
        assert plotter.permeability_plot is None


class TestDatasetPathsProperty:
    """Test dataset_paths property getter and setter."""

    def test_get_dataset_paths(self, mock_dataset_paths):
        """Test getting dataset paths."""
        plotter = DataPlotter(
            dataset_paths=mock_dataset_paths, dataset_names=["A", "B"]
        )

        assert plotter.dataset_paths == mock_dataset_paths

    def test_set_valid_dataset_paths(self, mock_dataset_paths):
        """Test setting valid dataset paths."""
        plotter = DataPlotter()
        plotter.dataset_paths = mock_dataset_paths

        assert plotter.dataset_paths == mock_dataset_paths

    def test_set_invalid_type_raises_error(self):
        """Test that setting invalid type raises ValueError."""
        plotter = DataPlotter()

        with pytest.raises(ValueError, match="must be a list of strings"):
            plotter.dataset_paths = "not a list"

    def test_set_non_string_items_raises_error(self, mock_dataset_path):
        """Test that non-string items raise ValueError."""
        plotter = DataPlotter()

        with pytest.raises(ValueError, match="must be a list of strings"):
            plotter.dataset_paths = [mock_dataset_path, 123]

    def test_set_nonexistent_path_raises_error(self):
        """Test that nonexistent paths raise ValueError."""
        plotter = DataPlotter()

        with pytest.raises(ValueError, match="does not exist"):
            plotter.dataset_paths = ["/nonexistent/path"]

    def test_set_duplicate_paths_raises_error(self, mock_dataset_path):
        """Test that duplicate paths raise ValueError."""
        plotter = DataPlotter()

        with pytest.raises(ValueError, match="must contain unique paths"):
            plotter.dataset_paths = [mock_dataset_path, mock_dataset_path]

    def test_set_path_without_csv_raises_error(self, tmp_path):
        """Test that path without CSV files raises FileNotFoundError."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        # Create metadata but no CSV
        metadata_file = empty_dir / "run_metadata.json"
        metadata_file.write_text('{"version": "1.3"}')

        plotter = DataPlotter()

        with pytest.raises(FileNotFoundError, match="No data CSV files found"):
            plotter.dataset_paths = [str(empty_dir)]

    def test_set_path_without_metadata_raises_error(self, tmp_path):
        """Test that path without metadata raises FileNotFoundError."""
        no_metadata_dir = tmp_path / "no_metadata"
        no_metadata_dir.mkdir()

        # Create CSV but no metadata
        csv_file = no_metadata_dir / "data.csv"
        csv_file.write_text("time,pressure\n")

        plotter = DataPlotter()

        with pytest.raises(FileNotFoundError, match="No run_metadata.json"):
            plotter.dataset_paths = [str(no_metadata_dir)]


class TestDatasetNamesProperty:
    """Test dataset_names property getter and setter."""

    def test_get_dataset_names(self, mock_dataset_paths):
        """Test getting dataset names."""
        plotter = DataPlotter(
            dataset_paths=mock_dataset_paths, dataset_names=["Name 1", "Name 2"]
        )

        assert plotter.dataset_names == ["Name 1", "Name 2"]

    def test_set_valid_dataset_names(self, mock_dataset_paths):
        """Test setting valid dataset names."""
        plotter = DataPlotter(
            dataset_paths=mock_dataset_paths, dataset_names=["A", "B"]
        )
        plotter.dataset_names = ["New 1", "New 2"]

        assert plotter.dataset_names == ["New 1", "New 2"]

    def test_set_invalid_type_raises_error(self, mock_dataset_paths):
        """Test that setting invalid type raises ValueError."""
        plotter = DataPlotter(
            dataset_paths=mock_dataset_paths, dataset_names=["A", "B"]
        )

        with pytest.raises(ValueError, match="must be a list of strings"):
            plotter.dataset_names = "not a list"

    def test_set_non_string_items_raises_error(self, mock_dataset_paths):
        """Test that non-string items raise ValueError."""
        plotter = DataPlotter(
            dataset_paths=mock_dataset_paths, dataset_names=["A", "B"]
        )

        with pytest.raises(ValueError, match="must be a list of strings"):
            plotter.dataset_names = ["Name 1", 123]

    def test_set_mismatched_length_raises_error(self, mock_dataset_paths):
        """Test that mismatched length raises ValueError."""
        plotter = DataPlotter(
            dataset_paths=mock_dataset_paths, dataset_names=["A", "B"]
        )

        with pytest.raises(ValueError, match="length.*must match dataset_paths length"):
            plotter.dataset_names = ["Only One"]

    def test_set_duplicate_names_raises_error(self, mock_dataset_paths):
        """Test that duplicate names raise ValueError."""
        plotter = DataPlotter(
            dataset_paths=mock_dataset_paths, dataset_names=["A", "B"]
        )

        with pytest.raises(ValueError, match="must contain unique names"):
            plotter.dataset_names = ["Same", "Same"]


class TestLoadData:
    """Test load_data method."""

    @patch("shield_das.data_plotter.Dataset")
    def test_load_data_creates_dataset(self, mock_dataset_class, mock_dataset_path):
        """Test that load_data creates a Dataset instance."""
        mock_dataset = Mock()
        mock_dataset_class.return_value = mock_dataset

        plotter = DataPlotter()
        plotter.load_data(mock_dataset_path, "Test Dataset")

        mock_dataset_class.assert_called_once_with(
            path=mock_dataset_path, name="Test Dataset"
        )

    @patch("shield_das.data_plotter.Dataset")
    def test_load_data_assigns_color(self, mock_dataset_class, mock_dataset_path):
        """Test that load_data assigns a color to the dataset."""
        mock_dataset = Mock()
        mock_dataset_class.return_value = mock_dataset

        plotter = DataPlotter()
        plotter.load_data(mock_dataset_path, "Test Dataset")

        assert mock_dataset.colour == "#000000"  # First color (black)

    @patch("shield_das.data_plotter.Dataset")
    def test_load_data_processes_data(self, mock_dataset_class, mock_dataset_path):
        """Test that load_data calls process_data."""
        mock_dataset = Mock()
        mock_dataset_class.return_value = mock_dataset

        plotter = DataPlotter()
        plotter.load_data(mock_dataset_path, "Test Dataset")

        mock_dataset.process_data.assert_called_once()

    @patch("shield_das.data_plotter.Dataset")
    def test_load_data_adds_to_datasets_list(
        self, mock_dataset_class, mock_dataset_path
    ):
        """Test that load_data adds dataset to datasets list."""
        mock_dataset = Mock()
        mock_dataset_class.return_value = mock_dataset

        plotter = DataPlotter()
        plotter.load_data(mock_dataset_path, "Test Dataset")

        assert len(plotter.datasets) == 1
        assert plotter.datasets[0] == mock_dataset

    @patch("shield_das.data_plotter.Dataset")
    def test_load_data_multiple_datasets(self, mock_dataset_class, mock_dataset_paths):
        """Test loading multiple datasets."""
        mock_datasets = [Mock(), Mock()]
        mock_dataset_class.side_effect = mock_datasets

        plotter = DataPlotter()
        plotter.load_data(mock_dataset_paths[0], "Dataset 1")
        plotter.load_data(mock_dataset_paths[1], "Dataset 2")

        assert len(plotter.datasets) == 2
        assert plotter.datasets == mock_datasets


class TestGetNextColor:
    """Test get_next_color method."""

    def test_get_next_color_first_index(self):
        """Test getting color for first dataset (index 0)."""
        plotter = DataPlotter()
        color = plotter.get_next_color(0)

        assert color == "#000000"  # Black

    def test_get_next_color_second_index(self):
        """Test getting color for second dataset (index 1)."""
        plotter = DataPlotter()
        color = plotter.get_next_color(1)

        assert color == "#DF1AD2"  # Magenta

    def test_get_next_color_wraps_around(self):
        """Test that color selection wraps around after palette is exhausted."""
        plotter = DataPlotter()
        # There are 8 colors in the palette
        color_at_8 = plotter.get_next_color(8)
        color_at_0 = plotter.get_next_color(0)

        assert color_at_8 == color_at_0  # Should wrap to first color

    def test_get_next_color_returns_hex_format(self):
        """Test that colors are returned in hex format."""
        plotter = DataPlotter()

        for i in range(10):
            color = plotter.get_next_color(i)
            assert color.startswith("#")
            assert len(color) == 7  # #RRGGBB format


class TestCreateLayout:
    """Test create_layout method."""

    @patch("shield_das.data_plotter.DatasetTable")
    def test_create_layout_returns_container(self, mock_table_class):
        """Test that create_layout returns a Container."""
        mock_table = Mock()
        mock_table.create.return_value = "<table>"
        mock_table_class.return_value = mock_table

        plotter = DataPlotter()
        plotter.datasets = []

        # Mock plot generators
        plotter._generate_upstream_plot = Mock(return_value=go.Figure())
        plotter._generate_downstream_plot = Mock(return_value=go.Figure())
        plotter._generate_temperature_plot = Mock(return_value=go.Figure())
        plotter._generate_permeability_plot = Mock(return_value=go.Figure())

        layout = plotter.create_layout()

        assert isinstance(layout, dbc.Container)

    @patch("shield_das.data_plotter.DatasetTable")
    def test_create_layout_uses_fluid_container(self, mock_table_class):
        """Test that layout uses fluid container."""
        mock_table = Mock()
        mock_table.create.return_value = "<table>"
        mock_table_class.return_value = mock_table

        plotter = DataPlotter()
        plotter.datasets = []

        # Mock plot generators
        plotter._generate_upstream_plot = Mock(return_value=go.Figure())
        plotter._generate_downstream_plot = Mock(return_value=go.Figure())
        plotter._generate_temperature_plot = Mock(return_value=go.Figure())
        plotter._generate_permeability_plot = Mock(return_value=go.Figure())

        layout = plotter.create_layout()

        assert layout.fluid is True

    @patch("shield_das.data_plotter.DatasetTable")
    def test_create_layout_creates_dataset_table(self, mock_table_class):
        """Test that create_layout creates DatasetTable with datasets."""
        mock_table = Mock()
        mock_table.create.return_value = "<table>"
        mock_table_class.return_value = mock_table

        plotter = DataPlotter()
        plotter.datasets = [Mock(), Mock()]

        # Mock plot generators
        plotter._generate_upstream_plot = Mock(return_value=go.Figure())
        plotter._generate_downstream_plot = Mock(return_value=go.Figure())
        plotter._generate_temperature_plot = Mock(return_value=go.Figure())
        plotter._generate_permeability_plot = Mock(return_value=go.Figure())

        plotter.create_layout()

        mock_table_class.assert_called_once_with(plotter.datasets)
        mock_table.create.assert_called_once()

    @patch("shield_das.data_plotter.DatasetTable")
    def test_create_layout_calls_plot_generators(self, mock_table_class):
        """Test that create_layout calls all plot generator methods."""
        mock_table = Mock()
        mock_table.create.return_value = "<table>"
        mock_table_class.return_value = mock_table

        plotter = DataPlotter()
        plotter._initialize_figure_generators = Mock()
        plotter.datasets = []
        plotter._generate_upstream_plot = Mock(return_value=go.Figure())
        plotter._generate_downstream_plot = Mock(return_value=go.Figure())
        plotter._generate_temperature_plot = Mock(return_value=go.Figure())
        plotter._generate_permeability_plot = Mock(return_value=go.Figure())

        plotter.create_layout()

        plotter._generate_upstream_plot.assert_called_once()
        plotter._generate_downstream_plot.assert_called_once()
        plotter._generate_temperature_plot.assert_called_once()
        plotter._generate_permeability_plot.assert_called_once()


class TestCreateDatasetDownload:
    """Test _create_dataset_download method."""

    def test_create_dataset_download_returns_dict(self, mock_dataset_path):
        """Test that method returns a dictionary."""
        plotter = DataPlotter()
        result = plotter._create_dataset_download(mock_dataset_path)

        assert isinstance(result, dict)

    def test_create_dataset_download_has_required_keys(self, mock_dataset_path):
        """Test that result has required keys."""
        plotter = DataPlotter()
        result = plotter._create_dataset_download(mock_dataset_path)

        assert "content" in result
        assert "filename" in result
        assert "type" in result

    def test_create_dataset_download_content_is_bytes(self, mock_dataset_path):
        """Test that content is bytes."""
        plotter = DataPlotter()
        result = plotter._create_dataset_download(mock_dataset_path)

        assert isinstance(result["content"], bytes)

    def test_create_dataset_download_creates_valid_zip(self, mock_dataset_path):
        """Test that created ZIP is valid and contains files."""
        plotter = DataPlotter()
        result = plotter._create_dataset_download(mock_dataset_path)

        # Verify it's a valid ZIP
        zip_bytes = io.BytesIO(result["content"])
        with zipfile.ZipFile(zip_bytes, "r") as zf:
            file_list = zf.namelist()
            # Should contain at least the CSV and metadata files
            assert len(file_list) >= 2

    def test_create_dataset_download_filename_format(self, mock_dataset_path):
        """Test that filename is based on folder name with .zip extension."""
        plotter = DataPlotter()
        result = plotter._create_dataset_download(mock_dataset_path)

        assert result["filename"].endswith(".zip")
        assert "test_dataset" in result["filename"]

    def test_create_dataset_download_mime_type(self, mock_dataset_path):
        """Test that MIME type is application/zip."""
        plotter = DataPlotter()
        result = plotter._create_dataset_download(mock_dataset_path)

        assert result["type"] == "application/zip"


class TestInitializeFigureGenerators:
    """Test _initialize_figure_generators method."""

    @patch("shield_das.data_plotter.PressurePlot")
    @patch("shield_das.data_plotter.TemperaturePlot")
    @patch("shield_das.data_plotter.PermeabilityPlot")
    def test_initialize_creates_all_plots(self, mock_perm, mock_temp, mock_pressure):
        """Test that all plot generators are created."""
        plotter = DataPlotter()
        plotter.datasets = [Mock()]

        plotter._initialize_figure_generators()

        # Should create 2 pressure plots (upstream + downstream)
        assert mock_pressure.call_count == 2
        mock_temp.assert_called_once()
        mock_perm.assert_called_once()

    @patch("shield_das.data_plotter.PressurePlot")
    def test_initialize_creates_upstream_plot(self, mock_pressure_class):
        """Test that upstream plot is created with correct parameters."""
        mock_plot = Mock()
        mock_pressure_class.return_value = mock_plot

        plotter = DataPlotter()
        plotter.datasets = [Mock()]

        plotter._initialize_figure_generators()

        # Check upstream plot creation
        calls = mock_pressure_class.call_args_list
        upstream_call = [c for c in calls if c[1].get("plot_type") == "upstream"][0]
        assert upstream_call[1]["plot_id"] == "upstream-plot"
        assert upstream_call[1]["datasets"] == plotter.datasets

    @patch("shield_das.data_plotter.PressurePlot")
    def test_initialize_creates_downstream_plot(self, mock_pressure_class):
        """Test that downstream plot is created with correct parameters."""
        mock_plot = Mock()
        mock_pressure_class.return_value = mock_plot

        plotter = DataPlotter()
        plotter.datasets = [Mock()]

        plotter._initialize_figure_generators()

        # Check downstream plot creation
        calls = mock_pressure_class.call_args_list
        downstream_call = [c for c in calls if c[1].get("plot_type") == "downstream"][0]
        assert downstream_call[1]["plot_id"] == "downstream-plot"
        assert downstream_call[1]["datasets"] == plotter.datasets

    @patch("shield_das.data_plotter.TemperaturePlot")
    def test_initialize_creates_temperature_plot(self, mock_temp_class):
        """Test that temperature plot is created with correct parameters."""
        mock_plot = Mock()
        mock_temp_class.return_value = mock_plot

        plotter = DataPlotter()
        plotter.datasets = [Mock()]

        plotter._initialize_figure_generators()

        mock_temp_class.assert_called_once_with(
            datasets=plotter.datasets, plot_id="temperature-plot"
        )

    @patch("shield_das.data_plotter.PermeabilityPlot")
    def test_initialize_creates_permeability_plot(self, mock_perm_class):
        """Test that permeability plot is created with correct parameters."""
        mock_plot = Mock()
        mock_perm_class.return_value = mock_plot

        plotter = DataPlotter()
        plotter.datasets = [Mock()]

        plotter._initialize_figure_generators()

        mock_perm_class.assert_called_once_with(
            datasets=plotter.datasets, plot_id="permeability-plot"
        )


class TestGeneratePlot:
    """Test _generate_plot method."""

    def test_generate_plot_upstream(self):
        """Test generating upstream plot."""
        plotter = DataPlotter()
        mock_plot = Mock()
        mock_plot.generate.return_value = go.Figure()
        mock_plot.figure_resampler = Mock()
        plotter.upstream_plot = mock_plot

        result = plotter._generate_plot("upstream")

        assert isinstance(result, go.Figure)
        mock_plot.generate.assert_called_once()

    def test_generate_plot_downstream(self):
        """Test generating downstream plot."""
        plotter = DataPlotter()
        mock_plot = Mock()
        mock_plot.generate.return_value = go.Figure()
        mock_plot.figure_resampler = Mock()
        plotter.downstream_plot = mock_plot

        result = plotter._generate_plot("downstream")

        assert isinstance(result, go.Figure)
        mock_plot.generate.assert_called_once()

    def test_generate_plot_temperature(self):
        """Test generating temperature plot."""
        plotter = DataPlotter()
        mock_plot = Mock()
        mock_plot.generate.return_value = go.Figure()
        mock_plot.figure_resampler = Mock()
        plotter.temperature_plot = mock_plot

        result = plotter._generate_plot("temperature")

        assert isinstance(result, go.Figure)
        mock_plot.generate.assert_called_once()

    def test_generate_plot_permeability(self):
        """Test generating permeability plot."""
        plotter = DataPlotter()
        mock_plot = Mock()
        mock_plot.generate.return_value = go.Figure()
        mock_plot.figure_resampler = Mock()
        plotter.permeability_plot = mock_plot

        result = plotter._generate_plot("permeability")

        assert isinstance(result, go.Figure)
        mock_plot.generate.assert_called_once()

    def test_generate_plot_invalid_type_raises_error(self):
        """Test that invalid plot type raises ValueError."""
        plotter = DataPlotter()

        with pytest.raises(ValueError, match="Unknown plot type"):
            plotter._generate_plot("invalid_type")

    def test_generate_plot_sets_parameters(self):
        """Test that plot parameters are set correctly."""
        plotter = DataPlotter()
        mock_plot = Mock()
        mock_plot.generate.return_value = go.Figure()
        mock_plot.figure_resampler = Mock()
        plotter.upstream_plot = mock_plot

        plotter._generate_plot(
            "upstream",
            show_error_bars=True,
            show_valve_times=True,
            x_scale="log",
            y_scale="linear",
            x_min=0,
            x_max=100,
            y_min=0,
            y_max=50,
        )

        expected_params = {
            "show_error_bars": True,
            "show_valve_times": True,
            "x_scale": "log",
            "y_scale": "linear",
            "x_min": 0,
            "x_max": 100,
            "y_min": 0,
            "y_max": 50,
        }
        assert mock_plot.plot_parameters == expected_params

    def test_generate_plot_valve_times_only_for_pressure(self):
        """Test that valve times are only used for pressure plots."""
        plotter = DataPlotter()

        # Temperature plot should ignore valve times
        mock_temp = Mock()
        mock_temp.generate.return_value = go.Figure()
        mock_temp.figure_resampler = Mock()
        plotter.temperature_plot = mock_temp

        plotter._generate_plot("temperature", show_valve_times=True)

        # Valve times should be False for temperature
        assert mock_temp.plot_parameters["show_valve_times"] is False

    def test_generate_plot_stores_figure_resampler(self):
        """Test that figure resampler is stored."""
        plotter = DataPlotter()
        mock_plot = Mock()
        mock_plot.generate.return_value = go.Figure()
        mock_resampler = Mock()
        mock_plot.figure_resampler = mock_resampler
        plotter.upstream_plot = mock_plot

        plotter._generate_plot("upstream")

        assert plotter.figure_resamplers["upstream-plot"] == mock_resampler


class TestGeneratePlotConvenienceMethods:
    """Test convenience wrapper methods for plot generation."""

    def test_generate_upstream_plot(self):
        """Test _generate_upstream_plot wrapper."""
        plotter = DataPlotter()
        plotter._generate_plot = Mock(return_value=go.Figure())

        result = plotter._generate_upstream_plot(show_error_bars=True)

        plotter._generate_plot.assert_called_once_with("upstream", show_error_bars=True)
        assert isinstance(result, go.Figure)

    def test_generate_downstream_plot(self):
        """Test _generate_downstream_plot wrapper."""
        plotter = DataPlotter()
        plotter._generate_plot = Mock(return_value=go.Figure())

        result = plotter._generate_downstream_plot(x_scale="log")

        plotter._generate_plot.assert_called_once_with("downstream", x_scale="log")
        assert isinstance(result, go.Figure)

    def test_generate_temperature_plot(self):
        """Test _generate_temperature_plot wrapper."""
        plotter = DataPlotter()
        plotter._generate_plot = Mock(return_value=go.Figure())

        result = plotter._generate_temperature_plot(y_min=0)

        plotter._generate_plot.assert_called_once_with("temperature", y_min=0)
        assert isinstance(result, go.Figure)

    def test_generate_permeability_plot(self):
        """Test _generate_permeability_plot wrapper."""
        plotter = DataPlotter()
        plotter._generate_plot = Mock(return_value=go.Figure())

        result = plotter._generate_permeability_plot(y_max=100)

        plotter._generate_plot.assert_called_once_with("permeability", y_max=100)
        assert isinstance(result, go.Figure)


class TestStart:
    """Test start method."""

    @patch("shield_das.data_plotter.register_all_callbacks")
    @patch("shield_das.data_plotter.threading.Timer")
    @patch("shield_das.data_plotter.webbrowser.open")
    @patch("shield_das.data_plotter.Dataset")
    def test_start_loads_all_datasets(
        self,
        mock_dataset_class,
        mock_browser,
        mock_timer,
        mock_callbacks,
        mock_dataset_paths,
    ):
        """Test that start loads all configured datasets."""

        # Create properly mocked datasets with all required attributes
        def create_mock_dataset():
            mock_ds = Mock()
            mock_ds.upstream_time = []
            mock_ds.upstream_pressure = []
            mock_ds.downstream_time = []
            mock_ds.downstream_pressure = []
            mock_ds.time_data = []
            mock_ds.thermocouple_data = None
            mock_ds.local_temperature_data = []
            return mock_ds

        mock_dataset1 = create_mock_dataset()
        mock_dataset2 = create_mock_dataset()
        mock_dataset_class.side_effect = [mock_dataset1, mock_dataset2]

        plotter = DataPlotter(
            dataset_paths=mock_dataset_paths, dataset_names=["Dataset 1", "Dataset 2"]
        )
        plotter.app.run = Mock()  # Don't actually start server

        plotter.start()

        # Should call load_data for each dataset
        assert len(plotter.datasets) == 2

    @patch("shield_das.data_plotter.register_all_callbacks")
    @patch("shield_das.data_plotter.threading.Timer")
    @patch("shield_das.data_plotter.webbrowser.open")
    @patch("shield_das.data_plotter.Dataset")
    def test_start_initializes_figure_generators(
        self,
        mock_dataset_class,
        mock_browser,
        mock_timer,
        mock_callbacks,
        mock_dataset_paths,
    ):
        """Test that start initializes figure generators."""

        def create_mock_dataset():
            mock_ds = Mock()
            mock_ds.upstream_time = []
            mock_ds.upstream_pressure = []
            mock_ds.downstream_time = []
            mock_ds.downstream_pressure = []
            mock_ds.time_data = []
            mock_ds.thermocouple_data = None
            mock_ds.local_temperature_data = []
            return mock_ds

        mock_datasets = [create_mock_dataset(), create_mock_dataset()]
        mock_dataset_class.side_effect = mock_datasets

        plotter = DataPlotter(
            dataset_paths=mock_dataset_paths, dataset_names=["Dataset 1", "Dataset 2"]
        )
        plotter.app.run = Mock()

        plotter.start()

        # Should initialize all plot generators
        assert plotter.upstream_plot is not None
        assert plotter.downstream_plot is not None
        assert plotter.temperature_plot is not None
        assert plotter.permeability_plot is not None

    @patch("shield_das.data_plotter.register_all_callbacks")
    @patch("shield_das.data_plotter.threading.Timer")
    @patch("shield_das.data_plotter.webbrowser.open")
    @patch("shield_das.data_plotter.Dataset")
    def test_start_creates_layout(
        self,
        mock_dataset_class,
        mock_browser,
        mock_timer,
        mock_callbacks,
        mock_dataset_paths,
    ):
        """Test that start creates app layout."""

        def create_mock_dataset():
            mock_ds = Mock()
            mock_ds.upstream_time = []
            mock_ds.upstream_pressure = []
            mock_ds.downstream_time = []
            mock_ds.downstream_pressure = []
            mock_ds.time_data = []
            mock_ds.thermocouple_data = None
            mock_ds.local_temperature_data = []
            return mock_ds

        mock_datasets = [create_mock_dataset(), create_mock_dataset()]
        mock_dataset_class.side_effect = mock_datasets

        plotter = DataPlotter(
            dataset_paths=mock_dataset_paths, dataset_names=["Dataset 1", "Dataset 2"]
        )
        plotter.app.run = Mock()

        plotter.start()

        # Should set app.layout
        assert plotter.app.layout is not None

    @patch("shield_das.data_plotter.register_all_callbacks")
    @patch("shield_das.data_plotter.threading.Timer")
    @patch("shield_das.data_plotter.webbrowser.open")
    @patch("shield_das.data_plotter.Dataset")
    def test_start_registers_callbacks(
        self,
        mock_dataset_class,
        mock_browser,
        mock_timer,
        mock_callbacks,
        mock_dataset_paths,
    ):
        """Test that start registers all callbacks."""

        def create_mock_dataset():
            mock_ds = Mock()
            mock_ds.upstream_time = []
            mock_ds.upstream_pressure = []
            mock_ds.downstream_time = []
            mock_ds.downstream_pressure = []
            mock_ds.time_data = []
            mock_ds.thermocouple_data = None
            mock_ds.local_temperature_data = []
            return mock_ds

        mock_datasets = [create_mock_dataset(), create_mock_dataset()]
        mock_dataset_class.side_effect = mock_datasets

        plotter = DataPlotter(
            dataset_paths=mock_dataset_paths, dataset_names=["Dataset 1", "Dataset 2"]
        )
        plotter.app.run = Mock()

        plotter.start()

        mock_callbacks.assert_called_once_with(plotter)

    @patch("shield_das.data_plotter.register_all_callbacks")
    @patch("shield_das.data_plotter.threading.Timer")
    @patch("shield_das.data_plotter.webbrowser.open")
    @patch("shield_das.data_plotter.Dataset")
    def test_start_schedules_browser_open(
        self,
        mock_dataset_class,
        mock_browser,
        mock_timer,
        mock_callbacks,
        mock_dataset_paths,
    ):
        """Test that start schedules browser to open."""

        def create_mock_dataset():
            mock_ds = Mock()
            mock_ds.upstream_time = []
            mock_ds.upstream_pressure = []
            mock_ds.downstream_time = []
            mock_ds.downstream_pressure = []
            mock_ds.time_data = []
            mock_ds.thermocouple_data = None
            mock_ds.local_temperature_data = []
            return mock_ds

        mock_datasets = [create_mock_dataset(), create_mock_dataset()]
        mock_dataset_class.side_effect = mock_datasets

        plotter = DataPlotter(
            dataset_paths=mock_dataset_paths,
            dataset_names=["Dataset 1", "Dataset 2"],
            port=9000,
        )
        plotter.app.run = Mock()

        plotter.start()

        # Should create a timer to open browser
        mock_timer.assert_called_once()
        assert mock_timer.call_args[0][0] == 0.1  # 0.1 second delay

    @patch("shield_das.data_plotter.register_all_callbacks")
    @patch("shield_das.data_plotter.threading.Timer")
    @patch("shield_das.data_plotter.webbrowser.open")
    @patch("shield_das.data_plotter.Dataset")
    def test_start_runs_server(
        self,
        mock_dataset_class,
        mock_browser,
        mock_timer,
        mock_callbacks,
        mock_dataset_paths,
    ):
        """Test that start runs the Dash server."""

        def create_mock_dataset():
            mock_ds = Mock()
            mock_ds.upstream_time = []
            mock_ds.upstream_pressure = []
            mock_ds.downstream_time = []
            mock_ds.downstream_pressure = []
            mock_ds.time_data = []
            mock_ds.thermocouple_data = None
            mock_ds.local_temperature_data = []
            return mock_ds

        mock_datasets = [create_mock_dataset(), create_mock_dataset()]
        mock_dataset_class.side_effect = mock_datasets

        plotter = DataPlotter(
            dataset_paths=mock_dataset_paths,
            dataset_names=["Dataset 1", "Dataset 2"],
            port=9000,
        )
        plotter.app.run = Mock()

        plotter.start()

        # Should call app.run with correct parameters
        plotter.app.run.assert_called_once_with(
            debug=False, host="127.0.0.1", port=9000
        )

    @patch("shield_das.data_plotter.register_all_callbacks")
    @patch("shield_das.data_plotter.threading.Timer")
    @patch("shield_das.data_plotter.webbrowser.open")
    @patch("shield_das.data_plotter.Dataset")
    def test_start_sets_custom_css(
        self,
        mock_dataset_class,
        mock_browser,
        mock_timer,
        mock_callbacks,
        mock_dataset_paths,
    ):
        """Test that start applies custom CSS."""

        def create_mock_dataset():
            mock_ds = Mock()
            mock_ds.upstream_time = []
            mock_ds.upstream_pressure = []
            mock_ds.downstream_time = []
            mock_ds.downstream_pressure = []
            mock_ds.time_data = []
            mock_ds.thermocouple_data = None
            mock_ds.local_temperature_data = []
            return mock_ds

        mock_datasets = [create_mock_dataset(), create_mock_dataset()]
        mock_dataset_class.side_effect = mock_datasets

        plotter = DataPlotter(
            dataset_paths=mock_dataset_paths, dataset_names=["Dataset 1", "Dataset 2"]
        )
        plotter.app.run = Mock()

        plotter.start()

        # Should have custom CSS for hover effects
        assert "dataset-name-input:hover" in plotter.app.index_string
        assert "color-picker-input:hover" in plotter.app.index_string
