"""Unit tests for TemperaturePlot class.

This module contains comprehensive unit tests for the TemperaturePlot class,
testing all methods and ensuring high code coverage.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import plotly.graph_objects as go

from shield_das.figures.temperature_plot import TemperaturePlot


class TestTemperaturePlotInit:
    """Tests for TemperaturePlot initialization."""

    def test_init_creates_instance_with_datasets(self):
        """Test that TemperaturePlot can be initialized with datasets."""
        mock_dataset = MagicMock()
        plot = TemperaturePlot(datasets=[mock_dataset], plot_id="test")
        assert plot.datasets == [mock_dataset]

    def test_init_creates_instance_with_empty_datasets(self):
        """Test that TemperaturePlot can be initialized with empty datasets."""
        plot = TemperaturePlot(datasets=[], plot_id="test")
        assert plot.datasets == []

    def test_init_inherits_from_base_graph(self):
        """Test that TemperaturePlot inherits from BaseGraph."""
        from shield_das.figures.base_graph import BaseGraph

        plot = TemperaturePlot(datasets=[], plot_id="test")
        assert isinstance(plot, BaseGraph)

    def test_init_sets_figure_resampler_to_none(self):
        """Test that figure_resampler is None before generation."""
        plot = TemperaturePlot(datasets=[], plot_id="test")
        assert plot.figure_resampler is None

    def test_init_sets_plot_parameters_with_defaults(self):
        """Test that plot_parameters has default values."""
        plot = TemperaturePlot(datasets=[], plot_id="test")
        assert plot.plot_parameters is not None
        assert isinstance(plot.plot_parameters, dict)

    def test_init_plot_parameters_can_be_overridden(self):
        """Test that plot_parameters defaults can be overridden."""
        mock_dataset = MagicMock()
        plot = TemperaturePlot(datasets=[mock_dataset], plot_id="test")
        plot.plot_parameters["x_scale"] = "log"
        assert plot.plot_parameters["x_scale"] == "log"

    def test_init_stores_multiple_datasets(self):
        """Test that multiple datasets are stored correctly."""
        mock_dataset1 = MagicMock()
        mock_dataset2 = MagicMock()
        plot = TemperaturePlot(datasets=[mock_dataset1, mock_dataset2], plot_id="test")
        assert len(plot.datasets) == 2
        assert mock_dataset1 in plot.datasets
        assert mock_dataset2 in plot.datasets


class TestTemperaturePlotGenerate:
    """Tests for TemperaturePlot.generate() method."""

    @patch(
        "shield_das.figures.temperature_plot.TemperaturePlot._create_figure_resampler"
    )
    @patch(
        "shield_das.figures.temperature_plot.TemperaturePlot._add_temperature_traces"
    )
    @patch("shield_das.figures.temperature_plot.TemperaturePlot._configure_layout")
    @patch("shield_das.figures.temperature_plot.TemperaturePlot._apply_axis_settings")
    @patch("shield_das.figures.temperature_plot.TemperaturePlot._clean_trace_names")
    def test_generate_calls_create_figure_resampler(
        self, mock_clean, mock_apply, mock_layout, mock_traces, mock_create
    ):
        """Test that generate() calls _create_figure_resampler."""
        mock_fig = MagicMock(spec=go.Figure)
        mock_create.return_value = mock_fig
        mock_dataset = MagicMock()
        mock_dataset.thermocouple_data = np.array([1.0, 2.0])
        mock_dataset.name = "Test"

        plot = TemperaturePlot(datasets=[mock_dataset], plot_id="test")
        plot.generate()

        mock_create.assert_called_once()

    @patch(
        "shield_das.figures.temperature_plot.TemperaturePlot._create_figure_resampler"
    )
    @patch(
        "shield_das.figures.temperature_plot.TemperaturePlot._add_temperature_traces"
    )
    @patch("shield_das.figures.temperature_plot.TemperaturePlot._configure_layout")
    @patch("shield_das.figures.temperature_plot.TemperaturePlot._apply_axis_settings")
    @patch("shield_das.figures.temperature_plot.TemperaturePlot._clean_trace_names")
    def test_generate_sets_figure_resampler(
        self, mock_clean, mock_apply, mock_layout, mock_traces, mock_create
    ):
        """Test that generate() sets self.figure_resampler."""
        mock_fig = MagicMock(spec=go.Figure)
        mock_create.return_value = mock_fig
        mock_dataset = MagicMock()
        mock_dataset.thermocouple_data = np.array([1.0, 2.0])
        mock_dataset.name = "Test"

        plot = TemperaturePlot(datasets=[mock_dataset], plot_id="test")
        plot.generate()

        assert plot.figure_resampler is mock_fig

    @patch(
        "shield_das.figures.temperature_plot.TemperaturePlot._create_figure_resampler"
    )
    @patch(
        "shield_das.figures.temperature_plot.TemperaturePlot._add_temperature_traces"
    )
    @patch("shield_das.figures.temperature_plot.TemperaturePlot._configure_layout")
    @patch("shield_das.figures.temperature_plot.TemperaturePlot._apply_axis_settings")
    @patch("shield_das.figures.temperature_plot.TemperaturePlot._clean_trace_names")
    def test_generate_adds_traces_for_datasets_with_temperature(
        self, mock_clean, mock_apply, mock_layout, mock_traces, mock_create
    ):
        """Test that generate() adds traces for datasets with temperature data."""
        mock_fig = MagicMock(spec=go.Figure)
        mock_create.return_value = mock_fig
        mock_dataset = MagicMock()
        mock_dataset.thermocouple_data = np.array([1.0, 2.0])
        mock_dataset.name = "Test"

        plot = TemperaturePlot(datasets=[mock_dataset], plot_id="test")
        plot.generate()

        mock_traces.assert_called_once_with(mock_fig, mock_dataset)

    @patch(
        "shield_das.figures.temperature_plot.TemperaturePlot._create_figure_resampler"
    )
    @patch(
        "shield_das.figures.temperature_plot.TemperaturePlot._add_temperature_traces"
    )
    @patch("shield_das.figures.temperature_plot.TemperaturePlot._configure_layout")
    @patch("shield_das.figures.temperature_plot.TemperaturePlot._apply_axis_settings")
    @patch("shield_das.figures.temperature_plot.TemperaturePlot._clean_trace_names")
    def test_generate_skips_datasets_without_temperature(
        self, mock_clean, mock_apply, mock_layout, mock_traces, mock_create
    ):
        """Test that generate() skips datasets without temperature data."""
        mock_fig = MagicMock(spec=go.Figure)
        mock_create.return_value = mock_fig
        mock_dataset = MagicMock()
        mock_dataset.thermocouple_data = None
        mock_dataset.name = "Test"

        plot = TemperaturePlot(datasets=[mock_dataset], plot_id="test")
        plot.generate()

        mock_traces.assert_not_called()

    @patch(
        "shield_das.figures.temperature_plot.TemperaturePlot._create_figure_resampler"
    )
    @patch(
        "shield_das.figures.temperature_plot.TemperaturePlot._add_temperature_traces"
    )
    @patch("shield_das.figures.temperature_plot.TemperaturePlot._configure_layout")
    @patch("shield_das.figures.temperature_plot.TemperaturePlot._apply_axis_settings")
    @patch(
        "shield_das.figures.temperature_plot.TemperaturePlot._add_missing_data_annotation"
    )
    @patch("shield_das.figures.temperature_plot.TemperaturePlot._clean_trace_names")
    def test_generate_adds_missing_annotation_when_no_data(
        self,
        mock_clean,
        mock_missing,
        mock_apply,
        mock_layout,
        mock_traces,
        mock_create,
    ):
        """Test that generate() adds missing annotation when no data."""
        mock_fig = MagicMock(spec=go.Figure)
        mock_create.return_value = mock_fig
        mock_dataset = MagicMock()
        mock_dataset.thermocouple_data = None
        mock_dataset.name = "Test"

        plot = TemperaturePlot(datasets=[mock_dataset], plot_id="test")
        plot.generate()

        mock_missing.assert_called_once_with(mock_fig, ["Test"], False)

    @patch(
        "shield_das.figures.temperature_plot.TemperaturePlot._create_figure_resampler"
    )
    @patch(
        "shield_das.figures.temperature_plot.TemperaturePlot._add_temperature_traces"
    )
    @patch("shield_das.figures.temperature_plot.TemperaturePlot._configure_layout")
    @patch("shield_das.figures.temperature_plot.TemperaturePlot._apply_axis_settings")
    @patch(
        "shield_das.figures.temperature_plot.TemperaturePlot._add_missing_data_annotation"
    )
    @patch("shield_das.figures.temperature_plot.TemperaturePlot._clean_trace_names")
    def test_generate_adds_missing_annotation_with_has_some_data_flag(
        self,
        mock_clean,
        mock_missing,
        mock_apply,
        mock_layout,
        mock_traces,
        mock_create,
    ):
        """Test that generate() correctly sets has_some_data flag."""
        mock_fig = MagicMock(spec=go.Figure)
        mock_create.return_value = mock_fig
        mock_dataset1 = MagicMock()
        mock_dataset1.thermocouple_data = np.array([1.0, 2.0])
        mock_dataset1.name = "Has Data"
        mock_dataset2 = MagicMock()
        mock_dataset2.thermocouple_data = None
        mock_dataset2.name = "No Data"

        plot = TemperaturePlot(datasets=[mock_dataset1, mock_dataset2], plot_id="test")
        plot.generate()

        mock_missing.assert_called_once_with(mock_fig, ["No Data"], True)

    @patch(
        "shield_das.figures.temperature_plot.TemperaturePlot._create_figure_resampler"
    )
    @patch(
        "shield_das.figures.temperature_plot.TemperaturePlot._add_temperature_traces"
    )
    @patch("shield_das.figures.temperature_plot.TemperaturePlot._configure_layout")
    @patch("shield_das.figures.temperature_plot.TemperaturePlot._apply_axis_settings")
    @patch("shield_das.figures.temperature_plot.TemperaturePlot._clean_trace_names")
    def test_generate_calls_configure_layout(
        self, mock_clean, mock_apply, mock_layout, mock_traces, mock_create
    ):
        """Test that generate() calls _configure_layout."""
        mock_fig = MagicMock(spec=go.Figure)
        mock_create.return_value = mock_fig
        mock_dataset = MagicMock()
        mock_dataset.thermocouple_data = np.array([1.0, 2.0])

        plot = TemperaturePlot(datasets=[mock_dataset], plot_id="test")
        plot.generate()

        mock_layout.assert_called_once_with(mock_fig)

    @patch(
        "shield_das.figures.temperature_plot.TemperaturePlot._create_figure_resampler"
    )
    @patch(
        "shield_das.figures.temperature_plot.TemperaturePlot._add_temperature_traces"
    )
    @patch("shield_das.figures.temperature_plot.TemperaturePlot._configure_layout")
    @patch("shield_das.figures.temperature_plot.TemperaturePlot._apply_axis_settings")
    @patch("shield_das.figures.temperature_plot.TemperaturePlot._clean_trace_names")
    def test_generate_calls_apply_axis_settings(
        self, mock_clean, mock_apply, mock_layout, mock_traces, mock_create
    ):
        """Test that generate() calls _apply_axis_settings."""
        mock_fig = MagicMock(spec=go.Figure)
        mock_create.return_value = mock_fig
        mock_dataset = MagicMock()
        mock_dataset.thermocouple_data = np.array([1.0, 2.0])

        plot = TemperaturePlot(datasets=[mock_dataset], plot_id="test")
        plot.generate()

        mock_apply.assert_called_once()

    @patch(
        "shield_das.figures.temperature_plot.TemperaturePlot._create_figure_resampler"
    )
    @patch(
        "shield_das.figures.temperature_plot.TemperaturePlot._add_temperature_traces"
    )
    @patch("shield_das.figures.temperature_plot.TemperaturePlot._configure_layout")
    @patch("shield_das.figures.temperature_plot.TemperaturePlot._apply_axis_settings")
    @patch("shield_das.figures.temperature_plot.TemperaturePlot._clean_trace_names")
    def test_generate_calls_clean_trace_names(
        self, mock_clean, mock_apply, mock_layout, mock_traces, mock_create
    ):
        """Test that generate() calls _clean_trace_names."""
        mock_fig = MagicMock(spec=go.Figure)
        mock_create.return_value = mock_fig
        mock_dataset = MagicMock()
        mock_dataset.thermocouple_data = np.array([1.0, 2.0])

        plot = TemperaturePlot(datasets=[mock_dataset], plot_id="test")
        plot.generate()

        mock_clean.assert_called_once()

    @patch(
        "shield_das.figures.temperature_plot.TemperaturePlot._create_figure_resampler"
    )
    @patch(
        "shield_das.figures.temperature_plot.TemperaturePlot._add_temperature_traces"
    )
    @patch("shield_das.figures.temperature_plot.TemperaturePlot._configure_layout")
    @patch("shield_das.figures.temperature_plot.TemperaturePlot._apply_axis_settings")
    @patch("shield_das.figures.temperature_plot.TemperaturePlot._clean_trace_names")
    def test_generate_returns_figure(
        self, mock_clean, mock_apply, mock_layout, mock_traces, mock_create
    ):
        """Test that generate() returns the figure."""
        mock_fig = MagicMock(spec=go.Figure)
        mock_create.return_value = mock_fig
        mock_dataset = MagicMock()
        mock_dataset.thermocouple_data = np.array([1.0, 2.0])

        plot = TemperaturePlot(datasets=[mock_dataset], plot_id="test")
        result = plot.generate()

        assert result is mock_fig

    @patch(
        "shield_das.figures.temperature_plot.TemperaturePlot._create_figure_resampler"
    )
    @patch(
        "shield_das.figures.temperature_plot.TemperaturePlot._add_temperature_traces"
    )
    @patch("shield_das.figures.temperature_plot.TemperaturePlot._configure_layout")
    @patch("shield_das.figures.temperature_plot.TemperaturePlot._apply_axis_settings")
    @patch(
        "shield_das.figures.temperature_plot.TemperaturePlot._add_missing_data_annotation"
    )
    @patch("shield_das.figures.temperature_plot.TemperaturePlot._clean_trace_names")
    def test_generate_handles_multiple_datasets_with_mixed_data(
        self,
        mock_clean,
        mock_missing,
        mock_apply,
        mock_layout,
        mock_traces,
        mock_create,
    ):
        """Test that generate() handles multiple datasets with mixed data."""
        mock_fig = MagicMock(spec=go.Figure)
        mock_create.return_value = mock_fig
        mock_dataset1 = MagicMock()
        mock_dataset1.thermocouple_data = np.array([1.0, 2.0])
        mock_dataset1.name = "Dataset1"
        mock_dataset2 = MagicMock()
        mock_dataset2.thermocouple_data = None
        mock_dataset2.name = "Dataset2"
        mock_dataset3 = MagicMock()
        mock_dataset3.thermocouple_data = np.array([3.0, 4.0])
        mock_dataset3.name = "Dataset3"

        plot = TemperaturePlot(
            datasets=[mock_dataset1, mock_dataset2, mock_dataset3],
            plot_id="test",
        )
        plot.generate()

        assert mock_traces.call_count == 2
        mock_missing.assert_called_once_with(mock_fig, ["Dataset2"], True)

    @patch(
        "shield_das.figures.temperature_plot.TemperaturePlot._create_figure_resampler"
    )
    @patch(
        "shield_das.figures.temperature_plot.TemperaturePlot._add_temperature_traces"
    )
    @patch("shield_das.figures.temperature_plot.TemperaturePlot._configure_layout")
    @patch("shield_das.figures.temperature_plot.TemperaturePlot._apply_axis_settings")
    @patch(
        "shield_das.figures.temperature_plot.TemperaturePlot._add_missing_data_annotation"
    )
    @patch("shield_das.figures.temperature_plot.TemperaturePlot._clean_trace_names")
    def test_generate_does_not_add_annotation_when_all_have_data(
        self,
        mock_clean,
        mock_missing,
        mock_apply,
        mock_layout,
        mock_traces,
        mock_create,
    ):
        """Test that annotation not added when all have data."""
        mock_fig = MagicMock(spec=go.Figure)
        mock_create.return_value = mock_fig
        mock_dataset1 = MagicMock()
        mock_dataset1.thermocouple_data = np.array([1.0, 2.0])
        mock_dataset1.name = "Dataset1"
        mock_dataset2 = MagicMock()
        mock_dataset2.thermocouple_data = np.array([3.0, 4.0])
        mock_dataset2.name = "Dataset2"

        plot = TemperaturePlot(datasets=[mock_dataset1, mock_dataset2], plot_id="test")
        plot.generate()

        mock_missing.assert_not_called()


class TestAddTemperatureTraces:
    """Tests for TemperaturePlot._add_temperature_traces() method."""

    def test_add_temperature_traces_adds_two_traces(self):
        """Test _add_temperature_traces adds two traces."""
        mock_fig = MagicMock(spec=go.Figure)
        mock_dataset = MagicMock()
        mock_dataset.time_data = np.array([1.0, 2.0, 3.0])
        mock_dataset.local_temperature_data = np.array([100.0, 200.0, 300.0])
        mock_dataset.thermocouple_data = np.array([90.0, 190.0, 290.0])
        mock_dataset.colour = "red"
        mock_dataset.name = "Test"
        mock_dataset.thermocouple_name = "TC1"

        plot = TemperaturePlot(datasets=[mock_dataset], plot_id="test")
        plot._add_temperature_traces(mock_fig, mock_dataset)

        assert mock_fig.add_trace.call_count == 2

    def test_add_temperature_traces_furnace_trace_has_dashed_line(self):
        """Test that furnace setpoint trace has dashed line style."""
        mock_fig = MagicMock(spec=go.Figure)
        mock_dataset = MagicMock()
        mock_dataset.time_data = np.array([1.0, 2.0, 3.0])
        mock_dataset.local_temperature_data = np.array([100.0, 200.0, 300.0])
        mock_dataset.thermocouple_data = np.array([90.0, 190.0, 290.0])
        mock_dataset.colour = "red"
        mock_dataset.name = "Test"
        mock_dataset.thermocouple_name = "TC1"

        plot = TemperaturePlot(datasets=[mock_dataset], plot_id="test")
        plot._add_temperature_traces(mock_fig, mock_dataset)

        first_call = mock_fig.add_trace.call_args_list[0]
        scatter_obj = first_call[0][0]
        assert scatter_obj.line["dash"] == "dash"

    def test_add_temperature_traces_thermocouple_trace_has_solid_line(self):
        """Test that thermocouple trace has solid line style (no dash)."""
        mock_fig = MagicMock(spec=go.Figure)
        mock_dataset = MagicMock()
        mock_dataset.time_data = np.array([1.0, 2.0, 3.0])
        mock_dataset.local_temperature_data = np.array([100.0, 200.0, 300.0])
        mock_dataset.thermocouple_data = np.array([90.0, 190.0, 290.0])
        mock_dataset.colour = "red"
        mock_dataset.name = "Test"
        mock_dataset.thermocouple_name = "TC1"

        plot = TemperaturePlot(datasets=[mock_dataset], plot_id="test")
        plot._add_temperature_traces(mock_fig, mock_dataset)

        second_call = mock_fig.add_trace.call_args_list[1]
        scatter_obj = second_call[0][0]
        assert getattr(scatter_obj.line, "dash", None) is None

    def test_add_temperature_traces_uses_dataset_colour(self):
        """Test that both traces use the dataset's colour."""
        mock_fig = MagicMock(spec=go.Figure)
        mock_dataset = MagicMock()
        mock_dataset.time_data = np.array([1.0, 2.0, 3.0])
        mock_dataset.local_temperature_data = np.array([100.0, 200.0, 300.0])
        mock_dataset.thermocouple_data = np.array([90.0, 190.0, 290.0])
        mock_dataset.colour = "blue"
        mock_dataset.name = "Test"
        mock_dataset.thermocouple_name = "TC1"

        plot = TemperaturePlot(datasets=[mock_dataset], plot_id="test")
        plot._add_temperature_traces(mock_fig, mock_dataset)

        for call in mock_fig.add_trace.call_args_list:
            scatter_obj = call[0][0]
            assert scatter_obj.line["color"] == "blue"

    def test_add_temperature_traces_furnace_name_includes_dataset_name(self):
        """Test that furnace trace name includes dataset name."""
        mock_fig = MagicMock(spec=go.Figure)
        mock_dataset = MagicMock()
        mock_dataset.time_data = np.array([1.0, 2.0, 3.0])
        mock_dataset.local_temperature_data = np.array([100.0, 200.0, 300.0])
        mock_dataset.thermocouple_data = np.array([90.0, 190.0, 290.0])
        mock_dataset.colour = "red"
        mock_dataset.name = "MyDataset"
        mock_dataset.thermocouple_name = "TC1"

        plot = TemperaturePlot(datasets=[mock_dataset], plot_id="test")
        plot._add_temperature_traces(mock_fig, mock_dataset)

        first_call = mock_fig.add_trace.call_args_list[0]
        scatter_obj = first_call[0][0]
        assert "MyDataset" in scatter_obj.name
        assert "Furnace setpoint" in scatter_obj.name

    def test_add_temperature_traces_thermocouple_name_includes_dataset_name(self):
        """Test that thermocouple trace name includes dataset name."""
        mock_fig = MagicMock(spec=go.Figure)
        mock_dataset = MagicMock()
        mock_dataset.time_data = np.array([1.0, 2.0, 3.0])
        mock_dataset.local_temperature_data = np.array([100.0, 200.0, 300.0])
        mock_dataset.thermocouple_data = np.array([90.0, 190.0, 290.0])
        mock_dataset.colour = "red"
        mock_dataset.name = "MyDataset"
        mock_dataset.thermocouple_name = "TC1"

        plot = TemperaturePlot(datasets=[mock_dataset], plot_id="test")
        plot._add_temperature_traces(mock_fig, mock_dataset)

        second_call = mock_fig.add_trace.call_args_list[1]
        scatter_obj = second_call[0][0]
        assert "MyDataset" in scatter_obj.name

    def test_add_temperature_traces_uses_custom_thermocouple_name(self):
        """Test that thermocouple trace uses custom thermocouple name."""
        mock_fig = MagicMock(spec=go.Figure)
        mock_dataset = MagicMock()
        mock_dataset.time_data = np.array([1.0, 2.0, 3.0])
        mock_dataset.local_temperature_data = np.array([100.0, 200.0, 300.0])
        mock_dataset.thermocouple_data = np.array([90.0, 190.0, 290.0])
        mock_dataset.colour = "red"
        mock_dataset.name = "Test"
        mock_dataset.thermocouple_name = "CustomTC"

        plot = TemperaturePlot(datasets=[mock_dataset], plot_id="test")
        plot._add_temperature_traces(mock_fig, mock_dataset)

        second_call = mock_fig.add_trace.call_args_list[1]
        scatter_obj = second_call[0][0]
        assert "CustomTC" in scatter_obj.name

    def test_add_temperature_traces_uses_default_thermocouple_name_when_none(
        self,
    ):
        """Test default thermocouple name used when None."""
        mock_fig = MagicMock(spec=go.Figure)
        mock_dataset = MagicMock()
        mock_dataset.time_data = np.array([1.0, 2.0, 3.0])
        mock_dataset.local_temperature_data = np.array([100.0, 200.0, 300.0])
        mock_dataset.thermocouple_data = np.array([90.0, 190.0, 290.0])
        mock_dataset.colour = "red"
        mock_dataset.name = "Test"
        mock_dataset.thermocouple_name = None

        plot = TemperaturePlot(datasets=[mock_dataset], plot_id="test")
        plot._add_temperature_traces(mock_fig, mock_dataset)

        second_call = mock_fig.add_trace.call_args_list[1]
        scatter_obj = second_call[0][0]
        assert "Thermocouple" in scatter_obj.name

    def test_add_temperature_traces_passes_time_data_as_hf_x(self):
        """Test that time_data is passed as hf_x parameter."""
        mock_fig = MagicMock(spec=go.Figure)
        mock_dataset = MagicMock()
        time_data = np.array([1.0, 2.0, 3.0])
        mock_dataset.time_data = time_data
        mock_dataset.local_temperature_data = np.array([100.0, 200.0, 300.0])
        mock_dataset.thermocouple_data = np.array([90.0, 190.0, 290.0])
        mock_dataset.colour = "red"
        mock_dataset.name = "Test"
        mock_dataset.thermocouple_name = "TC1"

        plot = TemperaturePlot(datasets=[mock_dataset], plot_id="test")
        plot._add_temperature_traces(mock_fig, mock_dataset)

        for call in mock_fig.add_trace.call_args_list:
            assert "hf_x" in call[1]
            np.testing.assert_array_equal(call[1]["hf_x"], time_data)

    def test_add_temperature_traces_passes_local_temp_as_hf_y_for_furnace(self):
        """Test that local_temperature_data is passed as hf_y for furnace trace."""
        mock_fig = MagicMock(spec=go.Figure)
        mock_dataset = MagicMock()
        local_temp = np.array([100.0, 200.0, 300.0])
        mock_dataset.time_data = np.array([1.0, 2.0, 3.0])
        mock_dataset.local_temperature_data = local_temp
        mock_dataset.thermocouple_data = np.array([90.0, 190.0, 290.0])
        mock_dataset.colour = "red"
        mock_dataset.name = "Test"
        mock_dataset.thermocouple_name = "TC1"

        plot = TemperaturePlot(datasets=[mock_dataset], plot_id="test")
        plot._add_temperature_traces(mock_fig, mock_dataset)

        first_call = mock_fig.add_trace.call_args_list[0]
        np.testing.assert_array_equal(first_call[1]["hf_y"], local_temp)

    def test_add_temperature_traces_passes_thermocouple_data_as_hf_y(self):
        """Test thermocouple_data passed as hf_y for thermocouple trace."""
        mock_fig = MagicMock(spec=go.Figure)
        mock_dataset = MagicMock()
        thermocouple_temp = np.array([90.0, 190.0, 290.0])
        mock_dataset.time_data = np.array([1.0, 2.0, 3.0])
        mock_dataset.local_temperature_data = np.array([100.0, 200.0, 300.0])
        mock_dataset.thermocouple_data = thermocouple_temp
        mock_dataset.colour = "red"
        mock_dataset.name = "Test"
        mock_dataset.thermocouple_name = "TC1"

        plot = TemperaturePlot(datasets=[mock_dataset], plot_id="test")
        plot._add_temperature_traces(mock_fig, mock_dataset)

        second_call = mock_fig.add_trace.call_args_list[1]
        np.testing.assert_array_equal(second_call[1]["hf_y"], thermocouple_temp)

    def test_add_temperature_traces_converts_data_to_contiguous_arrays(self):
        """Test that data is converted to contiguous arrays."""
        mock_fig = MagicMock(spec=go.Figure)
        mock_dataset = MagicMock()
        # Non-contiguous array
        mock_dataset.time_data = np.array([1.0, 2.0, 3.0, 4.0])[::2]
        local_temp_arr = np.array([100.0, 200.0, 300.0, 400.0])[::2]
        mock_dataset.local_temperature_data = local_temp_arr
        tc_arr = np.array([90.0, 190.0, 290.0, 390.0])[::2]
        mock_dataset.thermocouple_data = tc_arr
        mock_dataset.colour = "red"
        mock_dataset.name = "Test"
        mock_dataset.thermocouple_name = "TC1"

        plot = TemperaturePlot(datasets=[mock_dataset], plot_id="test")
        plot._add_temperature_traces(mock_fig, mock_dataset)

        # If it runs without error, the data was successfully converted
        assert mock_fig.add_trace.call_count == 2


class TestConfigureLayout:
    """Tests for TemperaturePlot._configure_layout() method."""

    def test_configure_layout_updates_layout(self):
        """Test that _configure_layout calls update_layout on figure_resampler."""
        mock_fig = MagicMock(spec=go.Figure)
        plot = TemperaturePlot(datasets=[], plot_id="test")
        plot.figure_resampler = MagicMock()

        plot._configure_layout(mock_fig)

        plot.figure_resampler.update_layout.assert_called_once()

    def test_configure_layout_sets_height(self):
        """Test that _configure_layout sets height to 500."""
        mock_fig = MagicMock(spec=go.Figure)
        plot = TemperaturePlot(datasets=[], plot_id="test")
        plot.figure_resampler = MagicMock()

        plot._configure_layout(mock_fig)

        call_kwargs = plot.figure_resampler.update_layout.call_args[1]
        assert call_kwargs["height"] == 500

    def test_configure_layout_sets_xaxis_title(self):
        """Test that _configure_layout sets x-axis title to 'Time (s)'."""
        mock_fig = MagicMock(spec=go.Figure)
        plot = TemperaturePlot(datasets=[], plot_id="test")
        plot.figure_resampler = MagicMock()

        plot._configure_layout(mock_fig)

        call_kwargs = plot.figure_resampler.update_layout.call_args[1]
        assert call_kwargs["xaxis_title"] == "Time (s)"

    def test_configure_layout_sets_yaxis_title(self):
        """Test that _configure_layout sets y-axis title to 'Temperature (°C)'."""
        mock_fig = MagicMock(spec=go.Figure)
        plot = TemperaturePlot(datasets=[], plot_id="test")
        plot.figure_resampler = MagicMock()

        plot._configure_layout(mock_fig)

        call_kwargs = plot.figure_resampler.update_layout.call_args[1]
        assert call_kwargs["yaxis_title"] == "Temperature (°C)"

    def test_configure_layout_sets_template(self):
        """Test that _configure_layout sets template to 'plotly_white'."""
        mock_fig = MagicMock(spec=go.Figure)
        plot = TemperaturePlot(datasets=[], plot_id="test")
        plot.figure_resampler = MagicMock()

        plot._configure_layout(mock_fig)

        call_kwargs = plot.figure_resampler.update_layout.call_args[1]
        assert call_kwargs["template"] == "plotly_white"

    def test_configure_layout_sets_margins(self):
        """Test that _configure_layout sets margins."""
        mock_fig = MagicMock(spec=go.Figure)
        plot = TemperaturePlot(datasets=[], plot_id="test")
        plot.figure_resampler = MagicMock()

        plot._configure_layout(mock_fig)

        call_kwargs = plot.figure_resampler.update_layout.call_args[1]
        assert "margin" in call_kwargs
        assert call_kwargs["margin"]["l"] == 60
        assert call_kwargs["margin"]["r"] == 30
        assert call_kwargs["margin"]["t"] == 40
        assert call_kwargs["margin"]["b"] == 60

    def test_configure_layout_sets_legend_orientation(self):
        """Test that _configure_layout sets legend orientation to horizontal."""
        mock_fig = MagicMock(spec=go.Figure)
        plot = TemperaturePlot(datasets=[], plot_id="test")
        plot.figure_resampler = MagicMock()

        plot._configure_layout(mock_fig)

        call_kwargs = plot.figure_resampler.update_layout.call_args[1]
        assert call_kwargs["legend"]["orientation"] == "h"

    def test_configure_layout_sets_legend_position(self):
        """Test that _configure_layout sets legend position above plot."""
        mock_fig = MagicMock(spec=go.Figure)
        plot = TemperaturePlot(datasets=[], plot_id="test")
        plot.figure_resampler = MagicMock()

        plot._configure_layout(mock_fig)

        call_kwargs = plot.figure_resampler.update_layout.call_args[1]
        assert call_kwargs["legend"]["yanchor"] == "bottom"
        assert call_kwargs["legend"]["y"] == 1.02
        assert call_kwargs["legend"]["xanchor"] == "center"
        assert call_kwargs["legend"]["x"] == 0.5


class TestAddMissingDataAnnotation:
    """Tests for TemperaturePlot._add_missing_data_annotation() method."""

    def test_add_missing_annotation_no_data_centers_message(self):
        """Test that annotation is centered when no temperature data exists."""
        mock_fig = MagicMock(spec=go.Figure)
        plot = TemperaturePlot(datasets=[], plot_id="test")

        plot._add_missing_data_annotation(mock_fig, ["Dataset1"], False)

        call_kwargs = mock_fig.add_annotation.call_args[1]
        assert call_kwargs["x"] == 0.5
        assert call_kwargs["y"] == 0.5
        assert call_kwargs["xanchor"] == "center"
        assert call_kwargs["yanchor"] == "middle"

    def test_add_missing_annotation_no_data_uses_larger_font(self):
        """Test that annotation uses larger font when no data exists."""
        mock_fig = MagicMock(spec=go.Figure)
        plot = TemperaturePlot(datasets=[], plot_id="test")

        plot._add_missing_data_annotation(mock_fig, ["Dataset1"], False)

        call_kwargs = mock_fig.add_annotation.call_args[1]
        assert call_kwargs["font"]["size"] == 16

    def test_add_missing_annotation_no_data_shows_generic_message(self):
        """Test that generic message is shown when no data exists."""
        mock_fig = MagicMock(spec=go.Figure)
        plot = TemperaturePlot(datasets=[], plot_id="test")

        plot._add_missing_data_annotation(mock_fig, ["Dataset1"], False)

        call_kwargs = mock_fig.add_annotation.call_args[1]
        assert call_kwargs["text"] == "No temperature data available"

    def test_add_missing_annotation_some_data_positions_in_corner(self):
        """Test that annotation is in corner when some data exists."""
        mock_fig = MagicMock(spec=go.Figure)
        plot = TemperaturePlot(datasets=[], plot_id="test")

        plot._add_missing_data_annotation(mock_fig, ["Dataset1"], True)

        call_kwargs = mock_fig.add_annotation.call_args[1]
        assert call_kwargs["x"] == 0.98
        assert call_kwargs["y"] == 0.02
        assert call_kwargs["xanchor"] == "right"
        assert call_kwargs["yanchor"] == "bottom"

    def test_add_missing_annotation_some_data_uses_smaller_font(self):
        """Test that annotation uses smaller font when some data exists."""
        mock_fig = MagicMock(spec=go.Figure)
        plot = TemperaturePlot(datasets=[], plot_id="test")

        plot._add_missing_data_annotation(mock_fig, ["Dataset1"], True)

        call_kwargs = mock_fig.add_annotation.call_args[1]
        assert call_kwargs["font"]["size"] == 10

    def test_add_missing_annotation_single_dataset_includes_name(self):
        """Test that annotation includes dataset name for single missing dataset."""
        mock_fig = MagicMock(spec=go.Figure)
        plot = TemperaturePlot(datasets=[], plot_id="test")

        plot._add_missing_data_annotation(mock_fig, ["MyDataset"], True)

        call_kwargs = mock_fig.add_annotation.call_args[1]
        assert "MyDataset" in call_kwargs["text"]
        assert "No temperature data for:" in call_kwargs["text"]

    def test_add_missing_annotation_multiple_datasets_lists_all(self):
        """Test that annotation lists all dataset names when multiple are missing."""
        mock_fig = MagicMock(spec=go.Figure)
        plot = TemperaturePlot(datasets=[], plot_id="test")

        plot._add_missing_data_annotation(mock_fig, ["Dataset1", "Dataset2"], True)

        call_kwargs = mock_fig.add_annotation.call_args[1]
        assert "Dataset1" in call_kwargs["text"]
        assert "Dataset2" in call_kwargs["text"]
        assert "No temperature data for:" in call_kwargs["text"]

    def test_add_missing_annotation_sets_background_color(self):
        """Test that annotation has white background with transparency."""
        mock_fig = MagicMock(spec=go.Figure)
        plot = TemperaturePlot(datasets=[], plot_id="test")

        plot._add_missing_data_annotation(mock_fig, ["Dataset1"], True)

        call_kwargs = mock_fig.add_annotation.call_args[1]
        assert call_kwargs["bgcolor"] == "rgba(255, 255, 255, 0.8)"

    def test_add_missing_annotation_sets_border(self):
        """Test that annotation has gray border."""
        mock_fig = MagicMock(spec=go.Figure)
        plot = TemperaturePlot(datasets=[], plot_id="test")

        plot._add_missing_data_annotation(mock_fig, ["Dataset1"], True)

        call_kwargs = mock_fig.add_annotation.call_args[1]
        assert call_kwargs["bordercolor"] == "gray"
        assert call_kwargs["borderwidth"] == 1

    def test_add_missing_annotation_sets_padding(self):
        """Test that annotation has border padding."""
        mock_fig = MagicMock(spec=go.Figure)
        plot = TemperaturePlot(datasets=[], plot_id="test")

        plot._add_missing_data_annotation(mock_fig, ["Dataset1"], True)

        call_kwargs = mock_fig.add_annotation.call_args[1]
        assert call_kwargs["borderpad"] == 4

    def test_add_missing_annotation_disables_arrow(self):
        """Test that annotation has no arrow."""
        mock_fig = MagicMock(spec=go.Figure)
        plot = TemperaturePlot(datasets=[], plot_id="test")

        plot._add_missing_data_annotation(mock_fig, ["Dataset1"], True)

        call_kwargs = mock_fig.add_annotation.call_args[1]
        assert call_kwargs["showarrow"] is False

    def test_add_missing_annotation_uses_paper_coordinates(self):
        """Test that annotation uses paper coordinates."""
        mock_fig = MagicMock(spec=go.Figure)
        plot = TemperaturePlot(datasets=[], plot_id="test")

        plot._add_missing_data_annotation(mock_fig, ["Dataset1"], True)

        call_kwargs = mock_fig.add_annotation.call_args[1]
        assert call_kwargs["xref"] == "paper"
        assert call_kwargs["yref"] == "paper"

    def test_add_missing_annotation_font_is_gray(self):
        """Test that annotation font color is gray."""
        mock_fig = MagicMock(spec=go.Figure)
        plot = TemperaturePlot(datasets=[], plot_id="test")

        plot._add_missing_data_annotation(mock_fig, ["Dataset1"], True)

        call_kwargs = mock_fig.add_annotation.call_args[1]
        assert call_kwargs["font"]["color"] == "gray"


class TestCleanTraceNames:
    """Tests for TemperaturePlot._clean_trace_names() method."""

    def test_clean_trace_names_removes_r_prefix(self):
        """Test that _clean_trace_names removes '[R] ' prefix."""
        mock_trace = MagicMock()
        mock_trace.name = "[R] Test trace"
        plot = TemperaturePlot(datasets=[], plot_id="test")
        plot.figure_resampler = MagicMock()
        plot.figure_resampler.data = [mock_trace]

        plot._clean_trace_names()

        assert mock_trace.name == "Test trace"

    def test_clean_trace_names_removes_r_suffix(self):
        """Test that _clean_trace_names removes '[R]' suffix."""
        mock_trace = MagicMock()
        mock_trace.name = "Test trace[R]"
        plot = TemperaturePlot(datasets=[], plot_id="test")
        plot.figure_resampler = MagicMock()
        plot.figure_resampler.data = [mock_trace]

        plot._clean_trace_names()

        assert mock_trace.name == "Test trace"

    def test_clean_trace_names_handles_traces_without_name_attribute(self):
        """Test that _clean_trace_names handles traces without name attribute."""
        mock_trace = MagicMock(spec=[])  # No 'name' attribute
        plot = TemperaturePlot(datasets=[], plot_id="test")
        plot.figure_resampler = MagicMock()
        plot.figure_resampler.data = [mock_trace]

        # Should not raise an error
        plot._clean_trace_names()

    def test_clean_trace_names_handles_none_names(self):
        """Test that _clean_trace_names handles None trace names."""
        mock_trace = MagicMock()
        mock_trace.name = None
        plot = TemperaturePlot(datasets=[], plot_id="test")
        plot.figure_resampler = MagicMock()
        plot.figure_resampler.data = [mock_trace]

        # Should not raise an error
        plot._clean_trace_names()

    def test_clean_trace_names_processes_multiple_traces(self):
        """Test that _clean_trace_names processes all traces."""
        mock_trace1 = MagicMock()
        mock_trace1.name = "[R] Trace 1"
        mock_trace2 = MagicMock()
        mock_trace2.name = "[R] Trace 2"
        plot = TemperaturePlot(datasets=[], plot_id="test")
        plot.figure_resampler = MagicMock()
        plot.figure_resampler.data = [mock_trace1, mock_trace2]

        plot._clean_trace_names()

        assert mock_trace1.name == "Trace 1"
        assert mock_trace2.name == "Trace 2"

    def test_clean_trace_names_leaves_other_names_unchanged(self):
        """Test that _clean_trace_names doesn't modify names without [R]."""
        mock_trace = MagicMock()
        mock_trace.name = "Normal trace name"
        plot = TemperaturePlot(datasets=[], plot_id="test")
        plot.figure_resampler = MagicMock()
        plot.figure_resampler.data = [mock_trace]

        plot._clean_trace_names()

        assert mock_trace.name == "Normal trace name"
