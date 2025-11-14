"""Unit tests for PermeabilityPlot class.

This module contains comprehensive unit tests for the PermeabilityPlot class,
testing all methods and ensuring high code coverage.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import plotly.graph_objects as go
from uncertainties import ufloat

from shield_das.figures.permeability_plot import PermeabilityPlot


class TestPermeabilityPlotInit:
    """Tests for PermeabilityPlot initialization."""

    def test_init_creates_instance_with_datasets(self):
        """Test that PermeabilityPlot can be initialized with datasets."""
        mock_dataset = MagicMock()
        plot = PermeabilityPlot(datasets=[mock_dataset], plot_id="test")
        assert plot.datasets == [mock_dataset]

    def test_init_creates_instance_with_empty_datasets(self):
        """Test that PermeabilityPlot can be initialized with empty datasets."""
        plot = PermeabilityPlot(datasets=[], plot_id="test")
        assert plot.datasets == []

    def test_init_inherits_from_base_graph(self):
        """Test that PermeabilityPlot inherits from BaseGraph."""
        from shield_das.figures.base_graph import BaseGraph

        plot = PermeabilityPlot(datasets=[], plot_id="test")
        assert isinstance(plot, BaseGraph)

    def test_init_sets_plot_id(self):
        """Test that plot_id is set during initialization."""
        plot = PermeabilityPlot(datasets=[], plot_id="test123")
        assert plot.plot_id == "test123"

    def test_init_sets_figure_resampler_to_none(self):
        """Test that figure_resampler is None before generation."""
        plot = PermeabilityPlot(datasets=[], plot_id="test")
        assert plot.figure_resampler is None


class TestPermeabilityPlotGenerate:
    """Tests for PermeabilityPlot.generate() method."""

    @patch(
        "shield_das.figures.permeability_plot.PermeabilityPlot._create_figure_resampler"
    )
    @patch("shield_das.figures.permeability_plot.evaluate_permeability_values")
    @patch("shield_das.figures.permeability_plot.PermeabilityPlot._add_data_points")
    @patch("shield_das.figures.permeability_plot.PermeabilityPlot._add_fit_line")
    @patch("shield_das.figures.permeability_plot.PermeabilityPlot._add_htm_data")
    @patch("shield_das.figures.permeability_plot.PermeabilityPlot._configure_layout")
    @patch("shield_das.figures.permeability_plot.PermeabilityPlot._apply_axis_settings")
    @patch("shield_das.figures.permeability_plot.PermeabilityPlot._clean_trace_names")
    def test_generate_calls_create_figure_resampler(
        self,
        mock_clean,
        mock_apply,
        mock_layout,
        mock_htm,
        mock_fit,
        mock_points,
        mock_eval,
        mock_create,
    ):
        """Test that generate() calls _create_figure_resampler."""
        mock_fig = MagicMock(spec=go.Figure)
        mock_create.return_value = mock_fig
        mock_eval.return_value = (
            [300.0, 400.0],
            [1e-8, 2e-8],
            [0.1, 0.1],
            [0.1, 0.1],
            [1e-9, 2e-9],
            [1e-9, 2e-9],
        )
        mock_dataset = MagicMock()

        plot = PermeabilityPlot(datasets=[mock_dataset], plot_id="test")
        plot.generate()

        mock_create.assert_called_once()

    @patch(
        "shield_das.figures.permeability_plot.PermeabilityPlot._create_figure_resampler"
    )
    @patch("shield_das.figures.permeability_plot.PermeabilityPlot._add_no_data_message")
    @patch("shield_das.figures.permeability_plot.PermeabilityPlot._configure_layout")
    @patch("shield_das.figures.permeability_plot.PermeabilityPlot._apply_axis_settings")
    def test_generate_handles_empty_datasets(
        self, mock_apply, mock_layout, mock_no_data, mock_create
    ):
        """Test that generate() handles empty dataset list."""
        mock_fig = MagicMock(spec=go.Figure)
        mock_create.return_value = mock_fig

        plot = PermeabilityPlot(datasets=[], plot_id="test")
        _result = plot.generate()

        mock_no_data.assert_called_once()
        assert _result is mock_fig

    @patch(
        "shield_das.figures.permeability_plot.PermeabilityPlot._create_figure_resampler"
    )
    @patch("shield_das.figures.permeability_plot.evaluate_permeability_values")
    @patch("shield_das.figures.permeability_plot.PermeabilityPlot._add_no_data_message")
    @patch("shield_das.figures.permeability_plot.PermeabilityPlot._configure_layout")
    def test_generate_handles_evaluation_exception(
        self, mock_layout, mock_no_data, mock_eval, mock_create
    ):
        """Test that generate() handles exception from evaluate_permeability_values."""
        mock_fig = MagicMock(spec=go.Figure)
        mock_create.return_value = mock_fig
        mock_eval.side_effect = Exception("Evaluation error")
        mock_dataset = MagicMock()

        plot = PermeabilityPlot(datasets=[mock_dataset], plot_id="test")
        plot.generate()

        mock_no_data.assert_called_once()

    @patch(
        "shield_das.figures.permeability_plot.PermeabilityPlot._create_figure_resampler"
    )
    @patch("shield_das.figures.permeability_plot.evaluate_permeability_values")
    @patch("shield_das.figures.permeability_plot.PermeabilityPlot._add_no_data_message")
    @patch("shield_das.figures.permeability_plot.PermeabilityPlot._configure_layout")
    def test_generate_handles_empty_data_from_eval(
        self, mock_layout, mock_no_data, mock_eval, mock_create
    ):
        """Test generate() when eval returns empty data."""
        mock_fig = MagicMock(spec=go.Figure)
        mock_create.return_value = mock_fig
        mock_eval.return_value = ([], [], [], [], [], [])
        mock_dataset = MagicMock()

        plot = PermeabilityPlot(datasets=[mock_dataset], plot_id="test")
        result = plot.generate()

        mock_no_data.assert_called_once()

    @patch(
        "shield_das.figures.permeability_plot.PermeabilityPlot._create_figure_resampler"
    )
    @patch("shield_das.figures.permeability_plot.evaluate_permeability_values")
    @patch("shield_das.figures.permeability_plot.PermeabilityPlot._add_data_points")
    @patch("shield_das.figures.permeability_plot.PermeabilityPlot._add_fit_line")
    @patch("shield_das.figures.permeability_plot.PermeabilityPlot._add_htm_data")
    @patch("shield_das.figures.permeability_plot.PermeabilityPlot._configure_layout")
    @patch("shield_das.figures.permeability_plot.PermeabilityPlot._apply_axis_settings")
    @patch("shield_das.figures.permeability_plot.PermeabilityPlot._clean_trace_names")
    def test_generate_adds_data_points(
        self,
        mock_clean,
        mock_apply,
        mock_layout,
        mock_htm,
        mock_fit,
        mock_points,
        mock_eval,
        mock_create,
    ):
        """Test that generate() calls _add_data_points."""
        mock_fig = MagicMock(spec=go.Figure)
        mock_create.return_value = mock_fig
        temps = [300.0, 400.0]
        perms = [ufloat(1e-8, 1e-9), ufloat(2e-8, 2e-9)]
        mock_eval.return_value = (
            temps,
            perms,
            [0.1, 0.1],
            [0.1, 0.1],
            [1e-9, 2e-9],
            [1e-9, 2e-9],
        )
        mock_dataset = MagicMock()

        plot = PermeabilityPlot(datasets=[mock_dataset], plot_id="test")
        plot.generate()

        mock_points.assert_called_once()

    @patch(
        "shield_das.figures.permeability_plot.PermeabilityPlot._create_figure_resampler"
    )
    @patch("shield_das.figures.permeability_plot.evaluate_permeability_values")
    @patch("shield_das.figures.permeability_plot.PermeabilityPlot._add_data_points")
    @patch("shield_das.figures.permeability_plot.PermeabilityPlot._add_fit_line")
    @patch("shield_das.figures.permeability_plot.PermeabilityPlot._add_htm_data")
    @patch("shield_das.figures.permeability_plot.PermeabilityPlot._configure_layout")
    @patch("shield_das.figures.permeability_plot.PermeabilityPlot._apply_axis_settings")
    @patch("shield_das.figures.permeability_plot.PermeabilityPlot._clean_trace_names")
    def test_generate_adds_fit_line_when_enough_points(
        self,
        mock_clean,
        mock_apply,
        mock_layout,
        mock_htm,
        mock_fit,
        mock_points,
        mock_eval,
        mock_create,
    ):
        """Test that generate() adds fit line when >= 2 points."""
        mock_fig = MagicMock(spec=go.Figure)
        mock_create.return_value = mock_fig
        temps = [300.0, 400.0]
        perms = [ufloat(1e-8, 1e-9), ufloat(2e-8, 2e-9)]
        mock_eval.return_value = (
            temps,
            perms,
            [0.1, 0.1],
            [0.1, 0.1],
            [1e-9, 2e-9],
            [1e-9, 2e-9],
        )
        mock_dataset = MagicMock()

        plot = PermeabilityPlot(datasets=[mock_dataset], plot_id="test")
        plot.generate()

        mock_fit.assert_called_once_with(temps, perms)

    @patch(
        "shield_das.figures.permeability_plot.PermeabilityPlot._create_figure_resampler"
    )
    @patch("shield_das.figures.permeability_plot.evaluate_permeability_values")
    @patch("shield_das.figures.permeability_plot.PermeabilityPlot._add_data_points")
    @patch("shield_das.figures.permeability_plot.PermeabilityPlot._add_fit_line")
    @patch("shield_das.figures.permeability_plot.PermeabilityPlot._add_htm_data")
    @patch("shield_das.figures.permeability_plot.PermeabilityPlot._configure_layout")
    @patch("shield_das.figures.permeability_plot.PermeabilityPlot._apply_axis_settings")
    @patch("shield_das.figures.permeability_plot.PermeabilityPlot._clean_trace_names")
    def test_generate_skips_fit_line_when_one_point(
        self,
        mock_clean,
        mock_apply,
        mock_layout,
        mock_htm,
        mock_fit,
        mock_points,
        mock_eval,
        mock_create,
    ):
        """Test that generate() skips fit line when < 2 points."""
        mock_fig = MagicMock(spec=go.Figure)
        mock_create.return_value = mock_fig
        temps = [300.0]
        perms = [ufloat(1e-8, 1e-9)]
        mock_eval.return_value = (temps, perms, [0.1], [0.1], [1e-9], [1e-9])
        mock_dataset = MagicMock()

        plot = PermeabilityPlot(datasets=[mock_dataset], plot_id="test")
        plot.generate()

        mock_fit.assert_not_called()

    @patch(
        "shield_das.figures.permeability_plot.PermeabilityPlot._create_figure_resampler"
    )
    @patch("shield_das.figures.permeability_plot.evaluate_permeability_values")
    @patch("shield_das.figures.permeability_plot.PermeabilityPlot._add_data_points")
    @patch("shield_das.figures.permeability_plot.PermeabilityPlot._add_fit_line")
    @patch("shield_das.figures.permeability_plot.PermeabilityPlot._add_htm_data")
    @patch("shield_das.figures.permeability_plot.PermeabilityPlot._configure_layout")
    @patch("shield_das.figures.permeability_plot.PermeabilityPlot._apply_axis_settings")
    @patch("shield_das.figures.permeability_plot.PermeabilityPlot._clean_trace_names")
    def test_generate_adds_htm_data(
        self,
        mock_clean,
        mock_apply,
        mock_layout,
        mock_htm,
        mock_fit,
        mock_points,
        mock_eval,
        mock_create,
    ):
        """Test that generate() calls _add_htm_data."""
        mock_fig = MagicMock(spec=go.Figure)
        mock_create.return_value = mock_fig
        temps = [300.0, 400.0]
        perms = [ufloat(1e-8, 1e-9), ufloat(2e-8, 2e-9)]
        mock_eval.return_value = (
            temps,
            perms,
            [0.1, 0.1],
            [0.1, 0.1],
            [1e-9, 2e-9],
            [1e-9, 2e-9],
        )
        mock_dataset = MagicMock()

        plot = PermeabilityPlot(datasets=[mock_dataset], plot_id="test")
        plot.generate()

        mock_htm.assert_called_once()

    @patch(
        "shield_das.figures.permeability_plot.PermeabilityPlot._create_figure_resampler"
    )
    @patch("shield_das.figures.permeability_plot.evaluate_permeability_values")
    @patch("shield_das.figures.permeability_plot.PermeabilityPlot._add_data_points")
    @patch("shield_das.figures.permeability_plot.PermeabilityPlot._add_fit_line")
    @patch("shield_das.figures.permeability_plot.PermeabilityPlot._add_htm_data")
    @patch("shield_das.figures.permeability_plot.PermeabilityPlot._configure_layout")
    @patch("shield_das.figures.permeability_plot.PermeabilityPlot._apply_axis_settings")
    @patch("shield_das.figures.permeability_plot.PermeabilityPlot._clean_trace_names")
    def test_generate_returns_figure(
        self,
        mock_clean,
        mock_apply,
        mock_layout,
        mock_htm,
        mock_fit,
        mock_points,
        mock_eval,
        mock_create,
    ):
        """Test that generate() returns the figure."""
        mock_fig = MagicMock(spec=go.Figure)
        mock_create.return_value = mock_fig
        temps = [300.0, 400.0]
        perms = [ufloat(1e-8, 1e-9), ufloat(2e-8, 2e-9)]
        mock_eval.return_value = (
            temps,
            perms,
            [0.1, 0.1],
            [0.1, 0.1],
            [1e-9, 2e-9],
            [1e-9, 2e-9],
        )
        mock_dataset = MagicMock()

        plot = PermeabilityPlot(datasets=[mock_dataset], plot_id="test")
        result = plot.generate()

        assert result is mock_fig


class TestAddNoDataMessage:
    """Tests for PermeabilityPlot._add_no_data_message() method."""

    def test_add_no_data_message_adds_annotation(self):
        """Test that _add_no_data_message adds centered annotation."""
        plot = PermeabilityPlot(datasets=[], plot_id="test")
        plot.figure_resampler = MagicMock()

        plot._add_no_data_message()

        plot.figure_resampler.add_annotation.assert_called_once()

    def test_add_no_data_message_centers_annotation(self):
        """Test that annotation is centered on plot."""
        plot = PermeabilityPlot(datasets=[], plot_id="test")
        plot.figure_resampler = MagicMock()

        plot._add_no_data_message()

        call_kwargs = plot.figure_resampler.add_annotation.call_args[1]
        assert call_kwargs["x"] == 0.5
        assert call_kwargs["y"] == 0.5

    def test_add_no_data_message_uses_paper_coordinates(self):
        """Test that annotation uses paper coordinates."""
        plot = PermeabilityPlot(datasets=[], plot_id="test")
        plot.figure_resampler = MagicMock()

        plot._add_no_data_message()

        call_kwargs = plot.figure_resampler.add_annotation.call_args[1]
        assert call_kwargs["xref"] == "paper"
        assert call_kwargs["yref"] == "paper"

    def test_add_no_data_message_no_arrow(self):
        """Test that annotation has no arrow."""
        plot = PermeabilityPlot(datasets=[], plot_id="test")
        plot.figure_resampler = MagicMock()

        plot._add_no_data_message()

        call_kwargs = plot.figure_resampler.add_annotation.call_args[1]
        assert call_kwargs["showarrow"] is False

    def test_add_no_data_message_has_text(self):
        """Test that annotation has appropriate text."""
        plot = PermeabilityPlot(datasets=[], plot_id="test")
        plot.figure_resampler = MagicMock()

        plot._add_no_data_message()

        call_kwargs = plot.figure_resampler.add_annotation.call_args[1]
        assert "No data available" in call_kwargs["text"]


class TestAddDataPoints:
    """Tests for PermeabilityPlot._add_data_points() method."""

    def test_add_data_points_adds_traces_for_each_point(self):
        """Test that _add_data_points adds trace for each data point."""
        plot = PermeabilityPlot(datasets=[], plot_id="test")
        plot.figure_resampler = MagicMock()

        temps = [300.0, 400.0, 500.0]
        perms = [ufloat(1e-8, 1e-9), ufloat(2e-8, 2e-9), ufloat(3e-8, 3e-9)]
        plot._add_data_points(
            temps, perms, [1e-9, 2e-9, 3e-9], [1e-9, 2e-9, 3e-9], True
        )

        assert plot.figure_resampler.add_trace.call_count == 3

    def test_add_data_points_uses_inverse_temperature(self):
        """Test that x-values use 1000/T conversion."""
        plot = PermeabilityPlot(datasets=[], plot_id="test")
        plot.figure_resampler = MagicMock()

        temps = [500.0]  # 1000/500 = 2.0
        perms = [ufloat(1e-8, 1e-9)]
        plot._add_data_points(temps, perms, [1e-9], [1e-9], False)

        call_args = plot.figure_resampler.add_trace.call_args[0][0]
        assert list(call_args.x) == [2.0]

    def test_add_data_points_extracts_nominal_value_from_ufloat(self):
        """Test that nominal values are extracted from ufloat objects."""
        plot = PermeabilityPlot(datasets=[], plot_id="test")
        plot.figure_resampler = MagicMock()

        temps = [300.0]
        perms = [ufloat(1.5e-8, 1e-9)]
        plot._add_data_points(temps, perms, [1e-9], [1e-9], False)

        call_args = plot.figure_resampler.add_trace.call_args[0][0]
        assert list(call_args.y) == [1.5e-8]

    def test_add_data_points_handles_float_permeability(self):
        """Test that plain float permeability values are handled."""
        plot = PermeabilityPlot(datasets=[], plot_id="test")
        plot.figure_resampler = MagicMock()

        temps = [300.0]
        perms = [1.5e-8]
        plot._add_data_points(temps, perms, [1e-9], [1e-9], False)

        call_args = plot.figure_resampler.add_trace.call_args[0][0]
        assert list(call_args.y) == [1.5e-8]

    def test_add_data_points_adds_error_bars_when_enabled(self):
        """Test that error bars are added when show_error_bars is True."""
        plot = PermeabilityPlot(datasets=[], plot_id="test")
        plot.figure_resampler = MagicMock()

        temps = [300.0]
        perms = [ufloat(1e-8, 1e-9)]
        plot._add_data_points(temps, perms, [1e-9], [2e-9], True)

        call_args = plot.figure_resampler.add_trace.call_args[0][0]
        assert call_args.error_y is not None
        assert list(call_args.error_y["array"]) == [2e-9]
        assert list(call_args.error_y["arrayminus"]) == [1e-9]

    def test_add_data_points_no_error_bars_when_disabled(self):
        """Test that error bars are None when show_error_bars is False."""
        plot = PermeabilityPlot(datasets=[], plot_id="test")
        plot.figure_resampler = MagicMock()

        temps = [300.0]
        perms = [ufloat(1e-8, 1e-9)]
        plot._add_data_points(temps, perms, [1e-9], [2e-9], False)

        call_args = plot.figure_resampler.add_trace.call_args[0][0]
        # When False, error_y_dict is set to None, but plotly creates default ErrorY
        assert call_args.error_y.visible is False or call_args.error_y.array is None

    def test_add_data_points_uses_markers_mode(self):
        """Test that traces use markers mode."""
        plot = PermeabilityPlot(datasets=[], plot_id="test")
        plot.figure_resampler = MagicMock()

        temps = [300.0]
        perms = [1e-8]
        plot._add_data_points(temps, perms, [1e-9], [1e-9], False)

        call_args = plot.figure_resampler.add_trace.call_args[0][0]
        assert call_args.mode == "markers"

    def test_add_data_points_no_legend(self):
        """Test that data point traces have showlegend=False."""
        plot = PermeabilityPlot(datasets=[], plot_id="test")
        plot.figure_resampler = MagicMock()

        temps = [300.0]
        perms = [1e-8]
        plot._add_data_points(temps, perms, [1e-9], [1e-9], False)

        call_args = plot.figure_resampler.add_trace.call_args[0][0]
        assert call_args.showlegend is False

    def test_add_data_points_error_bars_not_symmetric(self):
        """Test that error bars are asymmetric."""
        plot = PermeabilityPlot(datasets=[], plot_id="test")
        plot.figure_resampler = MagicMock()

        temps = [300.0]
        perms = [1e-8]
        plot._add_data_points(temps, perms, [1e-9], [2e-9], True)

        call_args = plot.figure_resampler.add_trace.call_args[0][0]
        assert call_args.error_y["symmetric"] is False


class TestAddFitLine:
    """Tests for PermeabilityPlot._add_fit_line() method."""

    @patch("shield_das.figures.permeability_plot.fit_permeability_data")
    def test_add_fit_line_calls_fit_function(self, mock_fit):
        """Test that _add_fit_line calls fit_permeability_data."""
        mock_fit.return_value = ([1.0, 2.0], [1e-8, 2e-8])
        plot = PermeabilityPlot(datasets=[], plot_id="test")
        plot.figure_resampler = MagicMock()

        temps = [300.0, 400.0]
        perms = [ufloat(1e-8, 1e-9), ufloat(2e-8, 2e-9)]
        plot._add_fit_line(temps, perms)

        mock_fit.assert_called_once_with(temps, perms)

    @patch("shield_das.figures.permeability_plot.fit_permeability_data")
    def test_add_fit_line_adds_trace(self, mock_fit):
        """Test that _add_fit_line adds a trace to the figure."""
        mock_fit.return_value = ([1.0, 2.0], [1e-8, 2e-8])
        plot = PermeabilityPlot(datasets=[], plot_id="test")
        plot.figure_resampler = MagicMock()

        temps = [300.0, 400.0]
        perms = [ufloat(1e-8, 1e-9), ufloat(2e-8, 2e-9)]
        plot._add_fit_line(temps, perms)

        plot.figure_resampler.add_trace.assert_called_once()

    @patch("shield_das.figures.permeability_plot.fit_permeability_data")
    def test_add_fit_line_uses_red_dashed_style(self, mock_fit):
        """Test that fit line is red and dashed."""
        fit_x = [1.0, 2.0]
        fit_y = [1e-8, 2e-8]
        mock_fit.return_value = (fit_x, fit_y)
        plot = PermeabilityPlot(datasets=[], plot_id="test")
        plot.figure_resampler = MagicMock()

        temps = [300.0, 400.0]
        perms = [ufloat(1e-8, 1e-9), ufloat(2e-8, 2e-9)]
        plot._add_fit_line(temps, perms)

        call_args = plot.figure_resampler.add_trace.call_args[0][0]
        assert call_args.line.color == "red"
        assert call_args.line.dash == "dash"

    @patch("shield_das.figures.permeability_plot.fit_permeability_data")
    def test_add_fit_line_has_name(self, mock_fit):
        """Test that fit line has name in legend."""
        mock_fit.return_value = ([1.0, 2.0], [1e-8, 2e-8])
        plot = PermeabilityPlot(datasets=[], plot_id="test")
        plot.figure_resampler = MagicMock()

        temps = [300.0, 400.0]
        perms = [ufloat(1e-8, 1e-9), ufloat(2e-8, 2e-9)]
        plot._add_fit_line(temps, perms)

        call_args = plot.figure_resampler.add_trace.call_args[0][0]
        assert call_args.name == "Arrhenius Fit"

    @patch("shield_das.figures.permeability_plot.fit_permeability_data")
    def test_add_fit_line_passes_hf_parameters(self, mock_fit):
        """Test that hf_x and hf_y are passed for downsampling."""
        fit_x = [1.0, 2.0, 3.0]
        fit_y = [1e-8, 2e-8, 3e-8]
        mock_fit.return_value = (fit_x, fit_y)
        plot = PermeabilityPlot(datasets=[], plot_id="test")
        plot.figure_resampler = MagicMock()

        temps = [300.0, 400.0]
        perms = [ufloat(1e-8, 1e-9), ufloat(2e-8, 2e-9)]
        plot._add_fit_line(temps, perms)

        call_kwargs = plot.figure_resampler.add_trace.call_args[1]
        assert call_kwargs["hf_x"] is fit_x
        assert call_kwargs["hf_y"] is fit_y


class TestConfigureLayout:
    """Tests for PermeabilityPlot._configure_layout() method."""

    def test_configure_layout_updates_layout(self):
        """Test that _configure_layout calls update_layout."""
        plot = PermeabilityPlot(datasets=[], plot_id="test")
        plot.figure_resampler = MagicMock()

        plot._configure_layout()

        plot.figure_resampler.update_layout.assert_called_once()

    def test_configure_layout_sets_xaxis_title(self):
        """Test that x-axis title is set correctly."""
        plot = PermeabilityPlot(datasets=[], plot_id="test")
        plot.figure_resampler = MagicMock()

        plot._configure_layout()

        call_kwargs = plot.figure_resampler.update_layout.call_args[1]
        assert "1000/T" in call_kwargs["xaxis_title"]
        assert "K⁻¹" in call_kwargs["xaxis_title"]

    def test_configure_layout_sets_yaxis_title(self):
        """Test that y-axis title is set correctly."""
        plot = PermeabilityPlot(datasets=[], plot_id="test")
        plot.figure_resampler = MagicMock()

        plot._configure_layout()

        call_kwargs = plot.figure_resampler.update_layout.call_args[1]
        assert "Permeability" in call_kwargs["yaxis_title"]

    def test_configure_layout_sets_template(self):
        """Test that template is set to plotly_white."""
        plot = PermeabilityPlot(datasets=[], plot_id="test")
        plot.figure_resampler = MagicMock()

        plot._configure_layout()

        call_kwargs = plot.figure_resampler.update_layout.call_args[1]
        assert call_kwargs["template"] == "plotly_white"

    def test_configure_layout_sets_hovermode(self):
        """Test that hovermode is set to closest."""
        plot = PermeabilityPlot(datasets=[], plot_id="test")
        plot.figure_resampler = MagicMock()

        plot._configure_layout()

        call_kwargs = plot.figure_resampler.update_layout.call_args[1]
        assert call_kwargs["hovermode"] == "closest"


class TestCleanTraceNames:
    """Tests for PermeabilityPlot._clean_trace_names() method."""

    def test_clean_trace_names_removes_r_suffix(self):
        """Test that _clean_trace_names removes ' [R]' suffix."""
        plot = PermeabilityPlot(datasets=[], plot_id="test")
        mock_trace = MagicMock()
        mock_trace.name = "Test trace [R]"
        plot.figure_resampler = MagicMock()
        plot.figure_resampler.data = [mock_trace]

        plot._clean_trace_names()

        assert mock_trace.name == "Test trace"

    def test_clean_trace_names_leaves_other_names_unchanged(self):
        """Test that names without [R] are unchanged."""
        plot = PermeabilityPlot(datasets=[], plot_id="test")
        mock_trace = MagicMock()
        mock_trace.name = "Normal trace"
        plot.figure_resampler = MagicMock()
        plot.figure_resampler.data = [mock_trace]

        plot._clean_trace_names()

        assert mock_trace.name == "Normal trace"

    def test_clean_trace_names_handles_multiple_traces(self):
        """Test that all traces are processed."""
        plot = PermeabilityPlot(datasets=[], plot_id="test")
        mock_trace1 = MagicMock()
        mock_trace1.name = "Trace 1 [R]"
        mock_trace2 = MagicMock()
        mock_trace2.name = "Trace 2 [R]"
        plot.figure_resampler = MagicMock()
        plot.figure_resampler.data = [mock_trace1, mock_trace2]

        plot._clean_trace_names()

        assert mock_trace1.name == "Trace 1"
        assert mock_trace2.name == "Trace 2"

    def test_clean_trace_names_handles_none_names(self):
        """Test that None trace names don't cause errors."""
        plot = PermeabilityPlot(datasets=[], plot_id="test")
        mock_trace = MagicMock()
        mock_trace.name = None
        plot.figure_resampler = MagicMock()
        plot.figure_resampler.data = [mock_trace]

        # Should not raise an error
        plot._clean_trace_names()


class TestAddHtmData:
    """Tests for PermeabilityPlot._add_htm_data() method."""

    @patch("shield_das.figures.permeability_plot.import_htm_data")
    def test_add_htm_data_calls_import_function(self, mock_import):
        """Test that _add_htm_data calls import_htm_data."""
        mock_import.return_value = ([], [], [])
        plot = PermeabilityPlot(datasets=[], plot_id="test")
        plot.figure_resampler = MagicMock()

        plot._add_htm_data()

        mock_import.assert_called_once_with("316l_steel")

    @patch("shield_das.figures.permeability_plot.import_htm_data")
    def test_add_htm_data_uses_custom_material(self, mock_import):
        """Test that custom material parameter is passed."""
        mock_import.return_value = ([], [], [])
        plot = PermeabilityPlot(datasets=[], plot_id="test")
        plot.figure_resampler = MagicMock()

        plot._add_htm_data(material="custom_material")

        mock_import.assert_called_once_with("custom_material")

    @patch("shield_das.figures.permeability_plot.import_htm_data")
    def test_add_htm_data_adds_traces_for_each_dataset(self, mock_import):
        """Test that traces are added for each HTM dataset."""
        htm_x = [np.array([300.0, 400.0]), np.array([350.0, 450.0])]
        htm_y = [np.array([1e-8, 2e-8]), np.array([1.5e-8, 2.5e-8])]
        htm_labels = ["HTM 1", "HTM 2"]
        mock_import.return_value = (htm_x, htm_y, htm_labels)
        plot = PermeabilityPlot(datasets=[], plot_id="test")
        plot.figure_resampler = MagicMock()

        plot._add_htm_data()

        assert plot.figure_resampler.add_trace.call_count == 2

    @patch("shield_das.figures.permeability_plot.import_htm_data")
    def test_add_htm_data_converts_temperature_to_inverse(self, mock_import):
        """Test that HTM temperatures are converted to 1000/T."""
        htm_x = [np.array([500.0])]  # 1000/500 = 2.0
        htm_y = [np.array([1e-8])]
        htm_labels = ["HTM"]
        mock_import.return_value = (htm_x, htm_y, htm_labels)
        plot = PermeabilityPlot(datasets=[], plot_id="test")
        plot.figure_resampler = MagicMock()

        plot._add_htm_data()

        call_args = plot.figure_resampler.add_trace.call_args[0][0]
        expected_x = 1000 / np.array([500.0])
        np.testing.assert_array_almost_equal(call_args.x, expected_x)

    @patch("shield_das.figures.permeability_plot.import_htm_data")
    def test_add_htm_data_uses_labels_as_names(self, mock_import):
        """Test that HTM labels are used as trace names."""
        htm_x = [np.array([300.0])]
        htm_y = [np.array([1e-8])]
        htm_labels = ["Custom Label"]
        mock_import.return_value = (htm_x, htm_y, htm_labels)
        plot = PermeabilityPlot(datasets=[], plot_id="test")
        plot.figure_resampler = MagicMock()

        plot._add_htm_data()

        call_args = plot.figure_resampler.add_trace.call_args[0][0]
        assert call_args.name == "Custom Label"
