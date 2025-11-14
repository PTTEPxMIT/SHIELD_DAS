"""Unit tests for shield_das.figures.base_graph module.

This test module verifies the functionality of the BaseGraph abstract base
class, which provides common plotting infrastructure for all figure types.
"""

import plotly.graph_objs as go
import pytest
from plotly_resampler import FigureResampler

from shield_das.figures.base_graph import BaseGraph


# Concrete implementation for testing the abstract base class
class ConcreteGraph(BaseGraph):
    """Concrete implementation of BaseGraph for testing."""

    def generate(self) -> go.Figure:
        """Simple generate implementation for testing."""
        self.figure_resampler = self._create_figure_resampler()
        return self.figure_resampler


class TestBaseGraphInit:
    """Tests for BaseGraph.__init__()"""

    def test_datasets_assigned_correctly(self):
        """Test that datasets parameter is assigned to instance attribute"""
        datasets = [{"name": "test1"}, {"name": "test2"}]
        graph = ConcreteGraph(datasets, "test-id")

        assert graph.datasets == datasets

    def test_plot_id_assigned_correctly(self):
        """Test that plot_id parameter is assigned to instance attribute"""
        graph = ConcreteGraph([], "my-plot-id")

        assert graph.plot_id == "my-plot-id"

    def test_figure_resampler_initialized_to_none(self):
        """Test that figure_resampler is initialized to None"""
        graph = ConcreteGraph([], "test-id")

        assert graph.figure_resampler is None

    def test_plot_parameters_has_show_error_bars_false_default(self):
        """Test that plot_parameters['show_error_bars'] defaults to False"""
        graph = ConcreteGraph([], "test-id")

        assert graph.plot_parameters["show_error_bars"] is False

    def test_plot_parameters_has_show_valve_times_false_default(self):
        """Test that plot_parameters['show_valve_times'] defaults to False"""
        graph = ConcreteGraph([], "test-id")

        assert graph.plot_parameters["show_valve_times"] is False

    def test_plot_parameters_has_x_scale_linear_default(self):
        """Test that plot_parameters['x_scale'] defaults to 'linear'"""
        graph = ConcreteGraph([], "test-id")

        assert graph.plot_parameters["x_scale"] == "linear"

    def test_plot_parameters_has_y_scale_linear_default(self):
        """Test that plot_parameters['y_scale'] defaults to 'linear'"""
        graph = ConcreteGraph([], "test-id")

        assert graph.plot_parameters["y_scale"] == "linear"

    def test_plot_parameters_has_x_min_none_default(self):
        """Test that plot_parameters['x_min'] defaults to None"""
        graph = ConcreteGraph([], "test-id")

        assert graph.plot_parameters["x_min"] is None

    def test_plot_parameters_has_x_max_none_default(self):
        """Test that plot_parameters['x_max'] defaults to None"""
        graph = ConcreteGraph([], "test-id")

        assert graph.plot_parameters["x_max"] is None

    def test_plot_parameters_has_y_min_none_default(self):
        """Test that plot_parameters['y_min'] defaults to None"""
        graph = ConcreteGraph([], "test-id")

        assert graph.plot_parameters["y_min"] is None

    def test_plot_parameters_has_y_max_none_default(self):
        """Test that plot_parameters['y_max'] defaults to None"""
        graph = ConcreteGraph([], "test-id")

        assert graph.plot_parameters["y_max"] is None

    def test_accepts_empty_dataset_list(self):
        """Test that initialization accepts empty dataset list"""
        graph = ConcreteGraph([], "test-id")

        assert graph.datasets == []


class TestCreateFigureResampler:
    """Tests for BaseGraph._create_figure_resampler()"""

    def test_returns_figure_resampler_instance(self):
        """Test that method returns a FigureResampler instance"""
        graph = ConcreteGraph([], "test-id")
        result = graph._create_figure_resampler()

        assert isinstance(result, FigureResampler)

    def test_creates_with_default_n_shown_samples_1000(self):
        """Test that FigureResampler is created with default_n_shown_samples=1000"""
        graph = ConcreteGraph([], "test-id")
        result = graph._create_figure_resampler()

        assert result._global_n_shown_samples == 1000

    def test_creates_new_instance_each_call(self):
        """Test that each call creates a new FigureResampler instance"""
        graph = ConcreteGraph([], "test-id")
        result1 = graph._create_figure_resampler()
        result2 = graph._create_figure_resampler()

        assert result1 is not result2


class TestApplyAxisSettings:
    """Tests for BaseGraph._apply_axis_settings()"""

    def test_returns_early_when_figure_resampler_is_none(self):
        """Test that method returns without error when figure_resampler is None"""
        graph = ConcreteGraph([], "test-id")
        graph.figure_resampler = None

        # Should not raise any exceptions
        graph._apply_axis_settings()

    def test_sets_linear_x_axis_type(self):
        """Test that x-axis type is set to 'linear' when x_scale is 'linear'"""
        graph = ConcreteGraph([], "test-id")
        graph.figure_resampler = graph._create_figure_resampler()
        graph.plot_parameters["x_scale"] = "linear"

        graph._apply_axis_settings()

        assert graph.figure_resampler.layout.xaxis.type == "linear"

    def test_sets_log_x_axis_type(self):
        """Test that x-axis type is set to 'log' when x_scale is 'log'"""
        graph = ConcreteGraph([], "test-id")
        graph.figure_resampler = graph._create_figure_resampler()
        graph.plot_parameters["x_scale"] = "log"

        graph._apply_axis_settings()

        assert graph.figure_resampler.layout.xaxis.type == "log"

    def test_sets_linear_y_axis_type(self):
        """Test that y-axis type is set to 'linear' when y_scale is 'linear'"""
        graph = ConcreteGraph([], "test-id")
        graph.figure_resampler = graph._create_figure_resampler()
        graph.plot_parameters["y_scale"] = "linear"

        graph._apply_axis_settings()

        assert graph.figure_resampler.layout.yaxis.type == "linear"

    def test_sets_log_y_axis_type(self):
        """Test that y-axis type is set to 'log' when y_scale is 'log'"""
        graph = ConcreteGraph([], "test-id")
        graph.figure_resampler = graph._create_figure_resampler()
        graph.plot_parameters["y_scale"] = "log"

        graph._apply_axis_settings()

        assert graph.figure_resampler.layout.yaxis.type == "log"

    def test_applies_valid_x_range_linear_scale(self):
        """Test that valid x-axis range is applied with linear scale"""
        graph = ConcreteGraph([], "test-id")
        graph.figure_resampler = graph._create_figure_resampler()
        graph.plot_parameters["x_scale"] = "linear"
        graph.plot_parameters["x_min"] = 10
        graph.plot_parameters["x_max"] = 100

        graph._apply_axis_settings()

        assert graph.figure_resampler.layout.xaxis.range == (10, 100)

    def test_applies_valid_y_range_linear_scale(self):
        """Test that valid y-axis range is applied with linear scale"""
        graph = ConcreteGraph([], "test-id")
        graph.figure_resampler = graph._create_figure_resampler()
        graph.plot_parameters["y_scale"] = "linear"
        graph.plot_parameters["y_min"] = 5
        graph.plot_parameters["y_max"] = 50

        graph._apply_axis_settings()

        assert graph.figure_resampler.layout.yaxis.range == (5, 50)

    def test_applies_valid_x_range_log_scale(self):
        """Test that valid x-axis range is applied with log scale (positive values)"""
        graph = ConcreteGraph([], "test-id")
        graph.figure_resampler = graph._create_figure_resampler()
        graph.plot_parameters["x_scale"] = "log"
        graph.plot_parameters["x_min"] = 1
        graph.plot_parameters["x_max"] = 1000

        graph._apply_axis_settings()

        assert graph.figure_resampler.layout.xaxis.range == (1, 1000)

    def test_applies_valid_y_range_log_scale(self):
        """Test that valid y-axis range is applied with log scale (positive values)"""
        graph = ConcreteGraph([], "test-id")
        graph.figure_resampler = graph._create_figure_resampler()
        graph.plot_parameters["y_scale"] = "log"
        graph.plot_parameters["y_min"] = 0.01
        graph.plot_parameters["y_max"] = 100

        graph._apply_axis_settings()

        assert graph.figure_resampler.layout.yaxis.range == (0.01, 100)

    def test_ignores_x_range_when_min_equals_max(self):
        """Test that x-axis range is not applied when x_min equals x_max"""
        graph = ConcreteGraph([], "test-id")
        graph.figure_resampler = graph._create_figure_resampler()
        graph.plot_parameters["x_min"] = 10
        graph.plot_parameters["x_max"] = 10

        graph._apply_axis_settings()

        assert graph.figure_resampler.layout.xaxis.range is None

    def test_ignores_y_range_when_min_equals_max(self):
        """Test that y-axis range is not applied when y_min equals y_max"""
        graph = ConcreteGraph([], "test-id")
        graph.figure_resampler = graph._create_figure_resampler()
        graph.plot_parameters["y_min"] = 5
        graph.plot_parameters["y_max"] = 5

        graph._apply_axis_settings()

        assert graph.figure_resampler.layout.yaxis.range is None

    def test_ignores_x_range_when_min_greater_than_max(self):
        """Test that x-axis range is not applied when x_min > x_max"""
        graph = ConcreteGraph([], "test-id")
        graph.figure_resampler = graph._create_figure_resampler()
        graph.plot_parameters["x_min"] = 100
        graph.plot_parameters["x_max"] = 10

        graph._apply_axis_settings()

        assert graph.figure_resampler.layout.xaxis.range is None

    def test_ignores_y_range_when_min_greater_than_max(self):
        """Test that y-axis range is not applied when y_min > y_max"""
        graph = ConcreteGraph([], "test-id")
        graph.figure_resampler = graph._create_figure_resampler()
        graph.plot_parameters["y_min"] = 50
        graph.plot_parameters["y_max"] = 5

        graph._apply_axis_settings()

        assert graph.figure_resampler.layout.yaxis.range is None

    def test_ignores_x_range_log_scale_with_zero_min(self):
        """Test that x-axis range is not applied for log scale when x_min is zero"""
        graph = ConcreteGraph([], "test-id")
        graph.figure_resampler = graph._create_figure_resampler()
        graph.plot_parameters["x_scale"] = "log"
        graph.plot_parameters["x_min"] = 0
        graph.plot_parameters["x_max"] = 100

        graph._apply_axis_settings()

        assert graph.figure_resampler.layout.xaxis.range is None

    def test_ignores_y_range_log_scale_with_negative_min(self):
        """Test that y-axis range is not applied for log scale when y_min is negative"""
        graph = ConcreteGraph([], "test-id")
        graph.figure_resampler = graph._create_figure_resampler()
        graph.plot_parameters["y_scale"] = "log"
        graph.plot_parameters["y_min"] = -10
        graph.plot_parameters["y_max"] = 100

        graph._apply_axis_settings()

        assert graph.figure_resampler.layout.yaxis.range is None

    def test_ignores_x_range_when_x_min_is_none(self):
        """Test that x-axis range is not applied when x_min is None"""
        graph = ConcreteGraph([], "test-id")
        graph.figure_resampler = graph._create_figure_resampler()
        graph.plot_parameters["x_min"] = None
        graph.plot_parameters["x_max"] = 100

        graph._apply_axis_settings()

        assert graph.figure_resampler.layout.xaxis.range is None

    def test_ignores_y_range_when_y_max_is_none(self):
        """Test that y-axis range is not applied when y_max is None"""
        graph = ConcreteGraph([], "test-id")
        graph.figure_resampler = graph._create_figure_resampler()
        graph.plot_parameters["y_min"] = 5
        graph.plot_parameters["y_max"] = None

        graph._apply_axis_settings()

        assert graph.figure_resampler.layout.yaxis.range is None

    def test_uses_default_linear_when_x_scale_missing(self):
        """Test that x-axis defaults to linear when x_scale key is missing"""
        graph = ConcreteGraph([], "test-id")
        graph.figure_resampler = graph._create_figure_resampler()
        del graph.plot_parameters["x_scale"]

        graph._apply_axis_settings()

        assert graph.figure_resampler.layout.xaxis.type == "linear"

    def test_uses_default_linear_when_y_scale_missing(self):
        """Test that y-axis defaults to linear when y_scale key is missing"""
        graph = ConcreteGraph([], "test-id")
        graph.figure_resampler = graph._create_figure_resampler()
        del graph.plot_parameters["y_scale"]

        graph._apply_axis_settings()

        assert graph.figure_resampler.layout.yaxis.type == "linear"


class TestCleanTraceNames:
    """Tests for BaseGraph._clean_trace_names()"""

    def test_returns_early_when_figure_resampler_is_none(self):
        """Test that method returns without error when figure_resampler is None"""
        graph = ConcreteGraph([], "test-id")
        graph.figure_resampler = None

        # Should not raise any exceptions
        graph._clean_trace_names()

    def test_removes_resampler_suffix_from_trace_name(self):
        """Test that ' [R]' suffix is removed from trace name"""
        graph = ConcreteGraph([], "test-id")
        graph.figure_resampler = graph._create_figure_resampler()

        # Add a trace with [R] suffix
        trace = go.Scatter(x=[1, 2, 3], y=[4, 5, 6], name="Test Trace [R]")
        graph.figure_resampler.add_trace(trace)

        graph._clean_trace_names()

        assert graph.figure_resampler.data[0].name == "Test Trace"

    def test_leaves_trace_name_unchanged_without_suffix(self):
        """Test that trace names without [R] suffix are left unchanged"""
        graph = ConcreteGraph([], "test-id")
        graph.figure_resampler = graph._create_figure_resampler()

        trace = go.Scatter(x=[1, 2, 3], y=[4, 5, 6], name="Normal Trace")
        graph.figure_resampler.add_trace(trace)

        graph._clean_trace_names()

        assert graph.figure_resampler.data[0].name == "Normal Trace"

    def test_handles_multiple_traces_with_suffix(self):
        """Test that [R] suffix is removed from multiple traces"""
        graph = ConcreteGraph([], "test-id")
        graph.figure_resampler = graph._create_figure_resampler()

        trace1 = go.Scatter(x=[1], y=[2], name="Trace A [R]")
        trace2 = go.Scatter(x=[3], y=[4], name="Trace B [R]")
        graph.figure_resampler.add_trace(trace1)
        graph.figure_resampler.add_trace(trace2)

        graph._clean_trace_names()

        assert graph.figure_resampler.data[0].name == "Trace A"
        assert graph.figure_resampler.data[1].name == "Trace B"

    def test_handles_trace_with_none_name(self):
        """Test that method handles traces with None name without error"""
        graph = ConcreteGraph([], "test-id")
        graph.figure_resampler = graph._create_figure_resampler()

        trace = go.Scatter(x=[1, 2], y=[3, 4], name=None)
        graph.figure_resampler.add_trace(trace)

        # Should not raise any exceptions
        graph._clean_trace_names()

    def test_handles_trace_with_empty_string_name(self):
        """Test that method handles traces with empty string name"""
        graph = ConcreteGraph([], "test-id")
        graph.figure_resampler = graph._create_figure_resampler()

        trace = go.Scatter(x=[1, 2], y=[3, 4], name="")
        graph.figure_resampler.add_trace(trace)

        graph._clean_trace_names()

        assert graph.figure_resampler.data[0].name == ""

    def test_handles_name_ending_with_r_but_no_brackets(self):
        """Test that trace name ending with 'R' (no brackets) is unchanged"""
        graph = ConcreteGraph([], "test-id")
        graph.figure_resampler = graph._create_figure_resampler()

        trace = go.Scatter(x=[1, 2], y=[3, 4], name="Test R")
        graph.figure_resampler.add_trace(trace)

        graph._clean_trace_names()

        assert graph.figure_resampler.data[0].name == "Test R"


class TestGenerateAbstract:
    """Tests for BaseGraph.generate() abstract method"""

    def test_concrete_subclass_can_implement_generate(self):
        """Test that concrete subclass can implement generate() method"""
        graph = ConcreteGraph([], "test-id")
        result = graph.generate()

        assert isinstance(result, FigureResampler)

    def test_cannot_instantiate_base_graph_directly(self):
        """Test that BaseGraph cannot be instantiated directly (is abstract)"""
        with pytest.raises(TypeError):
            BaseGraph([], "test-id")
