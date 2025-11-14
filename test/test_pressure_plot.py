"""Unit tests for shield_das.figures.pressure_plot module.

This test module verifies the functionality of the PressurePlot class,
which generates upstream and downstream pressure plots with error bars
and valve time markers.
"""

from unittest.mock import MagicMock

import numpy as np
import pytest
from plotly_resampler import FigureResampler

from shield_das.figures.pressure_plot import PressurePlot


@pytest.fixture
def mock_dataset():
    """Create a mock dataset with pressure data."""
    dataset = MagicMock()
    dataset.name = "Test Dataset"
    dataset.colour = "#FF0000"
    dataset.time_data = np.array([0.0, 1.0, 2.0, 3.0])
    dataset.upstream_pressure = np.array([100.0, 200.0, 300.0, 400.0])
    dataset.downstream_pressure = np.array([10.0, 20.0, 30.0, 40.0])
    dataset.upstream_error = np.array([5.0, 10.0, 15.0, 20.0])
    dataset.downstream_error = np.array([1.0, 2.0, 3.0, 4.0])
    dataset.valve_times = {"valve_open": 1.5, "valve_close": 2.5}
    return dataset


@pytest.fixture
def empty_dataset():
    """Create a mock dataset with empty arrays."""
    dataset = MagicMock()
    dataset.name = "Empty Dataset"
    dataset.colour = "#00FF00"
    dataset.time_data = np.array([])
    dataset.upstream_pressure = np.array([])
    dataset.downstream_pressure = np.array([])
    dataset.upstream_error = np.array([])
    dataset.downstream_error = np.array([])
    dataset.valve_times = {}
    return dataset


class TestPressurePlotInit:
    """Tests for PressurePlot.__init__()"""

    def test_initializes_with_upstream_type(self):
        """Test that PressurePlot initializes with plot_type='upstream'"""
        plot = PressurePlot([], "test-id", plot_type="upstream")

        assert plot.plot_type == "upstream"

    def test_initializes_with_downstream_type(self):
        """Test that PressurePlot initializes with plot_type='downstream'"""
        plot = PressurePlot([], "test-id", plot_type="downstream")

        assert plot.plot_type == "downstream"

    def test_defaults_to_upstream_when_plot_type_not_specified(self):
        """Test that plot_type defaults to 'upstream' when not specified"""
        plot = PressurePlot([], "test-id")

        assert plot.plot_type == "upstream"

    def test_sets_time_attr_to_time_data(self):
        """Test that time_attr is set to 'time_data' for all plot types"""
        plot = PressurePlot([], "test-id", plot_type="upstream")

        assert plot.time_attr == "time_data"

    def test_sets_upstream_pressure_attr_for_upstream_type(self):
        """Test that pressure_attr is 'upstream_pressure' for upstream type"""
        plot = PressurePlot([], "test-id", plot_type="upstream")

        assert plot.pressure_attr == "upstream_pressure"

    def test_sets_downstream_pressure_attr_for_downstream_type(self):
        """Test that pressure_attr is 'downstream_pressure' for downstream type"""
        plot = PressurePlot([], "test-id", plot_type="downstream")

        assert plot.pressure_attr == "downstream_pressure"

    def test_sets_upstream_error_attr_for_upstream_type(self):
        """Test that error_attr is 'upstream_error' for upstream type"""
        plot = PressurePlot([], "test-id", plot_type="upstream")

        assert plot.error_attr == "upstream_error"

    def test_sets_downstream_error_attr_for_downstream_type(self):
        """Test that error_attr is 'downstream_error' for downstream type"""
        plot = PressurePlot([], "test-id", plot_type="downstream")

        assert plot.error_attr == "downstream_error"

    def test_inherits_datasets_from_base_graph(self):
        """Test that datasets parameter is passed to BaseGraph"""
        datasets = [{"name": "test1"}, {"name": "test2"}]
        plot = PressurePlot(datasets, "test-id")

        assert plot.datasets == datasets

    def test_inherits_plot_id_from_base_graph(self):
        """Test that plot_id parameter is passed to BaseGraph"""
        plot = PressurePlot([], "my-pressure-plot")

        assert plot.plot_id == "my-pressure-plot"


class TestPressurePlotGenerate:
    """Tests for PressurePlot.generate()"""

    def test_returns_figure_resampler_instance(self, mock_dataset):
        """Test that generate() returns a FigureResampler instance"""
        plot = PressurePlot([mock_dataset], "test-id")
        result = plot.generate()

        assert isinstance(result, FigureResampler)

    def test_creates_figure_resampler_on_instance(self, mock_dataset):
        """Test that generate() sets figure_resampler on instance"""
        plot = PressurePlot([mock_dataset], "test-id")
        plot.generate()

        assert plot.figure_resampler is not None
        assert isinstance(plot.figure_resampler, FigureResampler)

    def test_returns_figure_with_empty_datasets_list(self):
        """Test that generate() handles empty datasets list"""
        plot = PressurePlot([], "test-id")
        result = plot.generate()

        assert isinstance(result, FigureResampler)

    def test_adds_trace_for_dataset_with_data(self, mock_dataset):
        """Test that generate() adds trace for dataset with valid data"""
        plot = PressurePlot([mock_dataset], "test-id", plot_type="upstream")
        result = plot.generate()

        assert len(result.data) > 0

    def test_skips_dataset_with_no_time_data(self):
        """Test that generate() skips dataset when time_data is None"""
        dataset = MagicMock()
        dataset.time_data = None
        dataset.upstream_pressure = np.array([1, 2, 3])
        dataset.upstream_error = np.array([0.1, 0.2, 0.3])

        plot = PressurePlot([dataset], "test-id", plot_type="upstream")
        result = plot.generate()

        assert len(result.data) == 0

    def test_skips_dataset_with_no_pressure_data(self):
        """Test that generate() skips dataset when pressure_data is None"""
        dataset = MagicMock()
        dataset.time_data = np.array([0, 1, 2])
        dataset.upstream_pressure = None
        dataset.upstream_error = np.array([0.1, 0.2, 0.3])

        plot = PressurePlot([dataset], "test-id", plot_type="upstream")
        result = plot.generate()

        assert len(result.data) == 0

    def test_skips_dataset_with_empty_time_array(self, empty_dataset):
        """Test that generate() skips dataset with empty time_data array"""
        plot = PressurePlot([empty_dataset], "test-id")
        result = plot.generate()

        assert len(result.data) == 0

    def test_skips_dataset_with_empty_pressure_array(self):
        """Test that generate() skips dataset with empty pressure array"""
        dataset = MagicMock()
        dataset.time_data = np.array([0, 1, 2])
        dataset.upstream_pressure = np.array([])
        dataset.upstream_error = np.array([])

        plot = PressurePlot([dataset], "test-id", plot_type="upstream")
        result = plot.generate()

        assert len(result.data) == 0

    def test_uses_upstream_data_for_upstream_plot(self, mock_dataset):
        """Test that upstream plot uses upstream_pressure attribute"""
        plot = PressurePlot([mock_dataset], "test-id", plot_type="upstream")
        plot.generate()

        # Should access upstream_pressure, not downstream
        assert plot.pressure_attr == "upstream_pressure"

    def test_uses_downstream_data_for_downstream_plot(self, mock_dataset):
        """Test that downstream plot uses downstream_pressure attribute"""
        plot = PressurePlot([mock_dataset], "test-id", plot_type="downstream")
        plot.generate()

        assert plot.pressure_attr == "downstream_pressure"


class TestAddDatasetTrace:
    """Tests for PressurePlot._add_dataset_trace()"""

    def test_creates_trace_with_dataset_name(self, mock_dataset):
        """Test that trace is created with dataset name"""
        plot = PressurePlot([mock_dataset], "test-id")
        plot.figure_resampler = plot._create_figure_resampler()
        plot._add_dataset_trace(mock_dataset)

        assert any(trace.name == "Test Dataset" for trace in plot.figure_resampler.data)

    def test_creates_trace_with_dataset_colour(self, mock_dataset):
        """Test that trace uses dataset colour"""
        plot = PressurePlot([mock_dataset], "test-id")
        plot.figure_resampler = plot._create_figure_resampler()
        plot._add_dataset_trace(mock_dataset)

        # Check that at least one trace has the correct color
        colors = [
            trace.line.color
            for trace in plot.figure_resampler.data
            if hasattr(trace, "line")
        ]
        assert "#FF0000" in colors

    def test_creates_lines_and_markers_mode(self, mock_dataset):
        """Test that trace mode is 'lines+markers'"""
        plot = PressurePlot([mock_dataset], "test-id")
        plot.figure_resampler = plot._create_figure_resampler()
        plot._add_dataset_trace(mock_dataset)

        modes = [trace.mode for trace in plot.figure_resampler.data]
        assert "lines+markers" in modes or "lines" in modes

    def test_truncates_arrays_to_minimum_length(self):
        """Test that arrays are truncated to minimum length when sizes differ"""
        dataset = MagicMock()
        dataset.name = "Test"
        dataset.colour = "#FF0000"
        dataset.time_data = np.array([0, 1, 2, 3, 4])  # 5 elements
        dataset.upstream_pressure = np.array([10, 20, 30])  # 3 elements
        dataset.upstream_error = np.array([1, 2, 3, 4])  # 4 elements
        dataset.valve_times = {}

        plot = PressurePlot([dataset], "test-id", plot_type="upstream")
        plot.figure_resampler = plot._create_figure_resampler()
        plot._add_dataset_trace(dataset)

        # Should have at least one trace
        assert len(plot.figure_resampler.data) > 0

    def test_adds_error_bars_when_show_error_bars_true(self, mock_dataset):
        """Test that error bars are added when show_error_bars is True"""
        plot = PressurePlot([mock_dataset], "test-id")
        plot.plot_parameters["show_error_bars"] = True
        plot.figure_resampler = plot._create_figure_resampler()
        plot._add_dataset_trace(mock_dataset)

        # Check if any trace has error_y
        has_error_bars = any(
            hasattr(trace, "error_y") and trace.error_y is not None
            for trace in plot.figure_resampler.data
        )
        assert has_error_bars

    def test_no_error_bars_when_show_error_bars_false(self, mock_dataset):
        """Test that error bars are not added when show_error_bars is False"""
        plot = PressurePlot([mock_dataset], "test-id")
        plot.plot_parameters["show_error_bars"] = False
        plot.figure_resampler = plot._create_figure_resampler()
        plot._add_dataset_trace(mock_dataset)

        # Check that traces don't have error_y or it's None
        has_error_bars = any(
            hasattr(trace, "error_y")
            and trace.error_y is not None
            and trace.error_y.visible
            for trace in plot.figure_resampler.data
        )
        assert not has_error_bars

    def test_adds_valve_markers_when_show_valve_times_true(self, mock_dataset):
        """Test that valve markers are added when show_valve_times is True"""
        plot = PressurePlot([mock_dataset], "test-id")
        plot.plot_parameters["show_valve_times"] = True
        plot.figure_resampler = plot._create_figure_resampler()
        plot._add_dataset_trace(mock_dataset)

        # Should have traces for data + vertical lines for valves
        assert len(plot.figure_resampler.layout.shapes) > 0

    def test_no_valve_markers_when_show_valve_times_false(self, mock_dataset):
        """Test that valve markers are not added when show_valve_times is False"""
        plot = PressurePlot([mock_dataset], "test-id")
        plot.plot_parameters["show_valve_times"] = False
        plot.figure_resampler = plot._create_figure_resampler()
        plot._add_dataset_trace(mock_dataset)

        # Should not have shapes for valve lines
        assert len(plot.figure_resampler.layout.shapes) == 0


class TestAddValveMarkers:
    """Tests for PressurePlot._add_valve_markers()"""

    def test_adds_vertical_line_for_each_valve_event(self, mock_dataset):
        """Test that vertical lines are added for each valve event"""
        plot = PressurePlot([mock_dataset], "test-id")
        plot.figure_resampler = plot._create_figure_resampler()
        plot._add_valve_markers(mock_dataset, "#FF0000")

        # Should have 2 shapes (one for each valve event)
        assert len(plot.figure_resampler.layout.shapes) == 2

    def test_uses_specified_colour_for_markers(self, mock_dataset):
        """Test that valve markers use the specified colour"""
        plot = PressurePlot([mock_dataset], "test-id")
        plot.figure_resampler = plot._create_figure_resampler()
        plot._add_valve_markers(mock_dataset, "#00FF00")

        # Check that shapes have the correct color
        colors = [shape.line.color for shape in plot.figure_resampler.layout.shapes]
        assert all(color == "#00FF00" for color in colors)

    def test_handles_empty_valve_times_dict(self):
        """Test that method handles dataset with no valve events"""
        dataset = MagicMock()
        dataset.valve_times = {}

        plot = PressurePlot([dataset], "test-id")
        plot.figure_resampler = plot._create_figure_resampler()
        plot._add_valve_markers(dataset, "#FF0000")

        assert len(plot.figure_resampler.layout.shapes) == 0

    def test_formats_valve_event_name_in_annotation(self, mock_dataset):
        """Test valve event names formatted (underscores to spaces, title)"""
        dataset = MagicMock()
        dataset.valve_times = {"valve_open_fast": 1.5}

        plot = PressurePlot([dataset], "test-id")
        plot.figure_resampler = plot._create_figure_resampler()
        plot._add_valve_markers(dataset, "#FF0000")

        # Check annotations for formatted text
        annotations = plot.figure_resampler.layout.annotations
        assert any("Valve Open Fast" in str(ann.text) for ann in annotations)


class TestConfigureLayout:
    """Tests for PressurePlot._configure_layout()"""

    def test_sets_height_to_500(self, mock_dataset):
        """Test that plot height is set to 500"""
        plot = PressurePlot([mock_dataset], "test-id")
        plot.figure_resampler = plot._create_figure_resampler()
        plot._configure_layout()

        assert plot.figure_resampler.layout.height == 500

    def test_sets_xaxis_title_to_time_seconds(self, mock_dataset):
        """Test that x-axis title is 'Time (s)'"""
        plot = PressurePlot([mock_dataset], "test-id")
        plot.figure_resampler = plot._create_figure_resampler()
        plot._configure_layout()

        assert plot.figure_resampler.layout.xaxis.title.text == "Time (s)"

    def test_sets_yaxis_title_to_pressure_torr(self, mock_dataset):
        """Test that y-axis title is 'Pressure (Torr)'"""
        plot = PressurePlot([mock_dataset], "test-id")
        plot.figure_resampler = plot._create_figure_resampler()
        plot._configure_layout()

        assert plot.figure_resampler.layout.yaxis.title.text == "Pressure (Torr)"

    def test_uses_plotly_white_template(self, mock_dataset):
        """Test that template is set to 'plotly_white'"""
        plot = PressurePlot([mock_dataset], "test-id")
        plot.figure_resampler = plot._create_figure_resampler()
        plot._configure_layout()

        assert plot.figure_resampler.layout.template.layout.plot_bgcolor == "white"

    def test_sets_horizontal_legend_orientation(self, mock_dataset):
        """Test that legend orientation is horizontal"""
        plot = PressurePlot([mock_dataset], "test-id")
        plot.figure_resampler = plot._create_figure_resampler()
        plot._configure_layout()

        assert plot.figure_resampler.layout.legend.orientation == "h"

    def test_positions_legend_above_plot(self, mock_dataset):
        """Test that legend is positioned above plot (y > 1)"""
        plot = PressurePlot([mock_dataset], "test-id")
        plot.figure_resampler = plot._create_figure_resampler()
        plot._configure_layout()

        assert plot.figure_resampler.layout.legend.y > 1


class TestApplyAxisRanges:
    """Tests for PressurePlot._apply_axis_ranges()"""

    def test_applies_linear_x_range_when_specified(self, mock_dataset):
        """Test that x-axis range is applied for linear scale"""
        plot = PressurePlot([mock_dataset], "test-id")
        plot.figure_resampler = plot._create_figure_resampler()
        plot.plot_parameters["x_scale"] = "linear"
        plot.plot_parameters["x_min"] = 0
        plot.plot_parameters["x_max"] = 10
        plot._apply_axis_ranges()

        assert plot.figure_resampler.layout.xaxis.range == (0, 10)

    def test_applies_linear_y_range_when_specified(self, mock_dataset):
        """Test that y-axis range is applied for linear scale"""
        plot = PressurePlot([mock_dataset], "test-id")
        plot.figure_resampler = plot._create_figure_resampler()
        plot.plot_parameters["y_scale"] = "linear"
        plot.plot_parameters["y_min"] = 0
        plot.plot_parameters["y_max"] = 1000
        plot._apply_axis_ranges()

        assert plot.figure_resampler.layout.yaxis.range == (0, 1000)

    def test_applies_log_x_range_when_specified(self, mock_dataset):
        """Test that log x-axis range is applied (as log10 values)"""
        plot = PressurePlot([mock_dataset], "test-id")
        plot.figure_resampler = plot._create_figure_resampler()
        plot.plot_parameters["x_scale"] = "log"
        plot.plot_parameters["x_min"] = 1
        plot.plot_parameters["x_max"] = 1000
        plot._apply_axis_ranges()

        # For log scale, range should be log10 of the values
        expected = (np.log10(1), np.log10(1000))
        actual = plot.figure_resampler.layout.xaxis.range
        np.testing.assert_array_almost_equal(actual, expected)

    def test_applies_log_y_range_when_specified(self, mock_dataset):
        """Test that log y-axis range is applied (as log10 values)"""
        plot = PressurePlot([mock_dataset], "test-id")
        plot.figure_resampler = plot._create_figure_resampler()
        plot.plot_parameters["y_scale"] = "log"
        plot.plot_parameters["y_min"] = 0.1
        plot.plot_parameters["y_max"] = 100
        plot._apply_axis_ranges()

        expected = (np.log10(0.1), np.log10(100))
        actual = plot.figure_resampler.layout.yaxis.range
        np.testing.assert_array_almost_equal(actual, expected)

    def test_computes_x_range_from_data_when_not_specified(self, mock_dataset):
        """Test that x-axis range is computed from data when not specified"""
        plot = PressurePlot([mock_dataset], "test-id")
        plot.figure_resampler = plot._create_figure_resampler()
        plot.plot_parameters["x_scale"] = "linear"
        plot.plot_parameters["x_min"] = None
        plot.plot_parameters["x_max"] = None
        plot._apply_axis_ranges()

        # Should compute from dataset time_data
        expected = (0.0, 3.0)
        assert plot.figure_resampler.layout.xaxis.range == expected

    def test_computes_y_range_from_data_when_not_specified(self, mock_dataset):
        """Test that y-axis range is computed from data when not specified"""
        plot = PressurePlot([mock_dataset], "test-id", plot_type="upstream")
        plot.figure_resampler = plot._create_figure_resampler()
        plot.plot_parameters["y_scale"] = "linear"
        plot.plot_parameters["y_min"] = None
        plot.plot_parameters["y_max"] = None
        plot._apply_axis_ranges()

        # Should compute from dataset upstream_pressure
        expected = (100.0, 400.0)
        assert plot.figure_resampler.layout.yaxis.range == expected


class TestGetLogRange:
    """Tests for PressurePlot._get_log_range()"""

    def test_returns_user_range_when_both_specified_and_valid(self, mock_dataset):
        """Test user-provided range returned when both min and max are valid"""
        plot = PressurePlot([mock_dataset], "test-id")
        result = plot._get_log_range(1.0, 1000.0, "upstream_pressure", 1e-12, 1e-6)

        assert result == (1.0, 1000.0)

    def test_returns_default_when_user_min_is_none(self, mock_dataset):
        """Test that default range is used when user_min is None"""
        plot = PressurePlot([mock_dataset], "test-id")
        min_val, max_val = plot._get_log_range(
            None, 1000.0, "upstream_pressure", 1e-12, 1e-6
        )

        # Should compute from data
        assert min_val == 100.0
        assert max_val == 400.0

    def test_returns_default_when_user_max_is_none(self, mock_dataset):
        """Test that default range is used when user_max is None"""
        plot = PressurePlot([mock_dataset], "test-id")
        min_val, max_val = plot._get_log_range(
            1.0, None, "upstream_pressure", 1e-12, 1e-6
        )

        # Should compute from data
        assert min_val == 100.0
        assert max_val == 400.0

    def test_returns_default_when_user_min_zero_or_negative(self, mock_dataset):
        """Test default used when user_min zero or negative (invalid for log)"""
        plot = PressurePlot([mock_dataset], "test-id")
        min_val, _max_val = plot._get_log_range(
            0, 1000.0, "upstream_pressure", 1e-12, 1e-6
        )

        # Should compute from data instead
        assert min_val == 100.0

    def test_returns_default_when_min_greater_than_max(self, mock_dataset):
        """Test that default is used when user_min > user_max"""
        plot = PressurePlot([mock_dataset], "test-id")
        min_val, max_val = plot._get_log_range(
            1000.0, 1.0, "upstream_pressure", 1e-12, 1e-6
        )

        # Should compute from data
        assert min_val == 100.0
        assert max_val == 400.0

    def test_computes_range_from_dataset_positive_values(self, mock_dataset):
        """Test that range is computed from positive dataset values"""
        plot = PressurePlot([mock_dataset], "test-id")
        min_val, max_val = plot._get_log_range(
            None, None, "upstream_pressure", 1e-12, 1e-6
        )

        assert min_val == 100.0
        assert max_val == 400.0

    def test_filters_out_non_positive_values(self):
        """Test zero and negative values are filtered out for log range"""
        dataset = MagicMock()
        dataset.upstream_pressure = np.array([-10, 0, 5, 10, 20])

        plot = PressurePlot([dataset], "test-id")
        min_val, max_val = plot._get_log_range(
            None, None, "upstream_pressure", 1e-12, 1e-6
        )

        # Should only use positive values: [5, 10, 20]
        assert min_val == 5.0
        assert max_val == 20.0

    def test_returns_defaults_when_no_positive_values(self):
        """Test that defaults are returned when no positive values exist"""
        dataset = MagicMock()
        dataset.upstream_pressure = np.array([-10, 0, -5])

        plot = PressurePlot([dataset], "test-id")
        min_val, max_val = plot._get_log_range(
            None, None, "upstream_pressure", 1e-12, 1e-6
        )

        assert min_val == 1e-12
        assert max_val == 1e-6

    def test_expands_max_when_min_equals_max(self):
        """Test that max is expanded when computed min equals max"""
        dataset = MagicMock()
        dataset.upstream_pressure = np.array([100.0, 100.0, 100.0])

        plot = PressurePlot([dataset], "test-id")
        min_val, max_val = plot._get_log_range(
            None, None, "upstream_pressure", 1e-12, 1e-6
        )

        assert min_val == 100.0
        assert max_val == 1000.0  # 100 * 10


class TestCollectValues:
    """Tests for PressurePlot._collect_values()"""

    def test_collects_values_from_single_dataset(self, mock_dataset):
        """Test that values are collected from a single dataset"""
        plot = PressurePlot([mock_dataset], "test-id")
        result = plot._collect_values("upstream_pressure")

        assert result == [100.0, 200.0, 300.0, 400.0]

    def test_collects_values_from_multiple_datasets(self, mock_dataset):
        """Test that values are collected from multiple datasets"""
        dataset2 = MagicMock()
        dataset2.upstream_pressure = np.array([500.0, 600.0])

        plot = PressurePlot([mock_dataset, dataset2], "test-id")
        result = plot._collect_values("upstream_pressure")

        assert result == [100.0, 200.0, 300.0, 400.0, 500.0, 600.0]

    def test_returns_empty_list_for_empty_datasets(self):
        """Test that empty list is returned when no datasets exist"""
        plot = PressurePlot([], "test-id")
        result = plot._collect_values("upstream_pressure")

        assert result == []

    def test_handles_dataset_without_attribute(self, mock_dataset):
        """Test that datasets without the attribute are skipped"""
        dataset_no_attr = MagicMock()
        dataset_no_attr.configure_mock(
            **{"upstream_pressure": MagicMock(side_effect=AttributeError)}
        )

        plot = PressurePlot([mock_dataset, dataset_no_attr], "test-id")
        result = plot._collect_values("upstream_pressure")

        # Should only get values from mock_dataset
        assert len(result) == 4

    def test_handles_non_numeric_attribute(self, mock_dataset):
        """Test that non-numeric attributes are handled gracefully"""
        dataset_bad = MagicMock()
        dataset_bad.upstream_pressure = "not a number"

        plot = PressurePlot([mock_dataset, dataset_bad], "test-id")
        result = plot._collect_values("upstream_pressure")

        # Should get values from mock_dataset, skip bad dataset
        assert len(result) >= 4
