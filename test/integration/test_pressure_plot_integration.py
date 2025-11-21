"""Integration tests for shield_das.figures.pressure_plot module.

This test module verifies the integration behavior of PressurePlot,
testing complete workflows for generating pressure plots with various
configurations and datasets.
"""

from unittest.mock import MagicMock

import numpy as np
from plotly_resampler import FigureResampler

from shield_das.figures.pressure_plot import PressurePlot


def create_mock_dataset(
    name="Dataset",
    colour="#FF0000",
    time_range=(0, 10),
    pressure_range=(100, 200),
    error_range=(5, 10),
    valve_times=None,
):
    """Helper to create a mock dataset with pressure data."""
    dataset = MagicMock()
    dataset.name = name
    dataset.colour = colour

    # Create data arrays
    num_points = 10
    dataset.time_data = np.linspace(*time_range, num_points)
    dataset.upstream_pressure = np.linspace(*pressure_range, num_points)
    dataset.downstream_pressure = dataset.upstream_pressure / 10
    dataset.upstream_error = np.linspace(*error_range, num_points)
    dataset.downstream_error = dataset.upstream_error / 10
    dataset.valve_times = valve_times or {}

    return dataset


class TestPressurePlotIntegrationUpstream:
    """Integration tests for upstream pressure plots."""

    def test_complete_upstream_plot_generation(self):
        """Test complete workflow for generating upstream pressure plot"""
        dataset = create_mock_dataset(
            name="Test Run", colour="#0000FF", valve_times={"open": 2.0}
        )
        plot = PressurePlot([dataset], "upstream-plot", plot_type="upstream")

        result = plot.generate()

        assert isinstance(result, FigureResampler)
        assert len(result.data) > 0
        assert result.layout.yaxis.title.text == "Pressure (Torr)"
        assert result.layout.height == 500

    def test_upstream_plot_with_multiple_datasets(self):
        """Test upstream plot with multiple datasets"""
        dataset1 = create_mock_dataset(name="Run 1", colour="#FF0000")
        dataset2 = create_mock_dataset(
            name="Run 2",
            colour="#00FF00",
            time_range=(0, 15),
            pressure_range=(150, 250),
        )

        plot = PressurePlot([dataset1, dataset2], "upstream-plot", plot_type="upstream")
        result = plot.generate()

        # Should have traces for both datasets
        assert len(result.data) >= 2

    def test_upstream_plot_with_error_bars_enabled(self):
        """Test upstream plot with error bars enabled"""
        dataset = create_mock_dataset()
        plot = PressurePlot([dataset], "upstream-plot", plot_type="upstream")
        plot.plot_parameters["show_error_bars"] = True

        result = plot.generate()

        # Check that error bars are present
        has_error = any(
            hasattr(trace, "error_y") and trace.error_y is not None
            for trace in result.data
        )
        assert has_error

    def test_upstream_plot_with_valve_times_enabled(self):
        """Test upstream plot with valve time markers"""
        dataset = create_mock_dataset(
            valve_times={"valve_open": 3.0, "valve_close": 7.0}
        )
        plot = PressurePlot([dataset], "upstream-plot", plot_type="upstream")
        plot.plot_parameters["show_valve_times"] = True

        result = plot.generate()

        # Should have vertical lines for valve events
        assert len(result.layout.shapes) == 2

    def test_upstream_plot_with_linear_scales_and_ranges(self):
        """Test upstream plot with linear scales and custom ranges"""
        dataset = create_mock_dataset()
        plot = PressurePlot([dataset], "upstream-plot", plot_type="upstream")
        plot.plot_parameters["x_scale"] = "linear"
        plot.plot_parameters["y_scale"] = "linear"
        plot.plot_parameters["x_min"] = 0
        plot.plot_parameters["x_max"] = 20
        plot.plot_parameters["y_min"] = 50
        plot.plot_parameters["y_max"] = 500

        result = plot.generate()

        assert result.layout.xaxis.type == "linear"
        assert result.layout.yaxis.type == "linear"
        assert result.layout.xaxis.range == (0, 20)
        assert result.layout.yaxis.range == (50, 500)

    def test_upstream_plot_with_log_scales(self):
        """Test upstream plot with logarithmic scales"""
        dataset = create_mock_dataset()
        plot = PressurePlot([dataset], "upstream-plot", plot_type="upstream")
        plot.plot_parameters["x_scale"] = "log"
        plot.plot_parameters["y_scale"] = "log"
        plot.plot_parameters["x_min"] = 1
        plot.plot_parameters["x_max"] = 100
        plot.plot_parameters["y_min"] = 10
        plot.plot_parameters["y_max"] = 1000

        result = plot.generate()

        assert result.layout.xaxis.type == "log"
        assert result.layout.yaxis.type == "log"


class TestPressurePlotIntegrationDownstream:
    """Integration tests for downstream pressure plots."""

    def test_complete_downstream_plot_generation(self):
        """Test complete workflow for generating downstream pressure plot"""
        dataset = create_mock_dataset(name="Test Run", colour="#00FF00")
        plot = PressurePlot([dataset], "downstream-plot", plot_type="downstream")

        result = plot.generate()

        assert isinstance(result, FigureResampler)
        assert len(result.data) > 0
        assert plot.pressure_attr == "downstream_pressure"
        assert plot.error_attr == "downstream_error"

    def test_downstream_plot_with_multiple_datasets(self):
        """Test downstream plot with multiple datasets"""
        dataset1 = create_mock_dataset(name="Run 1")
        dataset2 = create_mock_dataset(name="Run 2")

        plot = PressurePlot(
            [dataset1, dataset2], "downstream-plot", plot_type="downstream"
        )
        result = plot.generate()

        assert len(result.data) >= 2

    def test_downstream_plot_uses_downstream_pressure_data(self):
        """Test that downstream plot uses downstream_pressure attribute"""
        dataset = create_mock_dataset()
        plot = PressurePlot([dataset], "downstream-plot", plot_type="downstream")

        plot.generate()

        # Verify the correct attributes were accessed
        assert plot.pressure_attr == "downstream_pressure"
        assert plot.error_attr == "downstream_error"


class TestPressurePlotIntegrationEdgeCases:
    """Integration tests for edge cases and error handling."""

    def test_plot_with_empty_dataset_list(self):
        """Test that plot handles empty dataset list gracefully"""
        plot = PressurePlot([], "test-plot", plot_type="upstream")

        result = plot.generate()

        assert isinstance(result, FigureResampler)
        assert len(result.data) == 0

    def test_plot_with_mismatched_array_lengths(self):
        """Test that plot handles datasets with mismatched array lengths"""
        dataset = MagicMock()
        dataset.name = "Mismatched"
        dataset.colour = "#FF0000"
        dataset.time_data = np.array([1, 2, 3, 4, 5])
        dataset.upstream_pressure = np.array([10, 20, 30])  # Shorter
        dataset.upstream_error = np.array([1, 2, 3, 4])  # Different length
        dataset.valve_times = {}

        plot = PressurePlot([dataset], "test-plot", plot_type="upstream")
        result = plot.generate()

        # Should handle gracefully and create plot
        assert isinstance(result, FigureResampler)

    def test_plot_with_none_data_values(self):
        """Test that plot skips datasets with None data values"""
        dataset_none_time = MagicMock()
        dataset_none_time.time_data = None
        dataset_none_time.upstream_pressure = np.array([1, 2, 3])

        dataset_none_pressure = MagicMock()
        dataset_none_pressure.time_data = np.array([1, 2, 3])
        dataset_none_pressure.upstream_pressure = None

        plot = PressurePlot(
            [dataset_none_time, dataset_none_pressure],
            "test-plot",
            plot_type="upstream",
        )
        result = plot.generate()

        # Should skip both datasets
        assert len(result.data) == 0

    def test_plot_with_empty_arrays(self):
        """Test that plot skips datasets with empty arrays"""
        dataset = MagicMock()
        dataset.time_data = np.array([])
        dataset.upstream_pressure = np.array([])
        dataset.upstream_error = np.array([])
        dataset.valve_times = {}

        plot = PressurePlot([dataset], "test-plot", plot_type="upstream")
        result = plot.generate()

        assert len(result.data) == 0

    def test_plot_with_log_scale_and_auto_ranging(self):
        """Test log scale plot with automatic range calculation from data"""
        dataset = create_mock_dataset(pressure_range=(10, 100))
        plot = PressurePlot([dataset], "test-plot", plot_type="upstream")
        plot.plot_parameters["y_scale"] = "log"
        # Don't set y_min/y_max - should auto-calculate

        result = plot.generate()

        assert result.layout.yaxis.type == "log"
        # Range should be calculated from data

    def test_plot_with_negative_values_and_log_scale(self):
        """Test that negative pressure values are filtered for log scale"""
        dataset = MagicMock()
        dataset.name = "Test"
        dataset.colour = "#FF0000"
        dataset.time_data = np.array([0, 1, 2, 3, 4])
        dataset.upstream_pressure = np.array([-10, 0, 5, 10, 20])  # Mixed
        dataset.upstream_error = np.array([1, 1, 1, 1, 1])
        dataset.valve_times = {}

        plot = PressurePlot([dataset], "test-plot", plot_type="upstream")
        plot.plot_parameters["y_scale"] = "log"

        result = plot.generate()

        # Should handle gracefully (filter negative values)
        assert result.layout.yaxis.type == "log"


class TestPressurePlotIntegrationCombined:
    """Integration tests for combined features."""

    def test_plot_with_all_features_enabled(self):
        """Test plot with error bars, valve markers, and custom scales"""
        dataset = create_mock_dataset(
            name="Full Featured",
            valve_times={"start": 2.0, "peak": 5.0, "end": 8.0},
        )
        plot = PressurePlot([dataset], "full-plot", plot_type="upstream")
        plot.plot_parameters["show_error_bars"] = True
        plot.plot_parameters["show_valve_times"] = True
        plot.plot_parameters["x_scale"] = "linear"
        plot.plot_parameters["y_scale"] = "log"
        plot.plot_parameters["y_min"] = 50
        plot.plot_parameters["y_max"] = 500

        result = plot.generate()

        # Verify all features are present
        assert isinstance(result, FigureResampler)
        assert len(result.layout.shapes) == 3  # Three valve markers
        assert result.layout.xaxis.type == "linear"
        assert result.layout.yaxis.type == "log"

    def test_multiple_datasets_with_different_valve_times(self):
        """Test plot with multiple datasets having different valve events"""
        dataset1 = create_mock_dataset(
            name="Run 1",
            colour="#FF0000",
            valve_times={"open": 1.0, "close": 9.0},
        )
        dataset2 = create_mock_dataset(
            name="Run 2",
            colour="#00FF00",
            valve_times={"open": 1.5, "close": 8.5},
        )

        plot = PressurePlot([dataset1, dataset2], "multi-plot", plot_type="upstream")
        plot.plot_parameters["show_valve_times"] = True

        result = plot.generate()

        # Should have 4 valve markers total (2 per dataset)
        assert len(result.layout.shapes) == 4

    def test_regenerating_plot_with_updated_parameters(self):
        """Test that plot can be regenerated with updated parameters"""
        dataset = create_mock_dataset()
        plot = PressurePlot([dataset], "test-plot", plot_type="upstream")

        # First generation
        plot.plot_parameters["show_error_bars"] = False
        result1 = plot.generate()

        # Update and regenerate
        plot.plot_parameters["show_error_bars"] = True
        result2 = plot.generate()

        # Should be different figures
        assert result1 is not result2
        assert isinstance(result1, FigureResampler)
        assert isinstance(result2, FigureResampler)

    def test_plot_with_mixed_valid_and_invalid_datasets(self):
        """Test plot with mix of valid and invalid datasets"""
        valid_dataset = create_mock_dataset(name="Valid")

        invalid_dataset1 = MagicMock()
        invalid_dataset1.time_data = None
        invalid_dataset1.upstream_pressure = np.array([1, 2, 3])

        invalid_dataset2 = MagicMock()
        invalid_dataset2.time_data = np.array([])
        invalid_dataset2.upstream_pressure = np.array([])

        plot = PressurePlot(
            [invalid_dataset1, valid_dataset, invalid_dataset2],
            "test-plot",
            plot_type="upstream",
        )
        result = plot.generate()

        # Should only plot the valid dataset
        assert len(result.data) > 0

    def test_axis_range_computation_from_multiple_datasets(self):
        """Test that axis ranges are computed from all datasets"""
        dataset1 = create_mock_dataset(time_range=(0, 10), pressure_range=(100, 200))
        dataset2 = create_mock_dataset(time_range=(5, 15), pressure_range=(150, 300))

        plot = PressurePlot([dataset1, dataset2], "test-plot", plot_type="upstream")
        plot.plot_parameters["x_scale"] = "linear"
        plot.plot_parameters["y_scale"] = "linear"
        # Don't set explicit ranges - let it compute from data

        result = plot.generate()

        # Ranges should encompass all data
        x_range = result.layout.xaxis.range
        y_range = result.layout.yaxis.range

        assert x_range[0] <= 0  # Covers first dataset start
        assert x_range[1] >= 15  # Covers second dataset end
        assert y_range[0] <= 100  # Covers minimum pressure
        assert y_range[1] >= 300  # Covers maximum pressure
