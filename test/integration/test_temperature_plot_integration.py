"""Integration tests for TemperaturePlot class.

This module contains integration tests that verify complete temperature plot
generation workflows.
"""

from unittest.mock import MagicMock

import numpy as np

from shield_das.figures.temperature_plot import TemperaturePlot


def create_mock_dataset(
    name="Test Dataset",
    has_temp_data=True,
    time_points=100,
    thermocouple_name="TC1",
):
    """Create a mock dataset for testing.

    Args:
        name: Dataset name
        has_temp_data: Whether to include thermocouple data
        time_points: Number of data points
        thermocouple_name: Name of thermocouple

    Returns:
        MagicMock: Mock dataset object
    """
    dataset = MagicMock()
    dataset.name = name
    dataset.colour = "blue"
    dataset.time_data = np.linspace(0, 1000, time_points)
    dataset.local_temperature_data = 200 + 50 * np.sin(dataset.time_data / 100)

    if has_temp_data:
        # Thermocouple reads slightly lower than setpoint
        rng = np.random.default_rng(42)
        dataset.thermocouple_data = (
            dataset.local_temperature_data - 10 + rng.standard_normal(time_points)
        )
        dataset.thermocouple_name = thermocouple_name
    else:
        dataset.thermocouple_data = None
        dataset.thermocouple_name = None

    return dataset


class TestTemperaturePlotIntegrationWithData:
    """Integration tests for temperature plot with complete temperature data."""

    def test_complete_workflow_single_dataset(self):
        """Test complete plot generation workflow with single dataset."""
        dataset = create_mock_dataset()
        plot = TemperaturePlot(datasets=[dataset], plot_id="test")

        fig = plot.generate()

        assert fig is not None
        assert plot.figure_resampler is fig
        assert len(fig.data) == 2  # Furnace + thermocouple traces

    def test_complete_workflow_multiple_datasets(self):
        """Test complete plot generation workflow with multiple datasets."""
        dataset1 = create_mock_dataset(name="Dataset1", thermocouple_name="TC1")
        dataset2 = create_mock_dataset(name="Dataset2", thermocouple_name="TC2")
        plot = TemperaturePlot(datasets=[dataset1, dataset2], plot_id="test")

        fig = plot.generate()

        assert fig is not None
        # 2 traces per dataset (furnace + thermocouple)
        assert len(fig.data) == 4

    def test_trace_names_include_dataset_names(self):
        """Test that trace names include dataset names."""
        dataset = create_mock_dataset(name="MyExperiment")
        plot = TemperaturePlot(datasets=[dataset], plot_id="test")

        fig = plot.generate()

        trace_names = [trace.name for trace in fig.data]
        assert any("MyExperiment" in name for name in trace_names)
        assert any("Furnace setpoint" in name for name in trace_names)

    def test_custom_thermocouple_names_appear_in_traces(self):
        """Test that custom thermocouple names appear in trace names."""
        dataset = create_mock_dataset(thermocouple_name="CustomTC")
        plot = TemperaturePlot(datasets=[dataset], plot_id="test")

        fig = plot.generate()

        trace_names = [trace.name for trace in fig.data]
        assert any("CustomTC" in name for name in trace_names)

    def test_layout_configured_correctly(self):
        """Test that plot layout is configured correctly."""
        dataset = create_mock_dataset()
        plot = TemperaturePlot(datasets=[dataset], plot_id="test")

        fig = plot.generate()

        assert fig.layout.height == 500
        assert fig.layout.xaxis.title.text == "Time (s)"
        assert fig.layout.yaxis.title.text == "Temperature (°C)"
        # Template is an object, not a string
        assert fig.layout.template is not None

    def test_legend_positioned_horizontally_above_plot(self):
        """Test that legend is horizontal and positioned above plot."""
        dataset = create_mock_dataset()
        plot = TemperaturePlot(datasets=[dataset], plot_id="test")

        fig = plot.generate()

        assert fig.layout.legend.orientation == "h"
        assert fig.layout.legend.yanchor == "bottom"
        assert fig.layout.legend.y == 1.02

    def test_trace_names_cleaned_of_r_markers(self):
        """Test that [R] markers are removed from trace names."""
        dataset = create_mock_dataset()
        plot = TemperaturePlot(datasets=[dataset], plot_id="test")

        fig = plot.generate()

        for trace in fig.data:
            if trace.name:
                assert "[R]" not in trace.name

    def test_furnace_trace_is_dashed(self):
        """Test that furnace setpoint trace has dashed line style."""
        dataset = create_mock_dataset()
        plot = TemperaturePlot(datasets=[dataset], plot_id="test")

        fig = plot.generate()

        # First trace should be furnace (dashed)
        furnace_trace = fig.data[0]
        assert furnace_trace.line.dash == "dash"

    def test_thermocouple_trace_is_solid(self):
        """Test that thermocouple trace has solid line style."""
        dataset = create_mock_dataset()
        plot = TemperaturePlot(datasets=[dataset], plot_id="test")

        fig = plot.generate()

        # Second trace should be thermocouple (solid)
        thermocouple_trace = fig.data[1]
        assert getattr(thermocouple_trace.line, "dash", None) is None

    def test_multiple_generate_calls(self):
        """Test that generate() can be called multiple times."""
        dataset = create_mock_dataset()
        plot = TemperaturePlot(datasets=[dataset], plot_id="test")

        fig1 = plot.generate()
        fig2 = plot.generate()

        assert fig1 is not None
        assert fig2 is not None
        # Second call should create new figure
        assert plot.figure_resampler is fig2


class TestTemperaturePlotIntegrationWithoutData:
    """Integration tests for temperature plot without temperature data."""

    def test_plot_with_no_temperature_data(self):
        """Test plot generation when no datasets have temperature data."""
        dataset = create_mock_dataset(has_temp_data=False)
        plot = TemperaturePlot(datasets=[dataset], plot_id="test")

        fig = plot.generate()

        assert fig is not None
        # No traces should be added
        assert len(fig.data) == 0

    def test_annotation_shown_when_no_data(self):
        """Test that annotation is shown when no temperature data exists."""
        dataset = create_mock_dataset(name="NoData", has_temp_data=False)
        plot = TemperaturePlot(datasets=[dataset], plot_id="test")

        fig = plot.generate()

        assert len(fig.layout.annotations) > 0
        annotation = fig.layout.annotations[0]
        assert "No temperature data" in annotation.text

    def test_annotation_centered_when_no_data(self):
        """Test that annotation is centered when no data exists."""
        dataset = create_mock_dataset(has_temp_data=False)
        plot = TemperaturePlot(datasets=[dataset], plot_id="test")

        fig = plot.generate()

        annotation = fig.layout.annotations[0]
        assert annotation.x == 0.5
        assert annotation.y == 0.5
        assert annotation.xanchor == "center"
        assert annotation.yanchor == "middle"

    def test_annotation_font_size_large_when_no_data(self):
        """Test that annotation font is large when no data exists."""
        dataset = create_mock_dataset(has_temp_data=False)
        plot = TemperaturePlot(datasets=[dataset], plot_id="test")

        fig = plot.generate()

        annotation = fig.layout.annotations[0]
        assert annotation.font.size == 16


class TestTemperaturePlotIntegrationMixedData:
    """Integration tests for temperature plot with mixed data availability."""

    def test_plot_with_partial_temperature_data(self):
        """Test plot when some datasets have temperature data, others don't."""
        dataset1 = create_mock_dataset(name="HasData", has_temp_data=True)
        dataset2 = create_mock_dataset(name="NoData", has_temp_data=False)
        plot = TemperaturePlot(datasets=[dataset1, dataset2], plot_id="test")

        fig = plot.generate()

        assert fig is not None
        # Only 2 traces for dataset with data
        assert len(fig.data) == 2

    def test_annotation_for_missing_datasets(self):
        """Test annotation lists datasets without temperature data."""
        dataset1 = create_mock_dataset(name="HasData", has_temp_data=True)
        dataset2 = create_mock_dataset(name="NoData", has_temp_data=False)
        plot = TemperaturePlot(datasets=[dataset1, dataset2], plot_id="test")

        fig = plot.generate()

        assert len(fig.layout.annotations) > 0
        annotation = fig.layout.annotations[0]
        assert "NoData" in annotation.text
        assert "No temperature data for:" in annotation.text

    def test_annotation_positioned_in_corner_when_some_data(self):
        """Test annotation positioned in corner when some data exists."""
        dataset1 = create_mock_dataset(name="HasData", has_temp_data=True)
        dataset2 = create_mock_dataset(name="NoData", has_temp_data=False)
        plot = TemperaturePlot(datasets=[dataset1, dataset2], plot_id="test")

        fig = plot.generate()

        annotation = fig.layout.annotations[0]
        assert annotation.x == 0.98
        assert annotation.y == 0.02
        assert annotation.xanchor == "right"
        assert annotation.yanchor == "bottom"

    def test_annotation_font_smaller_when_some_data(self):
        """Test annotation font is smaller when some data exists."""
        dataset1 = create_mock_dataset(name="HasData", has_temp_data=True)
        dataset2 = create_mock_dataset(name="NoData", has_temp_data=False)
        plot = TemperaturePlot(datasets=[dataset1, dataset2], plot_id="test")

        fig = plot.generate()

        annotation = fig.layout.annotations[0]
        assert annotation.font.size == 10

    def test_multiple_missing_datasets_listed_in_annotation(self):
        """Test that all missing datasets are listed in annotation."""
        dataset1 = create_mock_dataset(name="HasData", has_temp_data=True)
        dataset2 = create_mock_dataset(name="NoData1", has_temp_data=False)
        dataset3 = create_mock_dataset(name="NoData2", has_temp_data=False)
        plot = TemperaturePlot(datasets=[dataset1, dataset2, dataset3], plot_id="test")

        fig = plot.generate()

        annotation = fig.layout.annotations[0]
        assert "NoData1" in annotation.text
        assert "NoData2" in annotation.text


class TestTemperaturePlotIntegrationEdgeCases:
    """Integration tests for edge cases in temperature plot generation."""

    def test_empty_dataset_list(self):
        """Test plot generation with empty dataset list."""
        plot = TemperaturePlot(datasets=[], plot_id="test")

        fig = plot.generate()

        assert fig is not None
        assert len(fig.data) == 0

    def test_single_data_point(self):
        """Test plot generation with single data point."""
        dataset = create_mock_dataset(time_points=1)
        plot = TemperaturePlot(datasets=[dataset], plot_id="test")

        fig = plot.generate()

        assert fig is not None
        assert len(fig.data) == 2

    def test_large_dataset(self):
        """Test plot generation with large dataset."""
        dataset = create_mock_dataset(time_points=10000)
        plot = TemperaturePlot(datasets=[dataset], plot_id="test")

        fig = plot.generate()

        assert fig is not None
        assert len(fig.data) == 2

    def test_plot_parameters_can_be_modified(self):
        """Test that plot parameters can be modified after initialization."""
        dataset = create_mock_dataset()
        plot = TemperaturePlot(datasets=[dataset], plot_id="test")

        plot.plot_parameters["x_scale"] = "log"
        fig = plot.generate()

        assert fig is not None
        assert plot.plot_parameters["x_scale"] == "log"

    def test_dataset_colour_applied_to_traces(self):
        """Test that dataset colour is applied to both traces."""
        dataset = create_mock_dataset()
        dataset.colour = "purple"
        plot = TemperaturePlot(datasets=[dataset], plot_id="test")

        fig = plot.generate()

        assert fig.data[0].line.color == "purple"
        assert fig.data[1].line.color == "purple"

    def test_different_colours_for_multiple_datasets(self):
        """Test that different datasets can have different colours."""
        dataset1 = create_mock_dataset(name="Dataset1")
        dataset1.colour = "red"
        dataset2 = create_mock_dataset(name="Dataset2")
        dataset2.colour = "green"
        plot = TemperaturePlot(datasets=[dataset1, dataset2], plot_id="test")

        fig = plot.generate()

        # First dataset traces (furnace + thermocouple)
        assert fig.data[0].line.color == "red"
        assert fig.data[1].line.color == "red"
        # Second dataset traces
        assert fig.data[2].line.color == "green"
        assert fig.data[3].line.color == "green"
