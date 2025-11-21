"""Integration tests for shield_das.figures.base_graph module.

This test module verifies the integration behavior of BaseGraph with
concrete implementations, testing the complete workflow of creating
and configuring plots.
"""

import plotly.graph_objs as go
from plotly_resampler import FigureResampler

from shield_das.figures.base_graph import BaseGraph


class ConcreteTestGraph(BaseGraph):
    """Concrete implementation for integration testing."""

    def generate(self) -> go.Figure:
        """Generate a simple plot with trace and layout."""
        self.figure_resampler = self._create_figure_resampler()

        # Add a test trace
        self.figure_resampler.add_trace(
            go.Scatter(x=[1, 2, 3], y=[4, 5, 6], name="Test Data [R]")
        )

        # Apply axis settings
        self._apply_axis_settings()

        # Clean trace names
        self._clean_trace_names()

        return self.figure_resampler


class TestBaseGraphIntegration:
    """Integration tests for BaseGraph complete workflow."""

    def test_generate_creates_complete_figure(self):
        """Test that generate() creates a complete figure with all components"""
        datasets = [{"name": "test1"}, {"name": "test2"}]
        graph = ConcreteTestGraph(datasets, "test-plot")

        result = graph.generate()

        assert isinstance(result, FigureResampler)
        assert len(result.data) > 0
        assert result.data[0].name == "Test Data"  # [R] should be removed

    def test_full_workflow_with_linear_scales(self):
        """Test complete workflow with linear x and y scales"""
        graph = ConcreteTestGraph([], "test-id")
        graph.plot_parameters["x_scale"] = "linear"
        graph.plot_parameters["y_scale"] = "linear"
        graph.plot_parameters["x_min"] = 0
        graph.plot_parameters["x_max"] = 10
        graph.plot_parameters["y_min"] = 0
        graph.plot_parameters["y_max"] = 100

        result = graph.generate()

        assert result.layout.xaxis.type == "linear"
        assert result.layout.yaxis.type == "linear"
        assert result.layout.xaxis.range == (0, 10)
        assert result.layout.yaxis.range == (0, 100)

    def test_full_workflow_with_log_scales(self):
        """Test complete workflow with logarithmic x and y scales"""
        graph = ConcreteTestGraph([], "test-id")
        graph.plot_parameters["x_scale"] = "log"
        graph.plot_parameters["y_scale"] = "log"
        graph.plot_parameters["x_min"] = 1
        graph.plot_parameters["x_max"] = 1000
        graph.plot_parameters["y_min"] = 0.1
        graph.plot_parameters["y_max"] = 100

        result = graph.generate()

        assert result.layout.xaxis.type == "log"
        assert result.layout.yaxis.type == "log"
        assert result.layout.xaxis.range == (1, 1000)
        assert result.layout.yaxis.range == (0.1, 100)

    def test_full_workflow_with_mixed_scales(self):
        """Test complete workflow with log x-axis and linear y-axis"""
        graph = ConcreteTestGraph([], "test-id")
        graph.plot_parameters["x_scale"] = "log"
        graph.plot_parameters["y_scale"] = "linear"
        graph.plot_parameters["x_min"] = 10
        graph.plot_parameters["x_max"] = 10000
        graph.plot_parameters["y_min"] = -50
        graph.plot_parameters["y_max"] = 50

        result = graph.generate()

        assert result.layout.xaxis.type == "log"
        assert result.layout.yaxis.type == "linear"
        assert result.layout.xaxis.range == (10, 10000)
        assert result.layout.yaxis.range == (-50, 50)

    def test_workflow_ignores_invalid_log_ranges(self):
        """Test that invalid log ranges (zero/negative) are ignored"""
        graph = ConcreteTestGraph([], "test-id")
        graph.plot_parameters["x_scale"] = "log"
        graph.plot_parameters["y_scale"] = "log"
        graph.plot_parameters["x_min"] = -10  # Invalid for log
        graph.plot_parameters["x_max"] = 100
        graph.plot_parameters["y_min"] = 0  # Invalid for log
        graph.plot_parameters["y_max"] = 1000

        result = graph.generate()

        # Ranges should not be applied due to invalid values
        assert result.layout.xaxis.range is None
        assert result.layout.yaxis.range is None

    def test_workflow_updates_plot_parameters_dynamically(self):
        """Test that plot parameters can be updated before generate()"""
        graph = ConcreteTestGraph([], "test-id")

        # Initial parameters
        graph.plot_parameters["show_error_bars"] = False
        graph.plot_parameters["x_scale"] = "linear"

        # Update parameters
        graph.plot_parameters["show_error_bars"] = True
        graph.plot_parameters["x_scale"] = "log"
        graph.plot_parameters["x_min"] = 1
        graph.plot_parameters["x_max"] = 100

        result = graph.generate()

        # Should use updated parameters
        assert result.layout.xaxis.type == "log"
        assert result.layout.xaxis.range == (1, 100)

    def test_multiple_generate_calls_create_new_figures(self):
        """Test that calling generate() multiple times creates new figures"""
        graph = ConcreteTestGraph([], "test-id")

        result1 = graph.generate()
        result2 = graph.generate()

        # Each call should create a new figure
        assert result1 is not result2
        assert isinstance(result1, FigureResampler)
        assert isinstance(result2, FigureResampler)

    def test_trace_name_cleaning_removes_all_suffixes(self):
        """Test that all [R] suffixes are removed from multiple traces"""

        class MultiTraceGraph(BaseGraph):
            def generate(self):
                self.figure_resampler = self._create_figure_resampler()
                self.figure_resampler.add_trace(
                    go.Scatter(x=[1, 2], y=[3, 4], name="Trace A [R]")
                )
                self.figure_resampler.add_trace(
                    go.Scatter(x=[5, 6], y=[7, 8], name="Trace B [R]")
                )
                self.figure_resampler.add_trace(
                    go.Scatter(x=[9, 10], y=[11, 12], name="Trace C [R]")
                )
                self._clean_trace_names()
                return self.figure_resampler

        graph = MultiTraceGraph([], "test-id")
        result = graph.generate()

        assert result.data[0].name == "Trace A"
        assert result.data[1].name == "Trace B"
        assert result.data[2].name == "Trace C"

    def test_workflow_with_no_range_specified_uses_defaults(self):
        """Test that when no range is specified, figure uses auto-ranging"""
        graph = ConcreteTestGraph([], "test-id")
        graph.plot_parameters["x_scale"] = "linear"
        graph.plot_parameters["y_scale"] = "linear"
        # Don't set min/max values

        result = graph.generate()

        # Should not have explicit ranges (will auto-range)
        assert result.layout.xaxis.range is None
        assert result.layout.yaxis.range is None

    def test_workflow_preserves_datasets_and_plot_id(self):
        """Test that datasets and plot_id are preserved through workflow"""
        datasets = [{"name": "A"}, {"name": "B"}]
        plot_id = "integration-test-plot"

        graph = ConcreteTestGraph(datasets, plot_id)
        graph.generate()

        assert graph.datasets == datasets
        assert graph.plot_id == plot_id

    def test_workflow_with_partial_range_specification(self):
        """Test that having only min or only max doesn't apply range"""
        graph = ConcreteTestGraph([], "test-id")
        graph.plot_parameters["x_min"] = 0  # Only min specified
        graph.plot_parameters["y_max"] = 100  # Only max specified

        result = graph.generate()

        # Ranges should not be applied (need both min and max)
        assert result.layout.xaxis.range is None
        assert result.layout.yaxis.range is None

    def test_workflow_with_inverted_range_ignored(self):
        """Test that ranges where min > max are ignored"""
        graph = ConcreteTestGraph([], "test-id")
        graph.plot_parameters["x_min"] = 100
        graph.plot_parameters["x_max"] = 10  # min > max
        graph.plot_parameters["y_min"] = 50
        graph.plot_parameters["y_max"] = 5  # min > max

        result = graph.generate()

        # Invalid ranges should not be applied
        assert result.layout.xaxis.range is None
        assert result.layout.yaxis.range is None
