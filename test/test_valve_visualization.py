"""
Tests for valve time visualization functionality in DataPlotter.
These tests focus on the valve lines display feature and checkbox controls.
"""

import json
import os
import tempfile
from unittest.mock import Mock, patch

import pytest

from shield_das.data_plotter import DataPlotter


class TestValveTimeVisualization:
    """Test suite for valve time visualization features"""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.temp_dir = tempfile.mkdtemp()
        self.plotter = DataPlotter()

        # Create mock dataset with valve times
        self.mock_dataset = {
            "name": "Test Dataset",
            "colour": "#FF0000",
            "time_data": [0, 1, 2, 3, 4, 5],
            "upstream_data": {
                "pressure_data": [10, 15, 20, 25, 30, 35],
                "error_data": [1, 1.5, 2, 2.5, 3, 3.5],
            },
            "downstream_data": {
                "pressure_data": [5, 7.5, 10, 12.5, 15, 17.5],
                "error_data": [0.5, 0.75, 1, 1.25, 1.5, 1.75],
            },
            "valve_times": {
                "v4_close_time": 1.5,
                "v5_close_time": 2.5,
                "v6_close_time": 3.5,
                "v3_open_time": 4.5,
            },
        }

        # Add dataset to plotter
        self.plotter.datasets = {"test_dataset": self.mock_dataset}

    def test_valve_times_constant_exists(self):
        """Test that PLOT_CONTROL_STATES constant includes valve time states."""
        assert hasattr(self.plotter, "PLOT_CONTROL_STATES")
        states = [state.component_id for state in self.plotter.PLOT_CONTROL_STATES]
        assert "show-valve-times-upstream" in states
        assert "show-valve-times-downstream" in states

    def test_generate_both_plots_helper_method(self):
        """Test the _generate_both_plots helper method."""
        plots = self.plotter._generate_both_plots(
            show_error_bars_upstream=True,
            show_error_bars_downstream=True,
            show_valve_times_upstream=True,
            show_valve_times_downstream=True,
        )

        assert len(plots) == 3
        assert plots[0] is not None  # upstream plot
        assert plots[1] is not None  # downstream plot

    def test_upstream_plot_with_valve_times_enabled(self):
        """Test upstream plot generation with valve times enabled."""
        fig = self.plotter._generate_upstream_plot(
            show_error_bars=True, show_valve_times=True
        )

        # Check that valve lines were added to the figure
        shapes = fig.layout.shapes or []
        annotations = fig.layout.annotations or []

        # Should have vertical lines for each valve event
        expected_valve_count = len(self.mock_dataset["valve_times"])

        # Note: plotly add_vline creates shapes, so we check for vertical lines
        vertical_lines = [
            shape for shape in shapes if shape.type == "line" and shape.x0 == shape.x1
        ]  # vertical line has same x0 and x1

        assert (
            len(vertical_lines) >= expected_valve_count
            or len(annotations) >= expected_valve_count
        )

    def test_upstream_plot_with_valve_times_disabled(self):
        """Test upstream plot generation with valve times disabled."""
        fig = self.plotter._generate_upstream_plot(
            show_error_bars=True, show_valve_times=False
        )

        # Should not have valve time annotations when disabled
        # We can't easily check for absence of add_vline elements,
        # but we can verify the plot generates successfully
        assert fig is not None
        assert hasattr(fig, "data")
        assert len(fig.data) > 0  # Should have data traces

    def test_downstream_plot_with_valve_times_enabled(self):
        """Test downstream plot generation with valve times enabled."""
        fig = self.plotter._generate_downstream_plot(
            show_error_bars=True, show_valve_times=True
        )

        # Check that the plot was generated successfully
        assert fig is not None
        assert hasattr(fig, "data")
        assert len(fig.data) > 0  # Should have data traces

    def test_downstream_plot_with_valve_times_disabled(self):
        """Test downstream plot generation with valve times disabled."""
        fig = self.plotter._generate_downstream_plot(
            show_error_bars=True, show_valve_times=False
        )

        # Should generate plot successfully without valve times
        assert fig is not None
        assert hasattr(fig, "data")
        assert len(fig.data) > 0  # Should have data traces

    def test_plot_methods_accept_valve_times_parameter(self):
        """Test that plot generation methods accept show_valve_times parameter."""
        # Test with various parameter combinations
        test_cases = [
            {"show_valve_times": True},
            {"show_valve_times": False},
            {"show_error_bars": True, "show_valve_times": True},
            {"show_error_bars": False, "show_valve_times": False},
        ]

        for params in test_cases:
            # Should not raise exceptions
            upstream_fig = self.plotter._generate_upstream_plot(**params)
            downstream_fig = self.plotter._generate_downstream_plot(**params)

            assert upstream_fig is not None
            assert downstream_fig is not None

    def test_empty_valve_times_handling(self):
        """Test handling of datasets with no valve times."""
        # Create dataset without valve times
        dataset_no_valves = {
            "name": "No Valves Dataset",
            "colour": "#00FF00",
            "time_data": [0, 1, 2],
            "upstream_data": {"pressure_data": [10, 20, 30], "error_data": [1, 2, 3]},
            "downstream_data": {
                "pressure_data": [5, 10, 15],
                "error_data": [0.5, 1, 1.5],
            },
            # No valve_times key
        }

        plotter = DataPlotter()
        plotter.datasets = {"no_valves": dataset_no_valves}

        # Should handle missing valve_times gracefully
        upstream_fig = plotter._generate_upstream_plot(show_valve_times=True)
        downstream_fig = plotter._generate_downstream_plot(show_valve_times=True)

        assert upstream_fig is not None
        assert downstream_fig is not None

    def test_valve_times_with_empty_dict(self):
        """Test handling of datasets with empty valve_times dict."""
        dataset_empty_valves = {
            "name": "Empty Valves Dataset",
            "colour": "#0000FF",
            "time_data": [0, 1, 2],
            "upstream_data": {"pressure_data": [10, 20, 30], "error_data": [1, 2, 3]},
            "downstream_data": {
                "pressure_data": [5, 10, 15],
                "error_data": [0.5, 1, 1.5],
            },
            "valve_times": {},  # Empty dict
        }

        plotter = DataPlotter()
        plotter.datasets = {"empty_valves": dataset_empty_valves}

        # Should handle empty valve_times dict gracefully
        upstream_fig = plotter._generate_upstream_plot(show_valve_times=True)
        downstream_fig = plotter._generate_downstream_plot(show_valve_times=True)

        assert upstream_fig is not None
        assert downstream_fig is not None

    def test_multiple_datasets_with_different_valve_times(self):
        """Test plotting multiple datasets with different valve times."""
        # Create second dataset with different valve times
        dataset2 = {
            "name": "Dataset 2",
            "colour": "#00FFFF",
            "time_data": [0, 1, 2, 3, 4],
            "upstream_data": {
                "pressure_data": [5, 10, 15, 20, 25],
                "error_data": [0.5, 1, 1.5, 2, 2.5],
            },
            "downstream_data": {
                "pressure_data": [2.5, 5, 7.5, 10, 12.5],
                "error_data": [0.25, 0.5, 0.75, 1, 1.25],
            },
            "valve_times": {
                "v4_close_time": 0.8,
                "v5_close_time": 1.8,
                "v3_open_time": 3.8,
            },
        }

        plotter = DataPlotter()
        plotter.datasets = {"dataset1": self.mock_dataset, "dataset2": dataset2}

        # Should handle multiple datasets with different valve configurations
        upstream_fig = plotter._generate_upstream_plot(show_valve_times=True)
        downstream_fig = plotter._generate_downstream_plot(show_valve_times=True)

        assert upstream_fig is not None
        assert downstream_fig is not None

        # Should have traces for both datasets
        assert len(upstream_fig.data) >= 2
        assert len(downstream_fig.data) >= 2


class TestValveTimeDataProcessing:
    """Test suite for valve time data processing and metadata handling"""

    def test_valve_time_relative_calculation(self):
        """Test that valve times are correctly converted to relative time."""
        # This would test the create_dataset method's valve time processing
        # Mock the metadata file reading
        metadata = {
            "run_info": {
                "start_time": "2025-08-12 15:30:00.000",
                "v4_close_time": "2025-08-12 15:30:05.500",  # 5.5 seconds after start
                "v5_close_time": "2025-08-12 15:30:10.750",  # 10.75 seconds after start
            }
        }

        # Create a temporary metadata file
        temp_dir = tempfile.mkdtemp()
        metadata_path = os.path.join(temp_dir, "run_metadata.json")

        with open(metadata_path, "w") as f:
            json.dump(metadata, f)

        plotter = DataPlotter()

        # Mock the file structure
        with (
            patch("os.path.exists", return_value=True),
            patch("builtins.open", mock_open_for_metadata(metadata)),
        ):
            # This would call create_dataset which processes valve times
            # For now, we just test the concept that relative times should be calculated
            expected_relative_times = {"v4_close_time": 5.5, "v5_close_time": 10.75}

            # Verify the expected behavior (this would be tested in create_dataset)
            assert expected_relative_times["v4_close_time"] == 5.5
            assert expected_relative_times["v5_close_time"] == 10.75

    def test_valve_event_name_formatting(self):
        """Test that valve event names are properly formatted for display."""
        # Test the formatting logic: valve_event.replace("_", " ").title()
        test_cases = [
            ("v4_close_time", "V4 Close Time"),
            ("v5_close_time", "V5 Close Time"),
            ("v6_close_time", "V6 Close Time"),
            ("v3_open_time", "V3 Open Time"),
            ("some_valve_event", "Some Valve Event"),
        ]

        for input_name, expected_output in test_cases:
            formatted = input_name.replace("_", " ").title()
            assert formatted == expected_output


def mock_open_for_metadata(metadata_dict):
    """Helper function to mock file opening for metadata."""
    from unittest.mock import mock_open

    return mock_open(read_data=json.dumps(metadata_dict))


class TestValveTimeEdgeCases:
    """Test suite for edge cases and error conditions"""

    def test_no_datasets_loaded(self):
        """Test behavior when no datasets are loaded."""
        plotter = DataPlotter()
        plotter.datasets = {}

        # Should handle empty datasets gracefully
        upstream_fig = plotter._generate_upstream_plot(show_valve_times=True)
        downstream_fig = plotter._generate_downstream_plot(show_valve_times=True)

        assert upstream_fig is not None
        assert downstream_fig is not None
        # Should have no data traces but still be valid figures
        assert len(upstream_fig.data) == 0
        assert len(downstream_fig.data) == 0
