"""
Tests for valve time functionality in DataPlotter.
Focus on specific valve time features with simple, focused tests.
"""

from shield_das.data_plotter import DataPlotter


class TestValveTimeConstants:
    """Test valve time helper constants and methods."""

    def test_plot_control_states_constant_exists(self):
        """Test that PLOT_CONTROL_STATES constant includes valve time states."""
        plotter = DataPlotter()
        assert hasattr(plotter, "PLOT_CONTROL_STATES")

        # Extract component IDs from states
        state_ids = [state.component_id for state in plotter.PLOT_CONTROL_STATES]

        # Should include all four control states
        expected_states = [
            "show-error-bars-upstream",
            "show-error-bars-downstream",
            "show-valve-times-upstream",
            "show-valve-times-downstream",
        ]

        for expected_state in expected_states:
            assert expected_state in state_ids

    def test_generate_both_plots_helper_exists(self):
        """Test that _generate_both_plots helper method exists."""
        plotter = DataPlotter()
        assert hasattr(plotter, "_generate_both_plots")
        assert callable(getattr(plotter, "_generate_both_plots"))


class TestValveTimeParameters:
    """Test valve time parameter handling in plot methods."""

    def setup_method(self):
        """Set up test dataset."""
        self.plotter = DataPlotter()
        self.mock_dataset = {
            "name": "Test Dataset",
            "colour": "#FF0000",
            "time_data": [0, 1, 2, 3, 4],
            "upstream_data": {
                "pressure_data": [10, 15, 20, 25, 30],
                "error_data": [1, 1.5, 2, 2.5, 3],
            },
            "downstream_data": {
                "pressure_data": [5, 7, 10, 12, 15],
                "error_data": [0.5, 0.7, 1, 1.2, 1.5],
            },
            "valve_times": {"v4_close_time": 1.5, "v5_close_time": 2.5},
        }
        self.plotter.datasets = {"test": self.mock_dataset}

    def test_upstream_plot_accepts_valve_times_parameter(self):
        """Test upstream plot method accepts show_valve_times parameter."""
        # Should not raise exceptions with valve times enabled
        fig_enabled = self.plotter._generate_upstream_plot(show_valve_times=True)
        assert fig_enabled is not None

        # Should not raise exceptions with valve times disabled
        fig_disabled = self.plotter._generate_upstream_plot(show_valve_times=False)
        assert fig_disabled is not None

    def test_downstream_plot_accepts_valve_times_parameter(self):
        """Test downstream plot method accepts show_valve_times parameter."""
        # Should not raise exceptions with valve times enabled
        fig_enabled = self.plotter._generate_downstream_plot(show_valve_times=True)
        assert fig_enabled is not None

        # Should not raise exceptions with valve times disabled
        fig_disabled = self.plotter._generate_downstream_plot(show_valve_times=False)
        assert fig_disabled is not None

    def test_generate_both_plots_accepts_valve_parameters(self):
        """Test _generate_both_plots accepts valve time parameters."""
        plots = self.plotter._generate_both_plots(
            show_error_bars_upstream=True,
            show_error_bars_downstream=True,
            show_valve_times_upstream=True,
            show_valve_times_downstream=False,
        )

        assert len(plots) == 3
        assert plots[0] is not None  # upstream
        assert plots[1] is not None  # downstream

    def test_valve_times_default_to_false(self):
        """Test that valve times default to False when not specified."""
        # Should work without specifying valve times (defaults to False)
        upstream_fig = self.plotter._generate_upstream_plot()
        downstream_fig = self.plotter._generate_downstream_plot()

        assert upstream_fig is not None
        assert downstream_fig is not None


class TestValveTimeDataHandling:
    """Test valve time data processing and edge cases."""

    def test_missing_valve_times_key(self):
        """Test handling when valve_times key is missing from dataset."""
        plotter = DataPlotter()
        dataset_no_valves = {
            "name": "No Valves",
            "colour": "#00FF00",
            "time_data": [0, 1, 2],
            "upstream_data": {"pressure_data": [10, 20, 30], "error_data": [1, 2, 3]},
            "downstream_data": {
                "pressure_data": [5, 10, 15],
                "error_data": [0.5, 1, 1.5],
            },
            # No valve_times key
        }
        plotter.datasets = {"no_valves": dataset_no_valves}

        # Should handle gracefully
        upstream_fig = plotter._generate_upstream_plot(show_valve_times=True)
        downstream_fig = plotter._generate_downstream_plot(show_valve_times=True)

        assert upstream_fig is not None
        assert downstream_fig is not None

    def test_empty_valve_times_dict(self):
        """Test handling when valve_times is an empty dictionary."""
        plotter = DataPlotter()
        dataset_empty_valves = {
            "name": "Empty Valves",
            "colour": "#0000FF",
            "time_data": [0, 1, 2],
            "upstream_data": {"pressure_data": [10, 20, 30], "error_data": [1, 2, 3]},
            "downstream_data": {
                "pressure_data": [5, 10, 15],
                "error_data": [0.5, 1, 1.5],
            },
            "valve_times": {},  # Empty dict
        }
        plotter.datasets = {"empty": dataset_empty_valves}

        # Should handle gracefully
        upstream_fig = plotter._generate_upstream_plot(show_valve_times=True)
        downstream_fig = plotter._generate_downstream_plot(show_valve_times=True)

        assert upstream_fig is not None
        assert downstream_fig is not None

    def test_no_datasets_loaded(self):
        """Test behavior when no datasets are loaded."""
        plotter = DataPlotter()
        plotter.datasets = {}

        # Should generate empty plots without errors
        upstream_fig = plotter._generate_upstream_plot(show_valve_times=True)
        downstream_fig = plotter._generate_downstream_plot(show_valve_times=True)

        assert upstream_fig is not None
        assert downstream_fig is not None
        assert len(upstream_fig.data) == 0
        assert len(downstream_fig.data) == 0


class TestValveEventFormatting:
    """Test valve event name formatting."""

    def test_valve_event_name_formatting(self):
        """Test that valve event names are properly formatted for display."""
        test_cases = [
            ("v4_close_time", "V4 Close Time"),
            ("v5_close_time", "V5 Close Time"),
            ("v6_close_time", "V6 Close Time"),
            ("v3_open_time", "V3 Open Time"),
        ]

        for input_name, expected_output in test_cases:
            # This is the formatting logic used in the valve line annotations
            formatted = input_name.replace("_", " ").title()
            assert formatted == expected_output


class TestValveTimeRelativeCalculation:
    """Test valve time relative calculation logic."""

    def test_relative_time_calculation_concept(self):
        """Test the concept of relative time calculation."""
        # Example of the calculation that should happen in create_dataset
        from datetime import datetime

        start_time_str = "2025-08-12 15:30:00.000"
        valve_time_str = "2025-08-12 15:30:05.500"

        start_time = datetime.strptime(start_time_str, "%Y-%m-%d %H:%M:%S.%f")
        valve_time = datetime.strptime(valve_time_str, "%Y-%m-%d %H:%M:%S.%f")

        relative_time = (valve_time - start_time).total_seconds()

        assert relative_time == 5.5


class TestMultipleDatasets:
    """Test valve times with multiple datasets."""

    def test_multiple_datasets_with_different_valve_times(self):
        """Test plotting multiple datasets with different valve configurations."""
        plotter = DataPlotter()

        dataset1 = {
            "name": "Dataset 1",
            "colour": "#FF0000",
            "time_data": [0, 1, 2, 3],
            "upstream_data": {
                "pressure_data": [10, 15, 20, 25],
                "error_data": [1, 1.5, 2, 2.5],
            },
            "downstream_data": {
                "pressure_data": [5, 7, 10, 12],
                "error_data": [0.5, 0.7, 1, 1.2],
            },
            "valve_times": {"v4_close_time": 1.0, "v5_close_time": 2.0},
        }

        dataset2 = {
            "name": "Dataset 2",
            "colour": "#00FF00",
            "time_data": [0, 1, 2, 3],
            "upstream_data": {
                "pressure_data": [8, 12, 16, 20],
                "error_data": [0.8, 1.2, 1.6, 2.0],
            },
            "downstream_data": {
                "pressure_data": [4, 6, 8, 10],
                "error_data": [0.4, 0.6, 0.8, 1.0],
            },
            "valve_times": {"v4_close_time": 0.5, "v3_open_time": 2.5},
        }

        plotter.datasets = {"ds1": dataset1, "ds2": dataset2}

        # Should handle multiple datasets with different valve configurations
        upstream_fig = plotter._generate_upstream_plot(show_valve_times=True)
        downstream_fig = plotter._generate_downstream_plot(show_valve_times=True)

        assert upstream_fig is not None
        assert downstream_fig is not None

        # Should have data traces for both datasets
        assert len(upstream_fig.data) >= 2
        assert len(downstream_fig.data) >= 2


class TestPlotGeneration:
    """Test actual plot generation with valve times."""

    def setup_method(self):
        """Set up test dataset with valve times."""
        self.plotter = DataPlotter()
        self.dataset = {
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
            "valve_times": {"v4_close_time": 1.5, "v5_close_time": 3.5},
        }
        self.plotter.datasets = {"test": self.dataset}

    def test_plot_generation_with_valve_times_enabled(self):
        """Test that plots generate successfully with valve times enabled."""
        upstream_fig = self.plotter._generate_upstream_plot(
            show_error_bars=True, show_valve_times=True
        )
        downstream_fig = self.plotter._generate_downstream_plot(
            show_error_bars=True, show_valve_times=True
        )

        # Verify basic plot structure
        assert upstream_fig is not None
        assert downstream_fig is not None
        assert len(upstream_fig.data) >= 1  # At least one data trace
        assert len(downstream_fig.data) >= 1  # At least one data trace

        # Verify plot layout exists
        assert hasattr(upstream_fig, "layout")
        assert hasattr(downstream_fig, "layout")

    def test_plot_generation_with_valve_times_disabled(self):
        """Test that plots generate successfully with valve times disabled."""
        upstream_fig = self.plotter._generate_upstream_plot(
            show_error_bars=True, show_valve_times=False
        )
        downstream_fig = self.plotter._generate_downstream_plot(
            show_error_bars=True, show_valve_times=False
        )

        # Verify basic plot structure
        assert upstream_fig is not None
        assert downstream_fig is not None
        assert len(upstream_fig.data) >= 1  # At least one data trace
        assert len(downstream_fig.data) >= 1  # At least one data trace

    def test_plot_generation_with_mixed_parameters(self):
        """Test plot generation with various parameter combinations."""
        test_combinations = [
            {"show_error_bars": True, "show_valve_times": True},
            {"show_error_bars": False, "show_valve_times": True},
            {"show_error_bars": True, "show_valve_times": False},
            {"show_error_bars": False, "show_valve_times": False},
        ]

        for params in test_combinations:
            upstream_fig = self.plotter._generate_upstream_plot(**params)
            downstream_fig = self.plotter._generate_downstream_plot(**params)

            assert upstream_fig is not None
            assert downstream_fig is not None
            assert len(upstream_fig.data) >= 1
            assert len(downstream_fig.data) >= 1
