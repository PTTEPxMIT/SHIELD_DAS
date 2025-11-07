"""Tests for valve time functionality in DataPlotter."""

from datetime import datetime

import pytest

from unittest.mock import MagicMock

import numpy as np

from shield_das.data_plotter import DataPlotter


def create_mock_dataset(
    name="Test Dataset",
    colour="#FF0000",
    time_data=None,
    upstream_pressure=None,
    downstream_pressure=None,
    upstream_error=None,
    downstream_error=None,
    valve_times=None,
):
    """Create a mock Dataset object for testing.

    Args:
        name: Dataset name
        colour: Dataset color
        time_data: Time array
        upstream_pressure: Upstream pressure array
        downstream_pressure: Downstream pressure array
        upstream_error: Upstream error array
        downstream_error: Downstream error array
        valve_times: Dictionary of valve event times

    Returns:
        Mock Dataset object with required attributes
    """
    dataset = MagicMock()
    dataset.name = name
    dataset.colour = colour
    default_time = np.array([0, 1, 2, 3, 4])
    dataset.time_data = time_data if time_data is not None else default_time
    dataset.upstream_pressure = (
        upstream_pressure
        if upstream_pressure is not None
        else np.array([10, 15, 20, 25, 30])
    )
    dataset.downstream_pressure = (
        downstream_pressure
        if downstream_pressure is not None
        else np.array([5, 7, 10, 12, 15])
    )
    dataset.upstream_error = (
        upstream_error if upstream_error is not None else np.array([1, 1.5, 2, 2.5, 3])
    )
    dataset.downstream_error = (
        downstream_error
        if downstream_error is not None
        else np.array([0.5, 0.7, 1, 1.2, 1.5])
    )
    dataset.valve_times = valve_times if valve_times is not None else {}
    # Add temperature data attributes (set to None by default for valve tests)
    dataset.thermocouple_data = None
    dataset.local_temperature_data = None
    dataset.thermocouple_name = None
    return dataset


class TestValveTimeConstants:
    """Test valve time helper constants and methods."""

    def test_plot_control_states_constant_exists(self):
        """Test that PLOT_CONTROL_STATES constant includes valve time states."""
        # PLOT_CONTROL_STATES is now in callbacks.states module, not on plotter
        from shield_das.callbacks.states import PLOT_CONTROL_STATES

        # Extract component IDs from states
        state_ids = [state.component_id for state in PLOT_CONTROL_STATES]

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
        self.mock_dataset = create_mock_dataset(
            name="Test Dataset",
            colour="#FF0000",
            valve_times={"v4_close_time": 1.5, "v5_close_time": 2.5},
        )
        self.plotter.datasets = [self.mock_dataset]

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
        """Test handling when valve_times attribute is empty."""
        plotter = DataPlotter()
        dataset_no_valves = create_mock_dataset(
            name="No Valves",
            colour="#00FF00",
            time_data=np.array([0, 1, 2]),
            upstream_pressure=np.array([10, 20, 30]),
            downstream_pressure=np.array([5, 10, 15]),
            valve_times={},  # Empty valve times
        )
        plotter.datasets = [dataset_no_valves]

        # Should handle gracefully
        upstream_fig = plotter._generate_upstream_plot(show_valve_times=True)
        downstream_fig = plotter._generate_downstream_plot(show_valve_times=True)

        assert upstream_fig is not None
        assert downstream_fig is not None

    def test_empty_valve_times_dict(self):
        """Test handling when valve_times is an empty dictionary."""
        plotter = DataPlotter()
        dataset_empty_valves = create_mock_dataset(
            name="Empty Valves",
            colour="#0000FF",
            time_data=np.array([0, 1, 2]),
            upstream_pressure=np.array([10, 20, 30]),
            downstream_pressure=np.array([5, 10, 15]),
            valve_times={},  # Empty dict
        )
        plotter.datasets = [dataset_empty_valves]

        # Should handle gracefully
        upstream_fig = plotter._generate_upstream_plot(show_valve_times=True)
        downstream_fig = plotter._generate_downstream_plot(show_valve_times=True)

        assert upstream_fig is not None
        assert downstream_fig is not None

    def test_no_datasets_loaded(self):
        """Test behavior when no datasets are loaded."""
        plotter = DataPlotter()
        plotter.datasets = []

        # Should generate empty plots without errors
        upstream_fig = plotter._generate_upstream_plot(show_valve_times=True)
        downstream_fig = plotter._generate_downstream_plot(show_valve_times=True)

        assert upstream_fig is not None
        assert downstream_fig is not None
        assert len(upstream_fig.data) == 0
        assert len(downstream_fig.data) == 0


@pytest.fixture
def plotter_with_dataset(plotter, mock_dataset):
    """Create a DataPlotter with a loaded dataset."""
    plotter.datasets = {"test": mock_dataset}
    return plotter


# =============================================================================
# Tests for Valve Time Constants
# =============================================================================


class TestMultipleDatasets:
    """Test valve times with multiple datasets."""

    def test_multiple_datasets_with_different_valve_times(self):
        """Test plotting multiple datasets with different valve configurations."""
        plotter = DataPlotter()

        dataset1 = create_mock_dataset(
            name="Dataset 1",
            colour="#FF0000",
            time_data=np.array([0, 1, 2, 3]),
            upstream_pressure=np.array([10, 15, 20, 25]),
            downstream_pressure=np.array([5, 7, 10, 12]),
            upstream_error=np.array([1, 1.5, 2, 2.5]),
            downstream_error=np.array([0.5, 0.7, 1, 1.2]),
            valve_times={"v4_close_time": 1.0, "v5_close_time": 2.0},
        )

        dataset2 = create_mock_dataset(
            name="Dataset 2",
            colour="#00FF00",
            time_data=np.array([0, 1, 2, 3]),
            upstream_pressure=np.array([8, 12, 16, 20]),
            downstream_pressure=np.array([4, 6, 8, 10]),
            upstream_error=np.array([0.8, 1.2, 1.6, 2.0]),
            downstream_error=np.array([0.4, 0.6, 0.8, 1.0]),
            valve_times={"v4_close_time": 0.5, "v3_open_time": 2.5},
        )

        plotter.datasets = [dataset1, dataset2]

        # Should handle multiple datasets with different valve configurations
        upstream_fig = plotter._generate_upstream_plot(show_valve_times=True)
        downstream_fig = plotter._generate_downstream_plot(show_valve_times=True)

        assert upstream_fig is not None
        assert downstream_fig is not None

        # Should have data traces for both datasets
        assert len(upstream_fig.data) >= 2
        assert len(downstream_fig.data) >= 2


def test_plot_control_states_includes_show_error_bars_upstream(plotter):
    """
    Test DataPlotter PLOT_CONTROL_STATES to verify it includes
    show-error-bars-upstream state.
    """
    state_ids = [state.component_id for state in plotter.PLOT_CONTROL_STATES]
    assert "show-error-bars-upstream" in state_ids

    def setup_method(self):
        """Set up test dataset with valve times."""
        self.plotter = DataPlotter()
        self.dataset = create_mock_dataset(
            name="Test Dataset",
            colour="#FF0000",
            time_data=np.array([0, 1, 2, 3, 4, 5]),
            upstream_pressure=np.array([10, 15, 20, 25, 30, 35]),
            downstream_pressure=np.array([5, 7.5, 10, 12.5, 15, 17.5]),
            upstream_error=np.array([1, 1.5, 2, 2.5, 3, 3.5]),
            downstream_error=np.array([0.5, 0.75, 1, 1.25, 1.5, 1.75]),
            valve_times={"v4_close_time": 1.5, "v5_close_time": 3.5},
        )
        self.plotter.datasets = [self.dataset]


def test_plot_control_states_includes_show_error_bars_downstream(plotter):
    """
    Test DataPlotter PLOT_CONTROL_STATES to verify it includes
    show-error-bars-downstream state.
    """
    state_ids = [state.component_id for state in plotter.PLOT_CONTROL_STATES]
    assert "show-error-bars-downstream" in state_ids


def test_plot_control_states_includes_show_valve_times_upstream(plotter):
    """
    Test DataPlotter PLOT_CONTROL_STATES to verify it includes
    show-valve-times-upstream state.
    """
    state_ids = [state.component_id for state in plotter.PLOT_CONTROL_STATES]
    assert "show-valve-times-upstream" in state_ids


def test_plot_control_states_includes_show_valve_times_downstream(plotter):
    """
    Test DataPlotter PLOT_CONTROL_STATES to verify it includes
    show-valve-times-downstream state.
    """
    state_ids = [state.component_id for state in plotter.PLOT_CONTROL_STATES]
    assert "show-valve-times-downstream" in state_ids


def test_generate_both_plots_helper_method_exists(plotter):
    """
    Test DataPlotter to verify _generate_both_plots helper method exists.
    """
    assert hasattr(plotter, "_generate_both_plots")


def test_generate_both_plots_helper_method_is_callable(plotter):
    """
    Test DataPlotter to verify _generate_both_plots method is callable.
    """
    assert callable(getattr(plotter, "_generate_both_plots"))


# =============================================================================
# Tests for Valve Time Parameters
# =============================================================================


def test_upstream_plot_accepts_valve_times_enabled(plotter_with_dataset):
    """
    Test DataPlotter _generate_upstream_plot to verify it accepts
    show_valve_times=True parameter.
    """
    fig = plotter_with_dataset._generate_upstream_plot(show_valve_times=True)
    assert fig is not None


def test_upstream_plot_accepts_valve_times_disabled(plotter_with_dataset):
    """
    Test DataPlotter _generate_upstream_plot to verify it accepts
    show_valve_times=False parameter.
    """
    fig = plotter_with_dataset._generate_upstream_plot(show_valve_times=False)
    assert fig is not None


def test_downstream_plot_accepts_valve_times_enabled(plotter_with_dataset):
    """
    Test DataPlotter _generate_downstream_plot to verify it accepts
    show_valve_times=True parameter.
    """
    fig = plotter_with_dataset._generate_downstream_plot(show_valve_times=True)
    assert fig is not None


def test_downstream_plot_accepts_valve_times_disabled(plotter_with_dataset):
    """
    Test DataPlotter _generate_downstream_plot to verify it accepts
    show_valve_times=False parameter.
    """
    fig = plotter_with_dataset._generate_downstream_plot(show_valve_times=False)
    assert fig is not None


def test_generate_both_plots_accepts_all_valve_parameters(plotter_with_dataset):
    """
    Test DataPlotter _generate_both_plots to verify it accepts all valve
    time parameters.
    """
    plots = plotter_with_dataset._generate_both_plots(
        show_error_bars_upstream=True,
        show_error_bars_downstream=True,
        show_valve_times_upstream=True,
        show_valve_times_downstream=False,
    )
    assert len(plots) == 3


def test_generate_both_plots_returns_upstream_plot(plotter_with_dataset):
    """
    Test DataPlotter _generate_both_plots to verify first element is
    upstream plot.
    """
    plots = plotter_with_dataset._generate_both_plots()
    assert plots[0] is not None


def test_generate_both_plots_returns_downstream_plot(plotter_with_dataset):
    """
    Test DataPlotter _generate_both_plots to verify second element is
    downstream plot.
    """
    plots = plotter_with_dataset._generate_both_plots()
    assert plots[1] is not None


def test_upstream_plot_valve_times_defaults_to_false(plotter_with_dataset):
    """
    Test DataPlotter _generate_upstream_plot to verify show_valve_times
    defaults to False when not specified.
    """
    fig = plotter_with_dataset._generate_upstream_plot()
    assert fig is not None


def test_downstream_plot_valve_times_defaults_to_false(plotter_with_dataset):
    """
    Test DataPlotter _generate_downstream_plot to verify show_valve_times
    defaults to False when not specified.
    """
    fig = plotter_with_dataset._generate_downstream_plot()
    assert fig is not None


# =============================================================================
# Tests for Valve Time Data Handling
# =============================================================================


def test_missing_valve_times_key_upstream_plot(plotter):
    """
    Test DataPlotter _generate_upstream_plot to verify it handles datasets
    with missing valve_times key.
    """
    dataset_no_valves = {
        "name": "No Valves",
        "colour": "#00FF00",
        "time_data": [0, 1, 2],
        "upstream_data": {"pressure_data": [10, 20, 30], "error_data": [1, 2, 3]},
        "downstream_data": {"pressure_data": [5, 10, 15], "error_data": [0.5, 1, 1.5]},
    }
    plotter.datasets = {"no_valves": dataset_no_valves}
    fig = plotter._generate_upstream_plot(show_valve_times=True)
    assert fig is not None


def test_missing_valve_times_key_downstream_plot(plotter):
    """
    Test DataPlotter _generate_downstream_plot to verify it handles datasets
    with missing valve_times key.
    """
    dataset_no_valves = {
        "name": "No Valves",
        "colour": "#00FF00",
        "time_data": [0, 1, 2],
        "upstream_data": {"pressure_data": [10, 20, 30], "error_data": [1, 2, 3]},
        "downstream_data": {"pressure_data": [5, 10, 15], "error_data": [0.5, 1, 1.5]},
    }
    plotter.datasets = {"no_valves": dataset_no_valves}
    fig = plotter._generate_downstream_plot(show_valve_times=True)
    assert fig is not None


def test_empty_valve_times_dict_upstream_plot(plotter):
    """
    Test DataPlotter _generate_upstream_plot to verify it handles datasets
    with empty valve_times dict.
    """
    dataset_empty_valves = {
        "name": "Empty Valves",
        "colour": "#0000FF",
        "time_data": [0, 1, 2],
        "upstream_data": {"pressure_data": [10, 20, 30], "error_data": [1, 2, 3]},
        "downstream_data": {"pressure_data": [5, 10, 15], "error_data": [0.5, 1, 1.5]},
        "valve_times": {},
    }
    plotter.datasets = {"empty": dataset_empty_valves}
    fig = plotter._generate_upstream_plot(show_valve_times=True)
    assert fig is not None


def test_empty_valve_times_dict_downstream_plot(plotter):
    """
    Test DataPlotter _generate_downstream_plot to verify it handles datasets
    with empty valve_times dict.
    """
    dataset_empty_valves = {
        "name": "Empty Valves",
        "colour": "#0000FF",
        "time_data": [0, 1, 2],
        "upstream_data": {"pressure_data": [10, 20, 30], "error_data": [1, 2, 3]},
        "downstream_data": {"pressure_data": [5, 10, 15], "error_data": [0.5, 1, 1.5]},
        "valve_times": {},
    }
    plotter.datasets = {"empty": dataset_empty_valves}
    fig = plotter._generate_downstream_plot(show_valve_times=True)
    assert fig is not None


def test_no_datasets_loaded_upstream_plot(plotter):
    """
    Test DataPlotter _generate_upstream_plot to verify it handles case when
    no datasets are loaded.
    """
    plotter.datasets = {}
    fig = plotter._generate_upstream_plot(show_valve_times=True)
    assert fig is not None


def test_no_datasets_loaded_downstream_plot(plotter):
    """
    Test DataPlotter _generate_downstream_plot to verify it handles case when
    no datasets are loaded.
    """
    plotter.datasets = {}
    fig = plotter._generate_downstream_plot(show_valve_times=True)
    assert fig is not None


def test_no_datasets_loaded_upstream_plot_has_no_data(plotter):
    """
    Test DataPlotter _generate_upstream_plot to verify empty plot has no
    data traces when no datasets loaded.
    """
    plotter.datasets = {}
    fig = plotter._generate_upstream_plot(show_valve_times=True)
    assert len(fig.data) == 0


def test_no_datasets_loaded_downstream_plot_has_no_data(plotter):
    """
    Test DataPlotter _generate_downstream_plot to verify empty plot has no
    data traces when no datasets loaded.
    """
    plotter.datasets = {}
    fig = plotter._generate_downstream_plot(show_valve_times=True)
    assert len(fig.data) == 0


# =============================================================================
# Tests for Valve Event Formatting
# =============================================================================


@pytest.mark.parametrize(
    "input_name,expected_output",
    [
        ("v4_close_time", "V4 Close Time"),
        ("v5_close_time", "V5 Close Time"),
        ("v6_close_time", "V6 Close Time"),
        ("v3_open_time", "V3 Open Time"),
    ],
)
def test_valve_event_name_formatting(input_name, expected_output):
    """
    Test valve event name formatting to verify underscores are replaced
    with spaces and title case is applied.
    """
    formatted = input_name.replace("_", " ").title()
    assert formatted == expected_output


# =============================================================================
# Tests for Valve Time Relative Calculation
# =============================================================================


def test_relative_time_calculation_with_5_5_second_difference():
    """
    Test valve time relative calculation to verify 5.5 second difference
    is correctly calculated.
    """
    start_time_str = "2025-08-12 15:30:00.000"
    valve_time_str = "2025-08-12 15:30:05.500"

    start_time = datetime.strptime(start_time_str, "%Y-%m-%d %H:%M:%S.%f")
    valve_time = datetime.strptime(valve_time_str, "%Y-%m-%d %H:%M:%S.%f")

    relative_time = (valve_time - start_time).total_seconds()

    assert relative_time == 5.5


# =============================================================================
# Tests for Multiple Datasets
# =============================================================================


def test_multiple_datasets_with_different_valve_times_upstream(plotter):
    """
    Test DataPlotter _generate_upstream_plot to verify it handles multiple
    datasets with different valve configurations.
    """
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
    fig = plotter._generate_upstream_plot(show_valve_times=True)
    assert fig is not None


def test_multiple_datasets_with_different_valve_times_downstream(plotter):
    """
    Test DataPlotter _generate_downstream_plot to verify it handles multiple
    datasets with different valve configurations.
    """
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
    fig = plotter._generate_downstream_plot(show_valve_times=True)
    assert fig is not None


def test_multiple_datasets_upstream_has_multiple_data_traces(plotter):
    """
    Test DataPlotter _generate_upstream_plot to verify multiple datasets
    result in multiple data traces.
    """
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
        "valve_times": {"v4_close_time": 1.0},
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
        "valve_times": {"v4_close_time": 0.5},
    }

    plotter.datasets = {"ds1": dataset1, "ds2": dataset2}
    fig = plotter._generate_upstream_plot(show_valve_times=True)
    assert len(fig.data) >= 2


def test_multiple_datasets_downstream_has_multiple_data_traces(plotter):
    """
    Test DataPlotter _generate_downstream_plot to verify multiple datasets
    result in multiple data traces.
    """
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
        "valve_times": {"v4_close_time": 1.0},
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
        "valve_times": {"v4_close_time": 0.5},
    }

    plotter.datasets = {"ds1": dataset1, "ds2": dataset2}
    fig = plotter._generate_downstream_plot(show_valve_times=True)
    assert len(fig.data) >= 2


# =============================================================================
# Tests for Plot Generation
# =============================================================================


@pytest.fixture
def plotter_with_full_dataset(plotter):
    """Create a DataPlotter with a full dataset including valve times."""
    dataset = {
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
    plotter.datasets = {"test": dataset}
    return plotter


def test_plot_generation_upstream_with_valve_times_enabled(plotter_with_full_dataset):
    """
    Test DataPlotter _generate_upstream_plot to verify plot generates
    successfully with valve times enabled.
    """
    fig = plotter_with_full_dataset._generate_upstream_plot(
        show_error_bars=True, show_valve_times=True
    )
    assert fig is not None


def test_plot_generation_upstream_has_data_traces(plotter_with_full_dataset):
    """
    Test DataPlotter _generate_upstream_plot to verify plot contains
    data traces.
    """
    fig = plotter_with_full_dataset._generate_upstream_plot(
        show_error_bars=True, show_valve_times=True
    )
    assert len(fig.data) >= 1


def test_plot_generation_upstream_has_layout(plotter_with_full_dataset):
    """
    Test DataPlotter _generate_upstream_plot to verify plot has layout
    attribute.
    """
    fig = plotter_with_full_dataset._generate_upstream_plot(
        show_error_bars=True, show_valve_times=True
    )
    assert hasattr(fig, "layout")


def test_plot_generation_downstream_with_valve_times_enabled(plotter_with_full_dataset):
    """
    Test DataPlotter _generate_downstream_plot to verify plot generates
    successfully with valve times enabled.
    """
    fig = plotter_with_full_dataset._generate_downstream_plot(
        show_error_bars=True, show_valve_times=True
    )
    assert fig is not None


def test_plot_generation_downstream_has_data_traces(plotter_with_full_dataset):
    """
    Test DataPlotter _generate_downstream_plot to verify plot contains
    data traces.
    """
    fig = plotter_with_full_dataset._generate_downstream_plot(
        show_error_bars=True, show_valve_times=True
    )
    assert len(fig.data) >= 1


def test_plot_generation_downstream_has_layout(plotter_with_full_dataset):
    """
    Test DataPlotter _generate_downstream_plot to verify plot has layout
    attribute.
    """
    fig = plotter_with_full_dataset._generate_downstream_plot(
        show_error_bars=True, show_valve_times=True
    )
    assert hasattr(fig, "layout")


def test_plot_generation_downstream_with_valve_times_disabled(
    plotter_with_full_dataset,
):
    """
    Test DataPlotter _generate_downstream_plot to verify plot generates
    successfully with valve times disabled.
    """
    fig = plotter_with_full_dataset._generate_downstream_plot(
        show_error_bars=True, show_valve_times=False
    )
    assert fig is not None


@pytest.mark.parametrize(
    "show_error_bars,show_valve_times",
    [
        (True, True),
        (False, True),
        (True, False),
        (False, False),
    ],
)
def test_plot_generation_with_mixed_parameters_upstream(
    plotter_with_full_dataset, show_error_bars, show_valve_times
):
    """
    Test DataPlotter _generate_upstream_plot to verify it handles various
    parameter combinations.
    """
    fig = plotter_with_full_dataset._generate_upstream_plot(
        show_error_bars=show_error_bars, show_valve_times=show_valve_times
    )
    assert fig is not None
    assert len(fig.data) >= 1


@pytest.mark.parametrize(
    "show_error_bars,show_valve_times",
    [
        (True, True),
        (False, True),
        (True, False),
        (False, False),
    ],
)
def test_plot_generation_with_mixed_parameters_downstream(
    plotter_with_full_dataset, show_error_bars, show_valve_times
):
    """
    Test DataPlotter _generate_downstream_plot to verify it handles various
    parameter combinations.
    """
    fig = plotter_with_full_dataset._generate_downstream_plot(
        show_error_bars=show_error_bars, show_valve_times=show_valve_times
    )
    assert fig is not None
    assert len(fig.data) >= 1
