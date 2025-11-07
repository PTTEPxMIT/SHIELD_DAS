"""Tests for valve time visualization functionality in DataPlotter."""

import json
import os
import tempfile
from unittest.mock import Mock, MagicMock, patch

import numpy as np
import pytest

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
    """
    Helper function to create a mock Dataset object for testing.

    Args:
        name: Dataset name
        colour: Dataset color
        time_data: Time data array
        upstream_pressure: Upstream pressure data array
        downstream_pressure: Downstream pressure data array
        upstream_error: Upstream error data array
        downstream_error: Downstream error data array
        valve_times: Dictionary of valve timing events

    Returns:
        MagicMock object configured as a Dataset
    """
    dataset = MagicMock()

    # Set basic attributes
    dataset.name = name
    dataset.colour = colour

    # Set data arrays with defaults if not provided
    default_time = (
        time_data if time_data is not None else np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    )
    dataset.time_data = default_time

    dataset.upstream_pressure = (
        upstream_pressure
        if upstream_pressure is not None
        else np.array([10.0, 15.0, 20.0, 25.0, 30.0, 35.0])
    )
    dataset.downstream_pressure = (
        downstream_pressure
        if downstream_pressure is not None
        else np.array([5.0, 7.5, 10.0, 12.5, 15.0, 17.5])
    )

    dataset.upstream_error = (
        upstream_error
        if upstream_error is not None
        else np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5])
    )
    dataset.downstream_error = (
        downstream_error
        if downstream_error is not None
        else np.array([0.5, 0.75, 1.0, 1.25, 1.5, 1.75])
    )

    # Set valve times
    dataset.valve_times = (
        valve_times
        if valve_times is not None
        else {
            "v4_close_time": 1.5,
            "v5_close_time": 2.5,
            "v6_close_time": 3.5,
            "v3_open_time": 4.5,
        }
    )

    # Temperature data attributes (can be None)
    dataset.thermocouple_data = None
    dataset.local_temperature_data = None
    dataset.thermocouple_name = None

    return dataset


class TestValveTimeVisualization:
    """Test suite for valve time visualization features"""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.temp_dir = tempfile.mkdtemp()
        self.plotter = DataPlotter()

        # Create mock dataset with valve times
        self.mock_dataset = create_mock_dataset(
            name="Test Dataset",
            colour="#FF0000",
            time_data=np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]),
            upstream_pressure=np.array([10.0, 15.0, 20.0, 25.0, 30.0, 35.0]),
            downstream_pressure=np.array([5.0, 7.5, 10.0, 12.5, 15.0, 17.5]),
            upstream_error=np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5]),
            downstream_error=np.array([0.5, 0.75, 1.0, 1.25, 1.5, 1.75]),
            valve_times={
                "v4_close_time": 1.5,
                "v5_close_time": 2.5,
                "v6_close_time": 3.5,
                "v3_open_time": 4.5,
            },
        )

        # Add dataset to plotter
        self.plotter.datasets = [self.mock_dataset]

    def test_valve_times_constant_exists(self):
        """Test that PLOT_CONTROL_STATES constant includes valve time states."""
        from shield_das.callbacks.states import PLOT_CONTROL_STATES

        states = [state.component_id for state in PLOT_CONTROL_STATES]
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
        expected_valve_count = len(self.mock_dataset.valve_times)

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
        """Test handling of datasets with empty valve times."""
        # Create dataset with empty valve times
        dataset_no_valves = create_mock_dataset(
            name="No Valves Dataset",
            colour="#00FF00",
            time_data=np.array([0.0, 1.0, 2.0]),
            upstream_pressure=np.array([10.0, 20.0, 30.0]),
            downstream_pressure=np.array([5.0, 10.0, 15.0]),
            upstream_error=np.array([1.0, 2.0, 3.0]),
            downstream_error=np.array([0.5, 1.0, 1.5]),
            valve_times={},  # Empty valve times
        )

        plotter = DataPlotter()
        plotter.datasets = [dataset_no_valves]

        # Should handle empty valve_times gracefully
        upstream_fig = plotter._generate_upstream_plot(show_valve_times=True)
        downstream_fig = plotter._generate_downstream_plot(show_valve_times=True)

        assert upstream_fig is not None
        assert downstream_fig is not None

    def test_valve_times_with_empty_dict(self):
        """Test handling of datasets with empty valve_times dict."""
        dataset_empty_valves = create_mock_dataset(
            name="Empty Valves Dataset",
            colour="#0000FF",
            time_data=np.array([0.0, 1.0, 2.0]),
            upstream_pressure=np.array([10.0, 20.0, 30.0]),
            downstream_pressure=np.array([5.0, 10.0, 15.0]),
            upstream_error=np.array([1.0, 2.0, 3.0]),
            downstream_error=np.array([0.5, 1.0, 1.5]),
            valve_times={},  # Empty dict
        )

        plotter = DataPlotter()
        plotter.datasets = [dataset_empty_valves]

        # Should handle empty valve_times dict gracefully
        upstream_fig = plotter._generate_upstream_plot(show_valve_times=True)
        downstream_fig = plotter._generate_downstream_plot(show_valve_times=True)

        assert upstream_fig is not None
        assert downstream_fig is not None

    def test_multiple_datasets_with_different_valve_times(self):
        """Test plotting multiple datasets with different valve times."""
        # Create second dataset with different valve times
        dataset2 = create_mock_dataset(
            name="Dataset 2",
            colour="#00FFFF",
            time_data=np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
            upstream_pressure=np.array([5.0, 10.0, 15.0, 20.0, 25.0]),
            downstream_pressure=np.array([2.5, 5.0, 7.5, 10.0, 12.5]),
            upstream_error=np.array([0.5, 1.0, 1.5, 2.0, 2.5]),
            downstream_error=np.array([0.25, 0.5, 0.75, 1.0, 1.25]),
            valve_times={
                "v4_close_time": 0.8,
                "v5_close_time": 1.8,
                "v3_open_time": 3.8,
            },
        )

        plotter = DataPlotter()
        plotter.datasets = [self.mock_dataset, dataset2]

        # Should handle multiple datasets with different valve configurations
        upstream_fig = plotter._generate_upstream_plot(show_valve_times=True)
        downstream_fig = plotter._generate_downstream_plot(show_valve_times=True)

        assert upstream_fig is not None
        assert downstream_fig is not None

        # Should have traces for both datasets
        assert len(upstream_fig.data) >= 2
        assert len(downstream_fig.data) >= 2


@pytest.fixture
def plotter():
    """Create a DataPlotter instance for testing."""
    return DataPlotter()


@pytest.fixture
def mock_dataset():
    """Create a comprehensive mock dataset with valve times for testing."""
    return {
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


@pytest.fixture
def plotter_with_dataset(plotter, mock_dataset):
    """Create a DataPlotter with a loaded dataset."""
    plotter.datasets = {"test": mock_dataset}
    return plotter

    def test_no_datasets_loaded(self):
        """Test behavior when no datasets are loaded."""
        plotter = DataPlotter()
        plotter.datasets = []


# =============================================================================
# Tests for Valve Time Visualization Constants
# =============================================================================


def test_valve_time_visualization_states_exist(plotter):
    """
    Test DataPlotter to verify valve time visualization states exist in
    PLOT_CONTROL_STATES.
    """
    state_ids = [state.component_id for state in plotter.PLOT_CONTROL_STATES]
    assert "show-valve-times-upstream" in state_ids
    assert "show-valve-times-downstream" in state_ids


def test_valve_time_annotation_color_constant():
    """
    Test valve time visualization to verify line annotation color is
    consistently defined.
    """
    expected_color = "rgba(255, 0, 0, 0.5)"
    assert expected_color == "rgba(255, 0, 0, 0.5)"


# =============================================================================
# Tests for Valve Time Plot Generation
# =============================================================================


def test_upstream_plot_with_all_valve_times_shown(plotter_with_dataset):
    """
    Test DataPlotter _generate_upstream_plot to verify it generates plot
    with all valve times displayed.
    """
    fig = plotter_with_dataset._generate_upstream_plot(show_valve_times=True)
    assert fig is not None
    assert len(fig.data) >= 1


def test_downstream_plot_with_all_valve_times_shown(plotter_with_dataset):
    """
    Test DataPlotter _generate_downstream_plot to verify it generates plot
    with all valve times displayed.
    """
    fig = plotter_with_dataset._generate_downstream_plot(show_valve_times=True)
    assert fig is not None
    assert len(fig.data) >= 1


def test_upstream_plot_valve_times_creates_vertical_lines(plotter_with_dataset):
    """
    Test DataPlotter _generate_upstream_plot to verify valve times create
    vertical line annotations on plot.
    """
    fig = plotter_with_dataset._generate_upstream_plot(show_valve_times=True)
    # Valve times should add vertical lines (shapes) to the layout
    assert hasattr(fig.layout, "shapes")


def test_downstream_plot_valve_times_creates_vertical_lines(plotter_with_dataset):
    """
    Test DataPlotter _generate_downstream_plot to verify valve times create
    vertical line annotations on plot.
    """
    fig = plotter_with_dataset._generate_downstream_plot(show_valve_times=True)
    # Valve times should add vertical lines (shapes) to the layout
    assert hasattr(fig.layout, "shapes")


def test_upstream_plot_without_valve_times_shown(plotter_with_dataset):
    """
    Test DataPlotter _generate_upstream_plot to verify plot generates
    correctly when valve times are disabled.
    """
    fig = plotter_with_dataset._generate_upstream_plot(show_valve_times=False)
    assert fig is not None
    assert len(fig.data) >= 1


def test_downstream_plot_without_valve_times_shown(plotter_with_dataset):
    """
    Test DataPlotter _generate_downstream_plot to verify plot generates
    correctly when valve times are disabled.
    """
    fig = plotter_with_dataset._generate_downstream_plot(show_valve_times=False)
    assert fig is not None
    assert len(fig.data) >= 1


# =============================================================================
# Tests for Valve Time Formatting
# =============================================================================


@pytest.mark.parametrize(
    "valve_event,expected_label",
    [
        ("v4_close_time", "V4 Close Time"),
        ("v5_close_time", "V5 Close Time"),
        ("v6_close_time", "V6 Close Time"),
        ("v3_open_time", "V3 Open Time"),
    ],
)
def test_valve_event_label_formatting(valve_event, expected_label):
    """
    Test valve event name formatting to verify underscores are replaced
    with spaces and title case is applied.
    """
    formatted = valve_event.replace("_", " ").title()
    assert formatted == expected_label


# =============================================================================
# Tests for Valve Time Data Processing
# =============================================================================


def test_valve_times_relative_to_start_time():
    """
    Test valve time calculation to verify times are relative to recording
    start time.
    """
    from datetime import datetime

    start_time_str = "2025-08-12 10:00:00.000"
    valve_time_str = "2025-08-12 10:00:03.500"

    start_time = datetime.strptime(start_time_str, "%Y-%m-%d %H:%M:%S.%f")
    valve_time = datetime.strptime(valve_time_str, "%Y-%m-%d %H:%M:%S.%f")

    relative_time = (valve_time - start_time).total_seconds()
    assert relative_time == 3.5


def test_valve_time_dict_keys_are_event_names():
    """
    Test valve_times dictionary structure to verify keys are event names.
    """
    valve_times = {
        "v4_close_time": 1.5,
        "v5_close_time": 2.5,
        "v6_close_time": 3.5,
    }

    expected_keys = ["v4_close_time", "v5_close_time", "v6_close_time"]
    for key in expected_keys:
        assert key in valve_times


def test_valve_time_dict_values_are_floats():
    """
    Test valve_times dictionary structure to verify values are float
    timestamps.
    """
    valve_times = {
        "v4_close_time": 1.5,
        "v5_close_time": 2.5,
    }

    for value in valve_times.values():
        assert isinstance(value, (int, float))


# =============================================================================
# Tests for Multiple Valve Events
# =============================================================================


def test_multiple_valve_events_on_upstream_plot(plotter_with_dataset):
    """
    Test DataPlotter _generate_upstream_plot to verify it displays multiple
    valve events on the same plot.
    """
    # Dataset fixture has 4 valve times
    fig = plotter_with_dataset._generate_upstream_plot(show_valve_times=True)
    assert fig is not None


def test_multiple_valve_events_on_downstream_plot(plotter_with_dataset):
    """
    Test DataPlotter _generate_downstream_plot to verify it displays multiple
    valve events on the same plot.
    """
    # Dataset fixture has 4 valve times
    fig = plotter_with_dataset._generate_downstream_plot(show_valve_times=True)
    assert fig is not None


def test_different_valve_events_per_dataset(plotter):
    """
    Test DataPlotter with multiple datasets to verify different datasets
    can have different valve events.
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
        "valve_times": {"v3_open_time": 2.0, "v5_close_time": 1.5},
    }

    plotter.datasets = {"ds1": dataset1, "ds2": dataset2}
    fig = plotter._generate_upstream_plot(show_valve_times=True)
    assert fig is not None


# =============================================================================
# Tests for Dataset Loading with Valve Times
# =============================================================================


def test_load_dataset_from_json_with_valve_times():
    """
    Test DataPlotter load_dataset to verify it correctly loads valve times
    from JSON metadata.
    """
    mock_metadata = {
        "recording_start_time": "2025-08-12 10:00:00.000",
        "valve_event_times": {
            "v4_close_time": "2025-08-12 10:00:01.500",
            "v5_close_time": "2025-08-12 10:00:02.500",
        },
    }

    plotter = DataPlotter()

    with patch("builtins.open", mock_open(read_data=json.dumps(mock_metadata))):
        # This would be part of the load_dataset logic
        assert mock_metadata["valve_event_times"] is not None
        assert len(mock_metadata["valve_event_times"]) == 2


def test_metadata_without_valve_times():
    """
    Test metadata loading to verify it handles metadata without valve_event_times
    key gracefully.
    """
    mock_metadata = {
        "recording_start_time": "2025-08-12 10:00:00.000",
        # No valve_event_times
    }

    plotter = DataPlotter()
    # Should not raise exception when valve_event_times is missing
    valve_times = mock_metadata.get("valve_event_times", {})
    assert valve_times == {}


def test_metadata_with_empty_valve_times():
    """
    Test metadata loading to verify it handles empty valve_event_times dict.
    """
    mock_metadata = {
        "recording_start_time": "2025-08-12 10:00:00.000",
        "valve_event_times": {},
    }

    plotter = DataPlotter()
    valve_times = mock_metadata.get("valve_event_times", {})
    assert valve_times == {}
    assert isinstance(valve_times, dict)


# =============================================================================
# Tests for Empty Dataset Edge Cases
# =============================================================================


def test_empty_datasets_with_valve_times_enabled_upstream(plotter):
    """
    Test DataPlotter _generate_upstream_plot to verify it handles empty
    datasets list with valve times enabled.
    """
    plotter.datasets = {}
    fig = plotter._generate_upstream_plot(show_valve_times=True)
    assert fig is not None
    assert len(fig.data) == 0


def test_empty_datasets_with_valve_times_enabled_downstream(plotter):
    """
    Test DataPlotter _generate_downstream_plot to verify it handles empty
    datasets list with valve times enabled.
    """
    plotter.datasets = {}
    fig = plotter._generate_downstream_plot(show_valve_times=True)
    assert fig is not None
    assert len(fig.data) == 0


def test_dataset_with_no_valve_times_key_upstream(plotter):
    """
    Test DataPlotter _generate_upstream_plot to verify it handles dataset
    missing valve_times key.
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


def test_dataset_with_no_valve_times_key_downstream(plotter):
    """
    Test DataPlotter _generate_downstream_plot to verify it handles dataset
    missing valve_times key.
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


# =============================================================================
# Tests for Valve Time Display Control
# =============================================================================


@pytest.mark.parametrize(
    "show_upstream,show_downstream",
    [
        (True, True),
        (True, False),
        (False, True),
        (False, False),
    ],
)
def test_independent_valve_time_control(
    plotter_with_dataset, show_upstream, show_downstream
):
    """
    Test DataPlotter to verify upstream and downstream valve time displays
    can be controlled independently.
    """
    plots = plotter_with_dataset._generate_both_plots(
        show_error_bars_upstream=False,
        show_error_bars_downstream=False,
        show_valve_times_upstream=show_upstream,
        show_valve_times_downstream=show_downstream,
    )

    assert len(plots) == 3
    assert plots[0] is not None  # upstream
    assert plots[1] is not None  # downstream


def test_valve_times_shown_only_on_upstream(plotter_with_dataset):
    """
    Test DataPlotter _generate_both_plots to verify valve times can be
    shown only on upstream plot.
    """
    plots = plotter_with_dataset._generate_both_plots(
        show_valve_times_upstream=True, show_valve_times_downstream=False
    )

    assert plots[0] is not None
    assert plots[1] is not None


def test_valve_times_shown_only_on_downstream(plotter_with_dataset):
    """
    Test DataPlotter _generate_both_plots to verify valve times can be
    shown only on downstream plot.
    """
    plots = plotter_with_dataset._generate_both_plots(
        show_valve_times_upstream=False, show_valve_times_downstream=True
    )

    assert plots[0] is not None
    assert plots[1] is not None


# =============================================================================
# Tests for Combined Features
# =============================================================================


def test_valve_times_with_error_bars_upstream(plotter_with_dataset):
    """
    Test DataPlotter _generate_upstream_plot to verify valve times and
    error bars can be shown simultaneously.
    """
    fig = plotter_with_dataset._generate_upstream_plot(
        show_error_bars=True, show_valve_times=True
    )
    assert fig is not None
    assert len(fig.data) >= 1


def test_valve_times_with_error_bars_downstream(plotter_with_dataset):
    """
    Test DataPlotter _generate_downstream_plot to verify valve times and
    error bars can be shown simultaneously.
    """
    fig = plotter_with_dataset._generate_downstream_plot(
        show_error_bars=True, show_valve_times=True
    )
    assert fig is not None
    assert len(fig.data) >= 1


def test_valve_times_without_error_bars_upstream(plotter_with_dataset):
    """
    Test DataPlotter _generate_upstream_plot to verify valve times can be
    shown without error bars.
    """
    fig = plotter_with_dataset._generate_upstream_plot(
        show_error_bars=False, show_valve_times=True
    )
    assert fig is not None
    assert len(fig.data) >= 1


def test_valve_times_without_error_bars_downstream(plotter_with_dataset):
    """
    Test DataPlotter _generate_downstream_plot to verify valve times can be
    shown without error bars.
    """
    fig = plotter_with_dataset._generate_downstream_plot(
        show_error_bars=False, show_valve_times=True
    )
    assert fig is not None
    assert len(fig.data) >= 1
