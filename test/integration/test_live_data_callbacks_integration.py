"""Integration tests for live data callbacks in SHIELD DAS.

This module tests the actual Dash callback execution for live data monitoring,
achieving coverage of the inner callback functions.
"""

from unittest.mock import MagicMock

import pytest

from shield_das.callbacks.live_data_callbacks import (
    register_live_data_callbacks,
)


@pytest.fixture
def mock_plotter():
    """Create a mock plotter with app and datasets for testing."""
    plotter = MagicMock()
    plotter.app = MagicMock()

    # Store callbacks for later execution
    plotter.app._callbacks = []

    def mock_callback(*args, **kwargs):
        """Mock callback decorator that stores callback info."""

        def decorator(func):
            plotter.app._callbacks.append(
                {
                    "func": func,
                    "outputs": args[0] if args else kwargs.get("output"),
                    "inputs": args[1] if len(args) > 1 else kwargs.get("input"),
                    "states": args[2] if len(args) > 2 else kwargs.get("state"),
                }
            )
            return func

        return decorator

    plotter.app.callback = mock_callback

    # Mock datasets
    plotter.datasets = []

    # Mock plot generation methods
    plotter._generate_both_plots = MagicMock(
        return_value=[
            {"data": [], "layout": {}},
            {"data": [], "layout": {}},
            {"data": [], "layout": {}},
        ]
    )
    plotter._generate_upstream_plot = MagicMock(return_value={"data": [], "layout": {}})
    plotter._generate_downstream_plot = MagicMock(
        return_value={"data": [], "layout": {}}
    )
    plotter._generate_temperature_plot = MagicMock(
        return_value={"data": [], "layout": {}}
    )

    return plotter


def test_handle_live_data_toggle_callback_execution(mock_plotter):
    """Test that handle_live_data_toggle callback executes correctly."""
    register_live_data_callbacks(mock_plotter)

    callback = next(
        (
            cb
            for cb in mock_plotter.app._callbacks
            if cb["func"].__name__ == "handle_live_data_toggle"
        ),
        None,
    )

    assert callback is not None


def test_handle_live_data_toggle_disables_interval_when_no_live(mock_plotter):
    """Test that interval is disabled when no datasets are live."""
    register_live_data_callbacks(mock_plotter)

    # Create mock datasets
    mock_dataset1 = MagicMock()
    mock_dataset2 = MagicMock()
    mock_plotter.datasets = [mock_dataset1, mock_dataset2]

    callback = next(
        cb
        for cb in mock_plotter.app._callbacks
        if cb["func"].__name__ == "handle_live_data_toggle"
    )

    # All checkboxes unchecked (no live data)
    result = callback["func"](
        [False, False],  # live_data_values
        True,  # show_error_bars_upstream
        True,  # show_error_bars_downstream
        False,  # show_valve_times_upstream
        False,  # show_valve_times_downstream
    )

    # Should return 4 items: 3 plots + interval_disabled=True
    assert len(result) == 4
    assert result[3] is True  # interval_disabled


def test_handle_live_data_toggle_enables_interval_when_live(mock_plotter):
    """Test that interval is enabled when at least one dataset is live."""
    register_live_data_callbacks(mock_plotter)

    mock_dataset1 = MagicMock()
    mock_dataset2 = MagicMock()
    mock_plotter.datasets = [mock_dataset1, mock_dataset2]

    callback = next(
        cb
        for cb in mock_plotter.app._callbacks
        if cb["func"].__name__ == "handle_live_data_toggle"
    )

    # One checkbox checked (has live data)
    result = callback["func"](
        [True, False],  # live_data_values
        True,
        True,
        False,
        False,
    )

    assert len(result) == 4
    assert result[3] is False  # interval_disabled (enabled)


def test_handle_live_data_toggle_updates_dataset_flags(mock_plotter):
    """Test that live_data flags are updated on datasets."""
    register_live_data_callbacks(mock_plotter)

    mock_dataset1 = MagicMock()
    mock_dataset2 = MagicMock()
    mock_plotter.datasets = [mock_dataset1, mock_dataset2]

    callback = next(
        cb
        for cb in mock_plotter.app._callbacks
        if cb["func"].__name__ == "handle_live_data_toggle"
    )

    callback["func"]([True, False], True, True, False, False)

    assert mock_dataset1.live_data is True
    assert mock_dataset2.live_data is False


def test_handle_live_data_toggle_calls_generate_plots(mock_plotter):
    """Test that plots are regenerated when toggling live data."""
    register_live_data_callbacks(mock_plotter)

    mock_dataset1 = MagicMock()
    mock_plotter.datasets = [mock_dataset1]

    callback = next(
        cb
        for cb in mock_plotter.app._callbacks
        if cb["func"].__name__ == "handle_live_data_toggle"
    )

    callback["func"]([True], True, False, True, False)

    assert mock_plotter._generate_upstream_plot.called
    assert mock_plotter._generate_downstream_plot.called
    assert mock_plotter._generate_temperature_plot.called


def test_handle_live_data_toggle_passes_plot_settings(mock_plotter):
    """Test that plot settings are passed to individual plot generation methods."""
    register_live_data_callbacks(mock_plotter)

    mock_dataset1 = MagicMock()
    mock_plotter.datasets = [mock_dataset1]

    callback = next(
        cb
        for cb in mock_plotter.app._callbacks
        if cb["func"].__name__ == "handle_live_data_toggle"
    )

    callback["func"](
        [True],
        True,  # show_error_bars_upstream
        False,  # show_error_bars_downstream
        True,  # show_valve_times_upstream
        False,  # show_valve_times_downstream
    )

    # Check upstream plot settings
    upstream_kwargs = mock_plotter._generate_upstream_plot.call_args[1]
    assert upstream_kwargs["show_error_bars"] is True
    assert upstream_kwargs["show_valve_times"] is True

    # Check downstream plot settings
    downstream_kwargs = mock_plotter._generate_downstream_plot.call_args[1]
    assert downstream_kwargs["show_error_bars"] is False
    assert downstream_kwargs["show_valve_times"] is False


def test_update_live_data_callback_execution(mock_plotter):
    """Test that update_live_data callback executes correctly."""
    register_live_data_callbacks(mock_plotter)

    callback = next(
        (
            cb
            for cb in mock_plotter.app._callbacks
            if cb["func"].__name__ == "update_live_data"
        ),
        None,
    )

    assert callback is not None


def test_update_live_data_raises_prevent_update_when_no_live(mock_plotter):
    """Test that PreventUpdate is raised when no datasets are live."""
    from dash.exceptions import PreventUpdate

    register_live_data_callbacks(mock_plotter)

    mock_dataset1 = MagicMock()
    mock_dataset1.live_data = False
    mock_plotter.datasets = [mock_dataset1]

    callback = next(
        cb
        for cb in mock_plotter.app._callbacks
        if cb["func"].__name__ == "update_live_data"
    )

    with pytest.raises(PreventUpdate):
        callback["func"](
            1,  # n_intervals
            True,
            True,
            False,
            False,
            "linear",
            "linear",
            0,
            100,
            0,
            100,
            "linear",
            "linear",
            0,
            100,
            0,
            100,
        )


def test_update_live_data_processes_live_datasets(mock_plotter):
    """Test that live datasets are reprocessed during update."""
    register_live_data_callbacks(mock_plotter)

    mock_dataset1 = MagicMock()
    mock_dataset1.live_data = True
    mock_plotter.datasets = [mock_dataset1]

    callback = next(
        cb
        for cb in mock_plotter.app._callbacks
        if cb["func"].__name__ == "update_live_data"
    )

    callback["func"](
        1,
        True,
        True,
        False,
        False,
        "linear",
        "linear",
        0,
        100,
        0,
        100,
        "linear",
        "linear",
        0,
        100,
        0,
        100,
    )

    # Verify process_data was called on live dataset
    assert mock_dataset1.process_data.called


def test_update_live_data_generates_all_plots(mock_plotter):
    """Test that all three plots are generated during update."""
    register_live_data_callbacks(mock_plotter)

    mock_dataset1 = MagicMock()
    mock_dataset1.live_data = True
    mock_plotter.datasets = [mock_dataset1]

    callback = next(
        cb
        for cb in mock_plotter.app._callbacks
        if cb["func"].__name__ == "update_live_data"
    )

    result = callback["func"](
        1,
        True,
        True,
        False,
        False,
        "linear",
        "linear",
        0,
        100,
        0,
        100,
        "linear",
        "linear",
        0,
        100,
        0,
        100,
    )

    assert mock_plotter._generate_upstream_plot.called
    assert mock_plotter._generate_downstream_plot.called
    assert mock_plotter._generate_temperature_plot.called
    assert len(result) == 3


def test_update_live_data_preserves_plot_settings(mock_plotter):
    """Test that plot settings are preserved during live update."""
    register_live_data_callbacks(mock_plotter)

    mock_dataset1 = MagicMock()
    mock_dataset1.live_data = True
    mock_plotter.datasets = [mock_dataset1]

    callback = next(
        cb
        for cb in mock_plotter.app._callbacks
        if cb["func"].__name__ == "update_live_data"
    )

    callback["func"](
        1,
        True,  # show_error_bars_upstream
        False,  # show_error_bars_downstream
        True,  # show_valve_times_upstream
        False,  # show_valve_times_downstream
        "log",  # upstream_x_scale
        "linear",  # upstream_y_scale
        1,  # upstream_x_min
        1000,  # upstream_x_max
        0,  # upstream_y_min
        100,  # upstream_y_max
        "linear",  # downstream_x_scale
        "log",  # downstream_y_scale
        0,  # downstream_x_min
        500,  # downstream_x_max
        1,  # downstream_y_min
        200,  # downstream_y_max
    )

    # Check upstream plot settings
    upstream_kwargs = mock_plotter._generate_upstream_plot.call_args[1]
    assert upstream_kwargs["show_error_bars"] is True
    assert upstream_kwargs["show_valve_times"] is True
    assert upstream_kwargs["x_scale"] == "log"
    assert upstream_kwargs["y_scale"] == "linear"
    assert upstream_kwargs["x_min"] == 1
    assert upstream_kwargs["x_max"] == 1000
    assert upstream_kwargs["y_min"] == 0
    assert upstream_kwargs["y_max"] == 100

    # Check downstream plot settings
    downstream_kwargs = mock_plotter._generate_downstream_plot.call_args[1]
    assert downstream_kwargs["show_error_bars"] is False
    assert downstream_kwargs["show_valve_times"] is False
    assert downstream_kwargs["x_scale"] == "linear"
    assert downstream_kwargs["y_scale"] == "log"
    assert downstream_kwargs["x_min"] == 0
    assert downstream_kwargs["x_max"] == 500
    assert downstream_kwargs["y_min"] == 1
    assert downstream_kwargs["y_max"] == 200


@pytest.mark.parametrize(
    "live_values,expected_disabled",
    [
        ([False, False, False], True),
        ([True, False, False], False),
        ([False, True, False], False),
        ([True, True, True], False),
    ],
)
def test_handle_live_data_toggle_interval_state(
    mock_plotter, live_values, expected_disabled
):
    """Test interval disabled state for various live data combinations.

    Args:
        mock_plotter: Mock plotter fixture
        live_values: List of live data checkbox values
        expected_disabled: Expected interval disabled state
    """
    register_live_data_callbacks(mock_plotter)

    # Create datasets matching the number of values
    mock_plotter.datasets = [MagicMock() for _ in live_values]

    callback = next(
        cb
        for cb in mock_plotter.app._callbacks
        if cb["func"].__name__ == "handle_live_data_toggle"
    )

    result = callback["func"](live_values, True, True, False, False)

    assert result[3] == expected_disabled
