"""Integration tests for plot control callbacks in SHIELD DAS.

This module tests the actual Dash callback execution for plot settings,
achieving coverage of the inner callback functions.
"""

from unittest.mock import MagicMock

import pytest

from shield_das.callbacks.plot_control_callbacks import (
    register_plot_control_callbacks,
)


@pytest.fixture
def mock_plotter():
    """Create a mock plotter with app and plot generation methods."""
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

    # Mock plot generation methods
    plotter._generate_upstream_plot = MagicMock(return_value={"data": [], "layout": {}})
    plotter._generate_downstream_plot = MagicMock(
        return_value={"data": [], "layout": {}}
    )
    plotter._generate_temperature_plot = MagicMock(
        return_value={"data": [], "layout": {}}
    )

    return plotter


def test_upstream_x_scale_callback_sets_min_to_zero_for_linear(mock_plotter):
    """Test that upstream X scale callback sets min to 0 for linear scale."""
    register_plot_control_callbacks(mock_plotter)

    callback = next(
        cb
        for cb in mock_plotter.app._callbacks
        if cb["func"].__name__ == "update_upstream_x_min"
    )

    result = callback["func"]("linear")

    assert result == [0]


def test_upstream_x_scale_callback_sets_min_to_none_for_log(mock_plotter):
    """Test that upstream X scale callback sets min to None for log scale."""
    register_plot_control_callbacks(mock_plotter)

    callback = next(
        cb
        for cb in mock_plotter.app._callbacks
        if cb["func"].__name__ == "update_upstream_x_min"
    )

    result = callback["func"]("log")

    assert result == [None]


def test_upstream_y_scale_callback_sets_min_to_zero_for_linear(mock_plotter):
    """Test that upstream Y scale callback sets min to 0 for linear scale."""
    register_plot_control_callbacks(mock_plotter)

    callback = next(
        cb
        for cb in mock_plotter.app._callbacks
        if cb["func"].__name__ == "update_upstream_y_min"
    )

    result = callback["func"]("linear")

    assert result == [0]


def test_upstream_y_scale_callback_sets_min_to_none_for_log(mock_plotter):
    """Test that upstream Y scale callback sets min to None for log scale."""
    register_plot_control_callbacks(mock_plotter)

    callback = next(
        cb
        for cb in mock_plotter.app._callbacks
        if cb["func"].__name__ == "update_upstream_y_min"
    )

    result = callback["func"]("log")

    assert result == [None]


def test_downstream_x_scale_callback_execution(mock_plotter):
    """Test that downstream X scale callback executes correctly."""
    register_plot_control_callbacks(mock_plotter)

    callback = next(
        (
            cb
            for cb in mock_plotter.app._callbacks
            if cb["func"].__name__ == "update_downstream_x_min"
        ),
        None,
    )

    assert callback is not None


def test_downstream_y_scale_callback_execution(mock_plotter):
    """Test that downstream Y scale callback executes correctly."""
    register_plot_control_callbacks(mock_plotter)

    callback = next(
        (
            cb
            for cb in mock_plotter.app._callbacks
            if cb["func"].__name__ == "update_downstream_y_min"
        ),
        None,
    )

    assert callback is not None


def test_upstream_settings_callback_calls_generate_plot(mock_plotter):
    """Test that upstream settings callback calls plot generation method."""
    register_plot_control_callbacks(mock_plotter)

    callback = next(
        cb
        for cb in mock_plotter.app._callbacks
        if cb["func"].__name__ == "update_upstream_plot_settings"
    )

    # Call with minimal parameters
    callback["func"](
        "linear",  # x_scale
        "linear",  # y_scale
        0,  # x_min
        100,  # x_max
        0,  # y_min
        100,  # y_max
        True,  # show_error_bars
        False,  # show_valve_times
        {"data": []},  # current_fig
        None,  # store_data
    )

    assert mock_plotter._generate_upstream_plot.called


def test_upstream_settings_callback_passes_scale_parameters(mock_plotter):
    """Test that upstream settings callback passes scale parameters."""
    register_plot_control_callbacks(mock_plotter)

    callback = next(
        cb
        for cb in mock_plotter.app._callbacks
        if cb["func"].__name__ == "update_upstream_plot_settings"
    )

    callback["func"](
        "log",  # x_scale
        "log",  # y_scale
        None,  # x_min
        1000,  # x_max
        None,  # y_min
        1000,  # y_max
        False,  # show_error_bars
        True,  # show_valve_times
        {"data": []},  # current_fig
        None,  # store_data
    )

    call_kwargs = mock_plotter._generate_upstream_plot.call_args[1]
    assert call_kwargs["x_scale"] == "log"
    assert call_kwargs["y_scale"] == "log"


def test_upstream_settings_callback_passes_range_values(mock_plotter):
    """Test that upstream settings callback passes range values."""
    register_plot_control_callbacks(mock_plotter)

    callback = next(
        cb
        for cb in mock_plotter.app._callbacks
        if cb["func"].__name__ == "update_upstream_plot_settings"
    )

    callback["func"](
        "linear",
        "linear",
        10,  # x_min
        100,  # x_max
        20,  # y_min
        200,  # y_max
        True,
        False,
        {"data": []},
        None,
    )

    call_kwargs = mock_plotter._generate_upstream_plot.call_args[1]
    assert call_kwargs["x_min"] == 10
    assert call_kwargs["x_max"] == 100
    assert call_kwargs["y_min"] == 20
    assert call_kwargs["y_max"] == 200


def test_upstream_settings_callback_returns_tuple(mock_plotter):
    """Test that upstream settings callback returns a tuple."""
    register_plot_control_callbacks(mock_plotter)

    callback = next(
        cb
        for cb in mock_plotter.app._callbacks
        if cb["func"].__name__ == "update_upstream_plot_settings"
    )

    result = callback["func"](
        "linear", "linear", 0, 100, 0, 100, True, False, {"data": []}, None
    )

    assert isinstance(result, list)
    assert len(result) == 2


def test_downstream_settings_callback_execution(mock_plotter):
    """Test that downstream settings callback executes correctly."""
    register_plot_control_callbacks(mock_plotter)

    callback = next(
        cb
        for cb in mock_plotter.app._callbacks
        if cb["func"].__name__ == "update_downstream_plot_settings"
    )

    result = callback["func"](
        "linear", "linear", 0, 100, 0, 100, True, False, {"data": []}, None
    )

    assert mock_plotter._generate_downstream_plot.called
    assert isinstance(result, list)


def test_temperature_settings_callback_execution(mock_plotter):
    """Test that temperature settings callback executes correctly."""
    register_plot_control_callbacks(mock_plotter)

    callback = next(
        cb
        for cb in mock_plotter.app._callbacks
        if cb["func"].__name__ == "update_temperature_plot_settings"
    )

    # Temperature plot doesn't take as many parameters
    result = callback["func"](
        "linear", "linear", 0, 100, 0, 100, {"data": []}, None, None
    )

    assert mock_plotter._generate_temperature_plot.called
    assert isinstance(result, list)


def test_temperature_x_scale_callback_execution(mock_plotter):
    """Test that temperature X scale callback executes correctly."""
    register_plot_control_callbacks(mock_plotter)

    callback = next(
        (
            cb
            for cb in mock_plotter.app._callbacks
            if cb["func"].__name__ == "update_temperature_x_min"
        ),
        None,
    )

    assert callback is not None


def test_temperature_y_scale_callback_execution(mock_plotter):
    """Test that temperature Y scale callback executes correctly."""
    register_plot_control_callbacks(mock_plotter)

    callback = next(
        (
            cb
            for cb in mock_plotter.app._callbacks
            if cb["func"].__name__ == "update_temperature_y_min"
        ),
        None,
    )

    assert callback is not None


def test_permeability_settings_callback_execution(mock_plotter):
    """Test that permeability settings callback executes correctly."""
    register_plot_control_callbacks(mock_plotter)

    # Mock permeability plot generation
    mock_plotter._generate_permeability_plot = MagicMock(
        return_value={"data": [], "layout": {}}
    )

    callback = next(
        cb
        for cb in mock_plotter.app._callbacks
        if cb["func"].__name__ == "update_permeability_plot_settings"
    )

    result = callback["func"](
        "linear", "linear", 0, 100, 0, 100, {"data": []}, None, None
    )

    assert mock_plotter._generate_permeability_plot.called
    assert isinstance(result, list)


def test_permeability_x_scale_callback_execution(mock_plotter):
    """Test that permeability X scale callback executes correctly."""
    register_plot_control_callbacks(mock_plotter)

    callback = next(
        (
            cb
            for cb in mock_plotter.app._callbacks
            if cb["func"].__name__ == "update_permeability_x_min"
        ),
        None,
    )

    assert callback is not None


def test_permeability_y_scale_callback_execution(mock_plotter):
    """Test that permeability Y scale callback executes correctly."""
    register_plot_control_callbacks(mock_plotter)

    callback = next(
        (
            cb
            for cb in mock_plotter.app._callbacks
            if cb["func"].__name__ == "update_permeability_y_min"
        ),
        None,
    )

    assert callback is not None


@pytest.mark.parametrize(
    "callback_name,scale,expected_min",
    [
        ("update_upstream_x_min", "linear", [0]),
        ("update_upstream_x_min", "log", [None]),
        ("update_upstream_y_min", "linear", [0]),
        ("update_upstream_y_min", "log", [None]),
        ("update_downstream_x_min", "linear", [0]),
        ("update_downstream_x_min", "log", [None]),
        ("update_downstream_y_min", "linear", [0]),
        ("update_downstream_y_min", "log", [None]),
        ("update_temperature_x_min", "linear", [0]),
        ("update_temperature_x_min", "log", [None]),
        ("update_temperature_y_min", "linear", [0]),
        ("update_temperature_y_min", "log", [None]),
        ("update_permeability_x_min", "linear", [0]),
        ("update_permeability_x_min", "log", [None]),
        ("update_permeability_y_min", "linear", [0]),
        ("update_permeability_y_min", "log", [None]),
    ],
)
def test_all_scale_callbacks_min_value_behavior(
    mock_plotter, callback_name, scale, expected_min
):
    """Test min value behavior for all scale callbacks.

    Args:
        mock_plotter: Mock plotter fixture
        callback_name: Name of the callback to test
        scale: Scale mode to test (linear or log)
        expected_min: Expected minimum value
    """
    register_plot_control_callbacks(mock_plotter)

    callback = next(
        cb for cb in mock_plotter.app._callbacks if cb["func"].__name__ == callback_name
    )

    result = callback["func"](scale)

    assert result == expected_min
