"""Tests for plot control callbacks in SHIELD Data Acquisition System.

This module tests the plot control functionality including scale modes,
axis ranges, error bars, and valve time markers.
"""

from unittest.mock import MagicMock

import pytest

from shield_das.callbacks.plot_control_callbacks import (
    _create_min_value_callback,
    _create_plot_settings_callback,
    _normalize_range_values,
    register_plot_control_callbacks,
)


# =============================================================================
# Tests for _normalize_range_values
# =============================================================================


def test_normalize_range_values_preserves_numeric_values():
    """Test that _normalize_range_values preserves numeric input values."""
    result = _normalize_range_values(0, 100, 10, 50)
    assert result == [0, 100, 10, 50]


def test_normalize_range_values_preserves_none():
    """Test that _normalize_range_values preserves None values."""
    result = _normalize_range_values(None, 100, None, 50)
    assert result == [None, 100, None, 50]


def test_normalize_range_values_handles_all_none():
    """Test that _normalize_range_values handles all None inputs."""
    result = _normalize_range_values(None, None, None, None)
    assert result == [None, None, None, None]


def test_normalize_range_values_handles_empty_input():
    """Test that _normalize_range_values handles empty input."""
    result = _normalize_range_values()
    assert result == []


def test_normalize_range_values_handles_float_values():
    """Test that _normalize_range_values handles float values."""
    result = _normalize_range_values(0.5, 99.9, 1.23, 45.67)
    assert result == [0.5, 99.9, 1.23, 45.67]


def test_normalize_range_values_handles_negative_values():
    """Test that _normalize_range_values handles negative values."""
    result = _normalize_range_values(-10, 10, -5, 5)
    assert result == [-10, 10, -5, 5]


def test_normalize_range_values_handles_zero():
    """Test that _normalize_range_values correctly handles zero."""
    result = _normalize_range_values(0, 0, 0, 0)
    assert result == [0, 0, 0, 0]


@pytest.mark.parametrize(
    "values,expected",
    [
        ((1, 2, 3, 4), [1, 2, 3, 4]),
        ((None, None, None, None), [None, None, None, None]),
        ((0, None, 100, None), [0, None, 100, None]),
        ((None, 10, None, 20), [None, 10, None, 20]),
        ((-5, 5, -10, 10), [-5, 5, -10, 10]),
    ],
)
def test_normalize_range_values_parameterized(values, expected):
    """Test _normalize_range_values with various input combinations.

    Args:
        values: Tuple of input values
        expected: Expected output list
    """
    result = _normalize_range_values(*values)
    assert result == expected


# =============================================================================
# Tests for _create_min_value_callback
# =============================================================================


def test_create_min_value_callback_returns_zero_for_linear():
    """Test that min value callback returns [0] for linear scale."""
    callback_func = _create_min_value_callback("x")
    result = callback_func("linear")
    assert result == [0]


def test_create_min_value_callback_returns_none_for_log():
    """Test that min value callback returns [None] for log scale."""
    callback_func = _create_min_value_callback("x")
    result = callback_func("log")
    assert result == [None]


def test_create_min_value_callback_returns_list():
    """Test that min value callback returns a list."""
    callback_func = _create_min_value_callback("y")
    result = callback_func("linear")
    assert isinstance(result, list)


def test_create_min_value_callback_list_has_one_element():
    """Test that min value callback returns list with exactly one element."""
    callback_func = _create_min_value_callback("x")
    result = callback_func("linear")
    assert len(result) == 1


def test_create_min_value_callback_works_for_x_axis():
    """Test that min value callback works for x-axis scale type."""
    callback_func = _create_min_value_callback("x")
    result = callback_func("linear")
    assert result == [0]


def test_create_min_value_callback_works_for_y_axis():
    """Test that min value callback works for y-axis scale type."""
    callback_func = _create_min_value_callback("y")
    result = callback_func("log")
    assert result == [None]


@pytest.mark.parametrize(
    "scale_mode,expected",
    [
        ("linear", [0]),
        ("log", [None]),
    ],
)
def test_create_min_value_callback_scale_modes(scale_mode, expected):
    """Test min value callback for different scale modes.

    Args:
        scale_mode: Scale mode ("linear" or "log")
        expected: Expected return value
    """
    callback_func = _create_min_value_callback("x")
    result = callback_func(scale_mode)
    assert result == expected


@pytest.mark.parametrize(
    "axis_type",
    ["x", "y", "z", "custom"],
)
def test_create_min_value_callback_axis_types(axis_type):
    """Test min value callback creation for various axis types.

    Args:
        axis_type: Axis type identifier
    """
    callback_func = _create_min_value_callback(axis_type)
    result = callback_func("linear")
    assert result == [0]


# =============================================================================
# Tests for _create_plot_settings_callback
# =============================================================================


def test_create_plot_settings_callback_calls_generate_method():
    """Test that plot settings callback calls the generate method."""
    mock_generate = MagicMock(return_value="mock_figure")
    callback_func = _create_plot_settings_callback("upstream", mock_generate)

    callback_func(
        x_scale="linear",
        y_scale="log",
        x_min=0,
        x_max=100,
        y_min=1,
        y_max=1000,
        show_error_bars=True,
        show_valve_times=False,
        current_fig=None,
        store_data=None,
    )

    assert mock_generate.called


def test_create_plot_settings_callback_returns_tuple():
    """Test that plot settings callback returns a tuple."""
    mock_generate = MagicMock(return_value="mock_figure")
    callback_func = _create_plot_settings_callback("upstream", mock_generate)

    result = callback_func(
        x_scale="linear",
        y_scale="log",
        x_min=0,
        x_max=100,
        y_min=1,
        y_max=1000,
        show_error_bars=True,
        show_valve_times=False,
        current_fig=None,
        store_data=None,
    )

    assert isinstance(result, list)


def test_create_plot_settings_callback_tuple_has_two_elements():
    """Test that plot settings callback returns list with two elements."""
    mock_generate = MagicMock(return_value="mock_figure")
    callback_func = _create_plot_settings_callback("upstream", mock_generate)

    result = callback_func(
        x_scale="linear",
        y_scale="log",
        x_min=0,
        x_max=100,
        y_min=1,
        y_max=1000,
        show_error_bars=True,
        show_valve_times=False,
        current_fig=None,
        store_data=None,
    )

    assert len(result) == 2


def test_create_plot_settings_callback_returns_figure():
    """Test that plot settings callback returns figure in first position."""
    mock_figure = "mock_figure"
    mock_generate = MagicMock(return_value=mock_figure)
    callback_func = _create_plot_settings_callback("upstream", mock_generate)

    result = callback_func(
        x_scale="linear",
        y_scale="log",
        x_min=0,
        x_max=100,
        y_min=1,
        y_max=1000,
        show_error_bars=True,
        show_valve_times=False,
        current_fig=None,
        store_data=None,
    )

    assert result[0] == mock_figure


def test_create_plot_settings_callback_returns_store_with_y_scale():
    """Test that plot settings callback returns store dict with y_scale."""
    mock_generate = MagicMock(return_value="mock_figure")
    callback_func = _create_plot_settings_callback("upstream", mock_generate)

    result = callback_func(
        x_scale="linear",
        y_scale="log",
        x_min=0,
        x_max=100,
        y_min=1,
        y_max=1000,
        show_error_bars=True,
        show_valve_times=False,
        current_fig=None,
        store_data=None,
    )

    assert result[1] == {"y_scale": "log"}


def test_create_plot_settings_callback_passes_error_bars():
    """Test that plot settings callback passes show_error_bars parameter."""
    mock_generate = MagicMock(return_value="mock_figure")
    callback_func = _create_plot_settings_callback("upstream", mock_generate)

    callback_func(
        x_scale="linear",
        y_scale="log",
        x_min=0,
        x_max=100,
        y_min=1,
        y_max=1000,
        show_error_bars=True,
        show_valve_times=False,
        current_fig=None,
        store_data=None,
    )

    mock_generate.assert_called_once()
    call_kwargs = mock_generate.call_args[1]
    assert call_kwargs["show_error_bars"] is True


def test_create_plot_settings_callback_converts_error_bars_to_bool():
    """Test that plot settings callback converts error_bars to boolean."""
    mock_generate = MagicMock(return_value="mock_figure")
    callback_func = _create_plot_settings_callback("upstream", mock_generate)

    callback_func(
        x_scale="linear",
        y_scale="log",
        x_min=0,
        x_max=100,
        y_min=1,
        y_max=1000,
        show_error_bars=1,  # Non-boolean truthy value
        show_valve_times=False,
        current_fig=None,
        store_data=None,
    )

    call_kwargs = mock_generate.call_args[1]
    assert call_kwargs["show_error_bars"] is True
    assert isinstance(call_kwargs["show_error_bars"], bool)


def test_create_plot_settings_callback_passes_valve_times():
    """Test that plot settings callback passes show_valve_times parameter."""
    mock_generate = MagicMock(return_value="mock_figure")
    callback_func = _create_plot_settings_callback("upstream", mock_generate)

    callback_func(
        x_scale="linear",
        y_scale="log",
        x_min=0,
        x_max=100,
        y_min=1,
        y_max=1000,
        show_error_bars=False,
        show_valve_times=True,
        current_fig=None,
        store_data=None,
    )

    call_kwargs = mock_generate.call_args[1]
    assert call_kwargs["show_valve_times"] is True


def test_create_plot_settings_callback_passes_scales():
    """Test that plot settings callback passes x_scale and y_scale."""
    mock_generate = MagicMock(return_value="mock_figure")
    callback_func = _create_plot_settings_callback("upstream", mock_generate)

    callback_func(
        x_scale="log",
        y_scale="linear",
        x_min=0,
        x_max=100,
        y_min=1,
        y_max=1000,
        show_error_bars=False,
        show_valve_times=False,
        current_fig=None,
        store_data=None,
    )

    call_kwargs = mock_generate.call_args[1]
    assert call_kwargs["x_scale"] == "log"
    assert call_kwargs["y_scale"] == "linear"


def test_create_plot_settings_callback_normalizes_range_values():
    """Test that plot settings callback normalizes None range values."""
    mock_generate = MagicMock(return_value="mock_figure")
    callback_func = _create_plot_settings_callback("upstream", mock_generate)

    callback_func(
        x_scale="linear",
        y_scale="log",
        x_min=None,
        x_max=None,
        y_min=None,
        y_max=None,
        show_error_bars=False,
        show_valve_times=False,
        current_fig=None,
        store_data=None,
    )

    call_kwargs = mock_generate.call_args[1]
    assert call_kwargs["x_min"] is None
    assert call_kwargs["x_max"] is None
    assert call_kwargs["y_min"] is None
    assert call_kwargs["y_max"] is None


def test_create_plot_settings_callback_passes_numeric_ranges():
    """Test that plot settings callback passes numeric range values."""
    mock_generate = MagicMock(return_value="mock_figure")
    callback_func = _create_plot_settings_callback("upstream", mock_generate)

    callback_func(
        x_scale="linear",
        y_scale="log",
        x_min=10,
        x_max=90,
        y_min=0.1,
        y_max=100,
        show_error_bars=False,
        show_valve_times=False,
        current_fig=None,
        store_data=None,
    )

    call_kwargs = mock_generate.call_args[1]
    assert call_kwargs["x_min"] == 10
    assert call_kwargs["x_max"] == 90
    assert call_kwargs["y_min"] == 0.1
    assert call_kwargs["y_max"] == 100


@pytest.mark.parametrize(
    "x_scale,y_scale",
    [
        ("linear", "linear"),
        ("linear", "log"),
        ("log", "linear"),
        ("log", "log"),
    ],
)
def test_create_plot_settings_callback_scale_combinations(x_scale, y_scale):
    """Test plot settings callback with various scale combinations.

    Args:
        x_scale: X-axis scale mode
        y_scale: Y-axis scale mode
    """
    mock_generate = MagicMock(return_value="mock_figure")
    callback_func = _create_plot_settings_callback("upstream", mock_generate)

    result = callback_func(
        x_scale=x_scale,
        y_scale=y_scale,
        x_min=0,
        x_max=100,
        y_min=1,
        y_max=1000,
        show_error_bars=False,
        show_valve_times=False,
        current_fig=None,
        store_data=None,
    )

    assert result[1]["y_scale"] == y_scale


@pytest.mark.parametrize(
    "show_error_bars,show_valve_times",
    [
        (True, True),
        (True, False),
        (False, True),
        (False, False),
        (1, 0),
        ([], [1]),
    ],
)
def test_create_plot_settings_callback_boolean_combinations(
    show_error_bars, show_valve_times
):
    """Test plot settings callback with various boolean combinations.

    Args:
        show_error_bars: Error bar display setting
        show_valve_times: Valve time marker display setting
    """
    mock_generate = MagicMock(return_value="mock_figure")
    callback_func = _create_plot_settings_callback("upstream", mock_generate)

    callback_func(
        x_scale="linear",
        y_scale="log",
        x_min=0,
        x_max=100,
        y_min=1,
        y_max=1000,
        show_error_bars=show_error_bars,
        show_valve_times=show_valve_times,
        current_fig=None,
        store_data=None,
    )

    call_kwargs = mock_generate.call_args[1]
    assert call_kwargs["show_error_bars"] == bool(show_error_bars)
    assert call_kwargs["show_valve_times"] == bool(show_valve_times)


# =============================================================================
# Tests for register_plot_control_callbacks
# =============================================================================


def test_register_plot_control_callbacks_registers_callbacks():
    """Test that register_plot_control_callbacks registers callbacks."""
    mock_plotter = MagicMock()
    mock_app = MagicMock()
    mock_plotter.app = mock_app

    register_plot_control_callbacks(mock_plotter)

    # Should register callbacks for 4 plots (upstream, downstream, temp, perm)
    # Each has: 1 settings callback + 2 auto-min callbacks = 3 per plot
    # Total: 4 plots * 3 callbacks = 12 callbacks
    assert mock_app.callback.call_count == 12


def test_register_plot_control_callbacks_accepts_plotter():
    """Test that register_plot_control_callbacks accepts plotter."""
    mock_plotter = MagicMock()
    mock_plotter.app = MagicMock()

    # Should not raise exception
    register_plot_control_callbacks(mock_plotter)


def test_register_plot_control_callbacks_uses_plotter_app():
    """Test that register_plot_control_callbacks uses plotter app."""
    mock_plotter = MagicMock()
    mock_app = MagicMock()
    mock_plotter.app = mock_app

    register_plot_control_callbacks(mock_plotter)

    assert mock_app.callback.called
