"""Tests for export callbacks in SHIELD Data Acquisition System.

This module tests the export and plot interaction functionality including
HTML export and FigureResampler zoom/pan callbacks.
"""

from unittest.mock import MagicMock

import dash
import plotly.graph_objects as go
import pytest
from dash.exceptions import PreventUpdate

from shield_das.callbacks.export_callbacks import (
    _create_relayout_callback,
    _export_figure_to_html,
    _is_autoscale_event,
    register_export_callbacks,
)


# =============================================================================
# Tests for _is_autoscale_event
# =============================================================================


def test_is_autoscale_event_returns_false_for_none():
    """Test that _is_autoscale_event returns False for None input."""
    result = _is_autoscale_event(None)
    assert result is False


def test_is_autoscale_event_returns_false_for_empty_dict():
    """Test that _is_autoscale_event returns False for empty dict."""
    result = _is_autoscale_event({})
    assert result is False


def test_is_autoscale_event_returns_true_for_autosize():
    """Test that _is_autoscale_event returns True for autosize key."""
    result = _is_autoscale_event({"autosize": True})
    assert result is True


def test_is_autoscale_event_returns_true_for_xaxis_autorange():
    """Test that _is_autoscale_event returns True for xaxis.autorange."""
    result = _is_autoscale_event({"xaxis.autorange": True})
    assert result is True


def test_is_autoscale_event_returns_true_for_yaxis_autorange():
    """Test that _is_autoscale_event returns True for yaxis.autorange."""
    result = _is_autoscale_event({"yaxis.autorange": True})
    assert result is True


def test_is_autoscale_event_returns_false_for_zoom_data():
    """Test that _is_autoscale_event returns False for zoom data."""
    result = _is_autoscale_event({"xaxis.range[0]": 10, "xaxis.range[1]": 100})
    assert result is False


def test_is_autoscale_event_returns_false_for_pan_data():
    """Test that _is_autoscale_event returns False for pan data."""
    result = _is_autoscale_event({"xaxis.range": [10, 100], "yaxis.range": [0, 50]})
    assert result is False


def test_is_autoscale_event_handles_autosize_false():
    """Test that _is_autoscale_event handles autosize=False."""
    result = _is_autoscale_event({"autosize": False})
    assert result is True  # Key presence triggers autoscale


@pytest.mark.parametrize(
    "relayout_data,expected",
    [
        (None, False),
        ({}, False),
        ({"autosize": True}, True),
        ({"xaxis.autorange": True}, True),
        ({"yaxis.autorange": True}, True),
        ({"xaxis.range[0]": 10}, False),
        ({"xaxis.range": [0, 100]}, False),
        ({"some_other_key": True}, False),
    ],
)
def test_is_autoscale_event_parameterized(relayout_data, expected):
    """Test _is_autoscale_event with various relayout data inputs.

    Args:
        relayout_data: Relayout event data dictionary
        expected: Expected boolean result
    """
    result = _is_autoscale_event(relayout_data)
    assert result == expected


# =============================================================================
# Tests for _export_figure_to_html
# =============================================================================


def test_export_figure_to_html_returns_dict():
    """Test that _export_figure_to_html returns a dictionary."""
    fig = go.Figure()
    result = _export_figure_to_html(fig, "test.html")
    assert isinstance(result, dict)


def test_export_figure_to_html_has_content_key():
    """Test that _export_figure_to_html result has 'content' key."""
    fig = go.Figure()
    result = _export_figure_to_html(fig, "test.html")
    assert "content" in result


def test_export_figure_to_html_has_filename_key():
    """Test that _export_figure_to_html result has 'filename' key."""
    fig = go.Figure()
    result = _export_figure_to_html(fig, "test.html")
    assert "filename" in result


def test_export_figure_to_html_has_type_key():
    """Test that _export_figure_to_html result has 'type' key."""
    fig = go.Figure()
    result = _export_figure_to_html(fig, "test.html")
    assert "type" in result


def test_export_figure_to_html_uses_provided_filename():
    """Test that _export_figure_to_html uses the provided filename."""
    fig = go.Figure()
    result = _export_figure_to_html(fig, "custom_name.html")
    assert result["filename"] == "custom_name.html"


def test_export_figure_to_html_type_is_text_html():
    """Test that _export_figure_to_html sets type to text/html."""
    fig = go.Figure()
    result = _export_figure_to_html(fig, "test.html")
    assert result["type"] == "text/html"


def test_export_figure_to_html_content_is_string():
    """Test that _export_figure_to_html content is a string."""
    fig = go.Figure()
    result = _export_figure_to_html(fig, "test.html")
    assert isinstance(result["content"], str)


def test_export_figure_to_html_content_contains_html():
    """Test that _export_figure_to_html content contains HTML."""
    fig = go.Figure()
    result = _export_figure_to_html(fig, "test.html")
    assert "<html>" in result["content"]


def test_export_figure_to_html_content_contains_plotly():
    """Test that _export_figure_to_html content includes Plotly."""
    fig = go.Figure()
    result = _export_figure_to_html(fig, "test.html")
    assert "plotly" in result["content"].lower()


def test_export_figure_to_html_converts_dict_to_figure():
    """Test that _export_figure_to_html converts dict to Figure."""
    fig_dict = {"data": [], "layout": {}}
    result = _export_figure_to_html(fig_dict, "test.html")
    assert isinstance(result, dict)


def test_export_figure_to_html_raises_on_invalid_figure():
    """Test that _export_figure_to_html raises PreventUpdate on error."""
    with pytest.raises((PreventUpdate, Exception)):
        _export_figure_to_html("invalid", "test.html")


def test_export_figure_to_html_handles_figure_with_data():
    """Test that _export_figure_to_html handles figure with trace data."""
    fig = go.Figure(data=[go.Scatter(x=[1, 2, 3], y=[4, 5, 6])])
    result = _export_figure_to_html(fig, "test.html")
    assert "content" in result


@pytest.mark.parametrize(
    "filename",
    [
        "test.html",
        "upstream_plot.html",
        "my_export_123.html",
        "plot-with-dashes.html",
    ],
)
def test_export_figure_to_html_filename_variations(filename):
    """Test _export_figure_to_html with various filename formats.

    Args:
        filename: Filename to use for export
    """
    fig = go.Figure()
    result = _export_figure_to_html(fig, filename)
    assert result["filename"] == filename


# =============================================================================
# Tests for _create_relayout_callback
# =============================================================================


def test_create_relayout_callback_returns_no_update_when_no_data():
    """Test relayout callback returns no_update when relayout_data is None."""
    mock_generate = MagicMock()
    callback_func = _create_relayout_callback("test-plot", mock_generate)

    mock_plotter = MagicMock()
    mock_plotter.figure_resamplers = {"test-plot": MagicMock()}

    result = callback_func(mock_plotter, None)
    assert result == dash.no_update


def test_create_relayout_callback_returns_no_update_when_plot_not_in_resamplers():
    """Test relayout callback returns no_update when plot not in resamplers."""
    mock_generate = MagicMock()
    callback_func = _create_relayout_callback("test-plot", mock_generate)

    mock_plotter = MagicMock()
    mock_plotter.figure_resamplers = {}

    result = callback_func(mock_plotter, {"xaxis.range": [0, 100]})
    assert result == dash.no_update


def test_create_relayout_callback_calls_generate_on_autoscale():
    """Test relayout callback calls generate method on autoscale event."""
    mock_figure = "regenerated_figure"
    mock_generate = MagicMock(return_value=mock_figure)
    callback_func = _create_relayout_callback("test-plot", mock_generate)

    mock_plotter = MagicMock()
    mock_plotter.figure_resamplers = {"test-plot": MagicMock()}

    result = callback_func(mock_plotter, {"autosize": True})
    assert result == mock_figure


def test_create_relayout_callback_uses_resampler_on_zoom():
    """Test relayout callback uses resampler on zoom event."""
    mock_generate = MagicMock()
    callback_func = _create_relayout_callback("test-plot", mock_generate)

    mock_resampler = MagicMock()
    mock_resampler.construct_update_data_patch = MagicMock(return_value="patched_data")

    mock_plotter = MagicMock()
    mock_plotter.figure_resamplers = {"test-plot": mock_resampler}

    relayout_data = {"xaxis.range[0]": 10, "xaxis.range[1]": 100}
    result = callback_func(mock_plotter, relayout_data)

    mock_resampler.construct_update_data_patch.assert_called_once_with(relayout_data)


def test_create_relayout_callback_does_not_call_generate_on_zoom():
    """Test relayout callback does not call generate on zoom event."""
    mock_generate = MagicMock()
    callback_func = _create_relayout_callback("test-plot", mock_generate)

    mock_resampler = MagicMock()
    mock_plotter = MagicMock()
    mock_plotter.figure_resamplers = {"test-plot": mock_resampler}

    relayout_data = {"xaxis.range": [0, 100]}
    callback_func(mock_plotter, relayout_data)

    mock_generate.assert_not_called()


def test_create_relayout_callback_returns_patch_data():
    """Test relayout callback returns patch data from resampler."""
    mock_generate = MagicMock()
    callback_func = _create_relayout_callback("test-plot", mock_generate)

    expected_patch = {"data": "patched"}
    mock_resampler = MagicMock()
    mock_resampler.construct_update_data_patch = MagicMock(return_value=expected_patch)

    mock_plotter = MagicMock()
    mock_plotter.figure_resamplers = {"test-plot": mock_resampler}

    result = callback_func(mock_plotter, {"xaxis.range": [0, 100]})
    assert result == expected_patch


@pytest.mark.parametrize(
    "relayout_data,should_regenerate",
    [
        ({"autosize": True}, True),
        ({"xaxis.autorange": True}, True),
        ({"yaxis.autorange": True}, True),
        ({"xaxis.range": [0, 100]}, False),
        ({"xaxis.range[0]": 10}, False),
    ],
)
def test_create_relayout_callback_autoscale_vs_zoom(relayout_data, should_regenerate):
    """Test relayout callback behavior for autoscale vs zoom events.

    Args:
        relayout_data: Relayout event data
        should_regenerate: Whether generate method should be called
    """
    mock_generate = MagicMock(return_value="regenerated")
    callback_func = _create_relayout_callback("test-plot", mock_generate)

    mock_resampler = MagicMock()
    mock_plotter = MagicMock()
    mock_plotter.figure_resamplers = {"test-plot": mock_resampler}

    callback_func(mock_plotter, relayout_data)

    if should_regenerate:
        mock_generate.assert_called_once()
    else:
        mock_generate.assert_not_called()


def test_create_relayout_callback_handles_empty_relayout_data():
    """Test relayout callback handles empty relayout data dict."""
    mock_generate = MagicMock()
    callback_func = _create_relayout_callback("test-plot", mock_generate)

    mock_plotter = MagicMock()
    mock_plotter.figure_resamplers = {"test-plot": MagicMock()}

    result = callback_func(mock_plotter, {})
    assert result == dash.no_update


# =============================================================================
# Tests for register_export_callbacks
# =============================================================================


def test_register_export_callbacks_registers_callbacks():
    """Test that register_export_callbacks registers callbacks with app."""
    mock_plotter = MagicMock()
    mock_app = MagicMock()
    mock_plotter.app = mock_app

    register_export_callbacks(mock_plotter)

    # Should register: 4 export + 4 relayout callbacks = 8 total
    assert mock_app.callback.call_count == 8


def test_register_export_callbacks_accepts_plotter_instance():
    """Test that register_export_callbacks accepts a plotter instance."""
    mock_plotter = MagicMock()
    mock_plotter.app = MagicMock()

    # Should not raise any exception
    register_export_callbacks(mock_plotter)


def test_register_export_callbacks_uses_plotter_app():
    """Test that register_export_callbacks uses the plotter's app attribute."""
    mock_plotter = MagicMock()
    mock_app = MagicMock()
    mock_plotter.app = mock_app

    register_export_callbacks(mock_plotter)

    # Verify that the app's callback method was called
    assert mock_app.callback.called
