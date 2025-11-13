"""Tests for dataset management callbacks in SHIELD DAS.

This module tests dataset CRUD operations including path validation,
alert creation, and triggered index extraction.
"""

from unittest.mock import MagicMock, patch

import dash_bootstrap_components as dbc
import pytest
from dash.exceptions import PreventUpdate

from shield_das.callbacks.dataset_callbacks import (
    _create_alert,
    _get_triggered_index,
    _validate_dataset_path,
    register_dataset_callbacks,
)

# =============================================================================
# Tests for _get_triggered_index
# =============================================================================


def test_get_triggered_index_raises_when_no_trigger():
    """Test that _get_triggered_index raises PreventUpdate when no trigger."""
    mock_ctx = MagicMock()
    mock_ctx.triggered = []

    with pytest.raises(PreventUpdate):
        _get_triggered_index(mock_ctx)


def test_get_triggered_index_extracts_index_from_prop_id():
    """Test that _get_triggered_index extracts index from button ID."""
    mock_ctx = MagicMock()
    mock_ctx.triggered = [{"prop_id": '{"index": 5}.n_clicks'}]

    result = _get_triggered_index(mock_ctx)
    assert result == 5


def test_get_triggered_index_handles_zero_index():
    """Test that _get_triggered_index correctly handles index 0."""
    mock_ctx = MagicMock()
    mock_ctx.triggered = [{"prop_id": '{"index": 0}.n_clicks'}]

    result = _get_triggered_index(mock_ctx)
    assert result == 0


def test_get_triggered_index_raises_on_invalid_json():
    """Test that _get_triggered_index raises PreventUpdate on invalid JSON."""
    mock_ctx = MagicMock()
    mock_ctx.triggered = [{"prop_id": "invalid_json.n_clicks"}]

    with pytest.raises(PreventUpdate):
        _get_triggered_index(mock_ctx)


def test_get_triggered_index_raises_on_missing_index_key():
    """Test that _get_triggered_index raises when index key is missing."""
    mock_ctx = MagicMock()
    mock_ctx.triggered = [{"prop_id": '{"type": "button"}.n_clicks'}]

    with pytest.raises(PreventUpdate):
        _get_triggered_index(mock_ctx)


def test_get_triggered_index_raises_on_non_integer_index():
    """Test that _get_triggered_index raises when index is not integer."""
    mock_ctx = MagicMock()
    mock_ctx.triggered = [{"prop_id": '{"index": "not_an_int"}.n_clicks'}]

    with pytest.raises(PreventUpdate):
        _get_triggered_index(mock_ctx)


def test_get_triggered_index_returns_integer():
    """Test that _get_triggered_index returns an integer type."""
    mock_ctx = MagicMock()
    mock_ctx.triggered = [{"prop_id": '{"index": 3}.n_clicks'}]

    result = _get_triggered_index(mock_ctx)
    assert isinstance(result, int)


@pytest.mark.parametrize(
    "index",
    [0, 1, 5, 10, 100],
)
def test_get_triggered_index_various_indices(index):
    """Test _get_triggered_index with various valid indices.

    Args:
        index: Index value to test
    """
    mock_ctx = MagicMock()
    mock_ctx.triggered = [{"prop_id": f'{{"index": {index}}}.n_clicks'}]

    result = _get_triggered_index(mock_ctx)
    assert result == index


# =============================================================================
# Tests for _create_alert
# =============================================================================


def test_create_alert_returns_alert_component():
    """Test that _create_alert returns a Bootstrap Alert component."""
    result = _create_alert("Test message")
    assert isinstance(result, dbc.Alert)


def test_create_alert_uses_provided_message():
    """Test that _create_alert uses the provided message text."""
    message = "This is a test alert"
    result = _create_alert(message)
    assert result.children == message


def test_create_alert_default_color_is_danger():
    """Test that _create_alert uses 'danger' color by default."""
    result = _create_alert("Test")
    assert result.color == "danger"


def test_create_alert_uses_provided_color():
    """Test that _create_alert uses the provided color."""
    result = _create_alert("Test", color="success")
    assert result.color == "success"


def test_create_alert_is_dismissable_by_default():
    """Test that _create_alert creates dismissable alerts."""
    result = _create_alert("Test")
    assert result.dismissable is True


def test_create_alert_default_duration_is_3000():
    """Test that _create_alert uses 3000ms duration by default."""
    result = _create_alert("Test")
    assert result.duration == 3000


def test_create_alert_uses_provided_duration():
    """Test that _create_alert uses the provided duration."""
    result = _create_alert("Test", duration=5000)
    assert result.duration == 5000


@pytest.mark.parametrize(
    "color",
    ["primary", "secondary", "success", "danger", "warning", "info"],
)
def test_create_alert_various_colors(color):
    """Test _create_alert with various Bootstrap colors.

    Args:
        color: Bootstrap color name
    """
    result = _create_alert("Test", color=color)
    assert result.color == color


@pytest.mark.parametrize(
    "duration",
    [1000, 2000, 3000, 5000, 10000],
)
def test_create_alert_various_durations(duration):
    """Test _create_alert with various durations.

    Args:
        duration: Duration in milliseconds
    """
    result = _create_alert("Test", duration=duration)
    assert result.duration == duration


# =============================================================================
# Tests for _validate_dataset_path
# =============================================================================


def test_validate_dataset_path_returns_false_for_nonexistent_path():
    """Test that _validate_dataset_path returns False for missing path."""
    is_valid, error = _validate_dataset_path("/nonexistent/path")
    assert is_valid is False


def test_validate_dataset_path_returns_error_message_for_missing_path():
    """Test that _validate_dataset_path returns error for missing path."""
    is_valid, error = _validate_dataset_path("/nonexistent/path")
    assert error == "Path does not exist."


def test_validate_dataset_path_returns_tuple():
    """Test that _validate_dataset_path returns a tuple."""
    result = _validate_dataset_path("/nonexistent/path")
    assert isinstance(result, tuple)


def test_validate_dataset_path_tuple_has_two_elements():
    """Test that _validate_dataset_path returns tuple with 2 elements."""
    result = _validate_dataset_path("/nonexistent/path")
    assert len(result) == 2


@patch("os.path.exists")
def test_validate_dataset_path_checks_metadata_file(mock_exists):
    """Test that _validate_dataset_path checks for run_metadata.json."""
    mock_exists.side_effect = lambda p: p == "test_path"

    is_valid, error = _validate_dataset_path("test_path")

    assert is_valid is False
    assert "run_metadata.json" in error


@patch("os.path.exists")
def test_validate_dataset_path_returns_true_for_valid_path(mock_exists):
    """Test that _validate_dataset_path returns True for valid path."""
    mock_exists.return_value = True

    is_valid, error = _validate_dataset_path("test_path")

    assert is_valid is True


@patch("os.path.exists")
def test_validate_dataset_path_returns_none_error_for_valid(mock_exists):
    """Test that _validate_dataset_path returns None error when valid."""
    mock_exists.return_value = True

    is_valid, error = _validate_dataset_path("test_path")

    assert error is None


@patch("os.path.exists")
def test_validate_dataset_path_joins_metadata_path_correctly(mock_exists):
    """Test that _validate_dataset_path constructs correct metadata path."""
    calls = []
    mock_exists.side_effect = lambda p: (calls.append(p), True)[1]

    _validate_dataset_path("test_folder")

    # Should check both the folder and the metadata file
    assert "test_folder" in calls
    assert any("run_metadata.json" in call for call in calls)


# =============================================================================
# Tests for register_dataset_callbacks
# =============================================================================


def test_register_dataset_callbacks_accepts_plotter():
    """Test that register_dataset_callbacks accepts plotter instance."""
    mock_plotter = MagicMock()
    mock_plotter.app = MagicMock()
    mock_plotter.datasets = []

    # Should not raise exception
    register_dataset_callbacks(mock_plotter)


def test_register_dataset_callbacks_uses_plotter_app():
    """Test that register_dataset_callbacks uses plotter app."""
    mock_plotter = MagicMock()
    mock_app = MagicMock()
    mock_plotter.app = mock_app
    mock_plotter.datasets = []

    register_dataset_callbacks(mock_plotter)

    # Verify that the app's callback method was called
    assert mock_app.callback.called


def test_register_dataset_callbacks_registers_callbacks():
    """Test that register_dataset_callbacks registers callbacks."""
    mock_plotter = MagicMock()
    mock_app = MagicMock()
    mock_plotter.app = mock_app
    mock_plotter.datasets = []

    register_dataset_callbacks(mock_plotter)

    # Should register multiple callbacks for dataset management
    # (name/color update, add, delete, download, toggle)
    assert mock_app.callback.call_count >= 5
