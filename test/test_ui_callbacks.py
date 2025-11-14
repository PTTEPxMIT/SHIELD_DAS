"""Tests for UI interaction callbacks in SHIELD Data Acquisition System.

This module tests the collapsible section toggle functionality for all UI
callback functions including chevron icon creation and toggle behavior.
"""

from unittest.mock import MagicMock

import pytest
from dash import html

from shield_das.callbacks.ui_callbacks import (
    _create_chevron_icon,
    _create_toggle_callback,
    register_ui_callbacks,
)

# =============================================================================
# Tests for _create_chevron_icon
# =============================================================================


def test_create_chevron_icon_returns_down_when_open():
    """Test that _create_chevron_icon returns chevron-down icon when section is open."""
    result = _create_chevron_icon(is_open=True)
    assert result.className == "fas fa-chevron-down"


def test_create_chevron_icon_returns_up_when_closed():
    """Test that _create_chevron_icon returns chevron-up icon when section is closed."""
    result = _create_chevron_icon(is_open=False)
    assert result.className == "fas fa-chevron-up"


def test_create_chevron_icon_returns_html_i_element():
    """Test that _create_chevron_icon returns a Dash HTML I element."""
    result = _create_chevron_icon(is_open=True)
    assert isinstance(result, html.I)


# =============================================================================
# Tests for _create_toggle_callback
# =============================================================================


def test_toggle_callback_maintains_state_when_no_clicks():
    """Test that toggle callback maintains current state when n_clicks is None."""
    callback_func = _create_toggle_callback("test-collapse", "test-button")
    new_state, icon = callback_func(n_clicks=None, is_open=True)
    assert new_state is True


def test_toggle_callback_maintains_state_when_zero_clicks():
    """Test that toggle callback maintains current state when n_clicks is 0."""
    callback_func = _create_toggle_callback("test-collapse", "test-button")
    new_state, icon = callback_func(n_clicks=0, is_open=False)
    assert new_state is False


def test_toggle_callback_toggles_from_true_to_false():
    """Test that toggle callback changes state from True to False on click."""
    callback_func = _create_toggle_callback("test-collapse", "test-button")
    new_state, icon = callback_func(n_clicks=1, is_open=True)
    assert new_state is False


def test_toggle_callback_toggles_from_false_to_true():
    """Test that toggle callback changes state from False to True on click."""
    callback_func = _create_toggle_callback("test-collapse", "test-button")
    new_state, icon = callback_func(n_clicks=1, is_open=False)
    assert new_state is True


def test_toggle_callback_returns_correct_icon_when_toggling_to_open():
    """Test toggle callback returns chevron-down when toggling to open."""
    callback_func = _create_toggle_callback("test-collapse", "test-button")
    new_state, icon = callback_func(n_clicks=1, is_open=False)
    assert icon.className == "fas fa-chevron-down"


def test_toggle_callback_returns_correct_icon_when_toggling_to_closed():
    """Test toggle callback returns chevron-up when toggling to closed."""
    callback_func = _create_toggle_callback("test-collapse", "test-button")
    new_state, icon = callback_func(n_clicks=1, is_open=True)
    assert icon.className == "fas fa-chevron-up"


def test_toggle_callback_returns_correct_icon_when_no_clicks():
    """Test toggle callback returns correct icon for state when no clicks."""
    callback_func = _create_toggle_callback("test-collapse", "test-button")
    new_state, icon = callback_func(n_clicks=None, is_open=True)
    assert icon.className == "fas fa-chevron-down"


def test_toggle_callback_returns_tuple():
    """Test that toggle callback returns a tuple of (state, icon)."""
    callback_func = _create_toggle_callback("test-collapse", "test-button")
    result = callback_func(n_clicks=1, is_open=True)
    assert isinstance(result, tuple)


def test_toggle_callback_tuple_has_two_elements():
    """Test that toggle callback returns exactly two elements in tuple."""
    callback_func = _create_toggle_callback("test-collapse", "test-button")
    result = callback_func(n_clicks=1, is_open=True)
    assert len(result) == 2


@pytest.mark.parametrize(
    "initial_state,expected_state",
    [
        (True, False),
        (False, True),
    ],
)
def test_toggle_callback_toggles_state_correctly(initial_state, expected_state):
    """Test that toggle callback correctly toggles between states on click.

    Args:
        initial_state: Starting collapse state
        expected_state: Expected state after toggle
    """
    callback_func = _create_toggle_callback("test-collapse", "test-button")
    new_state, icon = callback_func(n_clicks=1, is_open=initial_state)
    assert new_state == expected_state


@pytest.mark.parametrize(
    "is_open,expected_class",
    [
        (True, "fas fa-chevron-down"),
        (False, "fas fa-chevron-up"),
    ],
)
def test_create_chevron_icon_class_for_state(is_open, expected_class):
    """Test that _create_chevron_icon returns correct class for each state.

    Args:
        is_open: Collapse state (True for open, False for closed)
        expected_class: Expected CSS class for chevron icon
    """
    result = _create_chevron_icon(is_open)
    assert result.className == expected_class


@pytest.mark.parametrize(
    "n_clicks,is_open,expected_icon_class",
    [
        (None, True, "fas fa-chevron-down"),
        (None, False, "fas fa-chevron-up"),
        (0, True, "fas fa-chevron-down"),
        (0, False, "fas fa-chevron-up"),
        (1, True, "fas fa-chevron-up"),
        (1, False, "fas fa-chevron-down"),
        (5, True, "fas fa-chevron-up"),
        (10, False, "fas fa-chevron-down"),
    ],
)
def test_toggle_callback_icon_class_combinations(
    n_clicks, is_open, expected_icon_class
):
    """Test toggle callback icon class for click and state combinations.

    Args:
        n_clicks: Number of button clicks (None, 0, or positive integer)
        is_open: Current collapse state
        expected_icon_class: Expected CSS class for the returned icon
    """
    callback_func = _create_toggle_callback("test-collapse", "test-button")
    new_state, icon = callback_func(n_clicks=n_clicks, is_open=is_open)
    assert icon.className == expected_icon_class


# =============================================================================
# Tests for register_ui_callbacks
# =============================================================================


def test_register_ui_callbacks_registers_callbacks():
    """Test that register_ui_callbacks registers callbacks with app."""
    mock_plotter = MagicMock()
    mock_app = MagicMock()
    mock_plotter.app = mock_app

    register_ui_callbacks(mock_plotter)

    # Should have registered 5 callbacks (dataset, upstream, downstream,
    # temperature, permeability)
    assert mock_app.callback.call_count == 5


def test_register_ui_callbacks_accepts_plotter_instance():
    """Test that register_ui_callbacks accepts a plotter instance."""
    mock_plotter = MagicMock()
    mock_plotter.app = MagicMock()

    # Should not raise any exception
    register_ui_callbacks(mock_plotter)


def test_register_ui_callbacks_uses_plotter_app():
    """Test that register_ui_callbacks uses the plotter's app attribute."""
    mock_plotter = MagicMock()
    mock_app = MagicMock()
    mock_plotter.app = mock_app

    register_ui_callbacks(mock_plotter)

    # Verify that the app's callback method was called
    assert mock_app.callback.called
