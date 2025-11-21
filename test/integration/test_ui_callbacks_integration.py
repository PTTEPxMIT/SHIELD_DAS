"""Integration tests for UI callbacks in SHIELD DAS.

This module tests the actual Dash callback execution for UI toggle functionality,
achieving coverage of the inner callback functions.
"""

from unittest.mock import MagicMock

import pytest

from shield_das.callbacks.ui_callbacks import register_ui_callbacks


@pytest.fixture
def mock_plotter():
    """Create a mock plotter with a Dash app for testing."""
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
    return plotter


def test_toggle_dataset_collapse_callback_execution(mock_plotter):
    """Test that toggle_dataset_collapse callback executes correctly."""
    register_ui_callbacks(mock_plotter)

    # Find the dataset collapse callback
    callback = next(
        (
            cb
            for cb in mock_plotter.app._callbacks
            if cb["func"].__name__ == "toggle_dataset_collapse"
        ),
        None,
    )

    assert callback is not None


def test_toggle_dataset_collapse_opens_when_closed(mock_plotter):
    """Test that dataset collapse opens when currently closed."""
    register_ui_callbacks(mock_plotter)

    callback = next(
        cb
        for cb in mock_plotter.app._callbacks
        if cb["func"].__name__ == "toggle_dataset_collapse"
    )

    # Simulate click when closed (is_open=False)
    is_open, _ = callback["func"](n_clicks=1, is_open=False)

    assert is_open is True


def test_toggle_dataset_collapse_closes_when_open(mock_plotter):
    """Test that dataset collapse closes when currently open."""
    register_ui_callbacks(mock_plotter)

    callback = next(
        cb
        for cb in mock_plotter.app._callbacks
        if cb["func"].__name__ == "toggle_dataset_collapse"
    )

    # Simulate click when open (is_open=True)
    is_open, _ = callback["func"](n_clicks=1, is_open=True)

    assert is_open is False


def test_toggle_dataset_collapse_returns_correct_icon(mock_plotter):
    """Test that dataset collapse returns correct icon element."""
    register_ui_callbacks(mock_plotter)

    callback = next(
        cb
        for cb in mock_plotter.app._callbacks
        if cb["func"].__name__ == "toggle_dataset_collapse"
    )

    _, button_children = callback["func"](n_clicks=1, is_open=False)

    assert button_children is not None


def test_toggle_upstream_controls_callback_execution(mock_plotter):
    """Test that toggle_upstream_controls_collapse executes correctly."""
    register_ui_callbacks(mock_plotter)

    callback = next(
        (
            cb
            for cb in mock_plotter.app._callbacks
            if cb["func"].__name__ == "toggle_upstream_controls_collapse"
        ),
        None,
    )

    assert callback is not None


def test_toggle_upstream_controls_opens_when_closed(mock_plotter):
    """Test that upstream controls open when currently closed."""
    register_ui_callbacks(mock_plotter)

    callback = next(
        cb
        for cb in mock_plotter.app._callbacks
        if cb["func"].__name__ == "toggle_upstream_controls_collapse"
    )

    is_open, _ = callback["func"](n_clicks=1, is_open=False)

    assert is_open is True


def test_toggle_upstream_controls_closes_when_open(mock_plotter):
    """Test that upstream controls close when currently open."""
    register_ui_callbacks(mock_plotter)

    callback = next(
        cb
        for cb in mock_plotter.app._callbacks
        if cb["func"].__name__ == "toggle_upstream_controls_collapse"
    )

    is_open, _ = callback["func"](n_clicks=1, is_open=True)

    assert is_open is False


def test_toggle_downstream_controls_callback_execution(mock_plotter):
    """Test that toggle_downstream_controls_collapse executes correctly."""
    register_ui_callbacks(mock_plotter)

    callback = next(
        (
            cb
            for cb in mock_plotter.app._callbacks
            if cb["func"].__name__ == "toggle_downstream_controls_collapse"
        ),
        None,
    )

    assert callback is not None


def test_toggle_downstream_controls_toggles_state(mock_plotter):
    """Test that downstream controls toggle state correctly."""
    register_ui_callbacks(mock_plotter)

    callback = next(
        cb
        for cb in mock_plotter.app._callbacks
        if cb["func"].__name__ == "toggle_downstream_controls_collapse"
    )

    is_open, _ = callback["func"](n_clicks=1, is_open=False)
    assert is_open is True

    is_open, _ = callback["func"](n_clicks=2, is_open=True)
    assert is_open is False


def test_toggle_temperature_controls_callback_execution(mock_plotter):
    """Test that toggle_temperature_controls_collapse executes correctly."""
    register_ui_callbacks(mock_plotter)

    callback = next(
        (
            cb
            for cb in mock_plotter.app._callbacks
            if cb["func"].__name__ == "toggle_temperature_controls_collapse"
        ),
        None,
    )

    assert callback is not None


def test_toggle_temperature_controls_toggles_state(mock_plotter):
    """Test that temperature controls toggle state correctly."""
    register_ui_callbacks(mock_plotter)

    callback = next(
        cb
        for cb in mock_plotter.app._callbacks
        if cb["func"].__name__ == "toggle_temperature_controls_collapse"
    )

    is_open, _ = callback["func"](n_clicks=1, is_open=False)
    assert is_open is True


def test_toggle_permeability_controls_callback_execution(mock_plotter):
    """Test that toggle_permeability_controls_collapse executes correctly."""
    register_ui_callbacks(mock_plotter)

    callback = next(
        (
            cb
            for cb in mock_plotter.app._callbacks
            if cb["func"].__name__ == "toggle_permeability_controls_collapse"
        ),
        None,
    )

    assert callback is not None


def test_toggle_permeability_controls_toggles_state(mock_plotter):
    """Test that permeability controls toggle state correctly."""
    register_ui_callbacks(mock_plotter)

    callback = next(
        cb
        for cb in mock_plotter.app._callbacks
        if cb["func"].__name__ == "toggle_permeability_controls_collapse"
    )

    is_open, _ = callback["func"](n_clicks=1, is_open=True)
    assert is_open is False


@pytest.mark.parametrize(
    "callback_name,initial_state,expected_state",
    [
        ("toggle_dataset_collapse", False, True),
        ("toggle_dataset_collapse", True, False),
        ("toggle_upstream_controls_collapse", False, True),
        ("toggle_upstream_controls_collapse", True, False),
        ("toggle_downstream_controls_collapse", False, True),
        ("toggle_downstream_controls_collapse", True, False),
        ("toggle_temperature_controls_collapse", False, True),
        ("toggle_temperature_controls_collapse", True, False),
        ("toggle_permeability_controls_collapse", False, True),
        ("toggle_permeability_controls_collapse", True, False),
    ],
)
def test_all_toggle_callbacks_state_transitions(
    mock_plotter, callback_name, initial_state, expected_state
):
    """Test state transitions for all toggle callbacks.

    Args:
        mock_plotter: Mock plotter fixture
        callback_name: Name of the callback function to test
        initial_state: Initial is_open state
        expected_state: Expected is_open state after toggle
    """
    register_ui_callbacks(mock_plotter)

    callback = next(
        cb for cb in mock_plotter.app._callbacks if cb["func"].__name__ == callback_name
    )

    is_open, _ = callback["func"](n_clicks=1, is_open=initial_state)

    assert is_open == expected_state
