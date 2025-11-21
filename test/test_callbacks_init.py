"""Tests for callbacks module initialization in SHIELD DAS.

This module tests the register_all_callbacks convenience function.
"""

from unittest.mock import MagicMock, patch

from shield_das.callbacks import register_all_callbacks


def test_register_all_callbacks_calls_dataset_callbacks():
    """Test register_all_callbacks calls register_dataset_callbacks."""
    mock_plotter = MagicMock()

    with patch("shield_das.callbacks.register_dataset_callbacks") as mock_register:
        register_all_callbacks(mock_plotter)

        mock_register.assert_called_once_with(mock_plotter)


def test_register_all_callbacks_calls_plot_control_callbacks():
    """Test register_all_callbacks calls register_plot_control_callbacks."""
    mock_plotter = MagicMock()

    with patch("shield_das.callbacks.register_plot_control_callbacks") as mock_register:
        register_all_callbacks(mock_plotter)

        mock_register.assert_called_once_with(mock_plotter)


def test_register_all_callbacks_calls_ui_callbacks():
    """Test register_all_callbacks calls register_ui_callbacks."""
    mock_plotter = MagicMock()

    with patch("shield_das.callbacks.register_ui_callbacks") as mock_register:
        register_all_callbacks(mock_plotter)

        mock_register.assert_called_once_with(mock_plotter)


def test_register_all_callbacks_calls_live_data_callbacks():
    """Test register_all_callbacks calls register_live_data_callbacks."""
    mock_plotter = MagicMock()

    with patch("shield_das.callbacks.register_live_data_callbacks") as mock_register:
        register_all_callbacks(mock_plotter)

        mock_register.assert_called_once_with(mock_plotter)


def test_register_all_callbacks_calls_export_callbacks():
    """Test register_all_callbacks calls register_export_callbacks."""
    mock_plotter = MagicMock()

    with patch("shield_das.callbacks.register_export_callbacks") as mock_register:
        register_all_callbacks(mock_plotter)

        mock_register.assert_called_once_with(mock_plotter)


def test_register_all_callbacks_calls_all_five_registrars():
    """Test register_all_callbacks calls all five callback registrars."""
    mock_plotter = MagicMock()

    with (
        patch("shield_das.callbacks.register_dataset_callbacks") as mock_dataset,
        patch("shield_das.callbacks.register_plot_control_callbacks") as mock_plot,
        patch("shield_das.callbacks.register_ui_callbacks") as mock_ui,
        patch("shield_das.callbacks.register_live_data_callbacks") as mock_live,
        patch("shield_das.callbacks.register_export_callbacks") as mock_export,
    ):
        register_all_callbacks(mock_plotter)

        mock_dataset.assert_called_once()
        mock_plot.assert_called_once()
        mock_ui.assert_called_once()
        mock_live.assert_called_once()
        mock_export.assert_called_once()


def test_register_all_callbacks_accepts_plotter():
    """Test register_all_callbacks accepts a plotter object."""
    mock_plotter = MagicMock()

    with (
        patch("shield_das.callbacks.register_dataset_callbacks"),
        patch("shield_das.callbacks.register_plot_control_callbacks"),
        patch("shield_das.callbacks.register_ui_callbacks"),
        patch("shield_das.callbacks.register_live_data_callbacks"),
        patch("shield_das.callbacks.register_export_callbacks"),
    ):
        # Should not raise any exceptions
        register_all_callbacks(mock_plotter)
