"""Tests for live data monitoring callbacks in SHIELD DAS.

This module tests automatic data refreshing for ongoing experiments,
including live data flag management and periodic refresh logic.
"""

from unittest.mock import MagicMock

import pytest

from shield_das.callbacks.live_data_callbacks import (
    _refresh_live_datasets,
    _update_dataset_live_flags,
    register_live_data_callbacks,
)

# =============================================================================
# Tests for _update_dataset_live_flags
# =============================================================================


def test_update_dataset_live_flags_updates_dataset_flags():
    """Test that _update_dataset_live_flags updates dataset flags."""
    dataset1 = MagicMock()
    dataset2 = MagicMock()
    datasets = [dataset1, dataset2]

    _update_dataset_live_flags(datasets, [True, False])

    assert dataset1.live_data is True
    assert dataset2.live_data is False


def test_update_dataset_live_flags_returns_true_when_any_live():
    """Test that _update_dataset_live_flags returns True when any live."""
    datasets = [MagicMock(), MagicMock()]

    result = _update_dataset_live_flags(datasets, [True, False])

    assert result is True


def test_update_dataset_live_flags_returns_false_when_none_live():
    """Test that _update_dataset_live_flags returns False when none live."""
    datasets = [MagicMock(), MagicMock()]

    result = _update_dataset_live_flags(datasets, [False, False])

    assert result is False


def test_update_dataset_live_flags_handles_empty_values():
    """Test that _update_dataset_live_flags handles empty values list."""
    datasets = [MagicMock()]

    result = _update_dataset_live_flags(datasets, [])

    assert result is False


def test_update_dataset_live_flags_converts_to_bool():
    """Test that _update_dataset_live_flags converts values to bool."""
    dataset = MagicMock()
    datasets = [dataset]

    _update_dataset_live_flags(datasets, [1])

    assert dataset.live_data is True
    assert isinstance(dataset.live_data, bool)


def test_update_dataset_live_flags_handles_more_values_than_datasets():
    """Test that _update_dataset_live_flags handles extra values safely."""
    dataset = MagicMock()
    datasets = [dataset]

    # Should not raise error, just ignore extra values
    result = _update_dataset_live_flags(datasets, [True, True, True])

    assert result is True
    assert dataset.live_data is True


@pytest.mark.parametrize(
    "values,expected_any",
    [
        ([True, True, True], True),
        ([False, False, False], False),
        ([True, False, False], True),
        ([False, False, True], True),
        ([False, True, False], True),
    ],
)
def test_update_dataset_live_flags_various_combinations(values, expected_any):
    """Test _update_dataset_live_flags with various value combinations.

    Args:
        values: List of boolean values for datasets
        expected_any: Expected return value (any live?)
    """
    datasets = [MagicMock() for _ in values]

    result = _update_dataset_live_flags(datasets, values)

    assert result is expected_any


@pytest.mark.parametrize(
    "index,value",
    [(0, True), (1, False), (2, True)],
)
def test_update_dataset_live_flags_updates_correct_index(index, value):
    """Test that _update_dataset_live_flags updates correct dataset.

    Args:
        index: Index of dataset to verify
        value: Expected boolean value
    """
    datasets = [MagicMock(), MagicMock(), MagicMock()]

    _update_dataset_live_flags(datasets, [True, False, True])

    assert datasets[index].live_data is value


# =============================================================================
# Tests for _refresh_live_datasets
# =============================================================================


def test_refresh_live_datasets_processes_live_datasets():
    """Test that _refresh_live_datasets calls process_data on live datasets."""
    dataset1 = MagicMock()
    dataset1.live_data = True
    dataset2 = MagicMock()
    dataset2.live_data = False

    _refresh_live_datasets([dataset1, dataset2])

    dataset1.process_data.assert_called_once()
    dataset2.process_data.assert_not_called()


def test_refresh_live_datasets_returns_true_when_has_live():
    """Test that _refresh_live_datasets returns True when has live data."""
    dataset = MagicMock()
    dataset.live_data = True

    result = _refresh_live_datasets([dataset])

    assert result is True


def test_refresh_live_datasets_returns_false_when_no_live():
    """Test that _refresh_live_datasets returns False when no live data."""
    dataset = MagicMock()
    dataset.live_data = False

    result = _refresh_live_datasets([dataset])

    assert result is False


def test_refresh_live_datasets_handles_empty_list():
    """Test that _refresh_live_datasets handles empty dataset list."""
    result = _refresh_live_datasets([])

    assert result is False


def test_refresh_live_datasets_skips_processing_when_no_live():
    """Test that _refresh_live_datasets skips processing when no live data."""
    dataset = MagicMock()
    dataset.live_data = False

    _refresh_live_datasets([dataset])

    dataset.process_data.assert_not_called()


def test_refresh_live_datasets_processes_all_live():
    """Test that _refresh_live_datasets processes all live datasets."""
    dataset1 = MagicMock()
    dataset1.live_data = True
    dataset2 = MagicMock()
    dataset2.live_data = True

    _refresh_live_datasets([dataset1, dataset2])

    dataset1.process_data.assert_called_once()
    dataset2.process_data.assert_called_once()


@pytest.mark.parametrize(
    "live_states,expected_calls",
    [
        ([True, True, True], 3),
        ([True, False, True], 2),
        ([False, False, False], 0),
        ([True, False, False], 1),
    ],
)
def test_refresh_live_datasets_various_combinations(live_states, expected_calls):
    """Test _refresh_live_datasets with various dataset combinations.

    Args:
        live_states: List of live_data boolean values
        expected_calls: Expected number of process_data calls
    """
    datasets = []
    for is_live in live_states:
        dataset = MagicMock()
        dataset.live_data = is_live
        datasets.append(dataset)

    _refresh_live_datasets(datasets)

    total_calls = sum(d.process_data.call_count for d in datasets)
    assert total_calls == expected_calls


# =============================================================================
# Tests for register_live_data_callbacks
# =============================================================================


def test_register_live_data_callbacks_accepts_plotter():
    """Test that register_live_data_callbacks accepts plotter instance."""
    mock_plotter = MagicMock()
    mock_plotter.app = MagicMock()
    mock_plotter.datasets = []

    # Should not raise exception
    register_live_data_callbacks(mock_plotter)


def test_register_live_data_callbacks_uses_plotter_app():
    """Test that register_live_data_callbacks uses plotter app."""
    mock_plotter = MagicMock()
    mock_app = MagicMock()
    mock_plotter.app = mock_app
    mock_plotter.datasets = []

    register_live_data_callbacks(mock_plotter)

    # Verify that the app's callback method was called
    assert mock_app.callback.called


def test_register_live_data_callbacks_registers_callbacks():
    """Test that register_live_data_callbacks registers callbacks."""
    mock_plotter = MagicMock()
    mock_app = MagicMock()
    mock_plotter.app = mock_app
    mock_plotter.datasets = []

    register_live_data_callbacks(mock_plotter)

    # Should register 2 callbacks (toggle + interval update)
    assert mock_app.callback.call_count == 2
