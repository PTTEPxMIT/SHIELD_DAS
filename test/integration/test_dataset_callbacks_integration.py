"""Integration tests for dataset callbacks in SHIELD DAS.

This module tests the actual Dash callback execution for dataset management
operations including name updates, color updates, adding, deleting, downloading,
and toggling the add dataset UI.
"""

import json
import os
from unittest.mock import MagicMock, patch

import dash_bootstrap_components as dbc
import pytest
from dash.exceptions import PreventUpdate

from shield_das.callbacks.dataset_callbacks import register_dataset_callbacks


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
                    "prevent_initial_call": kwargs.get("prevent_initial_call", False),
                }
            )
            return func

        return decorator

    plotter.app.callback = mock_callback

    # Mock datasets
    plotter.datasets = [
        MagicMock(name="Dataset 1", colour="#ff0000", path="/path/to/dataset1"),
        MagicMock(name="Dataset 2", colour="#00ff00", path="/path/to/dataset2"),
        MagicMock(name="Dataset 3", colour="#0000ff", path="/path/to/dataset3"),
    ]

    # Mock methods
    plotter.create_dataset_table = MagicMock(return_value={"type": "table"})
    plotter._generate_upstream_plot = MagicMock(return_value={"data": [], "layout": {}})
    plotter._generate_downstream_plot = MagicMock(
        return_value={"data": [], "layout": {}}
    )
    plotter._generate_temperature_plot = MagicMock(
        return_value={"data": [], "layout": {}}
    )
    plotter._generate_both_plots = MagicMock(
        return_value=[{"data": [], "layout": {}}, {"data": [], "layout": {}}]
    )
    plotter.load_data = MagicMock()
    plotter._create_dataset_download = MagicMock(
        return_value={
            "content": b"test content",
            "filename": "dataset.zip",
            "type": "application/zip",
        }
    )

    return plotter


@pytest.fixture
def temp_dataset_dir(tmp_path):
    """Create a temporary dataset directory with metadata."""
    dataset_dir = tmp_path / "test_dataset"
    dataset_dir.mkdir()

    metadata = {
        "version": "1.3",
        "furnace_setpoint": 500,
        "material": "test_material",
        "thickness": 1.0,
    }

    metadata_file = dataset_dir / "run_metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f)

    return str(dataset_dir)


# =============================================================================
# Tests for update_dataset_names callback
# =============================================================================


def test_update_dataset_names_callback_registered(mock_plotter):
    """Test that update_dataset_names callback is registered."""
    register_dataset_callbacks(mock_plotter)

    callback = next(
        (
            cb
            for cb in mock_plotter.app._callbacks
            if cb["func"].__name__ == "update_dataset_names"
        ),
        None,
    )

    assert callback is not None


def test_update_dataset_names_updates_dataset_name(mock_plotter):
    """Test that update_dataset_names updates dataset names."""
    register_dataset_callbacks(mock_plotter)

    callback = next(
        cb
        for cb in mock_plotter.app._callbacks
        if cb["func"].__name__ == "update_dataset_names"
    )

    new_names = ["New Name 1", "Dataset 2", "Dataset 3"]
    callback["func"](new_names, False, False, False, False)

    assert mock_plotter.datasets[0].name == "New Name 1"


def test_update_dataset_names_skips_unchanged_datasets(mock_plotter):
    """Test that update_dataset_names only updates changed names."""
    register_dataset_callbacks(mock_plotter)

    callback = next(
        cb
        for cb in mock_plotter.app._callbacks
        if cb["func"].__name__ == "update_dataset_names"
    )

    # Keep first two unchanged, change third
    new_names = ["Dataset 1", "Dataset 2", "New Name 3"]
    callback["func"](new_names, False, False, False, False)

    # First two should remain unchanged
    assert mock_plotter.datasets[0].name == "Dataset 1"
    assert mock_plotter.datasets[1].name == "Dataset 2"
    # Third should be updated
    assert mock_plotter.datasets[2].name == "New Name 3"


def test_update_dataset_names_regenerates_plots(mock_plotter):
    """Test that update_dataset_names calls _generate_both_plots."""
    register_dataset_callbacks(mock_plotter)

    callback = next(
        cb
        for cb in mock_plotter.app._callbacks
        if cb["func"].__name__ == "update_dataset_names"
    )

    new_names = ["New Name 1", "Dataset 2", "Dataset 3"]
    callback["func"](new_names, True, False, True, False)

    mock_plotter._generate_both_plots.assert_called_once()


def test_update_dataset_names_passes_error_bar_states(mock_plotter):
    """Test that update_dataset_names passes error bar states to plot generation."""
    register_dataset_callbacks(mock_plotter)

    callback = next(
        cb
        for cb in mock_plotter.app._callbacks
        if cb["func"].__name__ == "update_dataset_names"
    )

    new_names = ["New Name 1", "Dataset 2", "Dataset 3"]
    callback["func"](new_names, True, False, True, False)

    mock_plotter._generate_both_plots.assert_called_once_with(
        show_error_bars_upstream=True,
        show_error_bars_downstream=False,
        show_valve_times_upstream=True,
        show_valve_times_downstream=False,
    )


def test_update_dataset_names_returns_table_and_plots(mock_plotter):
    """Test that update_dataset_names returns table and plots."""
    register_dataset_callbacks(mock_plotter)

    callback = next(
        cb
        for cb in mock_plotter.app._callbacks
        if cb["func"].__name__ == "update_dataset_names"
    )

    new_names = ["New Name 1", "Dataset 2", "Dataset 3"]
    result = callback["func"](new_names, False, False, False, False)

    assert len(result) == 3
    assert result[0] == mock_plotter.create_dataset_table.return_value


# =============================================================================
# Tests for update_dataset_colors callback
# =============================================================================


def test_update_dataset_colors_callback_registered(mock_plotter):
    """Test that update_dataset_colors callback is registered."""
    register_dataset_callbacks(mock_plotter)

    callback = next(
        (
            cb
            for cb in mock_plotter.app._callbacks
            if cb["func"].__name__ == "update_dataset_colors"
        ),
        None,
    )

    assert callback is not None


def test_update_dataset_colors_updates_dataset_color(mock_plotter):
    """Test that update_dataset_colors updates dataset colors."""
    register_dataset_callbacks(mock_plotter)

    callback = next(
        cb
        for cb in mock_plotter.app._callbacks
        if cb["func"].__name__ == "update_dataset_colors"
    )

    new_colors = ["#123456", "#00ff00", "#0000ff"]
    callback["func"](new_colors)

    assert mock_plotter.datasets[0].colour == "#123456"


def test_update_dataset_colors_regenerates_all_plots(mock_plotter):
    """Test that update_dataset_colors regenerates all four plots."""
    register_dataset_callbacks(mock_plotter)

    callback = next(
        cb
        for cb in mock_plotter.app._callbacks
        if cb["func"].__name__ == "update_dataset_colors"
    )

    new_colors = ["#123456", "#00ff00", "#0000ff"]
    callback["func"](new_colors)

    mock_plotter._generate_upstream_plot.assert_called_once()
    mock_plotter._generate_downstream_plot.assert_called_once()
    mock_plotter._generate_temperature_plot.assert_called_once()


def test_update_dataset_colors_returns_four_outputs(mock_plotter):
    """Test that update_dataset_colors returns table and three plots."""
    register_dataset_callbacks(mock_plotter)

    callback = next(
        cb
        for cb in mock_plotter.app._callbacks
        if cb["func"].__name__ == "update_dataset_colors"
    )

    new_colors = ["#123456", "#00ff00", "#0000ff"]
    result = callback["func"](new_colors)

    assert len(result) == 4


def test_update_dataset_colors_handles_empty_color(mock_plotter):
    """Test that update_dataset_colors skips empty color values."""
    register_dataset_callbacks(mock_plotter)

    callback = next(
        cb
        for cb in mock_plotter.app._callbacks
        if cb["func"].__name__ == "update_dataset_colors"
    )

    new_colors = ["", "#00ff00", "#0000ff"]
    callback["func"](new_colors)

    assert mock_plotter.datasets[0].colour == "#ff0000"


# =============================================================================
# Tests for add_new_dataset callback
# =============================================================================


def test_add_new_dataset_callback_registered(mock_plotter):
    """Test that add_new_dataset callback is registered."""
    register_dataset_callbacks(mock_plotter)

    callback = next(
        (
            cb
            for cb in mock_plotter.app._callbacks
            if cb["func"].__name__ == "add_new_dataset"
        ),
        None,
    )

    assert callback is not None


def test_add_new_dataset_returns_unchanged_when_no_clicks(mock_plotter):
    """Test that add_new_dataset returns unchanged state when no clicks."""
    register_dataset_callbacks(mock_plotter)

    callback = next(
        cb
        for cb in mock_plotter.app._callbacks
        if cb["func"].__name__ == "add_new_dataset"
    )

    result = callback["func"](None, "/some/path")

    assert result[3] == "/some/path"
    assert result[4] == ""


def test_add_new_dataset_returns_unchanged_when_no_path(mock_plotter):
    """Test that add_new_dataset returns unchanged state when no path."""
    register_dataset_callbacks(mock_plotter)

    callback = next(
        cb
        for cb in mock_plotter.app._callbacks
        if cb["func"].__name__ == "add_new_dataset"
    )

    result = callback["func"](1, "")

    assert result[3] == ""
    assert result[4] == ""


def test_add_new_dataset_validates_path_existence(mock_plotter):
    """Test that add_new_dataset validates path exists."""
    register_dataset_callbacks(mock_plotter)

    callback = next(
        cb
        for cb in mock_plotter.app._callbacks
        if cb["func"].__name__ == "add_new_dataset"
    )

    result = callback["func"](1, "/nonexistent/path")

    assert isinstance(result[4], dbc.Alert)
    assert "does not exist" in result[4].children


def test_add_new_dataset_validates_metadata_exists(mock_plotter, temp_dataset_dir):
    """Test that add_new_dataset validates metadata file exists."""
    register_dataset_callbacks(mock_plotter)

    callback = next(
        cb
        for cb in mock_plotter.app._callbacks
        if cb["func"].__name__ == "add_new_dataset"
    )

    # Remove metadata file
    os.remove(os.path.join(temp_dataset_dir, "run_metadata.json"))

    result = callback["func"](1, temp_dataset_dir)

    assert isinstance(result[4], dbc.Alert)
    assert "run_metadata.json" in result[4].children


def test_add_new_dataset_calls_load_data(mock_plotter, temp_dataset_dir):
    """Test that add_new_dataset calls load_data with correct arguments."""
    register_dataset_callbacks(mock_plotter)

    callback = next(
        cb
        for cb in mock_plotter.app._callbacks
        if cb["func"].__name__ == "add_new_dataset"
    )

    callback["func"](1, temp_dataset_dir)

    mock_plotter.load_data.assert_called_once()
    assert mock_plotter.load_data.call_args[0][0] == temp_dataset_dir


def test_add_new_dataset_clears_input_on_success(mock_plotter, temp_dataset_dir):
    """Test that add_new_dataset clears input field on successful add."""
    register_dataset_callbacks(mock_plotter)

    callback = next(
        cb
        for cb in mock_plotter.app._callbacks
        if cb["func"].__name__ == "add_new_dataset"
    )

    result = callback["func"](1, temp_dataset_dir)

    assert result[3] == ""


def test_add_new_dataset_shows_success_alert(mock_plotter, temp_dataset_dir):
    """Test that add_new_dataset shows success alert."""
    register_dataset_callbacks(mock_plotter)

    callback = next(
        cb
        for cb in mock_plotter.app._callbacks
        if cb["func"].__name__ == "add_new_dataset"
    )

    result = callback["func"](1, temp_dataset_dir)

    assert isinstance(result[4], dbc.Alert)
    assert result[4].color == "success"


def test_add_new_dataset_handles_load_error(mock_plotter, temp_dataset_dir):
    """Test that add_new_dataset handles errors during load_data."""
    register_dataset_callbacks(mock_plotter)

    callback = next(
        cb
        for cb in mock_plotter.app._callbacks
        if cb["func"].__name__ == "add_new_dataset"
    )

    mock_plotter.load_data.side_effect = Exception("Test error")

    result = callback["func"](1, temp_dataset_dir)

    assert isinstance(result[4], dbc.Alert)
    assert result[4].color == "danger"
    assert "Failed to add dataset" in result[4].children


def test_add_new_dataset_regenerates_plots_on_success(mock_plotter, temp_dataset_dir):
    """Test that add_new_dataset regenerates plots on success."""
    register_dataset_callbacks(mock_plotter)

    callback = next(
        cb
        for cb in mock_plotter.app._callbacks
        if cb["func"].__name__ == "add_new_dataset"
    )

    callback["func"](1, temp_dataset_dir)

    mock_plotter._generate_upstream_plot.assert_called_once()
    mock_plotter._generate_downstream_plot.assert_called_once()


# =============================================================================
# Tests for delete_dataset callback
# =============================================================================


def test_delete_dataset_callback_registered(mock_plotter):
    """Test that delete_dataset callback is registered."""
    register_dataset_callbacks(mock_plotter)

    callback = next(
        (
            cb
            for cb in mock_plotter.app._callbacks
            if cb["func"].__name__ == "delete_dataset"
        ),
        None,
    )

    assert callback is not None


@patch("shield_das.callbacks.dataset_callbacks.dash.callback_context")
def test_delete_dataset_removes_dataset_from_list(mock_ctx, mock_plotter):
    """Test that delete_dataset removes dataset from plotter.datasets."""
    register_dataset_callbacks(mock_plotter)

    callback = next(
        cb
        for cb in mock_plotter.app._callbacks
        if cb["func"].__name__ == "delete_dataset"
    )

    # Mock callback context
    mock_ctx.triggered = [
        {"prop_id": '{"type": "delete-dataset", "index": 1}.n_clicks'}
    ]

    initial_count = len(mock_plotter.datasets)
    callback["func"]([None, 1, None])

    assert len(mock_plotter.datasets) == initial_count - 1


@patch("shield_das.callbacks.dataset_callbacks.dash.callback_context")
def test_delete_dataset_removes_correct_dataset(mock_ctx, mock_plotter):
    """Test that delete_dataset removes the correct dataset."""
    register_dataset_callbacks(mock_plotter)

    callback = next(
        cb
        for cb in mock_plotter.app._callbacks
        if cb["func"].__name__ == "delete_dataset"
    )

    mock_ctx.triggered = [
        {"prop_id": '{"type": "delete-dataset", "index": 1}.n_clicks'}
    ]

    deleted_dataset = mock_plotter.datasets[1]
    callback["func"]([None, 1, None])

    assert deleted_dataset not in mock_plotter.datasets


def test_delete_dataset_raises_prevent_update_when_no_clicks(mock_plotter):
    """Test that delete_dataset raises PreventUpdate when no clicks."""
    register_dataset_callbacks(mock_plotter)

    callback = next(
        cb
        for cb in mock_plotter.app._callbacks
        if cb["func"].__name__ == "delete_dataset"
    )

    with pytest.raises(PreventUpdate):
        callback["func"]([None, None, None])


@patch("shield_das.callbacks.dataset_callbacks.dash.callback_context")
def test_delete_dataset_regenerates_all_plots(mock_ctx, mock_plotter):
    """Test that delete_dataset regenerates all plots."""
    register_dataset_callbacks(mock_plotter)

    callback = next(
        cb
        for cb in mock_plotter.app._callbacks
        if cb["func"].__name__ == "delete_dataset"
    )

    mock_ctx.triggered = [
        {"prop_id": '{"type": "delete-dataset", "index": 1}.n_clicks'}
    ]

    callback["func"]([None, 1, None])

    mock_plotter._generate_upstream_plot.assert_called_once()
    mock_plotter._generate_downstream_plot.assert_called_once()
    mock_plotter._generate_temperature_plot.assert_called_once()


@patch("shield_das.callbacks.dataset_callbacks.dash.callback_context")
def test_delete_dataset_returns_four_outputs(mock_ctx, mock_plotter):
    """Test that delete_dataset returns table and three plots."""
    register_dataset_callbacks(mock_plotter)

    callback = next(
        cb
        for cb in mock_plotter.app._callbacks
        if cb["func"].__name__ == "delete_dataset"
    )

    mock_ctx.triggered = [
        {"prop_id": '{"type": "delete-dataset", "index": 1}.n_clicks'}
    ]

    result = callback["func"]([None, 1, None])

    assert len(result) == 4


# =============================================================================
# Tests for download_dataset callback
# =============================================================================


def test_download_dataset_callback_registered(mock_plotter):
    """Test that download_dataset callback is registered."""
    register_dataset_callbacks(mock_plotter)

    callback = next(
        (
            cb
            for cb in mock_plotter.app._callbacks
            if cb["func"].__name__ == "download_dataset"
        ),
        None,
    )

    assert callback is not None


def test_download_dataset_raises_prevent_update_when_no_clicks(mock_plotter):
    """Test that download_dataset raises PreventUpdate when no clicks."""
    register_dataset_callbacks(mock_plotter)

    callback = next(
        cb
        for cb in mock_plotter.app._callbacks
        if cb["func"].__name__ == "download_dataset"
    )

    with pytest.raises(PreventUpdate):
        callback["func"]([None, None, None])


@patch("shield_das.callbacks.dataset_callbacks.os.path.exists")
@patch("shield_das.callbacks.dataset_callbacks.dash.callback_context")
def test_download_dataset_calls_create_dataset_download(
    mock_ctx, mock_exists, mock_plotter
):
    """Test that download_dataset calls _create_dataset_download."""
    register_dataset_callbacks(mock_plotter)

    callback = next(
        cb
        for cb in mock_plotter.app._callbacks
        if cb["func"].__name__ == "download_dataset"
    )

    mock_ctx.triggered = [
        {"prop_id": '{"type": "download-dataset", "index": 0}.n_clicks'}
    ]
    mock_exists.return_value = True

    callback["func"]([1, None, None])

    mock_plotter._create_dataset_download.assert_called_once()


@patch("shield_das.callbacks.dataset_callbacks.os.path.exists")
@patch("shield_das.callbacks.dataset_callbacks.dash.callback_context")
def test_download_dataset_returns_send_bytes_for_binary(
    mock_ctx, mock_exists, mock_plotter
):
    """Test that download_dataset returns dcc.send_bytes for binary content."""
    register_dataset_callbacks(mock_plotter)

    callback = next(
        cb
        for cb in mock_plotter.app._callbacks
        if cb["func"].__name__ == "download_dataset"
    )

    mock_ctx.triggered = [
        {"prop_id": '{"type": "download-dataset", "index": 0}.n_clicks'}
    ]
    mock_exists.return_value = True

    result = callback["func"]([1, None, None])

    assert result is not None


@patch("shield_das.callbacks.dataset_callbacks.dash.callback_context")
def test_download_dataset_raises_prevent_update_when_path_missing(
    mock_ctx, mock_plotter
):
    """Test that download_dataset raises PreventUpdate when dataset path missing."""
    register_dataset_callbacks(mock_plotter)

    callback = next(
        cb
        for cb in mock_plotter.app._callbacks
        if cb["func"].__name__ == "download_dataset"
    )

    mock_ctx.triggered = [
        {"prop_id": '{"type": "download-dataset", "index": 0}.n_clicks'}
    ]

    mock_plotter.datasets[0].path = None

    with pytest.raises(PreventUpdate):
        callback["func"]([1, None, None])


@patch("shield_das.callbacks.dataset_callbacks.dash.callback_context")
def test_download_dataset_raises_prevent_update_when_no_package(mock_ctx, mock_plotter):
    """Test that download_dataset raises PreventUpdate when package creation fails."""
    register_dataset_callbacks(mock_plotter)

    callback = next(
        cb
        for cb in mock_plotter.app._callbacks
        if cb["func"].__name__ == "download_dataset"
    )

    mock_ctx.triggered = [
        {"prop_id": '{"type": "download-dataset", "index": 0}.n_clicks'}
    ]

    mock_plotter._create_dataset_download.return_value = None

    with pytest.raises(PreventUpdate):
        callback["func"]([1, None, None])


# =============================================================================
# Tests for toggle_add_dataset_section callback
# =============================================================================


def test_toggle_add_dataset_section_callback_registered(mock_plotter):
    """Test that toggle_add_dataset_section callback is registered."""
    register_dataset_callbacks(mock_plotter)

    callback = next(
        (
            cb
            for cb in mock_plotter.app._callbacks
            if cb["func"].__name__ == "toggle_add_dataset_section"
        ),
        None,
    )

    assert callback is not None


def test_toggle_add_dataset_section_raises_prevent_update_when_no_clicks(mock_plotter):
    """Test that toggle raises PreventUpdate when no clicks."""
    register_dataset_callbacks(mock_plotter)

    callback = next(
        cb
        for cb in mock_plotter.app._callbacks
        if cb["func"].__name__ == "toggle_add_dataset_section"
    )

    with pytest.raises(PreventUpdate):
        callback["func"](None, False)


def test_toggle_add_dataset_section_toggles_from_false_to_true(mock_plotter):
    """Test that toggle changes state from False to True."""
    register_dataset_callbacks(mock_plotter)

    callback = next(
        cb
        for cb in mock_plotter.app._callbacks
        if cb["func"].__name__ == "toggle_add_dataset_section"
    )

    is_open, icon_class = callback["func"](1, False)

    assert is_open is True


def test_toggle_add_dataset_section_toggles_from_true_to_false(mock_plotter):
    """Test that toggle changes state from True to False."""
    register_dataset_callbacks(mock_plotter)

    callback = next(
        cb
        for cb in mock_plotter.app._callbacks
        if cb["func"].__name__ == "toggle_add_dataset_section"
    )

    is_open, icon_class = callback["func"](1, True)

    assert is_open is False


def test_toggle_add_dataset_section_returns_minus_icon_when_open(mock_plotter):
    """Test that toggle returns minus icon when opened."""
    register_dataset_callbacks(mock_plotter)

    callback = next(
        cb
        for cb in mock_plotter.app._callbacks
        if cb["func"].__name__ == "toggle_add_dataset_section"
    )

    is_open, icon_class = callback["func"](1, False)

    assert icon_class == "fas fa-minus"


def test_toggle_add_dataset_section_returns_plus_icon_when_closed(mock_plotter):
    """Test that toggle returns plus icon when closed."""
    register_dataset_callbacks(mock_plotter)

    callback = next(
        cb
        for cb in mock_plotter.app._callbacks
        if cb["func"].__name__ == "toggle_add_dataset_section"
    )

    is_open, icon_class = callback["func"](1, True)

    assert icon_class == "fas fa-plus"


def test_toggle_add_dataset_section_returns_tuple(mock_plotter):
    """Test that toggle returns a tuple of two elements."""
    register_dataset_callbacks(mock_plotter)

    callback = next(
        cb
        for cb in mock_plotter.app._callbacks
        if cb["func"].__name__ == "toggle_add_dataset_section"
    )

    result = callback["func"](1, False)

    assert isinstance(result, tuple)
    assert len(result) == 2
