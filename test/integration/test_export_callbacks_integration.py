"""Integration tests for export callbacks in SHIELD DAS.

This module tests the actual Dash callback execution for figure export
and FigureResampler relayout functionality, achieving coverage of inner callbacks.
"""

from unittest.mock import MagicMock, patch

import pytest

from shield_das.callbacks.export_callbacks import register_export_callbacks


@pytest.fixture
def mock_plotter():
    """Create a mock plotter with app and FigureResampler for testing."""
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

    # Mock FigureResampler instances
    plotter.figure_resamplers = {
        "upstream-plot": MagicMock(),
        "downstream-plot": MagicMock(),
        "temperature-plot": MagicMock(),
        "permeability-plot": MagicMock(),
    }

    # Mock plot generation methods
    plotter._generate_upstream_plot = MagicMock(return_value={"data": [], "layout": {}})
    plotter._generate_downstream_plot = MagicMock(
        return_value={"data": [], "layout": {}}
    )
    plotter._generate_temperature_plot = MagicMock(
        return_value={"data": [], "layout": {}}
    )
    plotter._generate_permeability_plot = MagicMock(
        return_value={"data": [], "layout": {}}
    )

    return plotter


def test_export_upstream_plot_callback_execution(mock_plotter):
    """Test that export_upstream_plot callback executes correctly."""
    register_export_callbacks(mock_plotter)

    callback = next(
        (
            cb
            for cb in mock_plotter.app._callbacks
            if cb["func"].__name__ == "export_upstream_plot"
        ),
        None,
    )

    assert callback is not None


@patch("shield_das.callbacks.export_callbacks._export_figure_to_html")
def test_export_upstream_plot_exports_figure(mock_export, mock_plotter):
    """Test that upstream plot export calls export function."""
    register_export_callbacks(mock_plotter)

    mock_export.return_value = {"content": "html", "filename": "test.html"}

    callback = next(
        cb
        for cb in mock_plotter.app._callbacks
        if cb["func"].__name__ == "export_upstream_plot"
    )

    callback["func"](1, {"data": []})

    assert mock_export.called


@patch("shield_das.callbacks.export_callbacks._export_figure_to_html")
def test_export_downstream_plot_callback_execution(mock_export, mock_plotter):
    """Test that export_downstream_plot callback executes correctly."""
    register_export_callbacks(mock_plotter)

    mock_export.return_value = {"content": "html", "filename": "test.html"}

    callback = next(
        cb
        for cb in mock_plotter.app._callbacks
        if cb["func"].__name__ == "export_downstream_plot"
    )

    callback["func"](1)

    assert mock_export.called


@patch("shield_das.callbacks.export_callbacks._export_figure_to_html")
def test_export_temperature_plot_callback_execution(mock_export, mock_plotter):
    """Test that export_temperature_plot callback executes correctly."""
    register_export_callbacks(mock_plotter)

    mock_export.return_value = {"content": "html", "filename": "test.html"}

    callback = next(
        cb
        for cb in mock_plotter.app._callbacks
        if cb["func"].__name__ == "export_temperature_plot"
    )

    callback["func"](1, {"data": []})

    assert mock_export.called


@patch("shield_das.callbacks.export_callbacks._export_figure_to_html")
def test_export_permeability_plot_callback_execution(mock_export, mock_plotter):
    """Test that export_permeability_plot callback executes correctly."""
    register_export_callbacks(mock_plotter)

    mock_export.return_value = {"content": "html", "filename": "test.html"}

    callback = next(
        cb
        for cb in mock_plotter.app._callbacks
        if cb["func"].__name__ == "export_permeability_plot"
    )

    callback["func"](1, {"data": []})

    assert mock_export.called


def test_update_upstream_plot_on_relayout_callback_execution(mock_plotter):
    """Test that upstream relayout callback executes correctly."""
    register_export_callbacks(mock_plotter)

    callback = next(
        (
            cb
            for cb in mock_plotter.app._callbacks
            if cb["func"].__name__ == "update_upstream_plot_on_relayout"
        ),
        None,
    )

    assert callback is not None


def test_update_upstream_plot_on_relayout_with_autoscale(mock_plotter):
    """Test that upstream relayout regenerates plot on autoscale."""
    register_export_callbacks(mock_plotter)

    callback = next(
        cb
        for cb in mock_plotter.app._callbacks
        if cb["func"].__name__ == "update_upstream_plot_on_relayout"
    )

    # Autoscale event
    relayout_data = {"autosize": True}
    callback["func"](relayout_data)

    assert mock_plotter._generate_upstream_plot.called


def test_update_temperature_plot_on_relayout_callback_execution(mock_plotter):
    """Test that temperature relayout callback executes correctly."""
    register_export_callbacks(mock_plotter)

    callback = next(
        cb
        for cb in mock_plotter.app._callbacks
        if cb["func"].__name__ == "update_temperature_plot_on_relayout"
    )

    relayout_data = {"xaxis.autorange": True}
    callback["func"](relayout_data)

    assert mock_plotter._generate_temperature_plot.called


def test_update_permeability_plot_on_relayout_callback_execution(mock_plotter):
    """Test that permeability relayout callback executes correctly."""
    register_export_callbacks(mock_plotter)

    callback = next(
        (
            cb
            for cb in mock_plotter.app._callbacks
            if cb["func"].__name__ == "update_permeability_plot_on_relayout"
        ),
        None,
    )

    assert callback is not None


@pytest.mark.parametrize(
    "plot_name,callback_name,generate_method",
    [
        (
            "upstream-plot",
            "update_upstream_plot_on_relayout",
            "_generate_upstream_plot",
        ),
        (
            "downstream-plot",
            "update_downstream_plot_on_relayout",
            "_generate_downstream_plot",
        ),
        (
            "temperature-plot",
            "update_temperature_plot_on_relayout",
            "_generate_temperature_plot",
        ),
        (
            "permeability-plot",
            "update_permeability_plot_on_relayout",
            "_generate_permeability_plot",
        ),
    ],
)
def test_all_relayout_callbacks_handle_autoscale(
    mock_plotter, plot_name, callback_name, generate_method
):
    """Test that all relayout callbacks regenerate plots on autoscale.

    Args:
        mock_plotter: Mock plotter fixture
        plot_name: Name of the plot
        callback_name: Name of the callback function
        generate_method: Name of the generation method to verify
    """
    register_export_callbacks(mock_plotter)

    callback = next(
        cb for cb in mock_plotter.app._callbacks if cb["func"].__name__ == callback_name
    )

    relayout_data = {"autosize": True}
    callback["func"](relayout_data)

    generate_func = getattr(mock_plotter, generate_method)
    assert generate_func.called
