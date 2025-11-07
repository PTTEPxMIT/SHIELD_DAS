"""Dataset management callbacks for the SHIELD Data Acquisition System.

This module handles all dataset CRUD operations:
- Updating dataset names and colors
- Adding new datasets from file paths
- Deleting existing datasets
- Downloading dataset files
- Toggling the add dataset UI section
"""

import json
import os

import dash
import dash_bootstrap_components as dbc
from dash import ALL, dcc
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

from .states import PLOT_CONTROL_STATES


def _get_triggered_index(ctx):
    """Extract dataset index from triggered callback context.

    Args:
        ctx: Dash callback context

    Returns:
        int: Index of the dataset that triggered the callback

    Raises:
        PreventUpdate: If context is invalid or index cannot be extracted
    """
    if not ctx.triggered:
        raise PreventUpdate

    button_id = ctx.triggered[0]["prop_id"]

    try:
        button_data = json.loads(button_id.split(".")[0])
        return int(button_data["index"])
    except (json.JSONDecodeError, KeyError, IndexError, ValueError):
        raise PreventUpdate


def _create_alert(message, color="danger", duration=3000):
    """Create a dismissable Bootstrap alert component.

    Args:
        message: Alert message text
        color: Bootstrap color (danger, success, warning, info)
        duration: Auto-dismiss duration in milliseconds

    Returns:
        dbc.Alert: Alert component
    """
    return dbc.Alert(message, color=color, dismissable=True, duration=duration)


def _validate_dataset_path(path):
    """Validate that a dataset path exists and contains required metadata.

    Args:
        path: File path to validate

    Returns:
        tuple: (is_valid, error_message)
            is_valid: True if path is valid
            error_message: Error description if invalid, None otherwise
    """
    if not os.path.exists(path):
        return False, "Path does not exist."

    metadata_path = os.path.join(path, "run_metadata.json")
    if not os.path.exists(metadata_path):
        return False, "run_metadata.json not found in dataset folder."

    return True, None


def register_dataset_callbacks(plotter):
    """Register all dataset management callbacks.

    Provides:
    - Name/color updates with plot regeneration
    - Add dataset with validation and error handling
    - Delete dataset with confirmation
    - Download dataset as zip/csv
    - Toggle add dataset section UI

    Args:
        plotter: DataPlotter instance with app, datasets, and helper methods
    """

    @plotter.app.callback(
        [
            Output("dataset-table-container", "children", allow_duplicate=True),
            Output("upstream-plot", "figure", allow_duplicate=True),
            Output("downstream-plot", "figure", allow_duplicate=True),
        ],
        [Input({"type": "dataset-name", "index": ALL}, "value")],
        PLOT_CONTROL_STATES,
        prevent_initial_call=True,
    )
    def update_dataset_names(
        names,
        show_error_bars_upstream,
        show_error_bars_downstream,
        show_valve_times_upstream,
        show_valve_times_downstream,
    ):
        """Update dataset names and regenerate table/plots.

        Only updates if names have changed to avoid unnecessary renders.

        Args:
            names: List of new names from input fields
            show_error_bars_upstream: Whether to show upstream error bars
            show_error_bars_downstream: Whether to show downstream error bars
            show_valve_times_upstream: Whether to show upstream valve markers
            show_valve_times_downstream: Whether to show downstream valve markers

        Returns:
            tuple: (updated_table, upstream_plot, downstream_plot)

        Raises:
            PreventUpdate: If names haven't changed
        """
        current_names = [ds.name for ds in plotter.datasets]

        if list(names) == current_names:
            raise PreventUpdate

        # Update only changed entries
        for i, name in enumerate(names):
            if i < len(plotter.datasets) and name and name != current_names[i]:
                plotter.datasets[i].name = name

        plots = plotter._generate_both_plots(
            show_error_bars_upstream=show_error_bars_upstream,
            show_error_bars_downstream=show_error_bars_downstream,
            show_valve_times_upstream=show_valve_times_upstream,
            show_valve_times_downstream=show_valve_times_downstream,
        )

        return [plotter.create_dataset_table(), *plots]

    @plotter.app.callback(
        [
            Output("dataset-table-container", "children", allow_duplicate=True),
            Output("upstream-plot", "figure", allow_duplicate=True),
            Output("downstream-plot", "figure", allow_duplicate=True),
            Output("temperature-plot", "figure", allow_duplicate=True),
        ],
        [Input({"type": "dataset-color", "index": ALL}, "value")],
        prevent_initial_call=True,
    )
    def update_dataset_colors(colors):
        """Update dataset colors and regenerate all plots.

        Args:
            colors: List of color values from color pickers

        Returns:
            tuple: (table, upstream_plot, downstream_plot, temperature_plot)
        """
        for i, color in enumerate(colors):
            if i < len(plotter.datasets) and color:
                plotter.datasets[i].colour = color

        return [
            plotter.create_dataset_table(),
            plotter._generate_upstream_plot(),
            plotter._generate_downstream_plot(),
            plotter._generate_temperature_plot(),
        ]

    @plotter.app.callback(
        [
            Output("dataset-table-container", "children", allow_duplicate=True),
            Output("upstream-plot", "figure", allow_duplicate=True),
            Output("downstream-plot", "figure", allow_duplicate=True),
            Output("new-dataset-path", "value"),
            Output("add-dataset-status", "children"),
        ],
        [Input("add-dataset-button", "n_clicks")],
        [State("new-dataset-path", "value")],
        prevent_initial_call=True,
    )
    def add_new_dataset(n_clicks, new_path):
        """Add a new dataset from file path with validation.

        Validates path existence and metadata presence before loading.
        Shows user-friendly error messages for common issues.

        Args:
            n_clicks: Number of button clicks (triggers callback)
            new_path: Path to dataset folder

        Returns:
            tuple: (table, upstream_plot, downstream_plot, cleared_input, status_alert)
        """
        if not n_clicks or not new_path:
            return [
                plotter.create_dataset_table(),
                plotter._generate_upstream_plot(),
                plotter._generate_downstream_plot(),
                new_path or "",
                "",
            ]

        # Validate path
        is_valid, error_msg = _validate_dataset_path(new_path)
        if not is_valid:
            return [
                plotter.create_dataset_table(),
                plotter._generate_upstream_plot(),
                plotter._generate_downstream_plot(),
                new_path,
                _create_alert(error_msg, duration=5000),
            ]

        # Generate dataset name from folder
        dataset_name = (
            os.path.basename(new_path) or f"dataset_{len(plotter.datasets) + 1}"
        )

        # Attempt to load dataset
        try:
            plotter.load_data(new_path, dataset_name)
            status = _create_alert(
                f"Dataset added successfully from {new_path}",
                color="success",
                duration=3000,
            )
            return [
                plotter.create_dataset_table(),
                plotter._generate_upstream_plot(),
                plotter._generate_downstream_plot(),
                "",  # Clear input field
                status,
            ]
        except Exception as e:
            msg = str(e) or "Unknown error while loading dataset"
            print(f"Error adding dataset from {new_path}: {msg}")
            return [
                plotter.create_dataset_table(),
                plotter._generate_upstream_plot(),
                plotter._generate_downstream_plot(),
                new_path,
                _create_alert(f"Failed to add dataset: {msg}", duration=7000),
            ]

    @plotter.app.callback(
        [
            Output("dataset-table-container", "children"),
            Output("upstream-plot", "figure"),
            Output("downstream-plot", "figure"),
            Output("temperature-plot", "figure"),
        ],
        [Input({"type": "delete-dataset", "index": ALL}, "n_clicks")],
        prevent_initial_call=True,
    )
    def delete_dataset(n_clicks_list):
        """Delete a dataset and regenerate all plots.

        Args:
            n_clicks_list: List of click counts for all delete buttons

        Returns:
            tuple: (table, upstream_plot, downstream_plot, temperature_plot)

        Raises:
            PreventUpdate: If no button was clicked
        """
        if not n_clicks_list or not any(n_clicks_list):
            raise PreventUpdate

        ctx = dash.callback_context
        delete_index = _get_triggered_index(ctx)

        if 0 <= delete_index < len(plotter.datasets):
            deleted = plotter.datasets.pop(delete_index)
            print(f"Deleted dataset: {deleted.name}")

        return [
            plotter.create_dataset_table(),
            plotter._generate_upstream_plot(),
            plotter._generate_downstream_plot(),
            plotter._generate_temperature_plot(),
        ]

    @plotter.app.callback(
        Output("download-dataset-output", "data", allow_duplicate=True),
        [Input({"type": "download-dataset", "index": ALL}, "n_clicks")],
        prevent_initial_call=True,
    )
    def download_dataset(n_clicks_list):
        """Package and download a dataset.

        Args:
            n_clicks_list: List of click counts for all download buttons

        Returns:
            dict or dcc.send_bytes: Download specification for zip or CSV

        Raises:
            PreventUpdate: If no button clicked or dataset not found
        """
        if not n_clicks_list or not any(n_clicks_list):
            raise PreventUpdate

        ctx = dash.callback_context
        download_index = _get_triggered_index(ctx)

        if 0 <= download_index < len(plotter.datasets):
            dataset = plotter.datasets[download_index]
            dataset_path = dataset.path

            if dataset_path and os.path.exists(dataset_path):
                packaged = plotter._create_dataset_download(dataset_path)
                if packaged:
                    content = packaged.get("content")
                    filename = packaged.get("filename")

                    if isinstance(content, bytes | bytearray):
                        return dcc.send_bytes(
                            lambda f, data=content: f.write(data), filename
                        )

                    return {
                        "content": content,
                        "filename": filename,
                        "type": packaged.get("type", "text/csv"),
                    }

        raise PreventUpdate

    @plotter.app.callback(
        [
            Output("collapse-add-dataset", "is_open"),
            Output("add-dataset-icon", "className"),
        ],
        [Input("toggle-add-dataset", "n_clicks")],
        [State("collapse-add-dataset", "is_open")],
        prevent_initial_call=True,
    )
    def toggle_add_dataset_section(n_clicks, is_open):
        """Toggle the add dataset form visibility.

        Args:
            n_clicks: Number of button clicks
            is_open: Current collapse state

        Returns:
            tuple: (new_is_open, icon_class)

        Raises:
            PreventUpdate: If button not clicked
        """
        if n_clicks is None:
            raise PreventUpdate

        new_is_open = not bool(is_open)
        new_icon_class = "fas fa-minus" if new_is_open else "fas fa-plus"

        print(
            f"toggle_add_dataset_section: clicked={n_clicks}, "
            f"is_open={is_open} -> {new_is_open}, icon={new_icon_class}"
        )

        return new_is_open, new_icon_class
