"""Live data callbacks for the SHIELD Data Acquisition System.

This module handles dynamic data updates for ongoing experiments:
- Enabling/disabling live data monitoring per dataset
- Periodic automatic data refreshing via interval component
"""

from dash.dependencies import ALL, Input, Output
from dash.exceptions import PreventUpdate

from .states import PLOT_CONTROL_STATES


def _update_dataset_live_flags(datasets, live_data_values):
    """Update live_data flag for each dataset based on checkbox values.

    Args:
        datasets: List of Dataset instances
        live_data_values: List of boolean values from checkboxes

    Returns:
        bool: True if any dataset has live data enabled
    """
    for i, is_live in enumerate(live_data_values):
        if i < len(datasets):
            datasets[i].live_data = bool(is_live)

    return any(live_data_values) if live_data_values else False


def _refresh_live_datasets(datasets):
    """Reload data for all datasets with live monitoring enabled.

    Args:
        datasets: List of Dataset instances

    Returns:
        bool: True if any dataset was refreshed
    """
    has_live_data = any(dataset.live_data for dataset in datasets)

    if not has_live_data:
        return False

    for dataset in datasets:
        if dataset.live_data:
            dataset.process_data()

    return True


def register_live_data_callbacks(plotter):
    """Register all live data monitoring callbacks.

    Provides:
    - Live data checkbox toggle (enables/disables interval updates)
    - Periodic data refresh (triggered by interval component)

    Args:
        plotter: DataPlotter instance with app and plot generation methods
    """

    @plotter.app.callback(
        [
            Output("upstream-plot", "figure", allow_duplicate=True),
            Output("downstream-plot", "figure", allow_duplicate=True),
            Output("temperature-plot", "figure", allow_duplicate=True),
            Output("live-data-interval", "disabled"),
        ],
        [Input({"type": "dataset-live-data", "index": ALL}, "value")],
        PLOT_CONTROL_STATES,
        prevent_initial_call=True,
    )
    def handle_live_data_toggle(
        live_data_values,
        show_error_bars_upstream,
        show_error_bars_downstream,
        show_valve_times_upstream,
        show_valve_times_downstream,
    ):
        """Toggle live data monitoring for datasets.

        Updates dataset live_data flags and regenerates plots.
        Enables/disables the interval component based on whether any dataset is live.

        Args:
            live_data_values: List of checkbox states (one per dataset)
            show_error_bars_upstream: Whether to show upstream error bars
            show_error_bars_downstream: Whether to show downstream error bars
            show_valve_times_upstream: Whether to show upstream valve markers
            show_valve_times_downstream: Whether to show downstream valve markers

        Returns:
            tuple: (upstream_plot, downstream_plot, temperature_plot, interval_disabled)
        """
        any_live = _update_dataset_live_flags(plotter.datasets, live_data_values)

        plots = plotter._generate_both_plots(
            show_error_bars_upstream=show_error_bars_upstream,
            show_error_bars_downstream=show_error_bars_downstream,
            show_valve_times_upstream=show_valve_times_upstream,
            show_valve_times_downstream=show_valve_times_downstream,
        )

        # Disable interval if no live data, enable if any dataset is live
        return [*plots, not any_live]

    @plotter.app.callback(
        [
            Output("upstream-plot", "figure", allow_duplicate=True),
            Output("downstream-plot", "figure", allow_duplicate=True),
            Output("temperature-plot", "figure", allow_duplicate=True),
        ],
        [Input("live-data-interval", "n_intervals")],
        PLOT_CONTROL_STATES,
        prevent_initial_call=True,
    )
    def update_live_data(
        n_intervals,
        show_error_bars_upstream,
        show_error_bars_downstream,
        show_valve_times_upstream,
        show_valve_times_downstream,
    ):
        """Periodically refresh data for live datasets.

        Called automatically by the interval component when enabled.
        Reprocesses data for all datasets with live_data=True and regenerates plots.

        Args:
            n_intervals: Number of intervals elapsed (unused, just triggers callback)
            show_error_bars_upstream: Whether to show upstream error bars
            show_error_bars_downstream: Whether to show downstream error bars
            show_valve_times_upstream: Whether to show upstream valve markers
            show_valve_times_downstream: Whether to show downstream valve markers

        Returns:
            tuple: (upstream_plot, downstream_plot, temperature_plot)

        Raises:
            PreventUpdate: If no datasets have live data enabled
        """
        if not _refresh_live_datasets(plotter.datasets):
            raise PreventUpdate

        return plotter._generate_both_plots(
            show_error_bars_upstream=show_error_bars_upstream,
            show_error_bars_downstream=show_error_bars_downstream,
            show_valve_times_upstream=show_valve_times_upstream,
            show_valve_times_downstream=show_valve_times_downstream,
        )
