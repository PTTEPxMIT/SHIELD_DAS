"""Live data callbacks for the SHIELD Data Acquisition System.

This module handles dynamic data updates for ongoing experiments:
- Enabling/disabling live data monitoring per dataset
- Periodic automatic data refreshing via interval component
"""

from dash.dependencies import ALL, Input, Output, State
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
        [
            *PLOT_CONTROL_STATES,
            # Upstream plot settings
            State("upstream-x-scale", "value"),
            State("upstream-y-scale", "value"),
            State("upstream-x-min", "value"),
            State("upstream-x-max", "value"),
            State("upstream-y-min", "value"),
            State("upstream-y-max", "value"),
            # Downstream plot settings
            State("downstream-x-scale", "value"),
            State("downstream-y-scale", "value"),
            State("downstream-x-min", "value"),
            State("downstream-x-max", "value"),
            State("downstream-y-min", "value"),
            State("downstream-y-max", "value"),
        ],
        prevent_initial_call=True,
    )
    def update_live_data(
        n_intervals,
        show_error_bars_upstream,
        show_error_bars_downstream,
        show_valve_times_upstream,
        show_valve_times_downstream,
        upstream_x_scale,
        upstream_y_scale,
        upstream_x_min,
        upstream_x_max,
        upstream_y_min,
        upstream_y_max,
        downstream_x_scale,
        downstream_y_scale,
        downstream_x_min,
        downstream_x_max,
        downstream_y_min,
        downstream_y_max,
    ):
        """Periodically refresh data for live datasets.

        Called automatically by the interval component when enabled.
        Reprocesses data for all datasets with live_data=True and regenerates plots.
        Preserves all current plot settings (scale, ranges, error bars, valve times).

        Args:
            n_intervals: Number of intervals elapsed (unused, just triggers callback)
            show_error_bars_upstream: Whether to show upstream error bars
            show_error_bars_downstream: Whether to show downstream error bars
            show_valve_times_upstream: Whether to show upstream valve markers
            show_valve_times_downstream: Whether to show downstream valve markers
            upstream_x_scale: Upstream X-axis scale mode
            upstream_y_scale: Upstream Y-axis scale mode
            upstream_x_min: Upstream X-axis minimum value
            upstream_x_max: Upstream X-axis maximum value
            upstream_y_min: Upstream Y-axis minimum value
            upstream_y_max: Upstream Y-axis maximum value
            downstream_x_scale: Downstream X-axis scale mode
            downstream_y_scale: Downstream Y-axis scale mode
            downstream_x_min: Downstream X-axis minimum value
            downstream_x_max: Downstream X-axis maximum value
            downstream_y_min: Downstream Y-axis minimum value
            downstream_y_max: Downstream Y-axis maximum value

        Returns:
            tuple: (upstream_plot, downstream_plot, temperature_plot)

        Raises:
            PreventUpdate: If no datasets have live data enabled
        """
        if not _refresh_live_datasets(plotter.datasets):
            raise PreventUpdate

        # Generate upstream plot with current settings
        upstream_plot = plotter._generate_upstream_plot(
            show_error_bars=show_error_bars_upstream,
            show_valve_times=show_valve_times_upstream,
            x_scale=upstream_x_scale,
            y_scale=upstream_y_scale,
            x_min=upstream_x_min,
            x_max=upstream_x_max,
            y_min=upstream_y_min,
            y_max=upstream_y_max,
        )

        # Generate downstream plot with current settings
        downstream_plot = plotter._generate_downstream_plot(
            show_error_bars=show_error_bars_downstream,
            show_valve_times=show_valve_times_downstream,
            x_scale=downstream_x_scale,
            y_scale=downstream_y_scale,
            x_min=downstream_x_min,
            x_max=downstream_x_max,
            y_min=downstream_y_min,
            y_max=downstream_y_max,
        )

        # Generate temperature plot
        temperature_plot = plotter._generate_temperature_plot()

        return [upstream_plot, downstream_plot, temperature_plot]
