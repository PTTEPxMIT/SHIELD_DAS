"""Plot control callbacks for the SHIELD Data Acquisition System.

This module manages plot display settings including:
- Scale modes (linear/log)
- Axis ranges (min/max values)
- Error bar visibility
- Valve time markers
"""

from dash.dependencies import Input, Output, State


def _normalize_range_values(*values):
    """Convert None or empty values to None for autosizing.

    Args:
        *values: Range values (min, max) that may be None

    Returns:
        list: Normalized values (None if input is None, otherwise unchanged)
    """
    return [val if val is not None else None for val in values]


def _create_plot_settings_callback(plot_type, generate_method):
    """Factory function to create consistent plot settings callbacks.

    Args:
        plot_type: "upstream" or "downstream"
        generate_method: Method to call for regenerating the plot

    Returns:
        function: Callback function handling all plot setting changes
    """

    def callback_func(
        x_scale,
        y_scale,
        x_min,
        x_max,
        y_min,
        y_max,
        show_error_bars,
        show_valve_times,
        current_fig,
        store_data,
    ):
        """Update plot with new display settings.

        Always regenerates the plot (no zoom preservation) when settings change.
        None values for min/max enable autosizing.

        Args:
            x_scale: "linear" or "log"
            y_scale: "linear" or "log"
            x_min: Minimum x value (None for auto)
            x_max: Maximum x value (None for auto)
            y_min: Minimum y value (None for auto)
            y_max: Maximum y value (None for auto)
            show_error_bars: Whether to display error bars
            show_valve_times: Whether to display valve time markers
            current_fig: Current figure state (unused, kept for compatibility)
            store_data: Stored settings (unused, kept for compatibility)

        Returns:
            tuple: (updated_figure, settings_store)
        """
        x_min, x_max, y_min, y_max = _normalize_range_values(x_min, x_max, y_min, y_max)

        new_store = {"y_scale": y_scale}

        updated_fig = generate_method(
            show_error_bars=bool(show_error_bars),
            show_valve_times=bool(show_valve_times),
            x_scale=x_scale,
            y_scale=y_scale,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
        )

        return [updated_fig, new_store]

    return callback_func


def _create_min_value_callback(scale_type):
    """Factory function for auto-updating min values based on scale mode.

    Linear scales default to 0, log scales to None (auto).

    Args:
        scale_type: "x" or "y"

    Returns:
        function: Callback function returning appropriate min value
    """

    def callback_func(scale_mode):
        """Set min value based on scale mode.

        Args:
            scale_mode: "linear" or "log"

        Returns:
            list: [0] for linear, [None] for log
        """
        return [0 if scale_mode == "linear" else None]

    return callback_func


def register_plot_control_callbacks(plotter):
    """Register all plot control callbacks.

    Provides:
    - 2 comprehensive settings callbacks (upstream, downstream)
    - 4 auto-min-value callbacks (x/y for both plots)

    Args:
        plotter: DataPlotter instance with app and plot generation methods
    """

    # === Plot Settings Callbacks ===

    @plotter.app.callback(
        [
            Output("upstream-plot", "figure", allow_duplicate=True),
            Output("upstream-settings-store", "data"),
        ],
        [
            Input("upstream-x-scale", "value"),
            Input("upstream-y-scale", "value"),
            Input("upstream-x-min", "value"),
            Input("upstream-x-max", "value"),
            Input("upstream-y-min", "value"),
            Input("upstream-y-max", "value"),
            Input("show-error-bars-upstream", "value"),
            Input("show-valve-times-upstream", "value"),
        ],
        [
            State("upstream-plot", "figure"),
            State("upstream-settings-store", "data"),
        ],
        prevent_initial_call=True,
    )
    def update_upstream_plot_settings(
        x_scale,
        y_scale,
        x_min,
        x_max,
        y_min,
        y_max,
        show_error_bars,
        show_valve_times,
        current_fig,
        store_data,
    ):
        """Apply all upstream plot display settings."""
        callback = _create_plot_settings_callback(
            "upstream", plotter._generate_upstream_plot
        )
        return callback(
            x_scale,
            y_scale,
            x_min,
            x_max,
            y_min,
            y_max,
            show_error_bars,
            show_valve_times,
            current_fig,
            store_data,
        )

    @plotter.app.callback(
        [
            Output("downstream-plot", "figure", allow_duplicate=True),
            Output("downstream-settings-store", "data"),
        ],
        [
            Input("downstream-x-scale", "value"),
            Input("downstream-y-scale", "value"),
            Input("downstream-x-min", "value"),
            Input("downstream-x-max", "value"),
            Input("downstream-y-min", "value"),
            Input("downstream-y-max", "value"),
            Input("show-error-bars-downstream", "value"),
            Input("show-valve-times-downstream", "value"),
        ],
        [
            State("downstream-plot", "figure"),
            State("downstream-settings-store", "data"),
        ],
        prevent_initial_call=True,
    )
    def update_downstream_plot_settings(
        x_scale,
        y_scale,
        x_min,
        x_max,
        y_min,
        y_max,
        show_error_bars,
        show_valve_times,
        current_fig,
        store_data,
    ):
        """Apply all downstream plot display settings."""
        callback = _create_plot_settings_callback(
            "downstream", plotter._generate_downstream_plot
        )
        return callback(
            x_scale,
            y_scale,
            x_min,
            x_max,
            y_min,
            y_max,
            show_error_bars,
            show_valve_times,
            current_fig,
            store_data,
        )

    # === Auto Min Value Callbacks ===
    # Linear scales default to 0, log scales to None (auto)

    @plotter.app.callback(
        [Output("upstream-x-min", "value")],
        [Input("upstream-x-scale", "value")],
        prevent_initial_call=True,
    )
    def update_upstream_x_min(x_scale):
        """Auto-set upstream X min: 0 for linear, None for log."""
        return _create_min_value_callback("x")(x_scale)

    @plotter.app.callback(
        [Output("upstream-y-min", "value")],
        [Input("upstream-y-scale", "value")],
        prevent_initial_call=True,
    )
    def update_upstream_y_min(y_scale):
        """Auto-set upstream Y min: 0 for linear, None for log."""
        return _create_min_value_callback("y")(y_scale)

    @plotter.app.callback(
        [Output("downstream-x-min", "value")],
        [Input("downstream-x-scale", "value")],
        prevent_initial_call=True,
    )
    def update_downstream_x_min(x_scale):
        """Auto-set downstream X min: 0 for linear, None for log."""
        return _create_min_value_callback("x")(x_scale)

    @plotter.app.callback(
        [Output("downstream-y-min", "value")],
        [Input("downstream-y-scale", "value")],
        prevent_initial_call=True,
    )
    def update_downstream_y_min(y_scale):
        """Auto-set downstream Y min: 0 for linear, None for log."""
        return _create_min_value_callback("y")(y_scale)
