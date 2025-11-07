"""Export and plot interaction callbacks for the SHIELD Data Acquisition System.

This module handles:
- Exporting plots to HTML files
- Interactive zoom/pan using FigureResampler for efficient large dataset rendering
"""

import dash
import plotly.graph_objects as go
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate


def _is_autoscale_event(relayout_data):
    """Check if relayout event is an autoscale/reset zoom (double-click).

    Args:
        relayout_data: Plotly relayout event data dictionary

    Returns:
        bool: True if this is an autoscale event, False for normal zoom/pan
    """
    if not relayout_data:
        return False

    autoscale_keys = ["autosize", "xaxis.autorange", "yaxis.autorange"]
    return any(
        key in relayout_data or relayout_data.get(key) is True for key in autoscale_keys
    )


def _export_figure_to_html(figure, filename):
    """Convert a Plotly figure to HTML download format.

    Args:
        figure: Plotly figure object or dict
        filename: Name for the downloaded HTML file

    Returns:
        dict: Download specification with content, filename, and type

    Raises:
        PreventUpdate: If figure conversion fails
    """
    try:
        if not isinstance(figure, go.Figure):
            figure = go.Figure(figure)
        html_str = figure.to_html(include_plotlyjs="inline")
        return {"content": html_str, "filename": filename, "type": "text/html"}
    except Exception:
        raise PreventUpdate


def _create_relayout_callback(plot_id, generate_method):
    """Factory function to create consistent relayout callbacks for FigureResampler.

    Args:
        plot_id: ID of the plot (e.g., "upstream-plot")
        generate_method: Method to call for regenerating the plot

    Returns:
        function: Callback function handling zoom/pan with resampling
    """

    def callback_func(plotter, relayout_data):
        """Handle plot relayout events (zoom, pan, autoscale).

        On autoscale (double-click): regenerate full plot
        On zoom/pan: use FigureResampler for efficient rendering

        Args:
            plotter: DataPlotter instance
            relayout_data: Plotly relayout event data

        Returns:
            Updated figure or dash.no_update
        """
        if plot_id not in plotter.figure_resamplers or not relayout_data:
            return dash.no_update

        if _is_autoscale_event(relayout_data):
            # Regenerate full plot for autoscale
            return generate_method()

        # Use resampler for efficient zoom/pan
        fig_resampler = plotter.figure_resamplers[plot_id]
        return fig_resampler.construct_update_data_patch(relayout_data)

    return callback_func


def register_export_callbacks(plotter):
    """Register all export and plot interaction callbacks.

    Provides:
    - HTML export for upstream and downstream plots
    - FigureResampler integration for 4 plots
      (upstream, downstream, temperature, permeability)

    Args:
        plotter: DataPlotter instance with app and plot generation methods
    """

    # === Export Callbacks ===

    @plotter.app.callback(
        Output("download-upstream-plot", "data", allow_duplicate=True),
        [Input("export-upstream-plot", "n_clicks")],
        [State("upstream-plot", "figure")],
        prevent_initial_call=True,
    )
    def export_upstream_plot(n_clicks, current_fig):
        """Export current upstream plot view to HTML file."""
        if not n_clicks or not current_fig:
            raise PreventUpdate
        return _export_figure_to_html(current_fig, "upstream_plot.html")

    @plotter.app.callback(
        Output("download-downstream-plot", "data", allow_duplicate=True),
        [Input("export-downstream-plot", "n_clicks")],
        prevent_initial_call=True,
    )
    def export_downstream_plot(n_clicks):
        """Export downstream plot with full data (no resampling) to HTML."""
        if not n_clicks:
            raise PreventUpdate
        fig = plotter._generate_downstream_plot_full_data(False)
        return _export_figure_to_html(fig, "downstream_plot_full_data.html")

    # === FigureResampler Interactive Callbacks ===

    @plotter.app.callback(
        Output("upstream-plot", "figure", allow_duplicate=True),
        Input("upstream-plot", "relayoutData"),
        prevent_initial_call=True,
    )
    def update_upstream_plot_on_relayout(relayout_data):
        """Handle upstream plot zoom/pan with resampling."""
        callback = _create_relayout_callback(
            "upstream-plot", plotter._generate_upstream_plot
        )
        return callback(plotter, relayout_data)

    @plotter.app.callback(
        Output("downstream-plot", "figure", allow_duplicate=True),
        Input("downstream-plot", "relayoutData"),
        prevent_initial_call=True,
    )
    def update_downstream_plot_on_relayout(relayout_data):
        """Handle downstream plot zoom/pan with resampling."""
        callback = _create_relayout_callback(
            "downstream-plot", plotter._generate_downstream_plot
        )
        return callback(plotter, relayout_data)

    @plotter.app.callback(
        Output("temperature-plot", "figure", allow_duplicate=True),
        Input("temperature-plot", "relayoutData"),
        prevent_initial_call=True,
    )
    def update_temperature_plot_on_relayout(relayout_data):
        """Handle temperature plot zoom/pan with resampling."""
        callback = _create_relayout_callback(
            "temperature-plot", plotter._generate_temperature_plot
        )
        return callback(plotter, relayout_data)

    @plotter.app.callback(
        Output("permeability-plot", "figure", allow_duplicate=True),
        Input("permeability-plot", "relayoutData"),
        prevent_initial_call=True,
    )
    def update_permeability_plot_on_relayout(relayout_data):
        """Handle permeability plot zoom/pan with resampling."""
        callback = _create_relayout_callback(
            "permeability-plot", plotter._generate_permeability_plot
        )
        return callback(plotter, relayout_data)
