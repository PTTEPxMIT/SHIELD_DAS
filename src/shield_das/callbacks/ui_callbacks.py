"""UI interaction callbacks for the SHIELD Data Acquisition System.

This module handles simple UI toggle interactions for collapsible sections.
All callbacks follow the same pattern: toggle open/closed state and update chevron icon.
"""

from dash import html
from dash.dependencies import Input, Output, State


def _create_chevron_icon(is_open):
    """Create a chevron icon based on collapse state.

    Args:
        is_open: Whether the section is expanded

    Returns:
        html.I: Dash HTML icon component (chevron-down if open, chevron-up if closed)
    """
    icon_class = "fas fa-chevron-down" if is_open else "fas fa-chevron-up"
    return html.I(className=icon_class)


def _create_toggle_callback(collapse_id, button_id):
    """Factory function to create a toggle callback with consistent behavior.

    Args:
        collapse_id: ID of the collapse component
        button_id: ID of the button triggering the toggle

    Returns:
        tuple: (callback decorator inputs, callback function)
    """

    def callback_func(n_clicks, is_open):
        """Toggle collapse state and update icon.

        Args:
            n_clicks: Number of times button was clicked
            is_open: Current collapse state

        Returns:
            tuple: (new_state, icon_component)
        """
        if not n_clicks:
            return is_open, _create_chevron_icon(is_open)

        new_state = not is_open
        return new_state, _create_chevron_icon(new_state)

    return callback_func


def register_ui_callbacks(plotter):
    """Register all UI interaction callbacks for collapsible sections.

    Creates three toggle callbacks with identical behavior:
    - Dataset management section
    - Upstream plot controls
    - Downstream plot controls

    Args:
        plotter: DataPlotter instance with app attribute
    """

    # Dataset management collapse toggle
    @plotter.app.callback(
        [
            Output("collapse-dataset", "is_open"),
            Output("collapse-dataset-button", "children"),
        ],
        [Input("collapse-dataset-button", "n_clicks")],
        [State("collapse-dataset", "is_open")],
        prevent_initial_call=True,
    )
    def toggle_dataset_collapse(n_clicks, is_open):
        """Toggle dataset management section visibility."""
        return _create_toggle_callback("collapse-dataset", "collapse-dataset-button")(
            n_clicks, is_open
        )

    # Upstream controls collapse toggle
    @plotter.app.callback(
        [
            Output("collapse-upstream-controls", "is_open"),
            Output("collapse-upstream-controls-button", "children"),
        ],
        [Input("collapse-upstream-controls-button", "n_clicks")],
        [State("collapse-upstream-controls", "is_open")],
        prevent_initial_call=True,
    )
    def toggle_upstream_controls_collapse(n_clicks, is_open):
        """Toggle upstream plot controls visibility."""
        return _create_toggle_callback(
            "collapse-upstream-controls", "collapse-upstream-controls-button"
        )(n_clicks, is_open)

    # Downstream controls collapse toggle
    @plotter.app.callback(
        [
            Output("collapse-downstream-controls", "is_open"),
            Output("collapse-downstream-controls-button", "children"),
        ],
        [Input("collapse-downstream-controls-button", "n_clicks")],
        [State("collapse-downstream-controls", "is_open")],
        prevent_initial_call=True,
    )
    def toggle_downstream_controls_collapse(n_clicks, is_open):
        """Toggle downstream plot controls visibility."""
        return _create_toggle_callback(
            "collapse-downstream-controls", "collapse-downstream-controls-button"
        )(n_clicks, is_open)

    # Temperature controls collapse toggle
    @plotter.app.callback(
        [
            Output("collapse-temperature-controls", "is_open"),
            Output("collapse-temperature-controls-button", "children"),
        ],
        [Input("collapse-temperature-controls-button", "n_clicks")],
        [State("collapse-temperature-controls", "is_open")],
        prevent_initial_call=True,
    )
    def toggle_temperature_controls_collapse(n_clicks, is_open):
        """Toggle temperature plot controls visibility."""
        return _create_toggle_callback(
            "collapse-temperature-controls", "collapse-temperature-controls-button"
        )(n_clicks, is_open)

    # Permeability controls collapse toggle
    @plotter.app.callback(
        [
            Output("collapse-permeability-controls", "is_open"),
            Output("collapse-permeability-controls-button", "children"),
        ],
        [Input("collapse-permeability-controls-button", "n_clicks")],
        [State("collapse-permeability-controls", "is_open")],
        prevent_initial_call=True,
    )
    def toggle_permeability_controls_collapse(n_clicks, is_open):
        """Toggle permeability plot controls visibility."""
        return _create_toggle_callback(
            "collapse-permeability-controls", "collapse-permeability-controls-button"
        )(n_clicks, is_open)
