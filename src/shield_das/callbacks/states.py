"""Shared callback state definitions for the SHIELD Data Acquisition System.

This module contains reusable State lists that are used across multiple callbacks
to maintain consistent plot control settings.
"""

from dash.dependencies import State

# Shared callback state for plot control settings
# Used across multiple callback modules to maintain consistent plot state
PLOT_CONTROL_STATES = [
    State("show-error-bars-upstream", "value"),
    State("show-error-bars-downstream", "value"),
    State("show-valve-times-upstream", "value"),
    State("show-valve-times-downstream", "value"),
]
