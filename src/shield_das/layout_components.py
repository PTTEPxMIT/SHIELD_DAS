"""UI component builders for the SHIELD Data Plotter.

This module contains functions that generate Dash/Bootstrap components
for different sections of the plotter UI layout.
"""

import dash_bootstrap_components as dbc
from dash import dcc, html


# Common styles used across components
COLLAPSE_BUTTON_STYLE = {
    "border": "1px solid #dee2e6",
    "background-color": "#f8f9fa",
    "box-shadow": "0 1px 3px rgba(0,0,0,0.1)",
    "width": "30px",
    "height": "30px",
    "padding": "0",
    "display": "flex",
    "align-items": "center",
    "justify-content": "center",
}

SEPARATOR_HR_STYLE = {
    "flex": "1",
    "margin": "0",
    "border-top": "1px solid #dee2e6",
}


def create_header():
    """Create the main header/title section.

    Returns:
        dbc.Row: Bootstrap row containing the title
    """
    return dbc.Row(
        [
            dbc.Col(
                html.H1(
                    "SHIELD Data Visualisation",
                    className="text-center",
                    style={
                        "fontSize": "3.5rem",
                        "fontWeight": "standard",
                        "marginTop": "2rem",
                        "marginBottom": "2rem",
                        "color": "#2c3e50",
                    },
                ),
                width=12,
            ),
        ],
        className="mb-4",
    )


def create_dataset_management_card(dataset_table_html):
    """Create the dataset management card with table and add dataset form.

    Args:
        dataset_table_html: HTML content for the dataset table

    Returns:
        dbc.Row: Bootstrap row containing the dataset management card
    """
    return dbc.Row(
        [
            dbc.Col(
                [
                    dbc.Card(
                        [
                            dbc.CardHeader(
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            "Dataset Management",
                                            className="d-flex align-items-center",
                                        ),
                                        dbc.Col(
                                            dbc.Button(
                                                html.I(className="fas fa-chevron-up"),
                                                id="collapse-dataset-button",
                                                color="light",
                                                size="sm",
                                                className="ms-auto",
                                                style=COLLAPSE_BUTTON_STYLE,
                                            ),
                                            width="auto",
                                            className="d-flex justify-content-end",
                                        ),
                                    ],
                                    className="g-0 align-items-center",
                                )
                            ),
                            dbc.Collapse(
                                dbc.CardBody(
                                    [
                                        # Dataset table
                                        html.Div(
                                            id="dataset-table-container",
                                            children=dataset_table_html,
                                        ),
                                        # Collapsible Add Dataset Section
                                        html.Div(
                                            [
                                                # Separator with centered plus button
                                                html.Div(
                                                    [
                                                        html.Hr(
                                                            style=SEPARATOR_HR_STYLE
                                                        ),
                                                        dbc.Button(
                                                            html.I(
                                                                id="add-dataset-icon",
                                                                className="fas fa-plus",
                                                            ),
                                                            id="toggle-add-dataset",
                                                            color="light",
                                                            size="sm",
                                                            style={
                                                                "margin": "0 10px",
                                                                "border-radius": "50%",
                                                                "width": "32px",
                                                                "height": "32px",
                                                                "padding": "0",
                                                                "border": (
                                                                    "1px solid #dee2e6"
                                                                ),
                                                            },
                                                            title="Add new dataset",
                                                        ),
                                                        html.Hr(
                                                            style=SEPARATOR_HR_STYLE
                                                        ),
                                                    ],
                                                    style={
                                                        "display": "flex",
                                                        "align-items": "center",
                                                        "margin": "20px 0 15px 0",
                                                    },
                                                ),
                                                # Collapsible add dataset form
                                                dbc.Collapse(
                                                    [
                                                        dbc.Row(
                                                            [
                                                                dbc.Col(
                                                                    [
                                                                        dbc.Input(
                                                                            id="new-dataset-path",
                                                                            type="text",
                                                                            placeholder=(
                                                                                "Enter dataset "
                                                                                "folder path..."
                                                                            ),
                                                                            style={
                                                                                "margin-bottom": "10px"
                                                                            },
                                                                        ),
                                                                    ],
                                                                    width=9,
                                                                ),
                                                                dbc.Col(
                                                                    [
                                                                        dbc.Button(
                                                                            [
                                                                                html.I(
                                                                                    className=(
                                                                                        "fas fa-plus me-2"
                                                                                    )
                                                                                ),
                                                                                "Add Dataset",
                                                                            ],
                                                                            id="add-dataset-button",
                                                                            color="primary",
                                                                            style={
                                                                                "width": "100%"
                                                                            },
                                                                        ),
                                                                    ],
                                                                    width=3,
                                                                ),
                                                            ],
                                                            className="g-2",
                                                        ),
                                                        # Status message
                                                        html.Div(
                                                            id="add-dataset-status",
                                                            style={
                                                                "margin-top": "10px"
                                                            },
                                                        ),
                                                    ],
                                                    id="collapse-add-dataset",
                                                    is_open=False,
                                                ),
                                            ]
                                        ),
                                    ]
                                ),
                                id="collapse-dataset",
                                is_open=True,
                            ),
                        ]
                    ),
                ],
                width=12,
            ),
        ],
        className="mb-3",
    )


def create_hidden_stores():
    """Create hidden Dash stores and status divs.

    Returns:
        list: List of Dash components (Store and Div elements)
    """
    return [
        # Hidden store to trigger plot updates
        dcc.Store(id="datasets-store"),
        # Hidden stores for plot settings
        dcc.Store(id="upstream-settings-store", data={}),
        dcc.Store(id="downstream-settings-store", data={}),
        # Status message for upload feedback (floating)
        html.Div(
            id="upload-status",
            style={
                "position": "fixed",
                "top": "20px",
                "right": "20px",
                "zIndex": "9999",
                "maxWidth": "400px",
                "minWidth": "300px",
            },
        ),
    ]


def create_pressure_plots_row(upstream_figure, downstream_figure):
    """Create the row containing upstream and downstream pressure plots.

    Args:
        upstream_figure: Plotly figure for upstream pressure
        downstream_figure: Plotly figure for downstream pressure

    Returns:
        dbc.Row: Bootstrap row containing both pressure plots
    """
    return dbc.Row(
        [
            dbc.Col(
                [
                    dbc.Card(
                        [
                            dbc.CardHeader("Upstream Pressure"),
                            dbc.CardBody(
                                [
                                    dcc.Graph(
                                        id="upstream-plot",
                                        figure=upstream_figure,
                                    )
                                ]
                            ),
                        ]
                    ),
                ],
                width=6,
            ),
            dbc.Col(
                [
                    dbc.Card(
                        [
                            dbc.CardHeader("Downstream Pressure"),
                            dbc.CardBody(
                                [
                                    dcc.Graph(
                                        id="downstream-plot",
                                        figure=downstream_figure,
                                    )
                                ]
                            ),
                        ]
                    ),
                ],
                width=6,
            ),
        ]
    )


def create_plot_controls_card(
    plot_name: str,
    card_title: str,
    collapse_button_id: str,
    collapse_id: str,
    x_scale_id: str,
    x_min_id: str,
    x_max_id: str,
    y_scale_id: str,
    y_min_id: str,
    y_max_id: str,
    show_error_bars_id: str,
    show_valve_times_id: str,
    export_button_id: str,
):
    """Create a plot controls card (used for both upstream and downstream).

    Args:
        plot_name: Name of the plot (e.g., "Upstream", "Downstream")
        card_title: Title for the card header
        collapse_button_id: ID for collapse toggle button
        collapse_id: ID for collapse component
        x_scale_id: ID for x-axis scale radio buttons
        x_min_id: ID for x-axis min input
        x_max_id: ID for x-axis max input
        y_scale_id: ID for y-axis scale radio buttons
        y_min_id: ID for y-axis min input
        y_max_id: ID for y-axis max input
        show_error_bars_id: ID for error bars checkbox
        show_valve_times_id: ID for valve times checkbox
        export_button_id: ID for export button

    Returns:
        dbc.Card: Bootstrap card containing plot controls
    """
    return dbc.Card(
        [
            dbc.CardHeader(
                dbc.Row(
                    [
                        dbc.Col(
                            card_title,
                            className="d-flex align-items-center",
                        ),
                        dbc.Col(
                            dbc.Button(
                                html.I(className="fas fa-chevron-up"),
                                id=collapse_button_id,
                                color="light",
                                size="sm",
                                className="ms-auto",
                                style=COLLAPSE_BUTTON_STYLE,
                            ),
                            width="auto",
                            className="d-flex justify-content-end",
                        ),
                    ],
                    className="g-0 align-items-center",
                )
            ),
            dbc.Collapse(
                dbc.CardBody(
                    [
                        dbc.Row(
                            [
                                # X-axis controls
                                dbc.Col(
                                    [
                                        html.H6("X-Axis", className="mb-2"),
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    [
                                                        dbc.Label("Scale:"),
                                                        dbc.RadioItems(
                                                            id=x_scale_id,
                                                            options=[
                                                                {
                                                                    "label": "Linear",
                                                                    "value": "linear",
                                                                },
                                                                {
                                                                    "label": "Log",
                                                                    "value": "log",
                                                                },
                                                            ],
                                                            value="linear",
                                                            inline=True,
                                                        ),
                                                    ],
                                                    width=12,
                                                ),
                                            ],
                                            className="mb-2",
                                        ),
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    [
                                                        dbc.Label("Min:"),
                                                        dbc.Input(
                                                            id=x_min_id,
                                                            type="number",
                                                            placeholder="Auto",
                                                            value=0,
                                                            size="sm",
                                                        ),
                                                    ],
                                                    width=6,
                                                ),
                                                dbc.Col(
                                                    [
                                                        dbc.Label("Max:"),
                                                        dbc.Input(
                                                            id=x_max_id,
                                                            type="number",
                                                            placeholder="Auto",
                                                            size="sm",
                                                        ),
                                                    ],
                                                    width=6,
                                                ),
                                            ]
                                        ),
                                    ],
                                    width=6,
                                ),
                                # Y-axis controls
                                dbc.Col(
                                    [
                                        html.H6("Y-Axis", className="mb-2"),
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    [
                                                        dbc.Label("Scale:"),
                                                        dbc.RadioItems(
                                                            id=y_scale_id,
                                                            options=[
                                                                {
                                                                    "label": "Linear",
                                                                    "value": "linear",
                                                                },
                                                                {
                                                                    "label": "Log",
                                                                    "value": "log",
                                                                },
                                                            ],
                                                            value="linear",
                                                            inline=True,
                                                        ),
                                                    ],
                                                    width=12,
                                                ),
                                            ],
                                            className="mb-2",
                                        ),
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    [
                                                        dbc.Label("Min:"),
                                                        dbc.Input(
                                                            id=y_min_id,
                                                            type="number",
                                                            placeholder="Auto",
                                                            value=0,
                                                            size="sm",
                                                        ),
                                                    ],
                                                    width=6,
                                                ),
                                                dbc.Col(
                                                    [
                                                        dbc.Label("Max:"),
                                                        dbc.Input(
                                                            id=y_max_id,
                                                            type="number",
                                                            placeholder="Auto",
                                                            size="sm",
                                                        ),
                                                    ],
                                                    width=6,
                                                ),
                                            ]
                                        ),
                                    ],
                                    width=6,
                                ),
                            ]
                        ),
                        # Options Row
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        html.H6(
                                            "Options",
                                            className="mb-2 mt-3",
                                        ),
                                        dbc.Checkbox(
                                            id=show_error_bars_id,
                                            label="Show error bars",
                                            value=True,
                                            className="mb-2",
                                        ),
                                        dbc.Checkbox(
                                            id=show_valve_times_id,
                                            label="Show valve operation times",
                                            value=False,
                                            className="mb-2",
                                        ),
                                        html.Hr(className="my-2"),
                                        dbc.Button(
                                            [
                                                html.I(
                                                    className="fas fa-download me-2"
                                                ),
                                                f"Export {plot_name} Plot",
                                            ],
                                            id=export_button_id,
                                            color="outline-secondary",
                                            size="sm",
                                            className="w-100",
                                        ),
                                    ],
                                    width=12,
                                ),
                            ]
                        ),
                    ]
                ),
                id=collapse_id,
                is_open=False,
            ),
        ]
    )


def create_upstream_controls_card():
    """Create the upstream plot controls card.

    Returns:
        dbc.Card: Bootstrap card with upstream plot controls
    """
    return create_plot_controls_card(
        plot_name="Upstream",
        card_title="Upstream Plot Controls",
        collapse_button_id="collapse-upstream-controls-button",
        collapse_id="collapse-upstream-controls",
        x_scale_id="upstream-x-scale",
        x_min_id="upstream-x-min",
        x_max_id="upstream-x-max",
        y_scale_id="upstream-y-scale",
        y_min_id="upstream-y-min",
        y_max_id="upstream-y-max",
        show_error_bars_id="show-error-bars-upstream",
        show_valve_times_id="show-valve-times-upstream",
        export_button_id="export-upstream-plot",
    )


def create_downstream_controls_card():
    """Create the downstream plot controls card.

    Returns:
        dbc.Card: Bootstrap card with downstream plot controls
    """
    return create_plot_controls_card(
        plot_name="Downstream",
        card_title="Downstream Plot Controls",
        collapse_button_id="collapse-downstream-controls-button",
        collapse_id="collapse-downstream-controls",
        x_scale_id="downstream-x-scale",
        x_min_id="downstream-x-min",
        x_max_id="downstream-x-max",
        y_scale_id="downstream-y-scale",
        y_min_id="downstream-y-min",
        y_max_id="downstream-y-max",
        show_error_bars_id="show-error-bars-downstream",
        show_valve_times_id="show-valve-times-downstream",
        export_button_id="export-downstream-plot",
    )


def create_plot_controls_row():
    """Create the row containing both upstream and downstream control cards.

    Returns:
        dbc.Row: Bootstrap row with control cards for both plots
    """
    return dbc.Row(
        [
            dbc.Col([create_upstream_controls_card()], width=6),
            dbc.Col([create_downstream_controls_card()], width=6),
        ],
        className="mt-3",
    )


def create_temperature_plot_card(temperature_figure):
    """Create the temperature plot card.

    Args:
        temperature_figure: Plotly figure for temperature plot

    Returns:
        dbc.Row: Bootstrap row containing temperature plot card
    """
    return dbc.Row(
        [
            dbc.Col(
                [
                    dbc.Card(
                        [
                            dbc.CardHeader("Temperature Data"),
                            dbc.CardBody(
                                [
                                    dcc.Graph(
                                        id="temperature-plot",
                                        figure=temperature_figure,
                                    )
                                ]
                            ),
                        ]
                    ),
                ],
                width=12,
            ),
        ],
        className="mt-3",
    )


def create_permeability_plot_card(permeability_figure):
    """Create the permeability plot card.

    Args:
        permeability_figure: Plotly figure for permeability plot

    Returns:
        dbc.Row: Bootstrap row containing permeability plot card
    """
    return dbc.Row(
        [
            dbc.Col(width=3),  # Left spacing
            dbc.Col(
                [
                    dbc.Card(
                        [
                            dbc.CardHeader("Measured Permeability"),
                            dbc.CardBody(
                                [
                                    dcc.Graph(
                                        id="permeability-plot",
                                        figure=permeability_figure,
                                    )
                                ]
                            ),
                        ]
                    ),
                ],
                width=6,
            ),
            dbc.Col(width=3),  # Right spacing
        ],
        className="mt-3",
    )


def create_bottom_spacing():
    """Create whitespace at the bottom of the page.

    Returns:
        dbc.Row: Bootstrap row with spacing
    """
    return dbc.Row(
        [
            dbc.Col(
                html.Div(style={"height": "100px"}),
                width=12,
            ),
        ],
    )


def create_download_components():
    """Create all Download components for data and plot exports.

    Returns:
        list: List of dcc.Download components
    """
    return [
        # Download component for dataset downloads
        dcc.Download(id="download-dataset-output"),
        # Download components for plot exports
        dcc.Download(id="download-upstream-plot"),
        dcc.Download(id="download-downstream-plot"),
        dcc.Download(id="download-temperature-plot"),
        dcc.Download(id="download-permeability-plot"),
    ]


def create_live_data_interval():
    """Create the interval component for live data updates.

    Returns:
        dcc.Interval: Dash interval component
    """
    return dcc.Interval(
        id="live-data-interval",
        interval=1000,  # Update every 1 second
        n_intervals=0,
        disabled=True,  # Start disabled, enable when needed
    )
