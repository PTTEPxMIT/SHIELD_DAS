"""Dataset table component for the SHIELD Data Acquisition System.

This module provides the DatasetTable class which handles the creation and
rendering of the dataset management table in the Dash UI.
"""

import dash_bootstrap_components as dbc
from dash import dcc, html


class DatasetTable:
    """Creates and manages the dataset table component.

    This class encapsulates all logic for rendering the dataset management
    table, including editable name fields, color pickers, live data checkboxes,
    and action buttons.

    Args:
        datasets: List of Dataset objects to display in the table
    """

    def __init__(self, datasets):
        self.datasets = datasets

    def create(self):
        """Create the complete dataset table HTML component.

        Returns:
            html.Div: The table wrapped in a Div container
        """
        rows = [self._create_header()]

        # Add dataset rows
        for i, dataset in enumerate(self.datasets):
            rows.append(self._create_row(i, dataset))

        # Create the table
        return html.Div(
            [
                html.Table(
                    rows,
                    className="table table-striped table-hover",
                    style={
                        "margin": "0",
                        "border": "1px solid #dee2e6",
                        "border-radius": "8px",
                        "overflow": "hidden",
                    },
                )
            ]
        )

    def _create_header(self):
        """Create the header row for the dataset table.

        Returns:
            html.Tr: Table header row with column names
        """
        # Common header cell style
        header_style = {
            "text-align": "center",
            "padding": "2px",
            "font-weight": "normal",
        }

        return html.Tr(
            [
                html.Th(
                    "Dataset Name",
                    style={**header_style, "text-align": "left", "width": "43.75%"},
                ),
                html.Th(
                    "Dataset Path",
                    style={**header_style, "text-align": "left", "width": "43.75%"},
                ),
                html.Th("Live", style={**header_style, "width": "2.5%"}),
                html.Th("Colour", style={**header_style, "width": "5%"}),
                html.Th("", style={**header_style, "width": "2.5%"}),  # Download
                html.Th("", style={**header_style, "width": "2.5%"}),  # Delete
            ]
        )

    def _create_row(self, index, dataset):
        """Create a table row for a single dataset.

        Args:
            index: Dataset index in the list
            dataset: Dataset object to display

        Returns:
            html.Tr: Complete table row with all cells
        """
        return html.Tr(
            [
                self._create_name_cell(index, dataset),
                self._create_path_cell(dataset),
                self._create_live_data_cell(index, dataset),
                self._create_color_cell(index, dataset),
                self._create_action_button_cell(index, dataset, "download"),
                self._create_action_button_cell(index, dataset, "delete"),
            ]
        )

    def _create_name_cell(self, index, dataset):
        """Create the dataset name input cell.

        Args:
            index: Dataset index
            dataset: Dataset object

        Returns:
            html.Td: Table cell with editable name input
        """
        return html.Td(
            dcc.Input(
                id={"type": "dataset-name", "index": index},
                value=dataset.name,
                style={
                    "width": "95%",
                    "border": "1px solid #ccc",
                    "padding": "4px",
                    "border-radius": "4px",
                    "transition": "all 0.2s ease",
                },
                className="dataset-name-input",
            ),
            style={"padding": "2px", "border": "none"},
        )

    def _create_path_cell(self, dataset):
        """Create the dataset path display cell.

        Args:
            dataset: Dataset object

        Returns:
            html.Td: Table cell with path display
        """
        return html.Td(
            html.Div(
                html.Span(
                    dataset.path,
                    style={
                        "font-family": "monospace",
                        "font-size": "0.9em",
                        "color": "#666",
                        "word-break": "break-all",
                    },
                    title=dataset.path,
                ),
                style={
                    "width": "100%",
                    "padding": "4px",
                    "min-height": "1.5em",
                    "display": "flex",
                    "align-items": "center",
                },
            ),
            style={"padding": "4px", "border": "none"},
        )

    def _create_live_data_cell(self, index, dataset):
        """Create the live data checkbox cell.

        Args:
            index: Dataset index
            dataset: Dataset object

        Returns:
            html.Td: Table cell with live data checkbox
        """
        return html.Td(
            html.Div(
                dbc.Checkbox(
                    id={"type": "dataset-live-data", "index": index},
                    value=dataset.live_data,
                    style={"transform": "scale(1.2)", "display": "inline-block"},
                ),
                style={"margin-left": "15px"},
            ),
            style={"padding": "4px", "text-align": "center", "border": "none"},
        )

    def _create_color_cell(self, index, dataset):
        """Create the color picker cell.

        Args:
            index: Dataset index
            dataset: Dataset object

        Returns:
            html.Td: Table cell with color picker input
        """
        return html.Td(
            dcc.Input(
                id={"type": "dataset-color", "index": index},
                type="color",
                value=dataset.colour,
                style={
                    "width": "32px",
                    "height": "32px",
                    "border": "2px solid transparent",
                    "border-radius": "4px",
                    "cursor": "pointer",
                    "transition": "all 0.2s ease",
                    "padding": "0",
                    "outline": "none",
                },
                className="color-picker-input",
            ),
            style={"text-align": "center", "padding": "4px", "border": "none"},
        )

    def _create_action_button_cell(self, index, dataset, action_type):
        """Create a cell with an action button (download or delete).

        Args:
            index: Dataset index
            dataset: Dataset object
            action_type: Either 'download' or 'delete'

        Returns:
            html.Td: Table cell with action button
        """
        if action_type == "download":
            icon_src = "/assets/download.svg"
            btn_class = "btn btn-outline-primary btn-sm"
            id_type = "download-dataset"
            title = f"Download {dataset.name}"
        else:  # delete
            icon_src = "/assets/delete.svg"
            btn_class = "btn btn-outline-danger btn-sm"
            id_type = "delete-dataset"
            title = f"Delete {dataset.name}"

        return html.Td(
            html.Div(
                html.Button(
                    html.Img(src=icon_src, style={"width": "16px", "height": "16px"}),
                    id={"type": id_type, "index": index},
                    className=btn_class,
                    style={
                        "width": "32px",
                        "height": "32px",
                        "padding": "0",
                        "border-radius": "4px",
                        "font-size": "14px",
                        "line-height": "1",
                        "display": "flex",
                        "align-items": "center",
                        "justify-content": "center",
                    },
                    title=title,
                ),
                style={"margin-left": "15px"},
            ),
            style={
                "text-align": "center",
                "padding": "4px",
                "vertical-align": "middle",
                "border": "none",
            },
        )
