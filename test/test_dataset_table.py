"""Tests for DatasetTable class in SHIELD DAS.

This module tests the DatasetTable class functionality including table creation,
header generation, row generation, and cell creation for dataset management UI.
"""

from unittest.mock import Mock

import pytest
from dash import dcc, html

from shield_das.dataset_table import DatasetTable

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_dataset():
    """Create a single mock Dataset object for testing.

    Returns:
        Mock Dataset with standard attributes
    """
    dataset = Mock()
    dataset.name = "Test Dataset"
    dataset.path = "/path/to/test/dataset"
    dataset.live_data = False
    dataset.colour = "#ff0000"
    return dataset


@pytest.fixture
def mock_datasets():
    """Create a list of mock Dataset objects for testing.

    Returns:
        List of three mock Dataset objects with different attributes
    """
    dataset1 = Mock()
    dataset1.name = "Dataset 1"
    dataset1.path = "/path/to/dataset1"
    dataset1.live_data = False
    dataset1.colour = "#ff0000"

    dataset2 = Mock()
    dataset2.name = "Dataset 2"
    dataset2.path = "/path/to/dataset2"
    dataset2.live_data = True
    dataset2.colour = "#00ff00"

    dataset3 = Mock()
    dataset3.name = "Dataset 3"
    dataset3.path = "/path/to/dataset3"
    dataset3.live_data = False
    dataset3.colour = "#0000ff"

    return [dataset1, dataset2, dataset3]


# =============================================================================
# Tests for DatasetTable Initialization
# =============================================================================


def test_dataset_table_initializes_with_datasets(mock_datasets):
    """
    Test DatasetTable initialization to verify it correctly stores the
    datasets parameter.
    """
    table = DatasetTable(datasets=mock_datasets)
    assert table.datasets == mock_datasets


def test_dataset_table_initializes_with_empty_list():
    """
    Test DatasetTable initialization to confirm it accepts an empty list
    of datasets.
    """
    table = DatasetTable(datasets=[])
    assert table.datasets == []


def test_dataset_table_initializes_with_single_dataset(mock_dataset):
    """
    Test DatasetTable initialization to verify it correctly handles a
    single dataset in a list.
    """
    table = DatasetTable(datasets=[mock_dataset])
    assert len(table.datasets) == 1
    assert table.datasets[0] == mock_dataset


# =============================================================================
# Tests for create Method
# =============================================================================


def test_dataset_table_create_returns_html_div(mock_datasets):
    """
    Test DatasetTable create method to verify it returns an html.Div
    component.
    """
    table = DatasetTable(datasets=mock_datasets)
    result = table.create()
    assert isinstance(result, html.Div)


def test_dataset_table_create_contains_html_table(mock_datasets):
    """
    Test DatasetTable create method to confirm the returned Div contains
    an html.Table component.
    """
    table = DatasetTable(datasets=mock_datasets)
    result = table.create()
    assert len(result.children) == 1
    assert isinstance(result.children[0], html.Table)


def test_dataset_table_create_applies_bootstrap_classes(mock_datasets):
    """
    Test DatasetTable create method to verify the table has Bootstrap
    CSS classes applied.
    """
    table = DatasetTable(datasets=mock_datasets)
    result = table.create()
    html_table = result.children[0]
    assert "table-striped" in html_table.className
    assert "table-hover" in html_table.className


def test_dataset_table_create_includes_header_row(mock_datasets):
    """
    Test DatasetTable create method to confirm the table includes a
    header row as the first row.
    """
    table = DatasetTable(datasets=mock_datasets)
    result = table.create()
    html_table = result.children[0]
    rows = html_table.children
    assert len(rows) > 0
    assert isinstance(rows[0], html.Tr)


def test_dataset_table_create_includes_all_dataset_rows(mock_datasets):
    """
    Test DatasetTable create method to verify the table includes a row
    for each dataset plus the header row (4 total for 3 datasets).
    """
    table = DatasetTable(datasets=mock_datasets)
    result = table.create()
    html_table = result.children[0]
    rows = html_table.children
    # 1 header + 3 datasets = 4 rows
    assert len(rows) == 4


def test_dataset_table_create_with_empty_datasets_includes_only_header():
    """
    Test DatasetTable create method to confirm an empty dataset list
    produces a table with only the header row.
    """
    table = DatasetTable(datasets=[])
    result = table.create()
    html_table = result.children[0]
    rows = html_table.children
    assert len(rows) == 1


def test_dataset_table_create_applies_custom_styles(mock_datasets):
    """
    Test DatasetTable create method to verify custom styles are applied
    to the table (border, border-radius, etc.).
    """
    table = DatasetTable(datasets=mock_datasets)
    result = table.create()
    html_table = result.children[0]
    style = html_table.style
    assert "border" in style
    assert "border-radius" in style
    assert "overflow" in style


# =============================================================================
# Tests for _create_header Method
# =============================================================================


def test_dataset_table_create_header_returns_html_tr(mock_datasets):
    """
    Test DatasetTable _create_header method to verify it returns an
    html.Tr component.
    """
    table = DatasetTable(datasets=mock_datasets)
    header = table._create_header()
    assert isinstance(header, html.Tr)


def test_dataset_table_create_header_has_six_columns(mock_datasets):
    """
    Test DatasetTable _create_header method to confirm the header row
    contains six html.Th elements (Name, Path, Live, Colour, Download, Delete).
    """
    table = DatasetTable(datasets=mock_datasets)
    header = table._create_header()
    assert len(header.children) == 6


def test_dataset_table_create_header_column_names(mock_datasets):
    """
    Test DatasetTable _create_header method to verify the header columns
    have correct text labels.
    """
    table = DatasetTable(datasets=mock_datasets)
    header = table._create_header()
    columns = header.children
    assert columns[0].children == "Dataset Name"
    assert columns[1].children == "Dataset Path"
    assert columns[2].children == "Live"
    assert columns[3].children == "Colour"
    assert columns[4].children == ""  # Download icon
    assert columns[5].children == ""  # Delete icon


def test_dataset_table_create_header_applies_center_alignment(mock_datasets):
    """
    Test DatasetTable _create_header method to confirm most header cells
    use center text alignment.
    """
    table = DatasetTable(datasets=mock_datasets)
    header = table._create_header()
    # Live, Colour, Download, Delete should be centered
    assert "center" in header.children[2].style["text-align"]
    assert "center" in header.children[3].style["text-align"]


def test_dataset_table_create_header_name_path_left_aligned(mock_datasets):
    """
    Test DatasetTable _create_header method to verify Name and Path
    columns use left text alignment.
    """
    table = DatasetTable(datasets=mock_datasets)
    header = table._create_header()
    assert "left" in header.children[0].style["text-align"]
    assert "left" in header.children[1].style["text-align"]


def test_dataset_table_create_header_column_widths(mock_datasets):
    """
    Test DatasetTable _create_header method to confirm columns have
    appropriate width percentages.
    """
    table = DatasetTable(datasets=mock_datasets)
    header = table._create_header()
    columns = header.children
    # Name and Path should be wider (43.75%)
    assert "43.75%" in columns[0].style["width"]
    assert "43.75%" in columns[1].style["width"]
    # Action buttons should be narrower (2.5% or 5%)
    assert "2.5%" in columns[2].style["width"]
    assert "5%" in columns[3].style["width"]


# =============================================================================
# Tests for _create_row Method
# =============================================================================


def test_dataset_table_create_row_returns_html_tr(mock_dataset):
    """
    Test DatasetTable _create_row method to verify it returns an html.Tr
    component.
    """
    table = DatasetTable(datasets=[mock_dataset])
    row = table._create_row(0, mock_dataset)
    assert isinstance(row, html.Tr)


def test_dataset_table_create_row_has_six_cells(mock_dataset):
    """
    Test DatasetTable _create_row method to confirm each row contains
    six html.Td elements.
    """
    table = DatasetTable(datasets=[mock_dataset])
    row = table._create_row(0, mock_dataset)
    assert len(row.children) == 6


def test_dataset_table_create_row_uses_correct_index(mock_datasets):
    """
    Test DatasetTable _create_row method to verify it uses the provided
    index parameter when creating cell components.
    """
    table = DatasetTable(datasets=mock_datasets)
    row = table._create_row(1, mock_datasets[1])
    # Should be able to create row without errors
    assert isinstance(row, html.Tr)


# =============================================================================
# Tests for _create_name_cell Method
# =============================================================================


def test_dataset_table_create_name_cell_returns_html_td(mock_dataset):
    """
    Test DatasetTable _create_name_cell method to verify it returns an
    html.Td component.
    """
    table = DatasetTable(datasets=[mock_dataset])
    cell = table._create_name_cell(0, mock_dataset)
    assert isinstance(cell, html.Td)


def test_dataset_table_create_name_cell_contains_dcc_input(mock_dataset):
    """
    Test DatasetTable _create_name_cell method to confirm the cell contains
    a dcc.Input component for editing the name.
    """
    table = DatasetTable(datasets=[mock_dataset])
    cell = table._create_name_cell(0, mock_dataset)
    assert isinstance(cell.children, dcc.Input)


def test_dataset_table_create_name_cell_input_has_dataset_name_id(mock_dataset):
    """
    Test DatasetTable _create_name_cell method to verify the input has
    the correct pattern-matching callback ID.
    """
    table = DatasetTable(datasets=[mock_dataset])
    cell = table._create_name_cell(0, mock_dataset)
    input_component = cell.children
    assert input_component.id["type"] == "dataset-name"
    assert input_component.id["index"] == 0


def test_dataset_table_create_name_cell_input_value_is_dataset_name(mock_dataset):
    """
    Test DatasetTable _create_name_cell method to confirm the input value
    is set to the dataset name.
    """
    table = DatasetTable(datasets=[mock_dataset])
    cell = table._create_name_cell(0, mock_dataset)
    input_component = cell.children
    assert input_component.value == "Test Dataset"


def test_dataset_table_create_name_cell_has_custom_class(mock_dataset):
    """
    Test DatasetTable _create_name_cell method to verify the input has
    the 'dataset-name-input' CSS class.
    """
    table = DatasetTable(datasets=[mock_dataset])
    cell = table._create_name_cell(0, mock_dataset)
    input_component = cell.children
    assert input_component.className == "dataset-name-input"


def test_dataset_table_create_name_cell_input_has_width_style(mock_dataset):
    """
    Test DatasetTable _create_name_cell method to confirm the input has
    width styling applied (95%).
    """
    table = DatasetTable(datasets=[mock_dataset])
    cell = table._create_name_cell(0, mock_dataset)
    input_component = cell.children
    assert "width" in input_component.style
    assert input_component.style["width"] == "95%"


@pytest.mark.parametrize("index", [0, 1, 2, 5, 10])
def test_dataset_table_create_name_cell_accepts_different_indices(mock_dataset, index):
    """
    Test DatasetTable _create_name_cell method to verify it correctly
    assigns different index values (0, 1, 2, 5, 10) to the input ID.
    """
    table = DatasetTable(datasets=[mock_dataset])
    cell = table._create_name_cell(index, mock_dataset)
    input_component = cell.children
    assert input_component.id["index"] == index


# =============================================================================
# Tests for _create_path_cell Method
# =============================================================================


def test_dataset_table_create_path_cell_returns_html_td(mock_dataset):
    """
    Test DatasetTable _create_path_cell method to verify it returns an
    html.Td component.
    """
    table = DatasetTable(datasets=[mock_dataset])
    cell = table._create_path_cell(mock_dataset)
    assert isinstance(cell, html.Td)


def test_dataset_table_create_path_cell_contains_nested_div_span(mock_dataset):
    """
    Test DatasetTable _create_path_cell method to confirm the cell contains
    nested Div and Span components for path display.
    """
    table = DatasetTable(datasets=[mock_dataset])
    cell = table._create_path_cell(mock_dataset)
    assert isinstance(cell.children, html.Div)
    assert isinstance(cell.children.children, html.Span)


def test_dataset_table_create_path_cell_displays_dataset_path(mock_dataset):
    """
    Test DatasetTable _create_path_cell method to verify the path text
    is displayed in the Span component.
    """
    table = DatasetTable(datasets=[mock_dataset])
    cell = table._create_path_cell(mock_dataset)
    span = cell.children.children
    assert span.children == "/path/to/test/dataset"


def test_dataset_table_create_path_cell_uses_monospace_font(mock_dataset):
    """
    Test DatasetTable _create_path_cell method to confirm the path uses
    monospace font family for better readability.
    """
    table = DatasetTable(datasets=[mock_dataset])
    cell = table._create_path_cell(mock_dataset)
    span = cell.children.children
    assert span.style["font-family"] == "monospace"


def test_dataset_table_create_path_cell_has_title_tooltip(mock_dataset):
    """
    Test DatasetTable _create_path_cell method to verify the Span has
    a title attribute (tooltip) with the full path.
    """
    table = DatasetTable(datasets=[mock_dataset])
    cell = table._create_path_cell(mock_dataset)
    span = cell.children.children
    assert span.title == "/path/to/test/dataset"


def test_dataset_table_create_path_cell_applies_word_break_style(mock_dataset):
    """
    Test DatasetTable _create_path_cell method to confirm word-break
    style is applied to handle long paths.
    """
    table = DatasetTable(datasets=[mock_dataset])
    cell = table._create_path_cell(mock_dataset)
    span = cell.children.children
    assert "word-break" in span.style
    assert span.style["word-break"] == "break-all"


# =============================================================================
# Tests for _create_live_data_cell Method
# =============================================================================


def test_dataset_table_create_live_data_cell_returns_html_td(mock_dataset):
    """
    Test DatasetTable _create_live_data_cell method to verify it returns
    an html.Td component.
    """
    table = DatasetTable(datasets=[mock_dataset])
    cell = table._create_live_data_cell(0, mock_dataset)
    assert isinstance(cell, html.Td)


def test_dataset_table_create_live_data_cell_contains_checkbox(mock_dataset):
    """
    Test DatasetTable _create_live_data_cell method to confirm the cell
    contains a dbc.Checkbox component.
    """
    table = DatasetTable(datasets=[mock_dataset])
    cell = table._create_live_data_cell(0, mock_dataset)
    # Cell -> Div -> Checkbox
    div = cell.children
    # Check that the checkbox component exists
    checkbox = div.children
    assert hasattr(checkbox, "id")
    assert hasattr(checkbox, "value")


def test_dataset_table_create_live_data_cell_checkbox_has_correct_id(mock_dataset):
    """
    Test DatasetTable _create_live_data_cell method to verify the checkbox
    has the correct pattern-matching callback ID.
    """
    table = DatasetTable(datasets=[mock_dataset])
    cell = table._create_live_data_cell(0, mock_dataset)
    div = cell.children
    checkbox = div.children
    assert checkbox.id["type"] == "dataset-live-data"
    assert checkbox.id["index"] == 0


def test_dataset_table_create_live_data_cell_checkbox_value_from_dataset(
    mock_dataset,
):
    """
    Test DatasetTable _create_live_data_cell method to confirm the checkbox
    value is set from the dataset's live_data attribute.
    """
    table = DatasetTable(datasets=[mock_dataset])
    cell = table._create_live_data_cell(0, mock_dataset)
    div = cell.children
    checkbox = div.children
    assert checkbox.value is False


@pytest.mark.parametrize("live_data_value", [True, False])
def test_dataset_table_create_live_data_cell_reflects_live_data_state(
    live_data_value,
):
    """
    Test DatasetTable _create_live_data_cell method to verify it correctly
    reflects both True and False live_data states.
    """
    dataset = Mock()
    dataset.name = "Test"
    dataset.path = "/test"
    dataset.live_data = live_data_value
    dataset.colour = "#ff0000"

    table = DatasetTable(datasets=[dataset])
    cell = table._create_live_data_cell(0, dataset)
    div = cell.children
    checkbox = div.children
    assert checkbox.value == live_data_value


def test_dataset_table_create_live_data_cell_centers_content(mock_dataset):
    """
    Test DatasetTable _create_live_data_cell method to confirm the cell
    has center text alignment.
    """
    table = DatasetTable(datasets=[mock_dataset])
    cell = table._create_live_data_cell(0, mock_dataset)
    assert "center" in cell.style["text-align"]


# =============================================================================
# Tests for _create_color_cell Method
# =============================================================================


def test_dataset_table_create_color_cell_returns_html_td(mock_dataset):
    """
    Test DatasetTable _create_color_cell method to verify it returns an
    html.Td component.
    """
    table = DatasetTable(datasets=[mock_dataset])
    cell = table._create_color_cell(0, mock_dataset)
    assert isinstance(cell, html.Td)


def test_dataset_table_create_color_cell_contains_color_input(mock_dataset):
    """
    Test DatasetTable _create_color_cell method to confirm the cell contains
    a dcc.Input with type='color'.
    """
    table = DatasetTable(datasets=[mock_dataset])
    cell = table._create_color_cell(0, mock_dataset)
    input_component = cell.children
    assert isinstance(input_component, dcc.Input)
    assert input_component.type == "color"


def test_dataset_table_create_color_cell_input_has_correct_id(mock_dataset):
    """
    Test DatasetTable _create_color_cell method to verify the color picker
    has the correct pattern-matching callback ID.
    """
    table = DatasetTable(datasets=[mock_dataset])
    cell = table._create_color_cell(0, mock_dataset)
    input_component = cell.children
    assert input_component.id["type"] == "dataset-color"
    assert input_component.id["index"] == 0


def test_dataset_table_create_color_cell_value_is_dataset_colour(mock_dataset):
    """
    Test DatasetTable _create_color_cell method to confirm the color picker
    value is set to the dataset colour.
    """
    table = DatasetTable(datasets=[mock_dataset])
    cell = table._create_color_cell(0, mock_dataset)
    input_component = cell.children
    assert input_component.value == "#ff0000"


def test_dataset_table_create_color_cell_has_custom_class(mock_dataset):
    """
    Test DatasetTable _create_color_cell method to verify the input has
    the 'color-picker-input' CSS class.
    """
    table = DatasetTable(datasets=[mock_dataset])
    cell = table._create_color_cell(0, mock_dataset)
    input_component = cell.children
    assert input_component.className == "color-picker-input"


def test_dataset_table_create_color_cell_has_fixed_dimensions(mock_dataset):
    """
    Test DatasetTable _create_color_cell method to confirm the color picker
    has fixed width and height (32px).
    """
    table = DatasetTable(datasets=[mock_dataset])
    cell = table._create_color_cell(0, mock_dataset)
    input_component = cell.children
    assert input_component.style["width"] == "32px"
    assert input_component.style["height"] == "32px"


def test_dataset_table_create_color_cell_centers_content(mock_dataset):
    """
    Test DatasetTable _create_color_cell method to verify the cell has
    center text alignment.
    """
    table = DatasetTable(datasets=[mock_dataset])
    cell = table._create_color_cell(0, mock_dataset)
    assert "center" in cell.style["text-align"]


@pytest.mark.parametrize(
    "colour",
    ["#ff0000", "#00ff00", "#0000ff", "#ffffff", "#000000"],
)
def test_dataset_table_create_color_cell_accepts_different_colors(colour):
    """
    Test DatasetTable _create_color_cell method to verify it correctly
    handles different hex color values.
    """
    dataset = Mock()
    dataset.name = "Test"
    dataset.path = "/test"
    dataset.live_data = False
    dataset.colour = colour

    table = DatasetTable(datasets=[dataset])
    cell = table._create_color_cell(0, dataset)
    input_component = cell.children
    assert input_component.value == colour


# =============================================================================
# Tests for _create_action_button_cell Method
# =============================================================================


def test_dataset_table_create_action_button_cell_returns_html_td(mock_dataset):
    """
    Test DatasetTable _create_action_button_cell method to verify it returns
    an html.Td component.
    """
    table = DatasetTable(datasets=[mock_dataset])
    cell = table._create_action_button_cell(0, mock_dataset, "download")
    assert isinstance(cell, html.Td)


def test_dataset_table_create_action_button_cell_contains_button(mock_dataset):
    """
    Test DatasetTable _create_action_button_cell method to confirm the cell
    contains a nested Div with an html.Button.
    """
    table = DatasetTable(datasets=[mock_dataset])
    cell = table._create_action_button_cell(0, mock_dataset, "download")
    div = cell.children
    assert isinstance(div.children, html.Button)


def test_dataset_table_create_action_button_download_type(mock_dataset):
    """
    Test DatasetTable _create_action_button_cell method with 'download' type
    to verify it creates a download button with correct properties.
    """
    table = DatasetTable(datasets=[mock_dataset])
    cell = table._create_action_button_cell(0, mock_dataset, "download")
    div = cell.children
    button = div.children
    assert button.id["type"] == "download-dataset"
    assert button.id["index"] == 0


def test_dataset_table_create_action_button_delete_type(mock_dataset):
    """
    Test DatasetTable _create_action_button_cell method with 'delete' type
    to verify it creates a delete button with correct properties.
    """
    table = DatasetTable(datasets=[mock_dataset])
    cell = table._create_action_button_cell(0, mock_dataset, "delete")
    div = cell.children
    button = div.children
    assert button.id["type"] == "delete-dataset"
    assert button.id["index"] == 0


def test_dataset_table_create_action_button_download_has_primary_class(mock_dataset):
    """
    Test DatasetTable _create_action_button_cell method to confirm download
    button has 'btn-outline-primary' Bootstrap class.
    """
    table = DatasetTable(datasets=[mock_dataset])
    cell = table._create_action_button_cell(0, mock_dataset, "download")
    div = cell.children
    button = div.children
    assert "btn-outline-primary" in button.className


def test_dataset_table_create_action_button_delete_has_danger_class(mock_dataset):
    """
    Test DatasetTable _create_action_button_cell method to confirm delete
    button has 'btn-outline-danger' Bootstrap class.
    """
    table = DatasetTable(datasets=[mock_dataset])
    cell = table._create_action_button_cell(0, mock_dataset, "delete")
    div = cell.children
    button = div.children
    assert "btn-outline-danger" in button.className


def test_dataset_table_create_action_button_download_icon_src(mock_dataset):
    """
    Test DatasetTable _create_action_button_cell method to verify download
    button uses correct icon source (/assets/download.svg).
    """
    table = DatasetTable(datasets=[mock_dataset])
    cell = table._create_action_button_cell(0, mock_dataset, "download")
    div = cell.children
    button = div.children
    img = button.children
    assert img.src == "/assets/download.svg"


def test_dataset_table_create_action_button_delete_icon_src(mock_dataset):
    """
    Test DatasetTable _create_action_button_cell method to verify delete
    button uses correct icon source (/assets/delete.svg).
    """
    table = DatasetTable(datasets=[mock_dataset])
    cell = table._create_action_button_cell(0, mock_dataset, "delete")
    div = cell.children
    button = div.children
    img = button.children
    assert img.src == "/assets/delete.svg"


def test_dataset_table_create_action_button_download_has_title(mock_dataset):
    """
    Test DatasetTable _create_action_button_cell method to confirm download
    button has tooltip title with dataset name.
    """
    table = DatasetTable(datasets=[mock_dataset])
    cell = table._create_action_button_cell(0, mock_dataset, "download")
    div = cell.children
    button = div.children
    assert "Download" in button.title
    assert "Test Dataset" in button.title


def test_dataset_table_create_action_button_delete_has_title(mock_dataset):
    """
    Test DatasetTable _create_action_button_cell method to confirm delete
    button has tooltip title with dataset name.
    """
    table = DatasetTable(datasets=[mock_dataset])
    cell = table._create_action_button_cell(0, mock_dataset, "delete")
    div = cell.children
    button = div.children
    assert "Delete" in button.title
    assert "Test Dataset" in button.title


def test_dataset_table_create_action_button_has_fixed_dimensions(mock_dataset):
    """
    Test DatasetTable _create_action_button_cell method to verify buttons
    have fixed dimensions (32px x 32px).
    """
    table = DatasetTable(datasets=[mock_dataset])
    cell = table._create_action_button_cell(0, mock_dataset, "download")
    div = cell.children
    button = div.children
    assert button.style["width"] == "32px"
    assert button.style["height"] == "32px"


def test_dataset_table_create_action_button_icon_has_correct_size(mock_dataset):
    """
    Test DatasetTable _create_action_button_cell method to confirm button
    icons have 16px x 16px dimensions.
    """
    table = DatasetTable(datasets=[mock_dataset])
    cell = table._create_action_button_cell(0, mock_dataset, "download")
    div = cell.children
    button = div.children
    img = button.children
    assert img.style["width"] == "16px"
    assert img.style["height"] == "16px"


@pytest.mark.parametrize("action_type", ["download", "delete"])
def test_dataset_table_create_action_button_accepts_both_action_types(
    mock_dataset, action_type
):
    """
    Test DatasetTable _create_action_button_cell method to verify it
    correctly handles both 'download' and 'delete' action types.
    """
    table = DatasetTable(datasets=[mock_dataset])
    cell = table._create_action_button_cell(0, mock_dataset, action_type)
    div = cell.children
    button = div.children
    assert f"{action_type}-dataset" in button.id["type"]


# =============================================================================
# Tests for Integration and Edge Cases
# =============================================================================


def test_dataset_table_integration_full_table_structure(mock_datasets):
    """
    Test DatasetTable complete integration to verify the full table structure
    is correctly assembled with all components.
    """
    table = DatasetTable(datasets=mock_datasets)
    result = table.create()

    # Verify structure: Div -> Table -> Rows
    assert isinstance(result, html.Div)
    html_table = result.children[0]
    assert isinstance(html_table, html.Table)
    rows = html_table.children
    assert len(rows) == 4  # 1 header + 3 datasets


def test_dataset_table_integration_all_cells_present_in_rows(mock_datasets):
    """
    Test DatasetTable integration to confirm each dataset row contains
    all six required cells.
    """
    table = DatasetTable(datasets=mock_datasets)
    result = table.create()
    html_table = result.children[0]
    rows = html_table.children[1:]  # Skip header

    for row in rows:
        assert len(row.children) == 6  # Name, Path, Live, Color, Download, Delete


def test_dataset_table_integration_dataset_values_correctly_displayed(mock_datasets):
    """
    Test DatasetTable integration to verify dataset values are correctly
    displayed in the corresponding cells.
    """
    table = DatasetTable(datasets=mock_datasets)
    result = table.create()
    html_table = result.children[0]
    first_data_row = html_table.children[1]  # First dataset row

    # Check name cell
    name_input = first_data_row.children[0].children
    assert name_input.value == "Dataset 1"

    # Check path cell
    path_span = first_data_row.children[1].children.children
    assert path_span.children == "/path/to/dataset1"

    # Check color cell
    color_input = first_data_row.children[3].children
    assert color_input.value == "#ff0000"


def test_dataset_table_handles_dataset_with_long_path():
    """
    Test DatasetTable to verify it correctly handles datasets with very
    long file paths without breaking layout.
    """
    dataset = Mock()
    dataset.name = "Long Path Dataset"
    dataset.path = "/very/long/path/" + "nested/" * 20 + "dataset"
    dataset.live_data = False
    dataset.colour = "#ff0000"

    table = DatasetTable(datasets=[dataset])
    result = table.create()

    # Should create without errors
    assert isinstance(result, html.Div)


def test_dataset_table_handles_dataset_with_special_characters_in_name():
    """
    Test DatasetTable to confirm it correctly handles dataset names with
    special characters (spaces, symbols, etc.).
    """
    dataset = Mock()
    dataset.name = "Test Dataset #1 (2025-11-13) [Special]"
    dataset.path = "/test/path"
    dataset.live_data = False
    dataset.colour = "#ff0000"

    table = DatasetTable(datasets=[dataset])
    cell = table._create_name_cell(0, dataset)
    input_component = cell.children

    assert input_component.value == "Test Dataset #1 (2025-11-13) [Special]"


def test_dataset_table_maintains_consistent_row_structure(mock_datasets):
    """
    Test DatasetTable to verify all dataset rows maintain consistent
    structure with the same number of cells.
    """
    table = DatasetTable(datasets=mock_datasets)
    result = table.create()
    html_table = result.children[0]
    rows = html_table.children[1:]  # Skip header

    # All rows should have same number of cells
    cell_counts = [len(row.children) for row in rows]
    assert all(count == 6 for count in cell_counts)


def test_dataset_table_unique_ids_for_multiple_datasets(mock_datasets):
    """
    Test DatasetTable to confirm each dataset row gets unique IDs for
    pattern-matching callbacks.
    """
    table = DatasetTable(datasets=mock_datasets)
    result = table.create()
    html_table = result.children[0]

    # Collect all dataset-name IDs
    name_ids = []
    for i, row in enumerate(html_table.children[1:], start=0):
        name_input = row.children[0].children
        name_ids.append(name_input.id["index"])

    # All IDs should be unique and sequential
    assert name_ids == [0, 1, 2]
    assert len(set(name_ids)) == 3
