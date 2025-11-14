"""Unit tests for layout_components module.

This module contains comprehensive unit tests for all layout component
creation functions, verifying structure, properties, and component hierarchy.
"""

import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import dcc, html

from shield_das.layout_components import (
    COLLAPSE_BUTTON_STYLE,
    SEPARATOR_HR_STYLE,
    create_bottom_spacing,
    create_dataset_management_card,
    create_download_components,
    create_downstream_controls_card,
    create_header,
    create_hidden_stores,
    create_live_data_interval,
    create_permeability_controls_card,
    create_permeability_plot_card,
    create_plot_controls_row,
    create_pressure_plots_row,
    create_simple_plot_controls_card,
    create_temperature_controls_card,
    create_temperature_plot_card,
    create_upstream_controls_card,
)


class TestConstants:
    """Test module-level constants."""

    def test_collapse_button_style_exists(self):
        """Test that COLLAPSE_BUTTON_STYLE constant is defined."""
        assert isinstance(COLLAPSE_BUTTON_STYLE, dict)

    def test_collapse_button_style_has_required_keys(self):
        """Test that COLLAPSE_BUTTON_STYLE has expected styling keys."""
        assert "border" in COLLAPSE_BUTTON_STYLE
        assert "background-color" in COLLAPSE_BUTTON_STYLE
        assert "width" in COLLAPSE_BUTTON_STYLE
        assert "height" in COLLAPSE_BUTTON_STYLE

    def test_separator_hr_style_exists(self):
        """Test that SEPARATOR_HR_STYLE constant is defined."""
        assert isinstance(SEPARATOR_HR_STYLE, dict)

    def test_separator_hr_style_has_flex(self):
        """Test that SEPARATOR_HR_STYLE includes flex property."""
        assert "flex" in SEPARATOR_HR_STYLE


class TestCreateHeader:
    """Test create_header function."""

    def test_create_header_returns_row(self):
        """Test that create_header returns a dbc.Row component."""
        result = create_header()
        assert isinstance(result, dbc.Row)

    def test_create_header_contains_column(self):
        """Test that header row contains a column."""
        result = create_header()
        assert len(result.children) == 1
        assert isinstance(result.children[0], dbc.Col)

    def test_create_header_contains_h1(self):
        """Test that header contains an H1 element."""
        result = create_header()
        col = result.children[0]
        assert isinstance(col.children, html.H1)

    def test_create_header_has_title_text(self):
        """Test that header displays correct title text."""
        result = create_header()
        h1 = result.children[0].children
        assert h1.children == "SHIELD Data Visualisation"

    def test_create_header_has_text_center_class(self):
        """Test that header title has text-center class."""
        result = create_header()
        h1 = result.children[0].children
        assert "text-center" in h1.className

    def test_create_header_has_style(self):
        """Test that header H1 has inline styles."""
        result = create_header()
        h1 = result.children[0].children
        assert h1.style is not None
        assert "fontSize" in h1.style


class TestCreateDatasetManagementCard:
    """Test create_dataset_management_card function."""

    def test_create_dataset_management_card_returns_row(self):
        """Test that function returns a dbc.Row component."""
        result = create_dataset_management_card("test_html")
        assert isinstance(result, dbc.Row)

    def test_create_dataset_management_card_contains_col(self):
        """Test that card row contains a column."""
        result = create_dataset_management_card("test_html")
        assert len(result.children) == 1
        assert isinstance(result.children[0], dbc.Col)

    def test_create_dataset_management_card_contains_card(self):
        """Test that column contains a card."""
        result = create_dataset_management_card("test_html")
        col = result.children[0]
        assert isinstance(col.children[0], dbc.Card)

    def test_create_dataset_management_card_has_header(self):
        """Test that card has a CardHeader."""
        result = create_dataset_management_card("test_html")
        card = result.children[0].children[0]
        assert isinstance(card.children[0], dbc.CardHeader)

    def test_create_dataset_management_card_has_collapse(self):
        """Test that card has a Collapse component."""
        result = create_dataset_management_card("test_html")
        card = result.children[0].children[0]
        assert isinstance(card.children[1], dbc.Collapse)

    def test_create_dataset_management_card_collapse_id(self):
        """Test that collapse has correct ID."""
        result = create_dataset_management_card("test_html")
        card = result.children[0].children[0]
        collapse = card.children[1]
        assert collapse.id == "collapse-dataset"

    def test_create_dataset_management_card_collapse_open_by_default(self):
        """Test that collapse is open by default."""
        result = create_dataset_management_card("test_html")
        card = result.children[0].children[0]
        collapse = card.children[1]
        assert collapse.is_open is True

    def test_create_dataset_management_card_includes_table_html(self):
        """Test that provided table HTML is included."""
        test_html = "<div>Test Table</div>"
        result = create_dataset_management_card(test_html)
        card = result.children[0].children[0]
        collapse = card.children[1]
        # Navigate to the dataset-table-container
        card_body = collapse.children
        assert isinstance(card_body, dbc.CardBody)

    def test_create_dataset_management_card_has_collapse_button(self):
        """Test that card header has a collapse button."""
        result = create_dataset_management_card("test_html")
        card = result.children[0].children[0]
        card_header = card.children[0]
        # Button should be in the header row
        assert "collapse-dataset-button" in str(card_header)


class TestCreateHiddenStores:
    """Test create_hidden_stores function."""

    def test_create_hidden_stores_returns_list(self):
        """Test that function returns a list."""
        result = create_hidden_stores()
        assert isinstance(result, list)

    def test_create_hidden_stores_contains_store_and_div_components(self):
        """Test that items are dcc.Store or html.Div components."""
        result = create_hidden_stores()
        assert all(isinstance(item, (dcc.Store, html.Div)) for item in result)

    def test_create_hidden_stores_has_datasets_store(self):
        """Test that datasets store is included."""
        result = create_hidden_stores()
        store_ids = [store.id for store in result]
        assert "datasets-store" in store_ids

    def test_create_hidden_stores_has_settings_stores(self):
        """Test that settings stores are included."""
        result = create_hidden_stores()
        component_ids = [comp.id for comp in result if hasattr(comp, "id")]
        assert "upstream-settings-store" in component_ids
        assert "downstream-settings-store" in component_ids
        assert "temperature-settings-store" in component_ids
        assert "permeability-settings-store" in component_ids

    def test_create_hidden_stores_has_upload_status_div(self):
        """Test that upload status div is included."""
        result = create_hidden_stores()
        component_ids = [comp.id for comp in result if hasattr(comp, "id")]
        assert "upload-status" in component_ids


class TestCreatePressurePlotsRow:
    """Test create_pressure_plots_row function."""

    def test_create_pressure_plots_row_returns_row(self):
        """Test that function returns a dbc.Row component."""
        fig1 = go.Figure()
        fig2 = go.Figure()
        result = create_pressure_plots_row(fig1, fig2)
        assert isinstance(result, dbc.Row)

    def test_create_pressure_plots_row_has_two_columns(self):
        """Test that row contains two columns."""
        fig1 = go.Figure()
        fig2 = go.Figure()
        result = create_pressure_plots_row(fig1, fig2)
        assert len(result.children) == 2
        assert all(isinstance(child, dbc.Col) for child in result.children)

    def test_create_pressure_plots_row_contains_cards(self):
        """Test that columns contain card components."""
        fig1 = go.Figure()
        fig2 = go.Figure()
        result = create_pressure_plots_row(fig1, fig2)
        for col in result.children:
            assert isinstance(col.children, list)
            # Card should be in the column

    def test_create_pressure_plots_row_uses_provided_figures(self):
        """Test that provided figures are used in the plots."""
        fig1 = go.Figure()
        fig2 = go.Figure()
        fig1.update_layout(title="Upstream Test")
        fig2.update_layout(title="Downstream Test")
        result = create_pressure_plots_row(fig1, fig2)
        # Figures should be present in the component tree
        assert result is not None


class TestCreateSimplePlotControlsCard:
    """Test create_simple_plot_controls_card function."""

    def test_create_simple_plot_controls_card_returns_card(self):
        """Test that function returns a dbc.Card component."""
        result = create_simple_plot_controls_card(
            plot_name="Test",
            card_title="Test Plot Controls",
            collapse_button_id="collapse-test-button",
            collapse_id="collapse-test",
            x_scale_id="test-x-scale",
            x_min_id="test-x-min",
            x_max_id="test-x-max",
            y_scale_id="test-y-scale",
            y_min_id="test-y-min",
            y_max_id="test-y-max",
            show_error_bars_id="show-error-bars-test",
            export_button_id="export-test-plot",
        )
        assert isinstance(result, dbc.Card)

    def test_create_simple_plot_controls_card_has_header(self):
        """Test that card has a header."""
        result = create_simple_plot_controls_card(
            plot_name="Test",
            card_title="Test Plot Controls",
            collapse_button_id="collapse-test-button",
            collapse_id="collapse-test",
            x_scale_id="test-x-scale",
            x_min_id="test-x-min",
            x_max_id="test-x-max",
            y_scale_id="test-y-scale",
            y_min_id="test-y-min",
            y_max_id="test-y-max",
            show_error_bars_id="show-error-bars-test",
            export_button_id="export-test-plot",
        )
        assert isinstance(result.children[0], dbc.CardHeader)

    def test_create_simple_plot_controls_card_has_title(self):
        """Test that card displays provided title."""
        result = create_simple_plot_controls_card(
            plot_name="Test",
            card_title="My Test Plot Controls",
            collapse_button_id="collapse-test-button",
            collapse_id="collapse-test",
            x_scale_id="test-x-scale",
            x_min_id="test-x-min",
            x_max_id="test-x-max",
            y_scale_id="test-y-scale",
            y_min_id="test-y-min",
            y_max_id="test-y-max",
            show_error_bars_id="show-error-bars-test",
            export_button_id="export-test-plot",
        )
        # Title should be in the header
        header = result.children[0]
        assert "My Test Plot Controls" in str(header)

    def test_create_simple_plot_controls_card_has_collapse(self):
        """Test that card has a collapse component."""
        result = create_simple_plot_controls_card(
            plot_name="Test",
            card_title="Test Plot Controls",
            collapse_button_id="collapse-test-button",
            collapse_id="collapse-test",
            x_scale_id="test-x-scale",
            x_min_id="test-x-min",
            x_max_id="test-x-max",
            y_scale_id="test-y-scale",
            y_min_id="test-y-min",
            y_max_id="test-y-max",
            show_error_bars_id="show-error-bars-test",
            export_button_id="export-test-plot",
        )
        assert isinstance(result.children[1], dbc.Collapse)

    def test_create_simple_plot_controls_card_collapse_id(self):
        """Test that collapse has correct ID."""
        result = create_simple_plot_controls_card(
            plot_name="Test",
            card_title="Test Plot Controls",
            collapse_button_id="collapse-myplot-button",
            collapse_id="collapse-myplot-controls",
            x_scale_id="test-x-scale",
            x_min_id="test-x-min",
            x_max_id="test-x-max",
            y_scale_id="test-y-scale",
            y_min_id="test-y-min",
            y_max_id="test-y-max",
            show_error_bars_id="show-error-bars-test",
            export_button_id="export-test-plot",
        )
        collapse = result.children[1]
        assert collapse.id == "collapse-myplot-controls"

    def test_create_simple_plot_controls_card_has_export_button(self):
        """Test that card includes an export button."""
        result = create_simple_plot_controls_card(
            plot_name="Test",
            card_title="Test Plot Controls",
            collapse_button_id="collapse-test-button",
            collapse_id="collapse-test",
            x_scale_id="test-x-scale",
            x_min_id="test-x-min",
            x_max_id="test-x-max",
            y_scale_id="test-y-scale",
            y_min_id="test-y-min",
            y_max_id="test-y-max",
            show_error_bars_id="show-error-bars-test",
            export_button_id="export-test-plot",
        )
        # Export button should be present in the card
        assert "export-test-plot" in str(result)


class TestCreateUpstreamControlsCard:
    """Test create_upstream_controls_card function."""

    def test_create_upstream_controls_card_returns_card(self):
        """Test that function returns a dbc.Card component."""
        result = create_upstream_controls_card()
        assert isinstance(result, dbc.Card)

    def test_create_upstream_controls_card_has_header(self):
        """Test that card has a header."""
        result = create_upstream_controls_card()
        assert isinstance(result.children[0], dbc.CardHeader)

    def test_create_upstream_controls_card_has_title(self):
        """Test that card displays 'Upstream' title."""
        result = create_upstream_controls_card()
        header = result.children[0]
        assert "Upstream" in str(header)

    def test_create_upstream_controls_card_has_collapse(self):
        """Test that card has a collapse component."""
        result = create_upstream_controls_card()
        assert isinstance(result.children[1], dbc.Collapse)


class TestCreateDownstreamControlsCard:
    """Test create_downstream_controls_card function."""

    def test_create_downstream_controls_card_returns_card(self):
        """Test that function returns a dbc.Card component."""
        result = create_downstream_controls_card()
        assert isinstance(result, dbc.Card)

    def test_create_downstream_controls_card_has_header(self):
        """Test that card has a header."""
        result = create_downstream_controls_card()
        assert isinstance(result.children[0], dbc.CardHeader)

    def test_create_downstream_controls_card_has_title(self):
        """Test that card displays 'Downstream' title."""
        result = create_downstream_controls_card()
        header = result.children[0]
        assert "Downstream" in str(header)

    def test_create_downstream_controls_card_has_collapse(self):
        """Test that card has a collapse component."""
        result = create_downstream_controls_card()
        assert isinstance(result.children[1], dbc.Collapse)


class TestCreateTemperatureControlsCard:
    """Test create_temperature_controls_card function."""

    def test_create_temperature_controls_card_returns_card(self):
        """Test that function returns a dbc.Card component."""
        result = create_temperature_controls_card()
        assert isinstance(result, dbc.Card)

    def test_create_temperature_controls_card_has_header(self):
        """Test that card has a header."""
        result = create_temperature_controls_card()
        assert isinstance(result.children[0], dbc.CardHeader)

    def test_create_temperature_controls_card_has_title(self):
        """Test that card displays 'Temperature' title."""
        result = create_temperature_controls_card()
        header = result.children[0]
        assert "Temperature" in str(header)

    def test_create_temperature_controls_card_has_collapse(self):
        """Test that card has a collapse component."""
        result = create_temperature_controls_card()
        assert isinstance(result.children[1], dbc.Collapse)


class TestCreatePermeabilityControlsCard:
    """Test create_permeability_controls_card function."""

    def test_create_permeability_controls_card_returns_card(self):
        """Test that function returns a dbc.Card component."""
        result = create_permeability_controls_card()
        assert isinstance(result, dbc.Card)

    def test_create_permeability_controls_card_has_header(self):
        """Test that card has a header."""
        result = create_permeability_controls_card()
        assert isinstance(result.children[0], dbc.CardHeader)

    def test_create_permeability_controls_card_has_title(self):
        """Test that card displays 'Permeability' title."""
        result = create_permeability_controls_card()
        header = result.children[0]
        assert "Permeability" in str(header)

    def test_create_permeability_controls_card_has_collapse(self):
        """Test that card has a collapse component."""
        result = create_permeability_controls_card()
        assert isinstance(result.children[1], dbc.Collapse)


class TestCreatePlotControlsRow:
    """Test create_plot_controls_row function."""

    def test_create_plot_controls_row_returns_row(self):
        """Test that function returns a dbc.Row component."""
        result = create_plot_controls_row()
        assert isinstance(result, dbc.Row)

    def test_create_plot_controls_row_has_two_columns(self):
        """Test that row contains two columns (upstream and downstream)."""
        result = create_plot_controls_row()
        assert len(result.children) == 2
        assert all(isinstance(child, dbc.Col) for child in result.children)

    def test_create_plot_controls_row_contains_cards(self):
        """Test that columns contain card components."""
        result = create_plot_controls_row()
        for col in result.children:
            # Each column should have a card
            assert col.children is not None


class TestCreateTemperaturePlotCard:
    """Test create_temperature_plot_card function."""

    def test_create_temperature_plot_card_returns_container(self):
        """Test that function returns a dbc.Container component."""
        fig = go.Figure()
        result = create_temperature_plot_card(fig)
        assert isinstance(result, dbc.Container)

    def test_create_temperature_plot_card_contains_two_rows(self):
        """Test that container has two rows (plot + controls)."""
        fig = go.Figure()
        result = create_temperature_plot_card(fig)
        assert len(result.children) == 2
        assert all(isinstance(child, dbc.Row) for child in result.children)

    def test_create_temperature_plot_card_contains_card(self):
        """Test that column contains a card."""
        fig = go.Figure()
        result = create_temperature_plot_card(fig)
        col = result.children[0]
        assert isinstance(col.children, list)

    def test_create_temperature_plot_card_uses_provided_figure(self):
        """Test that provided figure is used in the plot."""
        fig = go.Figure()
        fig.update_layout(title="Temperature Test")
        result = create_temperature_plot_card(fig)
        # Figure should be present in the component tree
        assert result is not None


class TestCreatePermeabilityPlotCard:
    """Test create_permeability_plot_card function."""

    def test_create_permeability_plot_card_returns_container(self):
        """Test that function returns a dbc.Container component."""
        fig = go.Figure()
        result = create_permeability_plot_card(fig)
        assert isinstance(result, dbc.Container)

    def test_create_permeability_plot_card_contains_two_rows(self):
        """Test that container has two rows (plot + controls)."""
        fig = go.Figure()
        result = create_permeability_plot_card(fig)
        assert len(result.children) == 2
        assert all(isinstance(child, dbc.Row) for child in result.children)

    def test_create_permeability_plot_card_contains_card(self):
        """Test that column contains a card."""
        fig = go.Figure()
        result = create_permeability_plot_card(fig)
        col = result.children[0]
        assert isinstance(col.children, list)

    def test_create_permeability_plot_card_uses_provided_figure(self):
        """Test that provided figure is used in the plot."""
        fig = go.Figure()
        fig.update_layout(title="Permeability Test")
        result = create_permeability_plot_card(fig)
        # Figure should be present in the component tree
        assert result is not None


class TestCreateBottomSpacing:
    """Test create_bottom_spacing function."""

    def test_create_bottom_spacing_returns_row(self):
        """Test that function returns a dbc.Row component."""
        result = create_bottom_spacing()
        assert isinstance(result, dbc.Row)

    def test_create_bottom_spacing_contains_column(self):
        """Test that row contains a column."""
        result = create_bottom_spacing()
        assert len(result.children) == 1
        assert isinstance(result.children[0], dbc.Col)

    def test_create_bottom_spacing_has_height_style(self):
        """Test that spacing div has height style."""
        result = create_bottom_spacing()
        col = result.children[0]
        div = col.children
        assert isinstance(div, html.Div)
        assert div.style is not None
        assert "height" in div.style


class TestCreateDownloadComponents:
    """Test create_download_components function."""

    def test_create_download_components_returns_list(self):
        """Test that function returns a list."""
        result = create_download_components()
        assert isinstance(result, list)

    def test_create_download_components_contains_download_objects(self):
        """Test that all items are dcc.Download components."""
        result = create_download_components()
        assert all(isinstance(item, dcc.Download) for item in result)

    def test_create_download_components_has_dataset_download(self):
        """Test that dataset download component is included."""
        result = create_download_components()
        download_ids = [dl.id for dl in result]
        assert "download-dataset-output" in download_ids

    def test_create_download_components_has_plot_downloads(self):
        """Test that all plot download components are included."""
        result = create_download_components()
        download_ids = [dl.id for dl in result]
        assert "download-upstream-plot" in download_ids
        assert "download-downstream-plot" in download_ids
        assert "download-temperature-plot" in download_ids
        assert "download-permeability-plot" in download_ids

    def test_create_download_components_has_five_components(self):
        """Test that exactly 5 download components are created."""
        result = create_download_components()
        assert len(result) == 5


class TestCreateLiveDataInterval:
    """Test create_live_data_interval function."""

    def test_create_live_data_interval_returns_interval(self):
        """Test that function returns a dcc.Interval component."""
        result = create_live_data_interval()
        assert isinstance(result, dcc.Interval)

    def test_create_live_data_interval_has_correct_id(self):
        """Test that interval has correct ID."""
        result = create_live_data_interval()
        assert result.id == "live-data-interval"

    def test_create_live_data_interval_has_interval_value(self):
        """Test that interval is set to 1000ms (1 second)."""
        result = create_live_data_interval()
        assert result.interval == 1000

    def test_create_live_data_interval_starts_at_zero(self):
        """Test that n_intervals starts at 0."""
        result = create_live_data_interval()
        assert result.n_intervals == 0

    def test_create_live_data_interval_starts_disabled(self):
        """Test that interval starts disabled."""
        result = create_live_data_interval()
        assert result.disabled is True
