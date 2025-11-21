"""Integration tests for PermeabilityPlot class.

Tests complete workflows with mock datasets to verify end-to-end functionality.
"""

from unittest.mock import MagicMock, patch

import numpy as np
from uncertainties import ufloat

from shield_das.figures.permeability_plot import PermeabilityPlot


def create_mock_dataset(
    has_permeability_data: bool = True,
    num_points: int = 5,
    use_ufloat: bool = True,
    include_errors: bool = True,
) -> MagicMock:
    """Create a mock dataset with permeability data for testing.

    Args:
        has_permeability_data: Whether dataset has permeability values
        num_points: Number of data points to generate
        use_ufloat: Whether to use ufloat objects (with uncertainties)
        include_errors: Whether to include error bars

    Returns:
        Mock dataset object
    """
    dataset = MagicMock()
    dataset.metadata = {"nickname": "Test Sample"}

    if has_permeability_data:
        # Generate realistic temperature and permeability data
        temps = np.linspace(300.0, 500.0, num_points)

        if use_ufloat and include_errors:
            # Create ufloat objects with uncertainties
            permeabilities = [
                ufloat(1e-8 * (1 + 0.1 * i), 1e-9 * (1 + 0.05 * i))
                for i in range(num_points)
            ]
        elif use_ufloat:
            # ufloat with zero uncertainty
            permeabilities = [
                ufloat(1e-8 * (1 + 0.1 * i), 0) for i in range(num_points)
            ]
        else:
            # Plain float values
            permeabilities = [1e-8 * (1 + 0.1 * i) for i in range(num_points)]

        dataset.permeability_values = list(zip(temps, permeabilities))
    else:
        dataset.permeability_values = []

    return dataset


class TestPermeabilityPlotIntegrationWithData:
    """Integration tests with datasets containing permeability data."""

    @patch("shield_das.figures.permeability_plot.evaluate_permeability_values")
    @patch("shield_das.figures.permeability_plot.fit_permeability_data")
    @patch("shield_das.figures.permeability_plot.import_htm_data")
    def test_complete_workflow_with_single_dataset(
        self, mock_htm, mock_fit, mock_evaluate
    ):
        """Test complete plot generation with one dataset."""
        # Setup
        dataset = create_mock_dataset(num_points=5)
        mock_evaluate.return_value = (
            [300.0, 400.0, 500.0],
            [1e-8, 1.5e-8, 2e-8],
            None,
            [1e-9, 1e-9, 1e-9],
            [1e-9, 1e-9, 1e-9],
            [1e-9, 1e-9, 1e-9],
        )
        mock_fit.return_value = (
            np.array([300.0, 500.0]),
            np.array([0.9e-8, 2.1e-8]),
        )
        mock_htm.return_value = (
            [np.array([300.0, 400.0])],
            [np.array([1.2e-8, 1.8e-8])],
            ["HTM Reference [R]"],
        )

        # Execute
        plot = PermeabilityPlot(datasets=[dataset], plot_id="integration_test")
        result = plot.generate()

        # Verify
        assert result is not None
        assert mock_evaluate.called
        assert mock_fit.called
        assert mock_htm.called
        # FigureResampler creates separate traces for each data point + fit + HTM
        # 3 data points + 1 fit line + 1 HTM trace = 5 traces
        assert len(result.data) == 5

    @patch("shield_das.figures.permeability_plot.evaluate_permeability_values")
    @patch("shield_das.figures.permeability_plot.fit_permeability_data")
    @patch("shield_das.figures.permeability_plot.import_htm_data")
    def test_complete_workflow_with_multiple_datasets(
        self, mock_htm, mock_fit, mock_evaluate
    ):
        """Test complete plot generation with multiple datasets."""
        # Setup
        dataset1 = create_mock_dataset(num_points=5)
        dataset2 = create_mock_dataset(num_points=7)
        dataset1.metadata["nickname"] = "Sample A"
        dataset2.metadata["nickname"] = "Sample B"

        mock_evaluate.side_effect = [
            (
                [300.0, 400.0],
                [1e-8, 1.5e-8],
                None,
                [1e-9, 1e-9],
                [1e-9, 1e-9],
                [1e-9, 1e-9],
            ),
            (
                [350.0, 450.0],
                [1.2e-8, 1.8e-8],
                None,
                [1e-9, 1e-9],
                [1e-9, 1e-9],
                [1e-9, 1e-9],
            ),
        ]
        mock_fit.side_effect = [
            (np.array([300.0, 400.0]), np.array([0.9e-8, 1.6e-8])),
            (np.array([350.0, 450.0]), np.array([1.1e-8, 1.9e-8])),
        ]
        mock_htm.return_value = (
            [np.array([300.0, 400.0])],
            [np.array([1.2e-8, 1.8e-8])],
            ["HTM Reference [R]"],
        )

        # Execute
        plot = PermeabilityPlot(
            datasets=[dataset1, dataset2], plot_id="integration_test"
        )
        result = plot.generate()

        # Verify
        assert result is not None
        # evaluate_permeability_values is called once with all datasets
        assert mock_evaluate.call_count == 1
        # fit is called once for the combined data
        assert mock_fit.call_count == 1
        # All data points + fit line + HTM data
        assert len(result.data) >= 3

    @patch("shield_das.figures.permeability_plot.evaluate_permeability_values")
    @patch("shield_das.figures.permeability_plot.fit_permeability_data")
    @patch("shield_das.figures.permeability_plot.import_htm_data")
    def test_workflow_with_error_bars(self, mock_htm, mock_fit, mock_evaluate):
        """Test workflow with error bars enabled."""
        # Setup
        dataset = create_mock_dataset(num_points=5, use_ufloat=True)
        mock_evaluate.return_value = (
            [300.0, 400.0],
            [1e-8, 1.5e-8],
            None,
            [1e-9, 1e-9],
            [1e-9, 1e-9],
            [1e-9, 1e-9],
        )
        mock_fit.return_value = (
            np.array([300.0, 400.0]),
            np.array([0.9e-8, 1.6e-8]),
        )
        mock_htm.return_value = ([], [], [])

        # Execute
        plot = PermeabilityPlot(datasets=[dataset], plot_id="integration_test")
        result = plot.generate()

        # Verify
        assert result is not None
        assert mock_evaluate.called
        # Should have data points (each as separate trace) + fit line
        # FigureResampler creates separate traces: 2 data + 1 fit = 3
        assert len(result.data) == 3

    @patch("shield_das.figures.permeability_plot.evaluate_permeability_values")
    @patch("shield_das.figures.permeability_plot.fit_permeability_data")
    @patch("shield_das.figures.permeability_plot.import_htm_data")
    def test_workflow_verifies_layout_configuration(
        self, mock_htm, mock_fit, mock_evaluate
    ):
        """Test that layout is configured correctly during workflow."""
        # Setup
        dataset = create_mock_dataset(num_points=5)
        mock_evaluate.return_value = (
            [300.0, 400.0],
            [1e-8, 1.5e-8],
            None,
            [1e-9, 1e-9],
            [1e-9, 1e-9],
            [1e-9, 1e-9],
        )
        mock_fit.return_value = (
            np.array([300.0, 400.0]),
            np.array([0.9e-8, 1.6e-8]),
        )
        mock_htm.return_value = ([], [], [])

        # Execute
        plot = PermeabilityPlot(datasets=[dataset], plot_id="integration_test")
        result = plot.generate()

        # Verify
        assert result is not None
        layout = result.layout
        assert layout.xaxis.title.text == "1000/T (K⁻¹)"
        assert layout.yaxis.title.text == "Permeability (mol/(m·s·Pa^0.5))"
        assert layout.yaxis.type == "linear"
        assert layout.hovermode == "closest"
        assert layout.template is not None


class TestPermeabilityPlotIntegrationWithoutData:
    """Integration tests with datasets lacking permeability data."""

    @patch("shield_das.figures.permeability_plot.import_htm_data")
    def test_workflow_with_no_permeability_data(self, mock_htm):
        """Test workflow when dataset has no permeability values."""
        # Setup
        dataset = create_mock_dataset(has_permeability_data=False)
        mock_htm.return_value = ([], [], [])

        # Execute
        plot = PermeabilityPlot(datasets=[dataset], plot_id="integration_test")
        result = plot.generate()

        # Verify
        assert result is not None
        # Should show "No data to display" message
        assert len(result.layout.annotations) > 0
        assert "No data" in result.layout.annotations[0].text

    @patch("shield_das.figures.permeability_plot.import_htm_data")
    def test_workflow_with_empty_dataset_list(self, mock_htm):
        """Test workflow with no datasets provided."""
        # Setup
        mock_htm.return_value = ([], [], [])

        # Execute
        plot = PermeabilityPlot(datasets=[], plot_id="integration_test")
        result = plot.generate()

        # Verify
        assert result is not None
        assert len(result.layout.annotations) > 0
        assert "No data" in result.layout.annotations[0].text

    @patch("shield_das.figures.permeability_plot.evaluate_permeability_values")
    @patch("shield_das.figures.permeability_plot.fit_permeability_data")
    @patch("shield_das.figures.permeability_plot.import_htm_data")
    def test_workflow_with_mixed_datasets_some_without_data(
        self, mock_htm, mock_fit, mock_evaluate
    ):
        """Test workflow when some datasets lack permeability data."""
        # Setup
        dataset1 = create_mock_dataset(has_permeability_data=True, num_points=5)
        dataset2 = create_mock_dataset(has_permeability_data=False)
        mock_evaluate.return_value = (
            [300.0, 400.0],
            [1e-8, 1.5e-8],
            None,
            [1e-9, 1e-9],
            [1e-9, 1e-9],
            [1e-9, 1e-9],
        )
        mock_fit.return_value = (
            np.array([300.0, 400.0]),
            np.array([0.9e-8, 1.6e-8]),
        )
        mock_htm.return_value = ([], [], [])

        # Execute
        plot = PermeabilityPlot(
            datasets=[dataset1, dataset2], plot_id="integration_test"
        )
        result = plot.generate()

        # Verify
        assert result is not None
        # Both datasets are passed together to evaluate_permeability_values
        assert mock_evaluate.call_count == 1
        assert mock_fit.call_count == 1
        # Should add traces for dataset1
        assert len(result.data) >= 1


class TestPermeabilityPlotIntegrationEdgeCases:
    """Integration tests for edge cases and error conditions."""

    @patch("shield_das.figures.permeability_plot.evaluate_permeability_values")
    @patch("shield_das.figures.permeability_plot.fit_permeability_data")
    @patch("shield_das.figures.permeability_plot.import_htm_data")
    def test_workflow_with_single_data_point(self, mock_htm, mock_fit, mock_evaluate):
        """Test workflow with only one data point (insufficient for fit)."""
        # Setup
        dataset = create_mock_dataset(num_points=1)
        mock_evaluate.return_value = (
            [300.0],
            [1e-8],
            None,
            [1e-9],
            [1e-9],
            [1e-9],
        )
        mock_fit.return_value = (None, None)  # Can't fit with 1 point
        mock_htm.return_value = ([], [], [])

        # Execute
        plot = PermeabilityPlot(datasets=[dataset], plot_id="integration_test")
        result = plot.generate()

        # Verify
        assert result is not None
        assert mock_evaluate.called
        # Fit should not be called with only 1 point (needs >= 2)
        assert not mock_fit.called
        # Should add only data point, no fit line
        assert len(result.data) == 1

    @patch("shield_das.figures.permeability_plot.import_htm_data")
    def test_workflow_with_htm_data_only(self, mock_htm):
        """Test workflow with only HTM reference data (no experimental data)."""
        # Setup
        dataset = create_mock_dataset(has_permeability_data=False)
        htm_x = [np.array([300.0, 400.0])]
        htm_y = [np.array([1e-8, 1.5e-8])]
        htm_labels = ["HTM Reference [R]"]
        mock_htm.return_value = (htm_x, htm_y, htm_labels)

        # Execute
        plot = PermeabilityPlot(datasets=[dataset], plot_id="integration_test")
        result = plot.generate()

        # Verify
        assert result is not None
        # HTM data is not added when there's no experimental data
        assert not mock_htm.called
        # Should show "No data" message
        assert len(result.layout.annotations) > 0
        # No HTM data added (requires experimental data first)
        assert len(result.data) == 0

    @patch("shield_das.figures.permeability_plot.evaluate_permeability_values")
    @patch("shield_das.figures.permeability_plot.fit_permeability_data")
    @patch("shield_das.figures.permeability_plot.import_htm_data")
    def test_workflow_handles_evaluate_returning_empty(
        self, mock_htm, mock_fit, mock_evaluate
    ):
        """Test workflow when evaluate_permeability_values returns empty data."""
        # Setup
        dataset = create_mock_dataset(has_permeability_data=True, num_points=5)
        mock_evaluate.return_value = ([], [], None, [], [], [])  # Empty evaluation
        mock_htm.return_value = ([], [], [])

        # Execute
        plot = PermeabilityPlot(datasets=[dataset], plot_id="integration_test")
        result = plot.generate()

        # Verify
        assert result is not None
        assert mock_evaluate.called
        assert not mock_fit.called  # Shouldn't attempt fit with no data
        assert len(result.data) == 0  # No traces added

    @patch("shield_das.figures.permeability_plot.evaluate_permeability_values")
    @patch("shield_das.figures.permeability_plot.fit_permeability_data")
    @patch("shield_das.figures.permeability_plot.import_htm_data")
    def test_workflow_with_very_large_dataset(self, mock_htm, mock_fit, mock_evaluate):
        """Test workflow with a large number of data points."""
        # Setup
        dataset = create_mock_dataset(num_points=100)
        mock_evaluate.return_value = (
            list(np.linspace(300.0, 500.0, 100)),
            list(np.linspace(1e-8, 2e-8, 100)),
            None,
            list(np.ones(100) * 1e-9),
            list(np.ones(100) * 1e-9),
            list(np.ones(100) * 1e-9),
        )
        mock_fit.return_value = (
            np.array([300.0, 500.0]),
            np.array([1e-8, 2e-8]),
        )
        mock_htm.return_value = ([], [], [])

        # Execute
        plot = PermeabilityPlot(datasets=[dataset], plot_id="integration_test")
        result = plot.generate()

        # Verify
        assert result is not None
        assert mock_evaluate.called
        assert mock_fit.called
        assert len(result.data) >= 2  # Data points + fit line

    @patch("shield_das.figures.permeability_plot.evaluate_permeability_values")
    @patch("shield_das.figures.permeability_plot.fit_permeability_data")
    @patch("shield_das.figures.permeability_plot.import_htm_data")
    def test_workflow_trace_names_cleaned_correctly(
        self, mock_htm, mock_fit, mock_evaluate
    ):
        """Test that trace names have [R] suffix removed in final plot."""
        # Setup
        dataset = create_mock_dataset(num_points=5)
        mock_evaluate.return_value = (
            [300.0, 400.0],
            [1e-8, 1.5e-8],
            None,
            [1e-9, 1e-9],
            [1e-9, 1e-9],
            [1e-9, 1e-9],
        )
        mock_fit.return_value = (
            np.array([300.0, 400.0]),
            np.array([0.9e-8, 1.6e-8]),
        )
        htm_x = [np.array([300.0, 400.0])]
        htm_y = [np.array([1.2e-8, 1.8e-8])]
        htm_labels = ["HTM Sample [R]"]
        mock_htm.return_value = (htm_x, htm_y, htm_labels)

        # Execute
        plot = PermeabilityPlot(datasets=[dataset], plot_id="integration_test")
        result = plot.generate()

        # Verify
        assert result is not None
        # Check that HTM trace name has [R] removed
        htm_trace = result.data[-1]  # HTM trace should be last
        assert htm_trace.name == "HTM Sample"
        assert "[R]" not in htm_trace.name
