"""
Tests for temperature plot functionality in DataPlotter (v1.2 features).

This module tests the new temperature plotting capabilities introduced in v1.2,
including:
- Temperature data processing (process_csv_v1_2)
- Dataset creation with temperature data (create_dataset_v1_2)
- Temperature plot generation (_generate_temperature_plot)
- Integration with existing plotting infrastructure
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from shield_das.data_plotter import DataPlotter


class TestProcessCSVV1_2:
    """Test CSV processing for v1.2 datasets with temperature data."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def v1_2_metadata(self):
        """Create valid v1.2 metadata with thermocouple information."""
        return {
            "version": "1.2",
            "run_info": {
                "start_time": "2025-11-04 10:00:00",
                "data_filename": "test_data.csv",
            },
            "gauges": [
                {
                    "name": "Upstream_Gauge",
                    "type": "Baratron626D_Gauge",
                    "gauge_location": "upstream",
                    "full_scale_torr": 1000.0,
                },
                {
                    "name": "Downstream_Gauge",
                    "type": "Baratron626D_Gauge",
                    "gauge_location": "downstream",
                    "full_scale_torr": 1.0,
                },
            ],
            "thermocouples": [
                {
                    "name": "TC1",
                    "type": "K",
                    "location": "chamber",
                }
            ],
        }

    @pytest.fixture
    def v1_2_csv_data(self, temp_dir, v1_2_metadata):
        """Create a temporary directory with v1.2 CSV and metadata files."""
        # Create metadata file
        metadata_path = os.path.join(temp_dir, "run_metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(v1_2_metadata, f)

        # Create CSV file with temperature data
        # Format matches actual v1.2 CSV:
        # numpy genfromtxt converts column names:
        # "Local_temperature (C)" -> "Local_temperature_C"
        # "Thermocouple_TC1 (mV)" -> "Thermocouple_TC1_mV"
        # But code expects "{name}_Voltage_mV", so column should be "TC1_Voltage (mV)"
        csv_path = os.path.join(temp_dir, "test_data.csv")
        csv_content = """RealTimestamp,Local_temperature (C),TC1_Voltage (mV),Upstream_Gauge_Voltage (V),Downstream_Gauge_Voltage (V)
2025-11-04 10:00:00.000,25.0,1.0,1.0,0.5
2025-11-04 10:00:01.000,26.0,1.1,1.1,0.6
2025-11-04 10:00:02.000,27.0,1.2,1.2,0.7
2025-11-04 10:00:03.000,28.0,1.3,1.3,0.8
"""
        with open(csv_path, "w") as f:
            f.write(csv_content)

        return temp_dir

    def test_process_csv_v1_2_returns_six_values(self, v1_2_csv_data, v1_2_metadata):
        """Test that process_csv_v1_2 returns 6 values."""
        plotter = DataPlotter()
        result = plotter.process_csv_v1_2(v1_2_metadata, v1_2_csv_data)

        assert len(result) == 6, "process_csv_v1_2 should return 6 values"

    def test_process_csv_v1_2_extracts_thermocouple_name(
        self, v1_2_csv_data, v1_2_metadata
    ):
        """Test that thermocouple name is extracted correctly."""
        plotter = DataPlotter()
        (
            time_data,
            upstream_data,
            downstream_data,
            local_temp,
            thermocouple_temp,
            thermocouple_name,
        ) = plotter.process_csv_v1_2(v1_2_metadata, v1_2_csv_data)

        assert thermocouple_name == "TC1"

    def test_process_csv_v1_2_temperature_data_types(
        self, v1_2_csv_data, v1_2_metadata
    ):
        """Test that temperature data is returned as numpy arrays."""
        plotter = DataPlotter()
        (
            time_data,
            upstream_data,
            downstream_data,
            local_temp,
            thermocouple_temp,
            thermocouple_name,
        ) = plotter.process_csv_v1_2(v1_2_metadata, v1_2_csv_data)

        assert isinstance(local_temp, np.ndarray)
        assert isinstance(thermocouple_temp, np.ndarray)
        assert len(local_temp) == 4
        assert len(thermocouple_temp) == 4

    def test_process_csv_v1_2_temperature_conversion(
        self, v1_2_csv_data, v1_2_metadata
    ):
        """Test that thermocouple voltage is converted to temperature."""
        plotter = DataPlotter()
        (
            time_data,
            upstream_data,
            downstream_data,
            local_temp,
            thermocouple_temp,
            thermocouple_name,
        ) = plotter.process_csv_v1_2(v1_2_metadata, v1_2_csv_data)

        # Local temperature should match CSV values
        assert np.allclose(local_temp, [25.0, 26.0, 27.0, 28.0])

        # Thermocouple temperature should be converted from mV
        # 1.0 mV ≈ 24.6°C, values should be reasonable
        assert all(0 < temp < 100 for temp in thermocouple_temp)


class TestCreateDatasetV1_2:
    """Test dataset creation with temperature data."""

    def test_create_dataset_v1_2_stores_temperature_data(self):
        """Test that temperature data is stored in dataset dictionary."""
        plotter = DataPlotter()

        time_data = np.array([0, 1, 2, 3])
        upstream_data = {
            "pressure_data": np.array([1.0, 1.1, 1.2, 1.3]),
            "error_data": np.array([0.1, 0.1, 0.1, 0.1]),
        }
        downstream_data = {
            "pressure_data": np.array([0.5, 0.6, 0.7, 0.8]),
            "error_data": np.array([0.05, 0.05, 0.05, 0.05]),
        }
        local_temp = np.array([25.0, 26.0, 27.0, 28.0])
        thermocouple_temp = np.array([24.5, 25.5, 26.5, 27.5])
        thermocouple_name = "TC1"

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create minimal metadata
            metadata_path = os.path.join(tmpdir, "run_metadata.json")
            with open(metadata_path, "w") as f:
                json.dump({"run_info": {}}, f)

            plotter.create_dataset_v1_2(
                dataset_path=tmpdir,
                dataset_name="Test Dataset",
                time_data=time_data,
                upstream_data=upstream_data,
                downstream_data=downstream_data,
                local_temperature_data=local_temp,
                thermocouple_data=thermocouple_temp,
                thermocouple_name=thermocouple_name,
            )

            # Check that dataset was created
            assert len(plotter.datasets) == 1

            dataset = plotter.datasets["dataset_1"]
            assert "local_temperature_data" in dataset
            assert "thermocouple_data" in dataset
            assert "thermocouple_name" in dataset

            np.testing.assert_array_equal(dataset["local_temperature_data"], local_temp)
            np.testing.assert_array_equal(
                dataset["thermocouple_data"], thermocouple_temp
            )
            assert dataset["thermocouple_name"] == "TC1"

    def test_create_dataset_v1_2_preserves_existing_fields(self):
        """Test that standard dataset fields are still present."""
        plotter = DataPlotter()

        with tempfile.TemporaryDirectory() as tmpdir:
            metadata_path = os.path.join(tmpdir, "run_metadata.json")
            with open(metadata_path, "w") as f:
                json.dump({"run_info": {}}, f)

            plotter.create_dataset_v1_2(
                dataset_path=tmpdir,
                dataset_name="Test Dataset",
                time_data=np.array([0, 1, 2]),
                upstream_data={
                    "pressure_data": np.array([1.0, 1.1, 1.2]),
                    "error_data": np.array([0.1, 0.1, 0.1]),
                },
                downstream_data={
                    "pressure_data": np.array([0.5, 0.6, 0.7]),
                    "error_data": np.array([0.05, 0.05, 0.05]),
                },
                local_temperature_data=np.array([25.0, 26.0, 27.0]),
                thermocouple_data=np.array([24.5, 25.5, 26.5]),
                thermocouple_name="TC1",
            )

            dataset = plotter.datasets["dataset_1"]
            # Check standard fields
            assert dataset["name"] == "Test Dataset"
            assert "colour" in dataset
            assert "dataset_path" in dataset
            assert dataset["live_data"] is False
            assert "time_data" in dataset
            assert "upstream_data" in dataset
            assert "downstream_data" in dataset


class TestGenerateTemperaturePlot:
    """Test temperature plot generation."""

    def test_temperature_plot_with_v1_2_dataset(self):
        """Test that temperature plot is generated for v1.2 datasets."""
        plotter = DataPlotter()

        with tempfile.TemporaryDirectory() as tmpdir:
            metadata_path = os.path.join(tmpdir, "run_metadata.json")
            with open(metadata_path, "w") as f:
                json.dump({"run_info": {}}, f)

            plotter.create_dataset_v1_2(
                dataset_path=tmpdir,
                dataset_name="Test Dataset",
                time_data=np.array([0, 1, 2, 3]),
                upstream_data={
                    "pressure_data": np.array([1.0, 1.1, 1.2, 1.3]),
                    "error_data": np.array([0.1, 0.1, 0.1, 0.1]),
                },
                downstream_data={
                    "pressure_data": np.array([0.5, 0.6, 0.7, 0.8]),
                    "error_data": np.array([0.05, 0.05, 0.05, 0.05]),
                },
                local_temperature_data=np.array([25.0, 26.0, 27.0, 28.0]),
                thermocouple_data=np.array([24.5, 25.5, 26.5, 27.5]),
                thermocouple_name="TC1",
            )

            fig = plotter._generate_temperature_plot()

            # Check that figure has data
            assert len(fig.data) > 0
            # Should have 2 traces (local + thermocouple)
            assert len(fig.data) == 2

    def test_temperature_plot_trace_names(self):
        """Test that trace names are correctly formatted."""
        plotter = DataPlotter()

        with tempfile.TemporaryDirectory() as tmpdir:
            metadata_path = os.path.join(tmpdir, "run_metadata.json")
            with open(metadata_path, "w") as f:
                json.dump({"run_info": {}}, f)

            plotter.create_dataset_v1_2(
                dataset_path=tmpdir,
                dataset_name="Test Dataset",
                time_data=np.array([0, 1, 2]),
                upstream_data={
                    "pressure_data": np.array([1.0, 1.1, 1.2]),
                    "error_data": np.array([0.1, 0.1, 0.1]),
                },
                downstream_data={
                    "pressure_data": np.array([0.5, 0.6, 0.7]),
                    "error_data": np.array([0.05, 0.05, 0.05]),
                },
                local_temperature_data=np.array([25.0, 26.0, 27.0]),
                thermocouple_data=np.array([24.5, 25.5, 26.5]),
                thermocouple_name="TC1",
            )

            fig = plotter._generate_temperature_plot()

            # Check trace names
            trace_names = [trace.name for trace in fig.data]
            assert "Local temperature (C)" in trace_names
            assert "TC1 (C)" in trace_names

    def test_temperature_plot_no_data_message(self):
        """Test that message is shown when no temperature data is available."""
        plotter = DataPlotter()

        # Create a dataset without temperature data (v1.0 style)
        plotter.datasets["dataset_1"] = {
            "name": "Old Dataset",
            "colour": "#000000",
            "dataset_path": "/fake/path",
            "live_data": False,
            "time_data": np.array([0, 1, 2]),
            "upstream_data": {
                "pressure_data": np.array([1.0, 1.1, 1.2]),
                "error_data": np.array([0.1, 0.1, 0.1]),
            },
            "downstream_data": {
                "pressure_data": np.array([0.5, 0.6, 0.7]),
                "error_data": np.array([0.05, 0.05, 0.05]),
            },
            "valve_times": {},
        }

        fig = plotter._generate_temperature_plot()

        # Should have annotation instead of data traces
        assert len(fig.layout.annotations) > 0
        assert any(
            "No temperature data available" in str(ann.text)
            for ann in fig.layout.annotations
        )

    def test_temperature_plot_with_multiple_datasets(self):
        """Test temperature plot with multiple v1.2 datasets."""
        plotter = DataPlotter()

        for i in range(2):
            with tempfile.TemporaryDirectory() as tmpdir:
                metadata_path = os.path.join(tmpdir, "run_metadata.json")
                with open(metadata_path, "w") as f:
                    json.dump({"run_info": {}}, f)

                plotter.create_dataset_v1_2(
                    dataset_path=tmpdir,
                    dataset_name=f"Dataset {i + 1}",
                    time_data=np.array([0, 1, 2]),
                    upstream_data={
                        "pressure_data": np.array([1.0, 1.1, 1.2]),
                        "error_data": np.array([0.1, 0.1, 0.1]),
                    },
                    downstream_data={
                        "pressure_data": np.array([0.5, 0.6, 0.7]),
                        "error_data": np.array([0.05, 0.05, 0.05]),
                    },
                    local_temperature_data=np.array([25.0, 26.0, 27.0]),
                    thermocouple_data=np.array([24.5, 25.5, 26.5]),
                    thermocouple_name=f"TC{i + 1}",
                )

        fig = plotter._generate_temperature_plot()

        # Should have 4 traces (2 datasets × 2 traces each)
        assert len(fig.data) == 4

    def test_temperature_plot_line_styles(self):
        """Test that local temperature is dashed and thermocouple is solid."""
        plotter = DataPlotter()

        with tempfile.TemporaryDirectory() as tmpdir:
            metadata_path = os.path.join(tmpdir, "run_metadata.json")
            with open(metadata_path, "w") as f:
                json.dump({"run_info": {}}, f)

            plotter.create_dataset_v1_2(
                dataset_path=tmpdir,
                dataset_name="Test Dataset",
                time_data=np.array([0, 1, 2]),
                upstream_data={
                    "pressure_data": np.array([1.0, 1.1, 1.2]),
                    "error_data": np.array([0.1, 0.1, 0.1]),
                },
                downstream_data={
                    "pressure_data": np.array([0.5, 0.6, 0.7]),
                    "error_data": np.array([0.05, 0.05, 0.05]),
                },
                local_temperature_data=np.array([25.0, 26.0, 27.0]),
                thermocouple_data=np.array([24.5, 25.5, 26.5]),
                thermocouple_name="TC1",
            )

            fig = plotter._generate_temperature_plot()

            # Find local temperature trace (should be dashed)
            local_trace = next(
                trace for trace in fig.data if "Local temperature" in trace.name
            )
            assert local_trace.line.dash == "dash"

            # Find thermocouple trace (should be solid, which is None or 'solid')
            tc_trace = next(trace for trace in fig.data if "TC1" in trace.name)
            assert tc_trace.line.dash in [None, "solid"]


class TestGenerateBothPlotsWithTemperature:
    """Test that _generate_both_plots now returns 3 plots."""

    def test_generate_both_plots_returns_three_plots(self):
        """Test that _generate_both_plots returns 3 plots (upstream, downstream, temperature)."""
        plotter = DataPlotter()

        with tempfile.TemporaryDirectory() as tmpdir:
            metadata_path = os.path.join(tmpdir, "run_metadata.json")
            with open(metadata_path, "w") as f:
                json.dump({"run_info": {}}, f)

            plotter.create_dataset_v1_2(
                dataset_path=tmpdir,
                dataset_name="Test Dataset",
                time_data=np.array([0, 1, 2]),
                upstream_data={
                    "pressure_data": np.array([1.0, 1.1, 1.2]),
                    "error_data": np.array([0.1, 0.1, 0.1]),
                },
                downstream_data={
                    "pressure_data": np.array([0.5, 0.6, 0.7]),
                    "error_data": np.array([0.05, 0.05, 0.05]),
                },
                local_temperature_data=np.array([25.0, 26.0, 27.0]),
                thermocouple_data=np.array([24.5, 25.5, 26.5]),
                thermocouple_name="TC1",
            )

            plots = plotter._generate_both_plots(
                show_error_bars_upstream=True,
                show_error_bars_downstream=True,
                show_valve_times_upstream=False,
                show_valve_times_downstream=False,
            )

            # Should now return 3 plots instead of 2
            assert len(plots) == 3

    def test_generate_both_plots_order(self):
        """Test that plots are returned in correct order: upstream, downstream, temperature."""
        plotter = DataPlotter()

        with tempfile.TemporaryDirectory() as tmpdir:
            metadata_path = os.path.join(tmpdir, "run_metadata.json")
            with open(metadata_path, "w") as f:
                json.dump({"run_info": {}}, f)

            plotter.create_dataset_v1_2(
                dataset_path=tmpdir,
                dataset_name="Test Dataset",
                time_data=np.array([0, 1, 2]),
                upstream_data={
                    "pressure_data": np.array([1.0, 1.1, 1.2]),
                    "error_data": np.array([0.1, 0.1, 0.1]),
                },
                downstream_data={
                    "pressure_data": np.array([0.5, 0.6, 0.7]),
                    "error_data": np.array([0.05, 0.05, 0.05]),
                },
                local_temperature_data=np.array([25.0, 26.0, 27.0]),
                thermocouple_data=np.array([24.5, 25.5, 26.5]),
                thermocouple_name="TC1",
            )

            upstream_plot, downstream_plot, temp_plot = plotter._generate_both_plots(
                show_error_bars_upstream=True,
                show_error_bars_downstream=True,
                show_valve_times_upstream=False,
                show_valve_times_downstream=False,
            )

            # Basic check that we got different plots
            # (they should have different y-axis titles)
            assert (
                upstream_plot.layout.yaxis.title.text
                != temp_plot.layout.yaxis.title.text
            )
            assert (
                downstream_plot.layout.yaxis.title.text
                != temp_plot.layout.yaxis.title.text
            )
