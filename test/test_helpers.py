import numpy as np
import pytest

from shield_das.helpers import (
    calculate_error,
    import_htm_data,
    voltage_to_pressure,
)


class TestVoltageToPressu:
    """Tests for voltage_to_pressure function"""

    def test_converts_voltage_correctly_1000_torr_scale(self):
        """Test voltage conversion for 1000 Torr full scale"""
        voltage = np.array([5.0])
        full_scale = 1000

        pressure = voltage_to_pressure(voltage, full_scale)

        assert pressure[0] == pytest.approx(500, rel=0.01)

    def test_converts_voltage_correctly_1_torr_scale(self):
        """Test voltage conversion for 1 Torr full scale"""
        voltage = np.array([5.0])
        full_scale = 1

        pressure = voltage_to_pressure(voltage, full_scale)

        assert pressure[0] == pytest.approx(0.5, rel=0.01)

    def test_clips_to_zero_below_threshold_1000_torr(self):
        """Test that values below 0.5 Torr are set to zero for 1000 scale"""
        voltage = np.array([0.004])  # 0.4 Torr (voltage * 100)
        full_scale = 1000

        pressure = voltage_to_pressure(voltage, full_scale)

        assert pressure[0] == 0

    def test_clips_to_zero_below_threshold_1_torr(self):
        """Test that values below 0.0005 Torr are set to zero for 1 scale"""
        voltage = np.array([0.00004])  # 0.000004 Torr
        full_scale = 1

        pressure = voltage_to_pressure(voltage, full_scale)

        assert pressure[0] == 0

    def test_clips_to_maximum_1000_torr(self):
        """Test that values above 1000 are clipped for 1000 scale"""
        voltage = np.array([15.0])  # 1500 Torr
        full_scale = 1000

        pressure = voltage_to_pressure(voltage, full_scale)

        assert pressure[0] == 1000

    def test_clips_to_maximum_1_torr(self):
        """Test that values above 1 are clipped for 1 scale"""
        voltage = np.array([15.0])  # 1.5 Torr
        full_scale = 1

        pressure = voltage_to_pressure(voltage, full_scale)

        assert pressure[0] == 1

    def test_handles_array_input(self):
        """Test that function works with numpy array"""
        voltage = np.array([1.0, 5.0, 10.0])
        full_scale = 1000

        pressure = voltage_to_pressure(voltage, full_scale)

        assert len(pressure) == 3
        assert pressure[1] == pytest.approx(500, rel=0.01)

    def test_handles_single_value(self):
        """Test that function works with single value"""
        voltage = np.array([5.0])
        full_scale = 1000

        pressure = voltage_to_pressure(voltage, full_scale)

        assert isinstance(pressure, np.ndarray)


class TestCalculateError:
    """Tests for calculate_error function"""

    def test_default_error_for_pressure_below_1(self):
        """Test that error is 0.5% for pressure below 1 Torr"""
        pressure = 0.5

        error = calculate_error(pressure)

        assert error == pytest.approx(0.5 * 0.005, rel=0.01)

    def test_reduced_error_for_pressure_above_1(self):
        """Test that error is 0.25% for pressure above 1 Torr"""
        pressure = 10.0

        error = calculate_error(pressure)

        assert error == pytest.approx(10.0 * 0.0025, rel=0.01)

    def test_handles_array_input(self):
        """Test that function works with array input"""
        pressure = np.array([0.5, 5.0, 10.0])

        error = calculate_error(pressure)

        assert len(error) == 3
        assert error[0] == pytest.approx(0.5 * 0.005, rel=0.01)
        assert error[1] == pytest.approx(5.0 * 0.0025, rel=0.01)

    def test_handles_zero_pressure(self):
        """Test that function handles zero pressure"""
        pressure = 0.0

        error = calculate_error(pressure)

        assert error == pytest.approx(0, abs=1e-10)

    def test_error_is_proportional_to_pressure(self):
        """Test that error scales with pressure"""
        pressure1 = 5.0
        pressure2 = 10.0

        error1 = calculate_error(pressure1)
        error2 = calculate_error(pressure2)

        assert error2 == pytest.approx(2 * error1, rel=0.01)

    def test_handles_list_input(self):
        """Test that function converts list to array"""
        pressure = [0.5, 5.0, 10.0]

        error = calculate_error(pressure)

        assert isinstance(error, np.ndarray)
        assert len(error) == 3


class TestImportHtmData:
    """Tests for import_htm_data function"""

    def test_returns_correct_structure_for_316l_steel(self):
        """Test that function returns three lists for 316l_steel"""
        x_values, y_values, labels = import_htm_data("316l_steel")

        assert isinstance(x_values, list)
        assert isinstance(y_values, list)
        assert isinstance(labels, list)
        assert len(x_values) == len(y_values) == len(labels)

    def test_returns_non_empty_data_for_316l_steel(self):
        """Test that 316l_steel returns non-empty data"""
        x_values, y_values, labels = import_htm_data("316l_steel")

        assert len(x_values) > 0
        assert len(y_values) > 0
        assert len(labels) > 0

    def test_x_values_are_numpy_arrays(self):
        """Test that x_values are numpy arrays"""
        x_values, y_values, labels = import_htm_data("316l_steel")

        for x in x_values:
            assert isinstance(x, np.ndarray)
            assert len(x) > 0

    def test_y_values_are_numpy_arrays(self):
        """Test that y_values are numpy arrays"""
        x_values, y_values, labels = import_htm_data("316l_steel")

        for y in y_values:
            assert isinstance(y, np.ndarray)
            assert len(y) > 0

    def test_labels_are_strings(self):
        """Test that labels are strings"""
        x_values, y_values, labels = import_htm_data("316l_steel")

        for label in labels:
            assert isinstance(label, str)
            assert len(label) > 0

    def test_x_and_y_arrays_have_same_length(self):
        """Test that each x and y array pair has the same length"""
        x_values, y_values, labels = import_htm_data("316l_steel")

        for x, y in zip(x_values, y_values):
            assert len(x) == len(y)

    def test_x_values_are_positive_temperatures(self):
        """Test that x_values (temperatures) are positive"""
        x_values, y_values, labels = import_htm_data("316l_steel")

        for x in x_values:
            assert np.all(x > 0)
            # Temperature should be in reasonable range (Kelvin)
            assert np.all(x >= 200)  # Above absolute zero
            assert np.all(x <= 2000)  # Below melting point

    def test_y_values_are_positive_permeabilities(self):
        """Test that y_values (permeabilities) are positive"""
        x_values, y_values, labels = import_htm_data("316l_steel")

        for y in y_values:
            assert np.all(y > 0)

    def test_labels_contain_author_and_year(self):
        """Test that labels contain author and year information"""
        x_values, y_values, labels = import_htm_data("316l_steel")

        for label in labels:
            # Should contain parentheses with year
            assert "(" in label
            assert ")" in label

    def test_permeability_increases_with_temperature(self):
        """Test that permeability generally increases with temperature"""
        x_values, y_values, labels = import_htm_data("316l_steel")

        for x, y in zip(x_values, y_values):
            # For Arrhenius behavior, permeability increases with temperature
            # Check that last value is greater than first value
            assert y[-1] > y[0]

    def test_temperature_values_are_sorted(self):
        """Test that temperature values are in ascending order"""
        x_values, y_values, labels = import_htm_data("316l_steel")

        for x in x_values:
            # Check if sorted
            assert np.all(x[:-1] <= x[1:])

    def test_function_works_with_316l_steel_input(self):
        """Test that function successfully executes with 316l_steel"""
        # This is the main test - ensure it doesn't raise any exceptions
        try:
            x_values, y_values, labels = import_htm_data("316l_steel")
            success = True
        except Exception:
            success = False

        assert success

    def test_x_values_have_100_points(self):
        """Test that x_values arrays have 100 points as specified"""
        x_values, y_values, labels = import_htm_data("316l_steel")

        for x in x_values:
            assert len(x) == 100

    def test_y_values_have_100_points(self):
        """Test that y_values arrays have 100 points as specified"""
        x_values, y_values, labels = import_htm_data("316l_steel")

        for y in y_values:
            assert len(y) == 100

    def test_labels_are_capitalized(self):
        """Test that labels are capitalized"""
        x_values, y_values, labels = import_htm_data("316l_steel")

        for label in labels:
            # First character should be uppercase
            assert label[0].isupper()
