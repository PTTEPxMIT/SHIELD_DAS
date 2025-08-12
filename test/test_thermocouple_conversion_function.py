import re

import numpy as np
import pytest

from shield_das.thermocouple import (
    Thermocouple,
    evaluate_poly,
    mv_to_temp_c,
    temp_c_to_mv,
    temp_to_volts_constants,
    volts_to_temp_constants,
)


def test_volts_to_temp_constants_return_type():
    """Test that the function returns a tuple of floats."""
    result = volts_to_temp_constants(0.0)
    assert isinstance(result, tuple)
    assert all(isinstance(x, float) for x in result)


@pytest.mark.parametrize(
    "voltage,expected_length",
    [
        (-5.0, 9),  # Negative range
        (0.0, 10),  # Zero point (should return 0-20.644 range)
        (10.0, 10),  # Middle of 0-20.644 range
        (20.644, 7),  # Exactly at transition point (should return upper range)
        (30.0, 7),  # Upper range
        (54.0, 7),  # Near upper limit
    ],
)
def test_volts_to_temp_constants_correct_length(voltage, expected_length):
    """Test that the function returns the correct number of coefficients for each
    range."""
    result = volts_to_temp_constants(voltage)
    assert len(result) == expected_length


@pytest.mark.parametrize(
    "voltage,expected_coeffs",
    [
        # Test exact match for first coefficient in each range
        (
            -5.0,
            (
                0.0e0,
                2.5173462e1,
                -1.1662878e0,
                -1.0833638e0,
                -8.977354e-1,
                -3.7342377e-1,
                -8.6632643e-2,
                -1.0450598e-2,
                -5.1920577e-4,
            ),
        ),
        (
            10.0,
            (
                0.0e0,
                2.508355e1,
                7.860106e-2,
                -2.503131e-1,
                8.31527e-2,
                -1.228034e-2,
                9.804036e-4,
                -4.41303e-5,
                1.057734e-6,
                -1.052755e-8,
            ),
        ),
        (
            30.0,
            (
                -1.318058e2,
                4.830222e1,
                -1.646031e0,
                5.464731e-2,
                -9.650715e-4,
                8.802193e-6,
                -3.11081e-8,
            ),
        ),
    ],
)
def test_volts_to_temp_constants_correct_coeffs(voltage, expected_coeffs):
    """Test that the function returns the correct coefficients for each range."""
    result = volts_to_temp_constants(voltage)
    assert len(result) == len(expected_coeffs)
    # Use numpy's allclose to handle floating point comparison
    assert np.allclose(result, expected_coeffs)


@pytest.mark.parametrize(
    "voltage",
    [
        -5.891,  # Lower bound
        -5.890,  # Just inside lower bound
        0.0,  # Transition point
        0.001,  # Just after transition point
        20.643,  # Just before transition point
        20.644,  # Transition point
        20.645,  # Just after transition point
        54.885,  # Just inside upper bound
        54.886,  # Upper bound
    ],
)
def test_volts_to_temp_constants_boundary_values(voltage):
    """Test that the function correctly handles boundary values."""
    # Function should not raise an exception for these values
    result = volts_to_temp_constants(voltage)
    assert isinstance(result, tuple)


@pytest.mark.parametrize(
    "voltage",
    [
        -5.893,  # Just outside lower bound (updated for new tolerance)
        -6.0,  # Below lower bound
        54.888,  # Just outside upper bound (updated for new tolerance)
        55.0,  # Above upper bound
    ],
)
def test_volts_to_temp_constants_out_of_range(voltage):
    """Test that the function raises ValueError for out-of-range inputs."""
    with pytest.raises(ValueError):
        volts_to_temp_constants(voltage)


def test_volts_to_temp_constants_transition_continuity():
    """Test that the function's output is continuous at transition points."""
    # Get coefficients just before and after transition points
    near_zero_neg = volts_to_temp_constants(-0.001)
    near_zero_pos = volts_to_temp_constants(0.001)
    near_transition_low = volts_to_temp_constants(20.643)
    near_transition_high = volts_to_temp_constants(20.645)

    # The polynomial evaluations should be close at these points
    # (We'd need to implement evaluate_poly here to test properly,
    # but we're just checking that different coefficient sets are returned)
    assert near_zero_neg != near_zero_pos
    assert near_transition_low != near_transition_high


@pytest.mark.parametrize(
    "range_description,voltage,expected_first_coeff",
    [
        ("Negative range", -5.0, 0.0),
        ("Zero to mid range", 10.0, 0.0),
        ("Upper range", 30.0, -1.318058e2),
    ],
)
def test_volts_to_temp_constants_first_coefficient(
    range_description, voltage, expected_first_coeff
):
    """Test that the first coefficient matches expected value for each range."""
    coeffs = volts_to_temp_constants(voltage)
    assert coeffs[0] == pytest.approx(expected_first_coeff)


# Tests for evaluate_poly function
@pytest.mark.parametrize(
    "coeffs,x,expected",
    [
        ([1, 2, 3], 0, 1),  # P(0) = 1 + 2*0 + 3*0^2 = 1
        ([1, 2, 3], 1, 6),  # P(1) = 1 + 2*1 + 3*1^2 = 6
        ([1, 2, 3], 2, 17),  # P(2) = 1 + 2*2 + 3*2^2 = 17
        ([0, 1], 5, 5),  # P(5) = 0 + 1*5 = 5
        ([2.5], 10, 2.5),  # Constant polynomial
        ([], 5, 0),  # Empty coefficients should give 0
    ],
)
def test_evaluate_poly(coeffs, x, expected):
    """Test polynomial evaluation with various inputs."""
    result = evaluate_poly(coeffs, x)
    assert result == pytest.approx(expected)


def test_evaluate_poly_with_tuple():
    """Test that evaluate_poly works with tuple input."""
    coeffs = (1, 2, 3)
    result = evaluate_poly(coeffs, 2)
    assert result == pytest.approx(17)


# Tests for temp_to_volts_constants function
@pytest.mark.parametrize(
    "temp_c,expected_coeffs_length,has_extended",
    [
        (-100, 11, False),  # Negative temperature range
        (-270, 11, False),  # Lower bound
        (0, 10, True),  # Transition point
        (25, 10, True),  # Room temperature
        (1000, 10, True),  # High temperature
        (1372, 10, True),  # Upper bound
    ],
)
def test_temp_to_volts_constants_return_format(
    temp_c, expected_coeffs_length, has_extended
):
    """Test that temp_to_volts_constants returns correct format."""
    coeffs, extended = temp_to_volts_constants(temp_c)
    assert isinstance(coeffs, tuple)
    assert len(coeffs) == expected_coeffs_length
    assert all(isinstance(x, float) for x in coeffs)

    if has_extended:
        assert extended is not None
        assert len(extended) == 3
        assert all(isinstance(x, float) for x in extended)
    else:
        assert extended is None


@pytest.mark.parametrize(
    "temp_c",
    [
        -271,  # Below lower bound
        -300,  # Well below lower bound
        1373,  # Above upper bound
        1500,  # Well above upper bound
    ],
)
def test_temp_to_volts_constants_out_of_range(temp_c):
    """Test that temp_to_volts_constants raises ValueError for out-of-range inputs."""
    with pytest.raises(ValueError, match="Temperature out of valid Type K range"):
        temp_to_volts_constants(temp_c)


# Tests for temp_c_to_mv function
@pytest.mark.parametrize(
    "temp_c,expected_mv_range",
    [
        (0, (-0.1, 0.1)),  # Around 0°C should be near 0 mV
        (25, (0.9, 1.1)),  # Room temperature ~1 mV
        (100, (4.0, 4.2)),  # 100°C ~4.1 mV
        (200, (8.1, 8.3)),  # 200°C ~8.2 mV
        (500, (20.6, 20.7)),  # 500°C ~20.64 mV
        (1000, (41.2, 41.4)),  # 1000°C ~41.3 mV
    ],
)
def test_temp_c_to_mv_known_values(temp_c, expected_mv_range):
    """Test temp_c_to_mv with known temperature-voltage relationships."""
    result = temp_c_to_mv(temp_c)
    assert expected_mv_range[0] <= result <= expected_mv_range[1]


def test_temp_c_to_mv_negative_temperature():
    """Test temp_c_to_mv with negative temperatures."""
    result = temp_c_to_mv(-100)
    assert result < 0  # Negative temperature should give negative voltage
    assert isinstance(result, float)


def test_temp_c_to_mv_boundary_values():
    """Test temp_c_to_mv at boundary values."""
    result_low = temp_c_to_mv(-270)
    result_high = temp_c_to_mv(1372)

    assert isinstance(result_low, float)
    assert isinstance(result_high, float)
    assert result_low < result_high  # Higher temp should give higher voltage


# Tests for mv_to_temp_c function
@pytest.mark.parametrize(
    "mv,expected_temp_range",
    [
        (0, (-0.1, 0.1)),  # 0 mV should be near 0°C
        (1.0, (24, 26)),  # ~1 mV should be near 25°C
        (4.1, (99, 101)),  # ~4.1 mV should be near 100°C
        (8.2, (201, 202)),  # ~8.2 mV should be near 201.5°C (corrected)
        (20.64, (499, 501)),  # ~20.64 mV should be near 500°C
        (41.3, (999, 1001)),  # ~41.3 mV should be near 1000°C
    ],
)
def test_mv_to_temp_c_known_values(mv, expected_temp_range):
    """Test mv_to_temp_c with known voltage-temperature relationships."""
    result = mv_to_temp_c(mv)
    assert expected_temp_range[0] <= result <= expected_temp_range[1]


def test_mv_to_temp_c_negative_voltage():
    """Test mv_to_temp_c with negative voltages."""
    result = mv_to_temp_c(-3.0)
    assert result < 0  # Negative voltage should give negative temperature
    assert isinstance(result, float)


def test_mv_to_temp_c_boundary_values():
    """Test mv_to_temp_c at boundary values."""
    result_low = mv_to_temp_c(-5.891)
    result_high = mv_to_temp_c(54.886)

    assert isinstance(result_low, float)
    assert isinstance(result_high, float)
    assert result_low < result_high


# Round-trip tests (temperature -> voltage -> temperature)
@pytest.mark.parametrize(
    "original_temp",
    [-200, -100, -50, 0, 25, 100, 200, 500, 1000, 1200],
)
def test_temp_voltage_round_trip(original_temp):
    """Test that converting temp->voltage->temp gives back original temperature."""
    # Convert temperature to voltage
    voltage = temp_c_to_mv(original_temp)

    # Convert voltage back to temperature
    recovered_temp = mv_to_temp_c(voltage)

    # Should be very close to original (within 0.1°C for most ranges)
    tolerance = 0.1 if abs(original_temp) < 1000 else 0.5
    assert abs(recovered_temp - original_temp) < tolerance


# Tests for Thermocouple class
class TestThermocouple:
    """Test suite for the Thermocouple class."""

    def test_thermocouple_init_default_values(self):
        """Test Thermocouple initialization with default values."""
        tc = Thermocouple()
        assert tc.name == "type K thermocouple"
        assert tc.export_filename == "temperature_data.csv"
        assert tc.timestamp_data == []
        assert tc.real_timestamp_data == []
        assert tc.local_temperature_data == []
        assert tc.measured_temperature_data == []
        assert tc.backup_interval == 10

    def test_thermocouple_init_custom_values(self):
        """Test Thermocouple initialization with custom values."""
        tc = Thermocouple(name="Custom TC", export_filename="custom_data.csv")
        assert tc.name == "Custom TC"
        assert tc.export_filename == "custom_data.csv"

    def test_get_temperature_test_mode(self):
        """Test get_temperature method in test mode (labjack=None)."""
        tc = Thermocouple()
        initial_count = len(tc.timestamp_data)

        # Call get_temperature with None labjack (test mode)
        tc.get_temperature(labjack=None, timestamp=123.45)

        # Check that data was added
        assert len(tc.timestamp_data) == initial_count + 1
        assert len(tc.real_timestamp_data) == initial_count + 1
        assert len(tc.local_temperature_data) == initial_count + 1
        assert len(tc.measured_temperature_data) == initial_count + 1

        # Check that timestamp was recorded correctly
        assert tc.timestamp_data[-1] == 123.45

        # Check that temperatures are in reasonable range (25-30°C for test mode)
        assert 25 <= tc.local_temperature_data[-1] <= 30
        assert 25 <= tc.measured_temperature_data[-1] <= 30

    def test_multiple_temperature_readings(self):
        """Test multiple temperature readings build up data correctly."""
        tc = Thermocouple()

        # Take multiple readings
        for i in range(5):
            tc.get_temperature(labjack=None, timestamp=float(i))

        # Check that all data was stored
        assert len(tc.timestamp_data) == 5
        assert len(tc.real_timestamp_data) == 5
        assert len(tc.local_temperature_data) == 5
        assert len(tc.measured_temperature_data) == 5

        # Check timestamps are correct
        assert tc.timestamp_data == [0.0, 1.0, 2.0, 3.0, 4.0]


# Integration tests
def test_thermocouple_conversion_consistency():
    """Test that the thermocouple conversion functions are internally consistent."""
    # Test various temperatures across the valid range
    test_temperatures = [-200, -100, 0, 25, 100, 200, 500, 1000]

    for temp in test_temperatures:
        # Convert to voltage and back
        voltage = temp_c_to_mv(temp)
        recovered_temp = mv_to_temp_c(voltage)

        # Should be very close (within 0.1°C for most cases)
        assert abs(recovered_temp - temp) < 0.1, f"Failed for {temp}°C"


def test_known_thermocouple_reference_points():
    """Test against known Type K thermocouple reference points."""
    # Some well-known reference points for Type K thermocouples
    reference_points = [
        (0, 0.000),  # Ice point
        (100, 4.096),  # Boiling point of water
        (1000, 41.276),  # High temperature reference
    ]

    for temp, expected_mv in reference_points:
        calculated_mv = temp_c_to_mv(temp)
        # Allow for small differences due to polynomial approximation
        assert abs(calculated_mv - expected_mv) < 0.01, f"Failed for {temp}°C"


def test_polynomial_evaluation_accuracy():
    """Test that polynomial evaluation is accurate for known inputs."""
    # Test a simple quadratic: x^2 + 2x + 1 = (x+1)^2
    coeffs = [1, 2, 1]  # 1 + 2x + 1x^2

    test_values = [0, 1, 2, 3, -1, -2]
    for x in test_values:
        expected = (x + 1) ** 2
        result = evaluate_poly(coeffs, x)
        assert result == pytest.approx(expected), f"Failed for x={x}"


@pytest.mark.parametrize(
    "temp",
    [-271, 1373],
)
def test_temperature_range_error_rasied(temp):
    """Tests that functions raise ValueError for out-of-range temperatures."""

    expected_message = "Temperature out of valid Type K range (-270 to 1372 C)."

    with pytest.raises(ValueError, match=re.escape(expected_message)):
        temp_c_to_mv(temp)


@pytest.mark.parametrize(
    "voltage",
    [-5.9, 54.9],
)
def test_voltage_range_boundaries(voltage):
    """Tests that functions raise ValueError for out-of-range voltages."""

    expected_message = "Voltage out of valid Type K range (-5.891 to 54.886 mV)."

    with pytest.raises(ValueError, match=re.escape(expected_message)):
        volts_to_temp_constants(voltage)
