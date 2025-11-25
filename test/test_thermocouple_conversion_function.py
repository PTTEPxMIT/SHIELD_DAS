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

# =============================================================================
# Tests for volts_to_temp_constants
# =============================================================================


def test_volts_to_temp_constants_returns_tuple():
    """
    Test volts_to_temp_constants to verify that it returns a tuple when called
    with a valid voltage value of 0.0 mV, confirming the correct return type.
    """
    result = volts_to_temp_constants(0.0)
    assert isinstance(result, tuple)


def test_volts_to_temp_constants_tuple_contains_floats():
    """
    Test volts_to_temp_constants to ensure all elements in the returned tuple
    are float values when called with 0.0 mV, verifying proper coefficient types.
    """
    result = volts_to_temp_constants(0.0)
    assert all(isinstance(x, float) for x in result)


@pytest.mark.parametrize(
    "voltage,expected_length",
    [
        (-5.0, 9),
        (0.0, 10),
        (10.0, 10),
        (20.644, 7),
        (30.0, 7),
        (54.0, 7),
    ],
)
def test_volts_to_temp_constants_returns_correct_coefficient_count(
    voltage, expected_length
):
    """
    Test volts_to_temp_constants to verify it returns the correct number of
    polynomial coefficients for different voltage ranges: 9 coefficients for
    negative voltages, 10 for mid-range (0-20.644 mV), and 7 for high range
    (>20.644 mV).
    """
    result = volts_to_temp_constants(voltage)
    assert len(result) == expected_length


@pytest.mark.parametrize(
    "voltage,expected_first_coeff",
    [
        (-5.0, 0.0e0),
        (10.0, 0.0e0),
        (30.0, -1.318058e2),
    ],
)
def test_volts_to_temp_constants_returns_correct_first_coefficient(
    voltage, expected_first_coeff
):
    """
    Test volts_to_temp_constants to verify the first polynomial coefficient
    matches expected NIST ITS-90 values: 0.0 for negative and mid-range voltages,
    -131.8058 for high-range voltages.
    """
    result = volts_to_temp_constants(voltage)
    assert result[0] == pytest.approx(expected_first_coeff)


def test_volts_to_temp_constants_negative_range_matches_nist():
    """
    Test volts_to_temp_constants to confirm that the complete set of polynomial
    coefficients for the negative voltage range (-5.891 to 0 mV) exactly matches
    the NIST ITS-90 standard values.
    """
    expected_coeffs = (
        0.0e0,
        2.5173462e1,
        -1.1662878e0,
        -1.0833638e0,
        -8.977354e-1,
        -3.7342377e-1,
        -8.6632643e-2,
        -1.0450598e-2,
        -5.1920577e-4,
    )
    result = volts_to_temp_constants(-5.0)
    assert np.allclose(result, expected_coeffs)


def test_volts_to_temp_constants_mid_range_matches_nist():
    """
    Test volts_to_temp_constants to confirm that the complete set of polynomial
    coefficients for the mid-range voltage (0 to 20.644 mV) exactly matches
    the NIST ITS-90 standard values.
    """
    expected_coeffs = (
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
    )
    result = volts_to_temp_constants(10.0)
    assert np.allclose(result, expected_coeffs)


def test_volts_to_temp_constants_high_range_matches_nist():
    """
    Test volts_to_temp_constants to confirm that the complete set of polynomial
    coefficients for the high-range voltage (20.644 to 54.886 mV) exactly matches
    the NIST ITS-90 standard values.
    """
    expected_coeffs = (
        -1.318058e2,
        4.830222e1,
        -1.646031e0,
        5.464731e-2,
        -9.650715e-4,
        8.802193e-6,
        -3.11081e-8,
    )
    result = volts_to_temp_constants(30.0)
    assert np.allclose(result, expected_coeffs)


@pytest.mark.parametrize(
    "voltage",
    [-5.891, -5.890, 0.0, 0.001, 20.643, 20.644, 20.645, 54.885, 54.886],
)
def test_volts_to_temp_constants_accepts_boundary_values(voltage):
    """
    Test volts_to_temp_constants to ensure it accepts and correctly processes
    voltage values at and near the boundaries of valid Type K ranges (-5.891 to
    54.886 mV) without raising exceptions.
    """
    result = volts_to_temp_constants(voltage)
    assert isinstance(result, tuple)


@pytest.mark.parametrize("voltage", [-5.893, -6.0, 54.888, 55.0])
def test_volts_to_temp_constants_raises_error_for_out_of_range_voltage(voltage):
    """
    Test volts_to_temp_constants to verify it raises ValueError when given
    voltage values outside the valid Type K range (-5.891 to 54.886 mV),
    with tolerance of 0.002 mV.
    """
    with pytest.raises(ValueError):
        volts_to_temp_constants(voltage)


def test_volts_to_temp_constants_uses_different_coefficients_across_zero():
    """
    Test volts_to_temp_constants to confirm it returns different coefficient
    sets for voltages just below and above 0 mV, demonstrating the transition
    between negative and mid-range polynomial regions.
    """
    coeffs_negative = volts_to_temp_constants(-0.001)
    coeffs_positive = volts_to_temp_constants(0.001)
    assert coeffs_negative != coeffs_positive


def test_volts_to_temp_constants_uses_different_coefficients_across_transition():
    """
    Test volts_to_temp_constants to confirm it returns different coefficient
    sets for voltages just below and above 20.644 mV, demonstrating the transition
    between mid-range and high-range polynomial regions.
    """
    coeffs_mid = volts_to_temp_constants(20.643)
    coeffs_high = volts_to_temp_constants(20.645)
    assert coeffs_mid != coeffs_high


# =============================================================================
# Tests for evaluate_poly
# =============================================================================


@pytest.mark.parametrize(
    "coeffs,x,expected",
    [
        ([1, 2, 3], 0, 1),
        ([1, 2, 3], 1, 6),
        ([1, 2, 3], 2, 17),
        ([0, 1], 5, 5),
        ([2.5], 10, 2.5),
    ],
)
def test_evaluate_poly_calculates_correct_polynomial_value(coeffs, x, expected):
    """
    Test evaluate_poly to verify it correctly evaluates polynomials using the
    form P(x) = a0 + a1*x + a2*x^2 + ... for various coefficient sets and
    input values.
    """
    result = evaluate_poly(coeffs, x)
    assert result == pytest.approx(expected)


def test_evaluate_poly_handles_empty_coefficients():
    """
    Test evaluate_poly to ensure it returns 0 when given an empty coefficient
    list, representing a null polynomial.
    """
    result = evaluate_poly([], 5)
    assert result == 0


def test_evaluate_poly_accepts_tuple_input():
    """
    Test evaluate_poly to verify it correctly processes coefficient data provided
    as a tuple rather than a list, evaluating (1, 2, 3) at x=2 to get 17.
    """
    coeffs = (1, 2, 3)
    result = evaluate_poly(coeffs, 2)
    assert result == pytest.approx(17)


def test_evaluate_poly_evaluates_quadratic_correctly():
    """
    Test evaluate_poly to confirm accurate evaluation of the quadratic polynomial
    (x+1)^2 = 1 + 2x + x^2 for multiple input values, verifying mathematical
    correctness.
    """
    coeffs = [1, 2, 1]
    x_values = [0, 1, 2, 3, -1, -2]
    for x in x_values:
        expected = (x + 1) ** 2
        result = evaluate_poly(coeffs, x)
        assert result == pytest.approx(expected)


def test_evaluate_poly_handles_array_input():
    """
    Test evaluate_poly to verify it correctly evaluates polynomials for numpy
    array inputs, processing multiple x values simultaneously.
    """
    coeffs = [1, 2, 1]
    x_array = np.array([0, 1, 2])
    result = evaluate_poly(coeffs, x_array)
    expected = np.array([1, 4, 9])
    assert np.allclose(result, expected)


# =============================================================================
# Tests for temp_to_volts_constants
# =============================================================================


@pytest.mark.parametrize(
    "temp_c,expected_coeffs_length",
    [
        (-100, 11),
        (-270, 11),
        (0, 10),
        (25, 10),
        (1000, 10),
        (1372, 10),
    ],
)
def test_temp_to_volts_constants_returns_correct_coefficient_count(
    temp_c, expected_coeffs_length
):
    """
    Test temp_to_volts_constants to verify it returns the correct number of
    polynomial coefficients: 11 for negative temperatures (-270 to 0°C) and
    10 for positive temperatures (0 to 1372°C).
    """
    coeffs, _ = temp_to_volts_constants(temp_c)
    assert len(coeffs) == expected_coeffs_length


def test_temp_to_volts_constants_returns_tuple_for_coefficients():
    """
    Test temp_to_volts_constants to ensure the first return value (coefficients)
    is a tuple when called with a valid temperature of 25°C.
    """
    coeffs, _ = temp_to_volts_constants(25)
    assert isinstance(coeffs, tuple)


def test_temp_to_volts_constants_coefficients_are_floats():
    """
    Test temp_to_volts_constants to verify all polynomial coefficients are
    float values when called with 25°C, ensuring proper numeric types.
    """
    coeffs, _ = temp_to_volts_constants(25)
    assert all(isinstance(x, float) for x in coeffs)


def test_temp_to_volts_constants_negative_range_has_no_exponential():
    """
    Test temp_to_volts_constants to confirm that negative temperature range
    (-270 to 0°C) returns None for exponential correction coefficients, as
    this correction only applies to positive temperatures.
    """
    _, extended = temp_to_volts_constants(-100)
    assert extended is None


def test_temp_to_volts_constants_positive_range_has_exponential():
    """
    Test temp_to_volts_constants to verify that positive temperature range
    (0 to 1372°C) returns exponential correction coefficients as a tuple of
    3 float values.
    """
    _, extended = temp_to_volts_constants(25)
    assert extended is not None


def test_temp_to_volts_constants_exponential_has_three_coefficients():
    """
    Test temp_to_volts_constants to verify that the exponential correction
    tuple contains exactly 3 coefficients for positive temperatures.
    """
    _, extended = temp_to_volts_constants(25)
    assert len(extended) == 3


def test_temp_to_volts_constants_exponential_coefficients_are_floats():
    """
    Test temp_to_volts_constants to ensure all three exponential correction
    coefficients (amplitude, exponent_factor, reference_temp) are float values
    for positive temperatures.
    """
    _, extended = temp_to_volts_constants(25)
    assert all(isinstance(x, float) for x in extended)


@pytest.mark.parametrize("temp_c", [-271, -300, 1373, 1500])
def test_temp_to_volts_constants_raises_error_for_out_of_range_temperature(temp_c):
    """
    Test temp_to_volts_constants to verify it raises ValueError with appropriate
    message when given temperatures outside valid Type K range (-270 to 1372°C).
    """
    with pytest.raises(ValueError, match="Temperature out of valid Type K range"):
        temp_to_volts_constants(temp_c)


# =============================================================================
# Tests for temp_c_to_mv - Scalar Inputs
# =============================================================================


@pytest.mark.parametrize(
    "temp_c,expected_mv_min,expected_mv_max",
    [
        (0, -0.1, 0.1),
        (25, 0.9, 1.1),
        (100, 4.0, 4.2),
        (200, 8.1, 8.3),
        (500, 20.6, 20.7),
        (1000, 41.2, 41.4),
    ],
)
def test_temp_c_to_mv_scalar_converts_known_temperatures_correctly(
    temp_c, expected_mv_min, expected_mv_max
):
    """
    Test temp_c_to_mv to verify it converts scalar temperature values to
    millivolt outputs within expected ranges for known Type K thermocouple
    reference points (0°C to 1000°C).
    """
    result = temp_c_to_mv(temp_c)
    assert expected_mv_min <= result <= expected_mv_max


def test_temp_c_to_mv_scalar_negative_temperature_produces_negative_voltage():
    """
    Test temp_c_to_mv to confirm that negative temperature input (-100°C)
    produces negative voltage output, following Type K thermocouple behavior.
    """
    result = temp_c_to_mv(-100)
    assert result < 0


def test_temp_c_to_mv_scalar_returns_float():
    """
    Test temp_c_to_mv to ensure it returns a float value when given a scalar
    temperature input of 25°C.
    """
    result = temp_c_to_mv(25)
    assert isinstance(result, float)


def test_temp_c_to_mv_scalar_lower_boundary_accepted():
    """
    Test temp_c_to_mv to verify it accepts the lower boundary temperature of
    -270°C without raising exceptions and returns a float value.
    """
    result = temp_c_to_mv(-270)
    assert isinstance(result, float)


def test_temp_c_to_mv_scalar_upper_boundary_accepted():
    """
    Test temp_c_to_mv to verify it accepts the upper boundary temperature of
    1372°C without raising exceptions and returns a float value.
    """
    result = temp_c_to_mv(1372)
    assert isinstance(result, float)


def test_temp_c_to_mv_scalar_monotonically_increasing():
    """
    Test temp_c_to_mv to confirm that higher temperatures produce higher
    voltages, verifying monotonic behavior across the valid range.
    """
    result_low = temp_c_to_mv(-270)
    result_high = temp_c_to_mv(1372)
    assert result_low < result_high


@pytest.mark.parametrize("temp_c", [-271, 1373])
def test_temp_c_to_mv_scalar_raises_error_for_out_of_range_temperature(temp_c):
    """
    Test temp_c_to_mv to verify it raises ValueError when given scalar
    temperatures outside the valid Type K range (-270 to 1372°C).
    """
    expected_message = "Temperature out of valid Type K range (-270 to 1372 C)."
    with pytest.raises(ValueError, match=re.escape(expected_message)):
        temp_c_to_mv(temp_c)


# =============================================================================
# Tests for temp_c_to_mv - Array Inputs
# =============================================================================


def test_temp_c_to_mv_array_returns_ndarray():
    """
    Test temp_c_to_mv to ensure it returns a numpy ndarray when given an
    array of temperature values.
    """
    temps = np.array([0, 25, 100])
    result = temp_c_to_mv(temps)
    assert isinstance(result, np.ndarray)


def test_temp_c_to_mv_array_preserves_shape():
    """
    Test temp_c_to_mv to verify the output array has the same shape as the
    input temperature array.
    """
    temps = np.array([0, 25, 100, 200])
    result = temp_c_to_mv(temps)
    assert result.shape == temps.shape


def test_temp_c_to_mv_array_handles_negative_temperatures():
    """
    Test temp_c_to_mv to confirm it correctly processes arrays containing
    negative temperatures, producing corresponding negative voltage values.
    """
    temps = np.array([-100, -50, 0])
    result = temp_c_to_mv(temps)
    assert result[0] < 0


def test_temp_c_to_mv_array_negative_temperature_second_element_is_negative():
    """
    Test temp_c_to_mv to verify the second element of an array with negative
    temperatures also produces a negative voltage.
    """
    temps = np.array([-100, -50, 0])
    result = temp_c_to_mv(temps)
    assert result[1] < 0


def test_temp_c_to_mv_array_handles_positive_temperatures():
    """
    Test temp_c_to_mv to verify it correctly processes arrays of positive
    temperatures with exponential correction term applied.
    """
    temps = np.array([25, 100, 500])
    result = temp_c_to_mv(temps)
    assert np.all(result > 0)


def test_temp_c_to_mv_array_handles_mixed_sign_temperatures():
    """
    Test temp_c_to_mv to ensure it correctly processes arrays containing both
    negative and positive temperatures, applying appropriate polynomial ranges.
    """
    temps = np.array([-100, 0, 100])
    result = temp_c_to_mv(temps)
    assert result[0] < 0


def test_temp_c_to_mv_array_mixed_sign_positive_element_is_positive():
    """
    Test temp_c_to_mv to verify that positive temperature elements in a mixed
    sign array produce positive voltage values.
    """
    temps = np.array([-100, 0, 100])
    result = temp_c_to_mv(temps)
    assert result[2] > 0


@pytest.mark.parametrize("temp_c", [-271, 1373])
def test_temp_c_to_mv_array_raises_error_for_out_of_range_values(temp_c):
    """
    Test temp_c_to_mv to verify it raises ValueError when any element in the
    input array falls outside the valid Type K range (-270 to 1372°C).
    """
    temps = np.array([0, 25, temp_c])
    expected_message = "Temperature out of valid Type K range (-270 to 1372 C)."
    with pytest.raises(ValueError, match=re.escape(expected_message)):
        temp_c_to_mv(temps)


def test_temp_c_to_mv_array_converts_0d_array_to_scalar():
    """
    Test temp_c_to_mv to confirm it converts 0-dimensional numpy array input
    (scalar wrapped in array) back to a scalar float output.
    """
    temp = np.array(25.0)
    result = temp_c_to_mv(temp)
    assert isinstance(result, float | np.floating)


# =============================================================================
# Tests for mv_to_temp_c - Scalar Inputs
# =============================================================================


@pytest.mark.parametrize(
    "mv,expected_temp_min,expected_temp_max",
    [
        (0, -0.1, 0.1),
        (1.0, 24, 26),
        (4.1, 99, 101),
        (8.2, 200, 203),
        (20.64, 499, 501),
        (41.3, 999, 1001),
    ],
)
def test_mv_to_temp_c_scalar_converts_known_voltages_correctly(
    mv, expected_temp_min, expected_temp_max
):
    """
    Test mv_to_temp_c to verify it converts scalar millivolt values to
    temperature outputs within expected ranges for known Type K thermocouple
    reference points (0 mV to 41.3 mV).
    """
    result = mv_to_temp_c(mv)
    assert expected_temp_min <= result <= expected_temp_max


def test_mv_to_temp_c_scalar_negative_voltage_produces_negative_temperature():
    """
    Test mv_to_temp_c to confirm that negative voltage input (-3.0 mV)
    produces negative temperature output, following Type K thermocouple behavior.
    """
    result = mv_to_temp_c(-3.0)
    assert result < 0


def test_mv_to_temp_c_scalar_returns_float():
    """
    Test mv_to_temp_c to ensure it returns a float value when given a scalar
    voltage input of 1.0 mV.
    """
    result = mv_to_temp_c(1.0)
    assert isinstance(result, float)


def test_mv_to_temp_c_scalar_lower_boundary_accepted():
    """
    Test mv_to_temp_c to verify it accepts the lower boundary voltage of
    -5.891 mV without raising exceptions and returns a float value.
    """
    result = mv_to_temp_c(-5.891)
    assert isinstance(result, float)


def test_mv_to_temp_c_scalar_upper_boundary_accepted():
    """
    Test mv_to_temp_c to verify it accepts the upper boundary voltage of
    54.886 mV without raising exceptions and returns a float value.
    """
    result = mv_to_temp_c(54.886)
    assert isinstance(result, float)


def test_mv_to_temp_c_scalar_monotonically_increasing():
    """
    Test mv_to_temp_c to confirm that higher voltages produce higher
    temperatures, verifying monotonic behavior across the valid range.
    """
    result_low = mv_to_temp_c(-5.891)
    result_high = mv_to_temp_c(54.886)
    assert result_low < result_high


@pytest.mark.parametrize("voltage", [-5.9, 54.9])
def test_mv_to_temp_c_scalar_raises_error_for_out_of_range_voltage(voltage):
    """
    Test mv_to_temp_c to verify it raises ValueError when given scalar voltages
    outside the valid Type K range (-5.891 to 54.886 mV).
    """
    expected_message = "Voltage out of valid Type K range (-5.891 to 54.886 mV)."
    with pytest.raises(ValueError, match=re.escape(expected_message)):
        mv_to_temp_c(voltage)


# =============================================================================
# Tests for mv_to_temp_c - Array Inputs
# =============================================================================


def test_mv_to_temp_c_array_returns_ndarray():
    """
    Test mv_to_temp_c to ensure it returns a numpy ndarray when given an
    array of voltage values.
    """
    voltages = np.array([0, 1.0, 4.1])
    result = mv_to_temp_c(voltages)
    assert isinstance(result, np.ndarray)


def test_mv_to_temp_c_array_preserves_shape():
    """
    Test mv_to_temp_c to verify the output array has the same shape as the
    input voltage array.
    """
    voltages = np.array([0, 1.0, 4.1, 8.2])
    result = mv_to_temp_c(voltages)
    assert result.shape == voltages.shape


def test_mv_to_temp_c_array_handles_negative_voltages():
    """
    Test mv_to_temp_c to confirm it correctly processes arrays containing
    negative voltages using the negative range polynomial coefficients.
    """
    voltages = np.array([-3.0, -1.0, 0])
    result = mv_to_temp_c(voltages)
    assert result[0] < 0


def test_mv_to_temp_c_array_negative_voltage_second_element_is_negative():
    """
    Test mv_to_temp_c to verify the second element of an array with negative
    voltages also produces a negative temperature.
    """
    voltages = np.array([-3.0, -1.0, 0])
    result = mv_to_temp_c(voltages)
    assert result[1] < 0


def test_mv_to_temp_c_array_handles_mid_range_voltages():
    """
    Test mv_to_temp_c to verify it correctly processes arrays of mid-range
    voltages (0 to 20.644 mV) using the mid-range polynomial coefficients.
    """
    voltages = np.array([1.0, 10.0, 20.0])
    result = mv_to_temp_c(voltages)
    assert np.all(result > 0)


def test_mv_to_temp_c_array_handles_high_range_voltages():
    """
    Test mv_to_temp_c to verify it correctly processes arrays of high-range
    voltages (>20.644 mV) using the high-range polynomial coefficients.
    """
    voltages = np.array([21.0, 30.0, 50.0])
    result = mv_to_temp_c(voltages)
    assert np.all(result > 500)


def test_mv_to_temp_c_array_handles_mixed_range_voltages():
    """
    Test mv_to_temp_c to ensure it correctly processes arrays containing
    voltages from all three ranges (negative, mid, high), applying appropriate
    polynomial coefficients for each.
    """
    voltages = np.array([-1.0, 10.0, 30.0])
    result = mv_to_temp_c(voltages)
    assert result[0] < 0


def test_mv_to_temp_c_array_mixed_range_mid_voltage_is_mid_temp():
    """
    Test mv_to_temp_c to verify mid-range voltage element in a mixed range
    array produces mid-range temperature (0-500°C).
    """
    voltages = np.array([-1.0, 10.0, 30.0])
    result = mv_to_temp_c(voltages)
    assert 0 < result[1] < 500


def test_mv_to_temp_c_array_mixed_range_high_voltage_is_high_temp():
    """
    Test mv_to_temp_c to verify high-range voltage element in a mixed range
    array produces high-range temperature (>500°C).
    """
    voltages = np.array([-1.0, 10.0, 30.0])
    result = mv_to_temp_c(voltages)
    assert result[2] > 500


def test_mv_to_temp_c_array_converts_0d_array_to_scalar():
    """
    Test mv_to_temp_c to confirm it converts 0-dimensional numpy array input
    (scalar wrapped in array) back to a scalar float output.
    """
    voltage = np.array(1.0)
    result = mv_to_temp_c(voltage)
    assert isinstance(result, float | np.floating)


# =============================================================================
# Tests for Round-Trip Conversion (temp -> voltage -> temp)
# =============================================================================


@pytest.mark.parametrize(
    "original_temp",
    [-200, -100, -50, 0, 25, 100, 200, 500, 1000, 1200],
)
def test_temperature_voltage_round_trip_within_tolerance(original_temp):
    """
    Test temp_c_to_mv and mv_to_temp_c together to verify round-trip conversion
    (temperature → voltage → temperature) recovers the original temperature
    within 0.1°C for most ranges and 0.5°C for high temperatures.
    """
    voltage = temp_c_to_mv(original_temp)
    recovered_temp = mv_to_temp_c(voltage)
    tolerance = 0.1 if abs(original_temp) < 1000 else 0.5
    assert abs(recovered_temp - original_temp) < tolerance


# =============================================================================
# Tests for Thermocouple Class
# =============================================================================


def test_thermocouple_initializes_with_default_name():
    """
    Test Thermocouple class to verify it initializes with the default name
    "type_K_thermocouple" when no custom name is provided.
    """
    tc = Thermocouple()
    assert tc.name == "type_K_thermocouple"


def test_thermocouple_initializes_with_empty_voltage_data():
    """
    Test Thermocouple class to ensure the voltage_data attribute initializes
    as an empty list when a new instance is created.
    """
    tc = Thermocouple()
    assert tc.voltage_data == []


def test_thermocouple_initializes_with_custom_name():
    """
    Test Thermocouple class to verify it correctly stores a custom name
    "Custom TC" when provided during initialization.
    """
    tc = Thermocouple(name="Custom TC")
    assert tc.name == "Custom TC"


def test_thermocouple_record_voltage_in_test_mode_appends_data():
    """
    Test Thermocouple.record_ain_channel_voltage to verify it appends voltage
    data when called in test mode (labjack=None), increasing the data list length.
    """
    tc = Thermocouple()
    initial_count = len(tc.voltage_data)
    tc.record_ain_channel_voltage(labjack=None)
    assert len(tc.voltage_data) == initial_count + 1


def test_thermocouple_record_voltage_in_test_mode_produces_reasonable_values():
    """
    Test Thermocouple.record_ain_channel_voltage to confirm test mode generates
    voltage values in a reasonable range (0.1-0.2 mV) corresponding to room
    temperature (~25-30°C).
    """
    tc = Thermocouple()
    tc.record_ain_channel_voltage(labjack=None)
    assert 0.1 <= tc.voltage_data[-1] <= 0.2


def test_thermocouple_multiple_recordings_accumulate_data():
    """
    Test Thermocouple.record_ain_channel_voltage to verify multiple calls
    correctly accumulate data, storing all 5 voltage readings in the data list.
    """
    tc = Thermocouple()
    for i in range(5):
        tc.record_ain_channel_voltage(labjack=None)
    assert len(tc.voltage_data) == 5


def test_thermocouple_initializes_with_empty_local_temperature_data():
    """
    Test Thermocouple class to ensure the local_temperature_data attribute
    initializes as an empty list when a new instance is created.
    """
    tc = Thermocouple()
    assert tc.local_temperature_data == []


def test_thermocouple_record_voltage_appends_local_temperature():
    """
    Test Thermocouple.record_ain_channel_voltage to verify it appends local
    temperature data (cold junction) when recording voltage in test mode.
    """
    tc = Thermocouple()
    initial_count = len(tc.local_temperature_data)
    tc.record_ain_channel_voltage(labjack=None)
    assert len(tc.local_temperature_data) == initial_count + 1


def test_thermocouple_local_temperature_in_reasonable_range():
    """
    Test Thermocouple.record_ain_channel_voltage to confirm test mode generates
    local temperature values in a reasonable range (20-25°C) for cold junction
    compensation.
    """
    tc = Thermocouple()
    tc.record_ain_channel_voltage(labjack=None)
    assert 20 <= tc.local_temperature_data[-1] <= 25


# =============================================================================
# Tests for Integration and Known Reference Points
# =============================================================================


def test_thermocouple_ice_point_reference():
    """
    Test temp_c_to_mv to verify it produces approximately 0.000 mV at the
    ice point (0°C), a fundamental Type K thermocouple reference point.
    """
    result = temp_c_to_mv(0)
    assert abs(result - 0.000) < 0.01


def test_thermocouple_boiling_point_reference():
    """
    Test temp_c_to_mv to verify it produces approximately 4.096 mV at the
    boiling point of water (100°C), a standard Type K thermocouple reference.
    """
    result = temp_c_to_mv(100)
    assert abs(result - 4.096) < 0.01


def test_thermocouple_high_temperature_reference():
    """
    Test temp_c_to_mv to verify it produces approximately 41.276 mV at 1000°C,
    a high-temperature Type K thermocouple reference point.
    """
    result = temp_c_to_mv(1000)
    assert abs(result - 41.276) < 0.01


def test_thermocouple_conversion_internal_consistency():
    """
    Test temp_c_to_mv and mv_to_temp_c together to verify internal consistency
    across multiple temperature points (-200°C to 1000°C), ensuring round-trip
    conversions are accurate within 0.1°C.
    """
    test_temperatures = [-200, -100, 0, 25, 100, 200, 500, 1000]
    for temp in test_temperatures:
        voltage = temp_c_to_mv(temp)
        recovered_temp = mv_to_temp_c(voltage)
        assert abs(recovered_temp - temp) < 0.1
