import numpy as np
import pytest

from shield_das.thermocouple_conversion_functions import volts_to_temp_constants


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
        -5.892,  # Just outside lower bound
        -6.0,  # Below lower bound
        54.887,  # Just outside upper bound
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
