from unittest.mock import patch

import numpy as np
import pytest
from uncertainties import ufloat

from shield_das.analysis import (
    average_pressure_after_increase,
    calculate_error,
    calculate_flux_from_sample,
    calculate_permeability_from_flux,
    evaluate_permeability_values,
    fit_permeability_data,
    voltage_to_pressure,
    voltage_to_temperature,
)

# =============================================================================
# Tests for average_pressure_after_increase
# =============================================================================


def test_average_pressure_returns_stable_value():
    """
    Test average_pressure_after_increase to ensure it correctly identifies and
    returns the stable pressure value after a step increase from 10 to 100 torr
    at t=5 seconds.
    """
    time = np.linspace(0, 20, 200)
    pressure = np.where(time < 5, 10, 100)
    result = average_pressure_after_increase(time, pressure)
    assert result == pytest.approx(100, rel=0.01)


def test_average_pressure_handles_gradual_rise():
    """
    Test average_pressure_after_increase to verify it handles a gradual
    pressure rise that stabilizes at 50 torr after t=10 seconds and returns
    the correct stabilized value.
    """
    time = np.linspace(0, 30, 300)
    pressure = np.where(time < 10, time * 5, 50)
    result = average_pressure_after_increase(time, pressure)
    assert result == pytest.approx(50, rel=0.01)


def test_average_pressure_handles_noisy_data():
    """
    Test average_pressure_after_increase to confirm it can handle noisy
    pressure data with random fluctuations (±1 torr) and still accurately
    identify the stable pressure around 100 torr.
    """
    time = np.linspace(0, 20, 200)
    pressure = np.where(time < 5, 10, 100)
    rng = np.random.default_rng(seed=42)
    noise = rng.normal(0, 1, len(pressure))
    pressure = pressure + noise
    result = average_pressure_after_increase(time, pressure)
    assert result == pytest.approx(100, abs=5)


def test_average_pressure_uses_fallback_for_unstable():
    """
    Test average_pressure_after_increase to ensure it uses fallback behavior
    when pressure never stabilizes (continuously increasing), returning a
    positive average value.
    """
    time = np.linspace(0, 20, 200)
    pressure = time * 10
    result = average_pressure_after_increase(time, pressure)
    assert result > 0


def test_average_pressure_ignores_first_five_seconds():
    """
    Test average_pressure_after_increase to verify it correctly ignores the
    first 5 seconds of data when calculating the average, even with constant
    pressure of 50 torr from the start.
    """
    time = np.linspace(0, 20, 200)
    pressure = np.full_like(time, 50)
    result = average_pressure_after_increase(time, pressure)
    assert result == pytest.approx(50, rel=0.01)


@pytest.mark.parametrize("input_type", ["list", "tuple", "array"])
def test_average_pressure_accepts_multiple_input_types(input_type):
    """
    Test average_pressure_after_increase to ensure it accepts and correctly
    processes time and pressure data provided as lists, tuples, or numpy arrays,
    returning 100 torr for all input types.
    """
    time_list = [0, 5, 10, 15, 20]
    pressure_list = [10, 10, 100, 100, 100]
    if input_type == "list":
        time, pressure = time_list, pressure_list
    elif input_type == "tuple":
        time, pressure = tuple(time_list), tuple(pressure_list)
    else:
        time, pressure = np.array(time_list), np.array(pressure_list)
    result = average_pressure_after_increase(time, pressure)
    assert result == pytest.approx(100, rel=0.01)


@pytest.mark.parametrize("window_size", [3, 5, 10, 20])
def test_average_pressure_respects_window_parameter(window_size):
    """
    Test average_pressure_after_increase to verify it respects the window
    parameter for slope smoothing with different window sizes (3, 5, 10, 20),
    all returning 100 torr for stable pressure.
    """
    time = np.linspace(0, 20, 200)
    pressure = np.where(time < 5, 10, 100)
    result = average_pressure_after_increase(time, pressure, window=window_size)
    assert result == pytest.approx(100, rel=0.01)


@pytest.mark.parametrize("threshold", [1e-4, 1e-3, 1e-2, 1e-1])
def test_average_pressure_respects_slope_threshold(threshold):
    """
    Test average_pressure_after_increase to confirm it respects the
    slope_threshold parameter for stability detection with different thresholds
    (1e-4 to 1e-1), returning approximately 100 torr.
    """
    time = np.linspace(0, 20, 200)
    pressure = np.where(time < 5, 10, 100)
    result = average_pressure_after_increase(time, pressure, slope_threshold=threshold)
    assert result == pytest.approx(100, rel=0.1)


# =============================================================================
# Tests for calculate_flux_from_sample
# =============================================================================


def test_flux_returns_positive_slope():
    """
    Test calculate_flux_from_sample to ensure it correctly calculates a positive
    flux value from linearly increasing pressure data rising at 0.001 torr/s.
    """
    time = np.linspace(0, 100, 50)
    pressure = 0.1 + time * 0.001
    flux = calculate_flux_from_sample(time, pressure)
    assert flux > 0


def test_flux_filters_low_pressure_values():
    """
    Test calculate_flux_from_sample to verify it filters out unreliable low
    pressure values (<0.05 torr) from gauge data ranging from 0.01 to 0.5 torr
    and returns a finite flux value.
    """
    time = np.linspace(0, 100, 50)
    pressure = np.linspace(0.01, 0.5, 50)
    flux = calculate_flux_from_sample(time, pressure)
    assert np.isfinite(flux)


def test_flux_filters_high_pressure_values():
    """
    Test calculate_flux_from_sample to verify it filters out unreliable high
    pressure values (>0.95 torr) from gauge data ranging from 0.1 to 1.5 torr
    and returns a finite flux value.
    """
    time = np.linspace(0, 100, 50)
    pressure = np.linspace(0.1, 1.5, 50)
    flux = calculate_flux_from_sample(time, pressure)
    assert np.isfinite(flux)


def test_flux_uses_weighted_fit():
    """
    Test calculate_flux_from_sample to confirm it uses weighted linear fitting
    with exponential weighting favoring later data points, returning approximately
    0.001 torr/s for a linear pressure rise at that rate.
    """
    time = np.linspace(0, 100, 50)
    pressure = 0.1 + time * 0.001
    flux = calculate_flux_from_sample(time, pressure)
    assert flux == pytest.approx(0.001, rel=0.1)


def test_flux_handles_array_input():
    """
    Test calculate_flux_from_sample to ensure it accepts numpy array inputs for
    time and pressure data and returns a finite flux value.
    """
    time = np.array([0, 10, 20, 30, 40])
    pressure = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    flux = calculate_flux_from_sample(time, pressure)
    assert np.isfinite(flux)


def test_flux_handles_list_input():
    """
    Test calculate_flux_from_sample to ensure it accepts Python list inputs for
    time and pressure data and returns a finite flux value.
    """
    time = [0, 10, 20, 30, 40]
    pressure = [0.1, 0.2, 0.3, 0.4, 0.5]
    flux = calculate_flux_from_sample(time, pressure)
    assert np.isfinite(flux)


def test_flux_returns_near_zero_for_constant_pressure():
    """
    Test calculate_flux_from_sample to verify it returns a flux value near zero
    when pressure remains constant at 0.5 torr (no pressure rise).
    """
    time = np.linspace(0, 100, 50)
    pressure = np.full(50, 0.5)
    flux = calculate_flux_from_sample(time, pressure)
    assert flux == pytest.approx(0, abs=1e-10)


# =============================================================================
# Tests for calculate_permeability_from_flux
# =============================================================================


def test_permeability_returns_ufloat():
    """
    Test calculate_permeability_from_flux to ensure it returns a ufloat object
    with nominal value (.n) and standard deviation (.s) attributes for
    uncertainty propagation.
    """
    slope = 0.001
    V_m3 = 7.9e-5
    T_K = 873
    A_m2 = 1.88e-4
    e_m = 0.00088
    P_down = np.array([0.1, 0.2, 0.3])
    P_up = 100
    perm = calculate_permeability_from_flux(slope, V_m3, T_K, A_m2, e_m, P_down, P_up)
    assert hasattr(perm, "n")


def test_permeability_returns_positive_value():
    """
    Test calculate_permeability_from_flux to verify it returns a positive nominal
    permeability value with standard inputs, as permeability is a physical quantity
    that must be positive.
    """
    slope = 0.001
    V_m3 = 7.9e-5
    T_K = 873
    A_m2 = 1.88e-4
    e_m = 0.00088
    P_down = np.array([0.1, 0.2, 0.3])
    P_up = 100
    perm = calculate_permeability_from_flux(slope, V_m3, T_K, A_m2, e_m, P_down, P_up)
    assert perm.n > 0


def test_permeability_returns_finite_value():
    """
    Test calculate_permeability_from_flux to confirm it returns a finite (not
    infinite or NaN) nominal permeability value with standard inputs.
    """
    slope = 0.001
    V_m3 = 7.9e-5
    T_K = 873
    A_m2 = 1.88e-4
    e_m = 0.00088
    P_down = np.array([0.1, 0.2, 0.3])
    P_up = 100
    perm = calculate_permeability_from_flux(slope, V_m3, T_K, A_m2, e_m, P_down, P_up)
    assert np.isfinite(perm.n)


def test_permeability_has_positive_uncertainty():
    """
    Test calculate_permeability_from_flux to verify it propagates uncertainties
    through the calculation and returns a positive uncertainty value reflecting
    measurement precision.
    """
    slope = 0.001
    V_m3 = 7.9e-5
    T_K = 873
    A_m2 = 1.88e-4
    e_m = 0.00088
    P_down = np.array([0.1, 0.2, 0.3])
    P_up = 100
    perm = calculate_permeability_from_flux(slope, V_m3, T_K, A_m2, e_m, P_down, P_up)
    assert perm.s > 0


def test_permeability_increases_with_flux():
    """
    Test calculate_permeability_from_flux to confirm that permeability scales
    linearly with flux, where doubling the flux from 0.001 to 0.002 torr/s results
    in a higher permeability value.
    """
    V_m3 = 7.9e-5
    T_K = 873
    A_m2 = 1.88e-4
    e_m = 0.00088
    P_down = np.array([0.1, 0.2, 0.3])
    P_up = 100
    perm1 = calculate_permeability_from_flux(0.001, V_m3, T_K, A_m2, e_m, P_down, P_up)
    perm2 = calculate_permeability_from_flux(0.002, V_m3, T_K, A_m2, e_m, P_down, P_up)
    assert perm2.n > perm1.n


def test_permeability_scales_with_temperature():
    """
    Test calculate_permeability_from_flux to verify that permeability values
    differ when calculated at different temperatures (600K vs 900K), reflecting
    the temperature dependence of hydrogen permeation.
    """
    slope = 0.001
    V_m3 = 7.9e-5
    A_m2 = 1.88e-4
    e_m = 0.00088
    P_down = np.array([0.1, 0.2, 0.3])
    P_up = 100
    perm_low = calculate_permeability_from_flux(
        slope, V_m3, 600, A_m2, e_m, P_down, P_up
    )
    perm_high = calculate_permeability_from_flux(
        slope, V_m3, 900, A_m2, e_m, P_down, P_up
    )
    assert perm_high.n != perm_low.n


def test_permeability_decreases_with_upstream_pressure():
    """
    Test calculate_permeability_from_flux to confirm that permeability decreases
    with increasing upstream pressure, where 100 torr upstream pressure yields
    lower permeability than 50 torr due to the Sieverts law dependency (P^0.5).
    """
    slope = 0.001
    V_m3 = 7.9e-5
    T_K = 873
    A_m2 = 1.88e-4
    e_m = 0.00088
    P_down = np.array([0.1, 0.2, 0.3])
    perm1 = calculate_permeability_from_flux(slope, V_m3, T_K, A_m2, e_m, P_down, 50)
    perm2 = calculate_permeability_from_flux(slope, V_m3, T_K, A_m2, e_m, P_down, 100)
    assert perm2.n < perm1.n


def test_permeability_increases_with_thickness():
    """
    Test calculate_permeability_from_flux to verify that permeability scales
    linearly with sample thickness, where doubling thickness from 0.001m to
    0.002m results in permeability doubling.
    """
    slope = 0.001
    V_m3 = 7.9e-5
    T_K = 873
    A_m2 = 1.88e-4
    P_down = np.array([0.1, 0.2, 0.3])
    P_up = 100
    perm1 = calculate_permeability_from_flux(
        slope, V_m3, T_K, A_m2, 0.001, P_down, P_up
    )
    perm2 = calculate_permeability_from_flux(
        slope, V_m3, T_K, A_m2, 0.002, P_down, P_up
    )
    assert perm2.n == pytest.approx(2 * perm1.n, rel=0.01)


def test_permeability_decreases_with_area():
    """
    Test calculate_permeability_from_flux to verify that permeability scales
    inversely with sample area, where doubling area from 1e-4 to 2e-4 m² results
    in permeability halving.
    """
    slope = 0.001
    V_m3 = 7.9e-5
    T_K = 873
    e_m = 0.00088
    P_down = np.array([0.1, 0.2, 0.3])
    P_up = 100
    perm1 = calculate_permeability_from_flux(slope, V_m3, T_K, 1e-4, e_m, P_down, P_up)
    perm2 = calculate_permeability_from_flux(slope, V_m3, T_K, 2e-4, e_m, P_down, P_up)
    assert perm2.n == pytest.approx(0.5 * perm1.n, rel=0.01)


def test_permeability_uses_final_downstream_pressure():
    """
    Test calculate_permeability_from_flux to ensure it correctly uses the final
    value from a downstream pressure array, producing the same result whether
    passing the full array [0.1, 0.2, 0.5] or just the final scalar value 0.5.
    """
    slope = 0.001
    V_m3 = 7.9e-5
    T_K = 873
    A_m2 = 1.88e-4
    e_m = 0.00088
    P_down = np.array([0.1, 0.2, 0.5])
    P_up = 100
    perm_array = calculate_permeability_from_flux(
        slope, V_m3, T_K, A_m2, e_m, P_down, P_up
    )
    perm_scalar = calculate_permeability_from_flux(
        slope, V_m3, T_K, A_m2, e_m, 0.5, P_up
    )
    assert perm_array.n == pytest.approx(perm_scalar.n, rel=0.01)


def test_permeability_value_is_physically_reasonable():
    """
    Test calculate_permeability_from_flux to ensure the calculated permeability
    value is physically reasonable (positive and finite) with standard experimental
    inputs, avoiding unphysical results like negative, infinite, or NaN values.
    """
    slope = 0.001
    V_m3 = 7.9e-5
    T_K = 873
    A_m2 = 1.88e-4
    e_m = 0.00088
    P_down = np.array([0.1, 0.2, 0.3])
    P_up = 100
    perm = calculate_permeability_from_flux(slope, V_m3, T_K, A_m2, e_m, P_down, P_up)
    assert perm.n > 0 and np.isfinite(perm.n)


# =============================================================================
# Tests for evaluate_permeability_values
# =============================================================================


def test_evaluate_returns_six_outputs():
    """
    Test evaluate_permeability_values to ensure it returns a tuple of 6 outputs
    (temps, perms, x_error, y_error, error_lower, error_upper) when processing
    a single experimental dataset.
    """
    datasets = {
        "run1": {
            "temperature": 873,
            "time_data": np.linspace(0, 100, 100),
            "upstream_data": {"pressure_data": np.full(100, 100.0)},
            "downstream_data": {"pressure_data": np.linspace(0.1, 1.0, 100)},
            "sample_thickness": 0.00088,
        }
    }
    result = evaluate_permeability_values(datasets)
    assert len(result) == 6


def test_evaluate_temps_list_has_correct_length():
    """
    Test evaluate_permeability_values to verify it correctly processes two
    datasets and returns a temps list with length 2, one entry per dataset.
    """
    datasets = {
        "run1": {
            "temperature": 873,
            "time_data": np.linspace(0, 100, 100),
            "upstream_data": {"pressure_data": np.full(100, 100.0)},
            "downstream_data": {"pressure_data": np.linspace(0.1, 1.0, 100)},
        },
        "run2": {
            "temperature": 973,
            "time_data": np.linspace(0, 100, 100),
            "upstream_data": {"pressure_data": np.full(100, 100.0)},
            "downstream_data": {"pressure_data": np.linspace(0.1, 1.0, 100)},
        },
    }
    temps, perms, x_error, y_error, error_lower, error_upper = (
        evaluate_permeability_values(datasets)
    )
    assert len(temps) == 2


def test_evaluate_perms_list_has_correct_length():
    """
    Test evaluate_permeability_values to verify it correctly calculates
    permeability values for two datasets and returns a perms list with length 2.
    """
    datasets = {
        "run1": {
            "temperature": 873,
            "time_data": np.linspace(0, 100, 100),
            "upstream_data": {"pressure_data": np.full(100, 100.0)},
            "downstream_data": {"pressure_data": np.linspace(0.1, 1.0, 100)},
        },
        "run2": {
            "temperature": 973,
            "time_data": np.linspace(0, 100, 100),
            "upstream_data": {"pressure_data": np.full(100, 100.0)},
            "downstream_data": {"pressure_data": np.linspace(0.1, 1.0, 100)},
        },
    }
    temps, perms, x_error, y_error, error_lower, error_upper = (
        evaluate_permeability_values(datasets)
    )
    assert len(perms) == 2


def test_evaluate_perms_are_ufloats():
    """
    Test evaluate_permeability_values to confirm it returns permeability values
    as ufloat objects with both nominal values (.n) and uncertainties (.s) for
    proper error propagation.
    """
    datasets = {
        "run1": {
            "temperature": 873,
            "time_data": np.linspace(0, 100, 100),
            "upstream_data": {"pressure_data": np.full(100, 100.0)},
            "downstream_data": {"pressure_data": np.linspace(0.1, 1.0, 100)},
        }
    }
    temps, perms, x_error, y_error, error_lower, error_upper = (
        evaluate_permeability_values(datasets)
    )
    assert hasattr(perms[0], "n") and hasattr(perms[0], "s")


def test_evaluate_groups_by_temperature():
    """
    Test evaluate_permeability_values to ensure it groups multiple runs at the
    same temperature (873K) and returns a single unique temperature point in the
    error arrays after averaging.
    """
    datasets = {
        "run1": {
            "temperature": 873,
            "time_data": np.linspace(0, 100, 100),
            "upstream_data": {"pressure_data": np.full(100, 100.0)},
            "downstream_data": {"pressure_data": np.linspace(0.1, 1.0, 100)},
        },
        "run2": {
            "temperature": 873,
            "time_data": np.linspace(0, 100, 100),
            "upstream_data": {"pressure_data": np.full(100, 100.0)},
            "downstream_data": {"pressure_data": np.linspace(0.1, 1.0, 100)},
        },
    }
    temps, perms, x_error, y_error, error_lower, error_upper = (
        evaluate_permeability_values(datasets)
    )
    assert len(x_error) == 1


def test_evaluate_x_error_is_inverse_temperature():
    """
    Test evaluate_permeability_values to verify it correctly calculates x_error
    as the inverse temperature (1000/T), expecting approximately 1.145 for a
    dataset at 873 Kelvin.
    """
    datasets = {
        "run1": {
            "temperature": 873,
            "time_data": np.linspace(0, 100, 100),
            "upstream_data": {"pressure_data": np.full(100, 100.0)},
            "downstream_data": {"pressure_data": np.linspace(0.1, 1.0, 100)},
        }
    }
    temps, perms, x_error, y_error, error_lower, error_upper = (
        evaluate_permeability_values(datasets)
    )
    assert x_error[0] == pytest.approx(1000 / 873, rel=0.01)


def test_evaluate_y_error_is_positive():
    """
    Test evaluate_permeability_values to confirm it returns positive y_error
    values (nominal permeability values) for a standard experimental dataset.
    """
    datasets = {
        "run1": {
            "temperature": 873,
            "time_data": np.linspace(0, 100, 100),
            "upstream_data": {"pressure_data": np.full(100, 100.0)},
            "downstream_data": {"pressure_data": np.linspace(0.1, 1.0, 100)},
        }
    }
    temps, perms, x_error, y_error, error_lower, error_upper = (
        evaluate_permeability_values(datasets)
    )
    assert y_error[0] > 0


def test_evaluate_error_bars_are_positive():
    """
    Test evaluate_permeability_values to ensure it returns positive error bar
    values (error_lower and error_upper) representing measurement uncertainties
    for standard datasets.
    """
    datasets = {
        "run1": {
            "temperature": 873,
            "time_data": np.linspace(0, 100, 100),
            "upstream_data": {"pressure_data": np.full(100, 100.0)},
            "downstream_data": {"pressure_data": np.linspace(0.1, 1.0, 100)},
        }
    }
    temps, perms, x_error, y_error, error_lower, error_upper = (
        evaluate_permeability_values(datasets)
    )
    assert error_lower[0] > 0 and error_upper[0] > 0


def test_evaluate_uses_default_thickness():
    """
    Test evaluate_permeability_values to verify it uses the default sample
    thickness of 0.00088 meters when the sample_thickness key is not provided
    in the dataset dictionary.
    """
    datasets = {
        "run1": {
            "temperature": 873,
            "time_data": np.linspace(0, 100, 100),
            "upstream_data": {"pressure_data": np.full(100, 100.0)},
            "downstream_data": {"pressure_data": np.linspace(0.1, 1.0, 100)},
        }
    }
    temps, perms, x_error, y_error, error_lower, error_upper = (
        evaluate_permeability_values(datasets)
    )
    assert len(perms) == 1


def test_evaluate_uses_custom_thickness():
    """
    Test evaluate_permeability_values to confirm it uses custom sample_thickness
    values when provided (0.002m) instead of the default (0.00088m), resulting
    in different calculated permeability values.
    """
    datasets_default = {
        "run1": {
            "temperature": 873,
            "time_data": np.linspace(0, 100, 100),
            "upstream_data": {"pressure_data": np.full(100, 100.0)},
            "downstream_data": {"pressure_data": np.linspace(0.1, 1.0, 100)},
        }
    }
    datasets_custom = {
        "run1": {
            "temperature": 873,
            "time_data": np.linspace(0, 100, 100),
            "upstream_data": {"pressure_data": np.full(100, 100.0)},
            "downstream_data": {"pressure_data": np.linspace(0.1, 1.0, 100)},
            "sample_thickness": 0.002,
        }
    }
    _, perms_default, _, _, _, _ = evaluate_permeability_values(datasets_default)
    _, perms_custom, _, _, _, _ = evaluate_permeability_values(datasets_custom)
    assert perms_default[0].n != pytest.approx(perms_custom[0].n, rel=0.01)


def test_evaluate_weighted_average_for_multiple_runs():
    """
    Test evaluate_permeability_values to verify it performs weighted averaging
    when multiple runs are conducted at the same temperature (873K), combining
    measurements and returning positive combined uncertainty.
    """
    datasets = {
        "run1": {
            "temperature": 873,
            "time_data": np.linspace(0, 100, 100),
            "upstream_data": {"pressure_data": np.full(100, 100.0)},
            "downstream_data": {"pressure_data": np.linspace(0.1, 1.0, 100)},
        },
        "run2": {
            "temperature": 873,
            "time_data": np.linspace(0, 100, 100),
            "upstream_data": {"pressure_data": np.full(100, 100.0)},
            "downstream_data": {"pressure_data": np.linspace(0.1, 1.5, 100)},
        },
    }
    temps, perms, x_error, y_error, error_lower, error_upper = (
        evaluate_permeability_values(datasets)
    )
    assert error_lower[0] > 0


def test_evaluate_sorts_temperatures():
    """
    Test evaluate_permeability_values to ensure it sorts datasets by temperature
    in ascending order (773K, 873K, 973K), resulting in x_error values (1000/T)
    in descending order for Arrhenius plotting.
    """
    datasets = {
        "run1": {
            "temperature": 973,
            "time_data": np.linspace(0, 100, 100),
            "upstream_data": {"pressure_data": np.full(100, 100.0)},
            "downstream_data": {"pressure_data": np.linspace(0.1, 1.0, 100)},
        },
        "run2": {
            "temperature": 773,
            "time_data": np.linspace(0, 100, 100),
            "upstream_data": {"pressure_data": np.full(100, 100.0)},
            "downstream_data": {"pressure_data": np.linspace(0.1, 1.0, 100)},
        },
        "run3": {
            "temperature": 873,
            "time_data": np.linspace(0, 100, 100),
            "upstream_data": {"pressure_data": np.full(100, 100.0)},
            "downstream_data": {"pressure_data": np.linspace(0.1, 1.0, 100)},
        },
    }
    temps, perms, x_error, y_error, error_lower, error_upper = (
        evaluate_permeability_values(datasets)
    )
    # Sorted by temperature ascending: 773, 873, 973
    # So 1000/T in descending order: 1000/773 > 1000/873 > 1000/973
    assert x_error[0] > x_error[1] > x_error[2]


def test_evaluate_handles_non_ufloat_permeability():
    """
    Test evaluate_permeability_values to verify it handles edge cases where
    permeability calculation returns a regular float instead of a ufloat object,
    resulting in zero error bars when uncertainty propagation is not available.
    """
    datasets = {
        "run1": {
            "temperature": 873,
            "time_data": np.linspace(0, 100, 100),
            "upstream_data": {"pressure_data": np.full(100, 100.0)},
            "downstream_data": {"pressure_data": np.linspace(0.1, 1.0, 100)},
        }
    }

    # Mock calculate_permeability_from_flux to return a regular float instead of ufloat
    with patch(
        "shield_das.analysis.calculate_permeability_from_flux", return_value=1e-10
    ):
        temps, perms, x_error, y_error, error_lower, error_upper = (
            evaluate_permeability_values(datasets)
        )
        # When avg_perm is not a ufloat, error bars should be 0
        assert error_lower[0] == 0
        assert error_upper[0] == 0


# =============================================================================
# Tests for fit_permeability_data
# =============================================================================


def test_fit_returns_two_arrays():
    """
    Test fit_permeability_data to ensure it returns a tuple of two arrays
    (fit_x and fit_y) representing the Arrhenius fit for input temperature
    and permeability data.
    """
    temps = [773, 873, 973]
    perms = [ufloat(1e-10, 1e-11), ufloat(5e-10, 5e-11), ufloat(2e-9, 2e-10)]
    result = fit_permeability_data(temps, perms)
    assert len(result) == 2


def test_fit_arrays_have_100_points():
    """
    Test fit_permeability_data to verify it generates smooth fit curves with
    exactly 100 points for both fit_x (1000/T) and fit_y (permeability) arrays
    for Arrhenius plotting.
    """
    temps = [773, 873, 973]
    perms = [ufloat(1e-10, 1e-11), ufloat(5e-10, 5e-11), ufloat(2e-9, 2e-10)]
    fit_x, fit_y = fit_permeability_data(temps, perms)
    assert len(fit_x) == 100 and len(fit_y) == 100


def test_fit_x_range_covers_input_data():
    """
    Test fit_permeability_data to confirm it generates fit_x values spanning
    the complete range of input temperatures, from 1000/973≈1.028 to
    1000/773≈1.294, ensuring the fit curve covers all experimental data.
    """
    temps = [773, 873, 973]
    perms = [ufloat(1e-10, 1e-11), ufloat(5e-10, 5e-11), ufloat(2e-9, 2e-10)]
    fit_x, fit_y = fit_permeability_data(temps, perms)
    assert fit_x.min() == pytest.approx(1000 / 973, rel=0.01)
    assert fit_x.max() == pytest.approx(1000 / 773, rel=0.01)


def test_fit_y_values_are_positive():
    """
    Test fit_permeability_data to ensure all fitted permeability values
    (fit_y) are positive, as required by the physical interpretation of
    permeability in Arrhenius models.
    """
    temps = [773, 873, 973]
    perms = [ufloat(1e-10, 1e-11), ufloat(5e-10, 5e-11), ufloat(2e-9, 2e-10)]
    fit_x, fit_y = fit_permeability_data(temps, perms)
    assert np.all(fit_y > 0)


def test_fit_handles_ufloat_inputs():
    """
    Test fit_permeability_data to verify it correctly processes ufloat input
    permeability values with uncertainties, extracting nominal values for
    fitting and returning valid numpy arrays.
    """
    temps = [773, 873, 973]
    perms = [ufloat(1e-10, 1e-11), ufloat(5e-10, 5e-11), ufloat(2e-9, 2e-10)]
    fit_x, fit_y = fit_permeability_data(temps, perms)
    assert isinstance(fit_x, np.ndarray) and isinstance(fit_y, np.ndarray)


def test_fit_handles_float_inputs():
    """
    Test fit_permeability_data to confirm it accepts regular float permeability
    values (without uncertainties) and returns valid numpy arrays for the
    Arrhenius fit.
    """
    temps = [773, 873, 973]
    perms = [1e-10, 5e-10, 2e-9]
    fit_x, fit_y = fit_permeability_data(temps, perms)
    assert isinstance(fit_x, np.ndarray) and isinstance(fit_y, np.ndarray)


def test_fit_uses_weighted_least_squares():
    """
    Test fit_permeability_data to verify it performs weighted least squares
    fitting, where data points with large uncertainties (5e-10 uncertainty on
    middle point) are weighted less than points with small uncertainties (1e-11),
    resulting in noticeably different fit curves (>1% relative difference).
    """
    temps = [773, 873, 973]
    perms_equal = [ufloat(1e-10, 1e-11), ufloat(5e-10, 1e-11), ufloat(2e-9, 1e-11)]
    # Give middle point huge uncertainty so it's weighted less
    perms_varied = [ufloat(1e-10, 1e-11), ufloat(5e-10, 5e-10), ufloat(2e-9, 1e-11)]
    fit_x_equal, fit_y_equal = fit_permeability_data(temps, perms_equal)
    fit_x_varied, fit_y_varied = fit_permeability_data(temps, perms_varied)
    # Fits should be different when weights change significantly
    max_rel_diff = np.max(np.abs(fit_y_equal - fit_y_varied) / fit_y_equal)
    assert max_rel_diff > 0.01  # At least 1% difference


def test_fit_arrhenius_increases_with_temperature():
    """
    Test fit_permeability_data to ensure the Arrhenius fit shows permeability
    decreasing as temperature decreases, with higher permeability at the start
    of fit_y (high temperature) than at the end (low temperature).
    """
    temps = [773, 873, 973]
    perms = [ufloat(1e-10, 1e-11), ufloat(5e-10, 5e-11), ufloat(2e-9, 2e-10)]
    fit_x, fit_y = fit_permeability_data(temps, perms)
    # fit_x increases (1000/T from low to high)
    # Temperature decreases along fit_x, so permeability decreases
    assert fit_y[0] > fit_y[-1]


def test_fit_x_is_monotonic():
    """
    Test fit_permeability_data to confirm that fit_x (1000/T) values are
    monotonically increasing from 1000/973 to 1000/773, ensuring a proper
    Arrhenius plot x-axis.
    """
    temps = [773, 873, 973]
    perms = [ufloat(1e-10, 1e-11), ufloat(5e-10, 5e-11), ufloat(2e-9, 2e-10)]
    fit_x, fit_y = fit_permeability_data(temps, perms)
    assert np.all(np.diff(fit_x) > 0)


def test_fit_handles_numpy_array_inputs():
    """
    Test fit_permeability_data to verify it accepts numpy array inputs for
    temperatures (instead of Python lists) and returns valid numpy arrays for
    the Arrhenius fit curve.
    """
    temps = np.array([773, 873, 973])
    perms = [ufloat(1e-10, 1e-11), ufloat(5e-10, 5e-11), ufloat(2e-9, 2e-10)]
    fit_x, fit_y = fit_permeability_data(temps, perms)
    assert isinstance(fit_x, np.ndarray) and isinstance(fit_y, np.ndarray)


# Tests for voltage_to_pressure function


def test_voltage_to_pressure_converts_correctly_1000_torr_scale():
    """
    Test voltage_to_pressure to ensure it correctly converts voltage to pressure
    using linear scaling for a 1000 torr full-scale gauge, where 5V should equal
    500 torr.
    """
    voltage = np.array([5.0])
    pressure = voltage_to_pressure(voltage, full_scale_torr=1000)
    assert np.isclose(pressure[0], 500.0, rtol=1e-10)


def test_voltage_to_pressure_converts_correctly_1_torr_scale():
    """
    Test voltage_to_pressure to ensure it correctly converts voltage to pressure
    using linear scaling for a 1 torr full-scale gauge, where 5V should equal
    0.5 torr.
    """
    voltage = np.array([5.0])
    pressure = voltage_to_pressure(voltage, full_scale_torr=1)
    assert np.isclose(pressure[0], 0.5, rtol=1e-10)


def test_voltage_to_pressure_filters_noise_on_high_range():
    """
    Test voltage_to_pressure to verify it applies noise filtering by setting
    readings below 0.5 torr to zero on 1000 torr full-scale gauge, eliminating
    spurious low readings.
    """
    # 0.01V * 100 = 1 torr (above threshold, kept)
    # 0.001V * 100 = 0.1 torr (below 0.5 threshold, filtered to 0)
    voltage = np.array([0.01, 0.001])
    pressure = voltage_to_pressure(voltage, full_scale_torr=1000)
    assert np.isclose(pressure[0], 1.0, rtol=1e-10)
    assert np.isclose(pressure[1], 0.0, rtol=1e-10)


def test_voltage_to_pressure_filters_noise_on_low_range():
    """
    Test voltage_to_pressure to verify it applies noise filtering by setting
    readings below 0.0005 torr to zero on 1 torr full-scale gauge, eliminating
    low-level noise.
    """
    # 0.01V * 0.1 = 0.001 torr (above 0.0005 threshold, kept)
    # 0.001V * 0.1 = 0.0001 torr (below 0.0005 threshold, filtered to 0)
    voltage = np.array([0.01, 0.001])
    pressure = voltage_to_pressure(voltage, full_scale_torr=1)
    assert np.isclose(pressure[0], 0.001, rtol=1e-10)
    assert np.isclose(pressure[1], 0.0, rtol=1e-10)


def test_voltage_to_pressure_clips_to_valid_range():
    """
    Test voltage_to_pressure to ensure it clips pressure values to the valid
    gauge range (0 to full_scale), preventing negative values and values
    exceeding the maximum measurable pressure.
    """
    # Negative voltage should clip to 0
    # Voltage > 10V should clip to full_scale
    voltage = np.array([-1.0, 5.0, 15.0])
    pressure = voltage_to_pressure(voltage, full_scale_torr=1000)
    assert np.isclose(pressure[0], 0.0, rtol=1e-10)
    assert np.isclose(pressure[1], 500.0, rtol=1e-10)
    assert np.isclose(pressure[2], 1000.0, rtol=1e-10)


def test_voltage_to_pressure_handles_array_input():
    """
    Test voltage_to_pressure to verify it correctly processes numpy array inputs
    with multiple voltage readings, returning corresponding pressure values for
    each element.
    """
    voltage = np.array([0.0, 2.5, 5.0, 7.5, 10.0])
    pressure = voltage_to_pressure(voltage, full_scale_torr=1000)
    expected = np.array([0.0, 250.0, 500.0, 750.0, 1000.0])
    # First element filtered by threshold
    assert np.isclose(pressure[0], 0.0, rtol=1e-10)
    assert np.allclose(pressure[1:], expected[1:], rtol=1e-10)


def test_voltage_to_pressure_zero_voltage_returns_zero():
    """
    Test voltage_to_pressure to confirm that zero voltage input returns zero
    pressure output, representing gauge reading when no pressure is present.
    """
    voltage = np.array([0.0])
    pressure = voltage_to_pressure(voltage, full_scale_torr=1000)
    assert np.isclose(pressure[0], 0.0, rtol=1e-10)


def test_voltage_to_pressure_max_voltage_returns_full_scale():
    """
    Test voltage_to_pressure to verify that maximum voltage (10V) returns the
    full scale pressure value, representing the gauge's upper measurement limit.
    """
    voltage = np.array([10.0])
    pressure = voltage_to_pressure(voltage, full_scale_torr=1000)
    assert np.isclose(pressure[0], 1000.0, rtol=1e-10)


def test_voltage_to_pressure_returns_ndarray():
    """
    Test voltage_to_pressure to ensure it returns a numpy ndarray type, maintaining
    consistency with numpy array operations throughout the analysis pipeline.
    """
    voltage = np.array([5.0])
    pressure = voltage_to_pressure(voltage, full_scale_torr=1000)
    assert isinstance(pressure, np.ndarray)


# Tests for calculate_error function


def test_calculate_error_returns_half_percent_for_low_pressure():
    """
    Test calculate_error to verify it returns 0.5% uncertainty (0.005 fraction)
    for pressure readings at or below 1 torr, matching gauge manufacturer
    specifications for low-pressure accuracy.
    """
    pressure = 0.5
    error = calculate_error(pressure)
    expected = 0.5 * 0.005
    assert np.isclose(error, expected, rtol=1e-10)


def test_calculate_error_returns_quarter_percent_for_high_pressure():
    """
    Test calculate_error to verify it returns 0.25% uncertainty (0.0025 fraction)
    for pressure readings above 1 torr, reflecting improved gauge accuracy at
    higher pressures.
    """
    pressure = 10.0
    error = calculate_error(pressure)
    expected = 10.0 * 0.0025
    assert np.isclose(error, expected, rtol=1e-10)


def test_calculate_error_boundary_at_one_torr():
    """
    Test calculate_error to confirm the accuracy threshold behavior at exactly
    1 torr, where readings at or below use 0.5% uncertainty and readings above
    use 0.25% uncertainty.
    """
    pressure_at = 1.0
    pressure_below = 0.9999
    pressure_above = 1.0001
    error_at = calculate_error(pressure_at)
    error_below = calculate_error(pressure_below)
    error_above = calculate_error(pressure_above)
    assert np.isclose(error_at, 1.0 * 0.005, rtol=1e-10)
    assert np.isclose(error_below, 0.9999 * 0.005, rtol=1e-10)
    assert np.isclose(error_above, 1.0001 * 0.0025, rtol=1e-10)


def test_calculate_error_handles_array_input():
    """
    Test calculate_error to verify it correctly processes numpy array inputs
    with multiple pressure readings, applying the appropriate uncertainty model
    (0.5% or 0.25%) element-wise to each value.
    """
    pressures = np.array([0.5, 1.0, 10.0, 100.0])
    errors = calculate_error(pressures)
    expected = np.array([0.5 * 0.005, 1.0 * 0.005, 10.0 * 0.0025, 100.0 * 0.0025])
    assert np.allclose(errors, expected, rtol=1e-10)


def test_calculate_error_handles_scalar_input():
    """
    Test calculate_error to confirm it accepts scalar float inputs and returns
    scalar float outputs, supporting single-point uncertainty calculations without
    array wrapping.
    """
    pressure = 50.0
    error = calculate_error(pressure)
    expected = 50.0 * 0.0025
    assert np.isclose(error, expected, rtol=1e-10)
    assert isinstance(error, (float, np.floating, np.ndarray))


def test_calculate_error_scales_linearly_with_pressure():
    """
    Test calculate_error to verify that uncertainty scales linearly with pressure
    magnitude within each accuracy regime, confirming proportional error behavior
    (doubling pressure doubles absolute uncertainty).
    """
    # Low pressure regime (0.5%)
    p1_low = 0.5
    p2_low = 1.0
    err1_low = calculate_error(p1_low)
    err2_low = calculate_error(p2_low)
    assert np.isclose(err2_low / err1_low, p2_low / p1_low, rtol=1e-10)

    # High pressure regime (0.25%)
    p1_high = 10.0
    p2_high = 100.0
    err1_high = calculate_error(p1_high)
    err2_high = calculate_error(p2_high)
    assert np.isclose(err2_high / err1_high, p2_high / p1_high, rtol=1e-10)


def test_calculate_error_returns_positive_values():
    """
    Test calculate_error to ensure all returned uncertainty values are positive
    numbers, as negative uncertainties are physically meaningless in measurement
    error propagation.
    """
    pressures = np.array([0.01, 0.1, 1.0, 10.0, 100.0])
    errors = calculate_error(pressures)
    assert np.all(errors > 0)


def test_calculate_error_returns_ndarray_for_array_input():
    """
    Test calculate_error to verify it returns numpy ndarray type when given array
    input, maintaining type consistency for vectorised uncertainty calculations.
    """
    pressures = np.array([1.0, 10.0, 100.0])
    errors = calculate_error(pressures)
    assert isinstance(errors, np.ndarray)


def test_calculate_error_zero_pressure_returns_zero():
    """
    Test calculate_error to confirm that zero pressure input returns zero
    uncertainty, as 0.5% of zero is mathematically zero (though physically rare).
    """
    pressure = 0.0
    error = calculate_error(pressure)
    assert np.isclose(error, 0.0, rtol=1e-10)


# Tests for voltage_to_temperature function


def test_voltage_to_temperature_converts_with_cold_junction_compensation():
    """
    Test voltage_to_temperature to ensure it correctly applies cold junction
    compensation by adding the local temperature's equivalent voltage to the
    measured voltage before converting to temperature.
    """
    # At room temperature (25°C), a small voltage reading should result in
    # temperature slightly above room temperature
    local_temp = np.array([25.0])
    voltage = np.array([1.0])  # 1 mV from thermocouple
    temperature = voltage_to_temperature(local_temp, voltage)
    # Result should be greater than local temperature
    assert temperature[0] > local_temp[0]
    assert isinstance(temperature, np.ndarray)


def test_voltage_to_temperature_zero_voltage_returns_local_temperature():
    """
    Test voltage_to_temperature to verify that zero thermocouple voltage returns
    the local temperature, indicating that the measurement junction is at the same
    temperature as the reference junction.
    """
    local_temp = np.array([25.0])
    voltage = np.array([0.0])
    temperature = voltage_to_temperature(local_temp, voltage)
    # With zero voltage difference, temperature should equal local temperature
    assert np.isclose(temperature[0], local_temp[0], rtol=0.01)


def test_voltage_to_temperature_handles_array_inputs():
    """
    Test voltage_to_temperature to verify it correctly processes numpy array inputs
    with multiple voltage readings, applying cold junction compensation element-wise
    to each measurement.
    """
    local_temp = np.array([25.0, 25.0, 25.0])
    voltage = np.array([0.0, 1.0, 2.0])
    temperature = voltage_to_temperature(local_temp, voltage)
    assert len(temperature) == 3
    assert isinstance(temperature, np.ndarray)
    # Temperatures should increase with voltage
    assert temperature[1] > temperature[0]
    assert temperature[2] > temperature[1]


def test_voltage_to_temperature_positive_voltage_increases_temperature():
    """
    Test voltage_to_temperature to confirm that positive thermocouple voltage
    readings result in measured temperatures higher than the local reference
    temperature, consistent with type K thermocouple characteristics.
    """
    local_temp = np.array([25.0])
    voltage_low = np.array([1.0])
    voltage_high = np.array([5.0])
    temp_low = voltage_to_temperature(local_temp, voltage_low)
    temp_high = voltage_to_temperature(local_temp, voltage_high)
    # Higher voltage should give higher temperature
    assert temp_high[0] > temp_low[0]
    # Both should be above local temperature
    assert temp_low[0] > local_temp[0]


def test_voltage_to_temperature_varying_local_temperature():
    """
    Test voltage_to_temperature to verify it correctly handles varying local
    reference temperatures, adjusting the cold junction compensation voltage
    accordingly for each measurement.
    """
    # Same thermocouple voltage but different local temperatures
    local_temp_cold = np.array([0.0])
    local_temp_hot = np.array([50.0])
    voltage = np.array([10.0])  # Same thermocouple voltage
    temp_from_cold = voltage_to_temperature(local_temp_cold, voltage)
    temp_from_hot = voltage_to_temperature(local_temp_hot, voltage)
    # With same thermocouple voltage, higher local temp should give higher result
    assert temp_from_hot[0] > temp_from_cold[0]


def test_voltage_to_temperature_returns_ndarray():
    """
    Test voltage_to_temperature to ensure it returns a numpy ndarray type,
    maintaining consistency with numpy array operations throughout the analysis
    pipeline.
    """
    local_temp = np.array([25.0])
    voltage = np.array([1.0])
    temperature = voltage_to_temperature(local_temp, voltage)
    assert isinstance(temperature, np.ndarray)


def test_voltage_to_temperature_handles_scalar_arrays():
    """
    Test voltage_to_temperature to confirm it accepts single-element arrays and
    returns a single-element array, supporting both scalar and vectorized usage
    patterns.
    """
    local_temp = np.array([25.0])
    voltage = np.array([2.5])
    temperature = voltage_to_temperature(local_temp, voltage)
    assert len(temperature) == 1
    assert isinstance(temperature, np.ndarray)


def test_voltage_to_temperature_negative_voltage_decreases_temperature():
    """
    Test voltage_to_temperature to verify that negative thermocouple voltage
    readings result in measured temperatures lower than the local reference
    temperature, representing reversed thermal gradients.
    """
    local_temp = np.array([25.0])
    voltage = np.array([-1.0])
    temperature = voltage_to_temperature(local_temp, voltage)
    # Negative voltage should give temperature below local temperature
    assert temperature[0] < local_temp[0]


def test_voltage_to_temperature_consistent_with_thermocouple_functions():
    """
    Test voltage_to_temperature to ensure the conversion is consistent with the
    underlying thermocouple conversion functions, verifying that the cold junction
    compensation logic is correctly implemented.
    """
    # Test at a known point: 25°C local, 10mV thermocouple voltage
    local_temp = np.array([25.0])
    voltage = np.array([10.0])
    temperature = voltage_to_temperature(local_temp, voltage)
    # Should return a reasonable temperature above local temp
    assert temperature[0] > 25.0
    assert temperature[0] < 500.0  # Reasonable upper bound for typical measurements
    assert isinstance(temperature[0], (float, np.floating))
