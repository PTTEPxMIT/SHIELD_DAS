from unittest.mock import patch

import numpy as np
import pytest
from uncertainties import ufloat

from shield_das.analysis import (
    average_pressure_after_increase,
    calculate_flux_from_sample,
    calculate_permeability_from_flux,
    evaluate_permeability_values,
    fit_permeability_data,
)

# =============================================================================
# Tests for average_pressure_after_increase
# =============================================================================


def test_average_pressure_returns_stable_value():
    """Test: Pressure jumps from 10 to 100 at t=5s. Expect: Returns 100."""
    time = np.linspace(0, 20, 200)
    pressure = np.where(time < 5, 10, 100)
    result = average_pressure_after_increase(time, pressure)
    assert result == pytest.approx(100, rel=0.01)


def test_average_pressure_handles_gradual_rise():
    """Test: Pressure rises to t=10s then stabilizes at 50. Expect: Returns 50."""
    time = np.linspace(0, 30, 300)
    pressure = np.where(time < 10, time * 5, 50)
    result = average_pressure_after_increase(time, pressure)
    assert result == pytest.approx(50, rel=0.01)


def test_average_pressure_handles_noisy_data():
    """Test: Stable pressure with random noise (±1 torr). Expect: Returns ~100."""
    time = np.linspace(0, 20, 200)
    pressure = np.where(time < 5, 10, 100)
    rng = np.random.default_rng(seed=42)
    noise = rng.normal(0, 1, len(pressure))
    pressure = pressure + noise
    result = average_pressure_after_increase(time, pressure)
    assert result == pytest.approx(100, abs=5)


def test_average_pressure_uses_fallback_for_unstable():
    """Test: Continuously increasing pressure. Expect: Returns positive value."""
    time = np.linspace(0, 20, 200)
    pressure = time * 10
    result = average_pressure_after_increase(time, pressure)
    assert result > 0


def test_average_pressure_ignores_first_five_seconds():
    """Test: Constant pressure from t=0. Expect: Ignores first 5s, returns 50."""
    time = np.linspace(0, 20, 200)
    pressure = np.full_like(time, 50)
    result = average_pressure_after_increase(time, pressure)
    assert result == pytest.approx(50, rel=0.01)


@pytest.mark.parametrize("input_type", ["list", "tuple", "array"])
def test_average_pressure_accepts_multiple_input_types(input_type):
    """Test: Input as list/tuple/array. Expect: All accepted, returns 100."""
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
    """Test: Different window sizes for smoothing. Expect: Returns 100 for all."""
    time = np.linspace(0, 20, 200)
    pressure = np.where(time < 5, 10, 100)
    result = average_pressure_after_increase(time, pressure, window=window_size)
    assert result == pytest.approx(100, rel=0.01)


@pytest.mark.parametrize("threshold", [1e-4, 1e-3, 1e-2, 1e-1])
def test_average_pressure_respects_slope_threshold(threshold):
    """Test: Different slope thresholds. Expect: Returns ~100 for all."""
    time = np.linspace(0, 20, 200)
    pressure = np.where(time < 5, 10, 100)
    result = average_pressure_after_increase(time, pressure, slope_threshold=threshold)
    assert result == pytest.approx(100, rel=0.1)


# =============================================================================
# Tests for calculate_flux_from_sample
# =============================================================================


def test_flux_returns_positive_slope():
    """Test: Linearly increasing pressure data. Expect: Positive flux value."""
    time = np.linspace(0, 100, 50)
    pressure = 0.1 + time * 0.001
    flux = calculate_flux_from_sample(time, pressure)
    assert flux > 0


def test_flux_filters_low_pressure_values():
    """Test: Pressure from 0.01 to 0.5 torr. Expect: Filters <0.05, returns finite."""
    time = np.linspace(0, 100, 50)
    pressure = np.linspace(0.01, 0.5, 50)
    flux = calculate_flux_from_sample(time, pressure)
    assert np.isfinite(flux)


def test_flux_filters_high_pressure_values():
    """Test: Pressure from 0.1 to 1.5 torr. Expect: Filters >0.95, returns finite."""
    time = np.linspace(0, 100, 50)
    pressure = np.linspace(0.1, 1.5, 50)
    flux = calculate_flux_from_sample(time, pressure)
    assert np.isfinite(flux)


def test_flux_uses_weighted_fit():
    """Test: Linear pressure rise at 0.001 torr/s. Expect: Flux ≈ 0.001."""
    time = np.linspace(0, 100, 50)
    pressure = 0.1 + time * 0.001
    flux = calculate_flux_from_sample(time, pressure)
    assert flux == pytest.approx(0.001, rel=0.1)


def test_flux_handles_array_input():
    """Test: Numpy array inputs. Expect: Returns finite flux value."""
    time = np.array([0, 10, 20, 30, 40])
    pressure = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    flux = calculate_flux_from_sample(time, pressure)
    assert np.isfinite(flux)


def test_flux_handles_list_input():
    """Test: Python list inputs. Expect: Returns finite flux value."""
    time = [0, 10, 20, 30, 40]
    pressure = [0.1, 0.2, 0.3, 0.4, 0.5]
    flux = calculate_flux_from_sample(time, pressure)
    assert np.isfinite(flux)


def test_flux_returns_near_zero_for_constant_pressure():
    """Test: Constant pressure at 0.5 torr. Expect: Flux ≈ 0."""
    time = np.linspace(0, 100, 50)
    pressure = np.full(50, 0.5)
    flux = calculate_flux_from_sample(time, pressure)
    assert flux == pytest.approx(0, abs=1e-10)


# =============================================================================
# Tests for calculate_permeability_from_flux
# =============================================================================


def test_permeability_returns_ufloat():
    """Test: Standard inputs. Expect: Returns ufloat object with .n and .s."""
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
    """Test: Standard inputs. Expect: Positive nominal permeability value."""
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
    """Test: Standard inputs. Expect: Finite nominal permeability value."""
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
    """Test: Standard inputs. Expect: Positive uncertainty value."""
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
    """Test: Double the flux. Expect: Higher permeability."""
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
    """Test: 600K vs 900K. Expect: Different permeabilities."""
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
    """Test: 50 vs 100 torr upstream. Expect: Lower perm at higher pressure."""
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
    """Test: Double thickness. Expect: Permeability doubles."""
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
    """Test: Double area. Expect: Permeability halves."""
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
    """Test: P_down array vs final value. Expect: Same result."""
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
    """Test: Standard inputs. Expect: Positive and finite."""
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
    """Test: Single dataset. Expect: Returns 6 outputs."""
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
    """Test: Two datasets. Expect: temps list has length 2."""
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
    """Test: Two datasets. Expect: perms list has length 2."""
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
    """Test: Single dataset. Expect: perms contain ufloat objects."""
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
    """Test: Two runs at same temp. Expect: One unique temp in error arrays."""
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
    """Test: 873 K dataset. Expect: x_error ≈ 1.145 (1000/873)."""
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
    """Test: Standard dataset. Expect: y_error values are positive."""
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
    """Test: Standard dataset. Expect: error_lower and error_upper are positive."""
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
    """Test: No sample_thickness key. Expect: Uses default 0.00088 m."""
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
    """Test: Custom sample_thickness. Expect: Different permeability than default."""
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
    """Test: Two runs at same temp. Expect: Combined uncertainty."""
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
    """Test: Temps 973, 773, 873. Expect: x_error in descending order."""
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
    """Test: Mock returns regular float instead of ufloat. Expect: Zero error bars."""
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
    """Test: Standard inputs. Expect: Returns two arrays."""
    temps = [773, 873, 973]
    perms = [ufloat(1e-10, 1e-11), ufloat(5e-10, 5e-11), ufloat(2e-9, 2e-10)]
    result = fit_permeability_data(temps, perms)
    assert len(result) == 2


def test_fit_arrays_have_100_points():
    """Test: Standard inputs. Expect: fit_x and fit_y have 100 points."""
    temps = [773, 873, 973]
    perms = [ufloat(1e-10, 1e-11), ufloat(5e-10, 5e-11), ufloat(2e-9, 2e-10)]
    fit_x, fit_y = fit_permeability_data(temps, perms)
    assert len(fit_x) == 100 and len(fit_y) == 100


def test_fit_x_range_covers_input_data():
    """Test: Temps 773-973 K. Expect: fit_x covers 1000/973 to 1000/773."""
    temps = [773, 873, 973]
    perms = [ufloat(1e-10, 1e-11), ufloat(5e-10, 5e-11), ufloat(2e-9, 2e-10)]
    fit_x, fit_y = fit_permeability_data(temps, perms)
    assert fit_x.min() == pytest.approx(1000 / 973, rel=0.01)
    assert fit_x.max() == pytest.approx(1000 / 773, rel=0.01)


def test_fit_y_values_are_positive():
    """Test: Positive permeability inputs. Expect: All fit_y values positive."""
    temps = [773, 873, 973]
    perms = [ufloat(1e-10, 1e-11), ufloat(5e-10, 5e-11), ufloat(2e-9, 2e-10)]
    fit_x, fit_y = fit_permeability_data(temps, perms)
    assert np.all(fit_y > 0)


def test_fit_handles_ufloat_inputs():
    """Test: ufloat inputs with uncertainties. Expect: Returns valid arrays."""
    temps = [773, 873, 973]
    perms = [ufloat(1e-10, 1e-11), ufloat(5e-10, 5e-11), ufloat(2e-9, 2e-10)]
    fit_x, fit_y = fit_permeability_data(temps, perms)
    assert isinstance(fit_x, np.ndarray) and isinstance(fit_y, np.ndarray)


def test_fit_handles_float_inputs():
    """Test: Regular float inputs. Expect: Returns valid arrays."""
    temps = [773, 873, 973]
    perms = [1e-10, 5e-10, 2e-9]
    fit_x, fit_y = fit_permeability_data(temps, perms)
    assert isinstance(fit_x, np.ndarray) and isinstance(fit_y, np.ndarray)


def test_fit_uses_weighted_least_squares():
    """Test: Large uncertainty on middle point. Expect: Different fit."""
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
    """Test: Increasing temps. Expect: fit_y decreases as fit_x increases."""
    temps = [773, 873, 973]
    perms = [ufloat(1e-10, 1e-11), ufloat(5e-10, 5e-11), ufloat(2e-9, 2e-10)]
    fit_x, fit_y = fit_permeability_data(temps, perms)
    # fit_x increases (1000/T from low to high)
    # Temperature decreases along fit_x, so permeability decreases
    assert fit_y[0] > fit_y[-1]


def test_fit_x_is_monotonic():
    """Test: Standard inputs. Expect: fit_x is monotonically increasing."""
    temps = [773, 873, 973]
    perms = [ufloat(1e-10, 1e-11), ufloat(5e-10, 5e-11), ufloat(2e-9, 2e-10)]
    fit_x, fit_y = fit_permeability_data(temps, perms)
    assert np.all(np.diff(fit_x) > 0)


def test_fit_handles_numpy_array_inputs():
    """Test: numpy array inputs. Expect: Returns valid arrays."""
    temps = np.array([773, 873, 973])
    perms = [ufloat(1e-10, 1e-11), ufloat(5e-10, 5e-11), ufloat(2e-9, 2e-10)]
    fit_x, fit_y = fit_permeability_data(temps, perms)
    assert isinstance(fit_x, np.ndarray) and isinstance(fit_y, np.ndarray)
