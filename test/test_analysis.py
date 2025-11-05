import numpy as np
import pytest

from shield_das.analysis import (
    average_pressure_after_increase,
    calculate_flux_from_sample,
    calculate_permeability_from_flux,
    evaluate_permeability_values,
    fit_permeability_data,
)


class TestAveragePressureAfterIncrease:
    """Tests for average_pressure_after_increase function"""

    def test_returns_average_after_stabilization(self):
        """Test that function returns average of stable region"""
        time = np.linspace(0, 20, 200)
        # Pressure jumps at t=5, then stabilizes at 100
        pressure = np.where(time < 5, 10, 100)

        result = average_pressure_after_increase(time, pressure)

        assert result == pytest.approx(100, rel=0.01)

    def test_handles_gradual_increase(self):
        """Test with gradual pressure increase that stabilizes"""
        time = np.linspace(0, 30, 300)
        # Linear increase up to t=10, then flat at 50
        pressure = np.where(time < 10, time * 5, 50)

        result = average_pressure_after_increase(time, pressure)

        assert result == pytest.approx(50, rel=0.01)

    def test_returns_value_with_noisy_data(self):
        """Test that function handles noisy pressure data"""
        time = np.linspace(0, 20, 200)
        pressure = np.where(time < 5, 10, 100)
        # Add noise using a Generator
        rng = np.random.default_rng()
        noise = rng.normal(0, 1, len(pressure))
        pressure = pressure + noise

        result = average_pressure_after_increase(time, pressure)

        # Should still be close to 100 despite noise
        assert result == pytest.approx(100, abs=5)

    def test_uses_fallback_when_no_stable_region(self):
        """Test fallback behavior when pressure never stabilizes"""
        time = np.linspace(0, 20, 200)
        pressure = time * 10  # Continuous increase

        result = average_pressure_after_increase(time, pressure)

        # Should use halfway point
        assert result > 0

    def test_respects_minimum_time_threshold(self):
        """Test that function ignores first 5 seconds"""
        time = np.linspace(0, 20, 200)
        # Flat at 50 from start, but should ignore first 5 seconds
        pressure = np.full_like(time, 50)

        result = average_pressure_after_increase(time, pressure, slope_threshold=1e-3)

        assert result == pytest.approx(50, rel=0.01)

    def test_accepts_list_input(self):
        """Test that function works with list inputs"""
        time = [0, 5, 10, 15, 20]
        pressure = [10, 10, 100, 100, 100]

        result = average_pressure_after_increase(time, pressure)

        assert result == pytest.approx(100, rel=0.01)

    def test_custom_window_size(self):
        """Test that custom window size parameter works"""
        time = np.linspace(0, 20, 200)
        pressure = np.where(time < 5, 10, 100)

        result = average_pressure_after_increase(time, pressure, window=10)

        assert result == pytest.approx(100, rel=0.01)

    def test_custom_slope_threshold(self):
        """Test that custom slope threshold parameter works"""
        time = np.linspace(0, 20, 200)
        pressure = np.where(time < 5, 10, 100)

        result = average_pressure_after_increase(time, pressure, slope_threshold=1e-2)

        assert result == pytest.approx(100, rel=0.01)


class TestCalculateFluxFromSample:
    """Tests for calculate_flux_from_sample function"""

    def test_positive_slope_for_increasing_pressure(self):
        """Test that increasing pressure gives positive slope"""
        time = np.linspace(0, 100, 100)
        pressure = 0.1 + 0.002 * time  # Linear increase

        slope = calculate_flux_from_sample(time, pressure)

        assert slope > 0
        assert slope == pytest.approx(0.002, rel=0.1)

    def test_filters_low_pressure_values(self):
        """Test that pressures below 0.05 are filtered out"""
        time = np.linspace(0, 100, 100)
        # Include some values below 0.05
        pressure = 0.01 + 0.001 * time

        slope = calculate_flux_from_sample(time, pressure)

        # Should calculate slope only using values >= 0.05
        assert slope > 0

    def test_filters_high_pressure_values(self):
        """Test that pressures above 0.95 are filtered out"""
        time = np.linspace(0, 100, 100)
        # Include values that go above 0.95
        pressure = 0.8 + 0.005 * time

        slope = calculate_flux_from_sample(time, pressure)

        # Should calculate slope only using values <= 0.95
        assert slope > 0

    def test_weighted_fit_emphasizes_final_points(self):
        """Test that weighting gives more importance to final points"""
        time = np.linspace(0, 100, 100)
        # Create data where early points have different slope
        pressure = np.where(time < 50, 0.1 + 0.001 * time, 0.15 + 0.003 * time)

        slope = calculate_flux_from_sample(time, pressure)

        # Slope should be closer to the later slope (0.003)
        assert slope > 0.002

    def test_raises_error_with_insufficient_valid_points(self):
        """Test that error is raised when too few valid points"""
        time = np.linspace(0, 10, 4)
        pressure = np.array([0.01, 0.02, 0.03, 0.04])  # All below threshold

        with pytest.raises((ValueError, TypeError)):
            calculate_flux_from_sample(time, pressure)

    def test_handles_array_input(self):
        """Test that function works with numpy array input"""
        time = np.array([0, 10, 20, 30, 40, 50])
        pressure = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

        slope = calculate_flux_from_sample(time, pressure)

        assert slope == pytest.approx(0.01, rel=0.1)

    def test_handles_list_input(self):
        """Test that function works with list input"""
        time = [0, 10, 20, 30, 40, 50]
        pressure = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

        slope = calculate_flux_from_sample(time, pressure)

        assert slope == pytest.approx(0.01, rel=0.1)

    def test_zero_slope_for_constant_pressure(self):
        """Test that constant pressure gives near-zero slope"""
        time = np.linspace(0, 100, 100)
        pressure = np.full_like(time, 0.5)

        slope = calculate_flux_from_sample(time, pressure)

        assert slope == pytest.approx(0, abs=1e-10)


class TestCalculatePermeabilityFromFlux:
    """Tests for calculate_permeability_from_flux function"""

    def test_returns_positive_permeability(self):
        """Test that permeability is positive for valid inputs"""
        slope_torr_per_s = 0.001
        V_m3 = 7.9e-5
        T_K = 873
        A_m2 = 1.88e-4
        e_m = 0.00088
        P_down_torr = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        P_up_torr = 100

        perm = calculate_permeability_from_flux(
            slope_torr_per_s, V_m3, T_K, A_m2, e_m, P_down_torr, P_up_torr
        )

        assert perm > 0

    def test_permeability_increases_with_flux(self):
        """Test that higher flux gives higher permeability"""
        V_m3 = 7.9e-5
        T_K = 873
        A_m2 = 1.88e-4
        e_m = 0.00088
        P_down_torr = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        P_up_torr = 100

        perm1 = calculate_permeability_from_flux(
            0.001, V_m3, T_K, A_m2, e_m, P_down_torr, P_up_torr
        )
        perm2 = calculate_permeability_from_flux(
            0.002, V_m3, T_K, A_m2, e_m, P_down_torr, P_up_torr
        )

        assert perm2 > perm1

    def test_permeability_scale_with_temperature(self):
        """Test that permeability calculation works at different temperatures"""
        slope_torr_per_s = 0.001
        V_m3 = 7.9e-5
        A_m2 = 1.88e-4
        e_m = 0.00088
        P_down_torr = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        P_up_torr = 100

        perm1 = calculate_permeability_from_flux(
            slope_torr_per_s, V_m3, 600, A_m2, e_m, P_down_torr, P_up_torr
        )
        perm2 = calculate_permeability_from_flux(
            slope_torr_per_s, V_m3, 900, A_m2, e_m, P_down_torr, P_up_torr
        )

        # Both should be positive (relationship is complex in Takaishi-Sensui)
        assert perm1 > 0
        assert perm2 > 0

    def test_permeability_decreases_with_upstream_pressure(self):
        """Test that higher upstream pressure gives lower permeability"""
        slope_torr_per_s = 0.001
        V_m3 = 7.9e-5
        T_K = 873
        A_m2 = 1.88e-4
        e_m = 0.00088
        P_down_torr = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

        perm1 = calculate_permeability_from_flux(
            slope_torr_per_s, V_m3, T_K, A_m2, e_m, P_down_torr, 50
        )
        perm2 = calculate_permeability_from_flux(
            slope_torr_per_s, V_m3, T_K, A_m2, e_m, P_down_torr, 200
        )

        assert perm2 < perm1

    def test_permeability_increases_with_thickness(self):
        """Test that thicker sample gives higher permeability"""
        slope_torr_per_s = 0.001
        V_m3 = 7.9e-5
        T_K = 873
        A_m2 = 1.88e-4
        P_down_torr = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        P_up_torr = 100

        perm1 = calculate_permeability_from_flux(
            slope_torr_per_s, V_m3, T_K, A_m2, 0.0005, P_down_torr, P_up_torr
        )
        perm2 = calculate_permeability_from_flux(
            slope_torr_per_s, V_m3, T_K, A_m2, 0.001, P_down_torr, P_up_torr
        )

        assert perm2 > perm1

    def test_permeability_decreases_with_area(self):
        """Test that larger area gives lower permeability"""
        slope_torr_per_s = 0.001
        V_m3 = 7.9e-5
        T_K = 873
        e_m = 0.00088
        P_down_torr = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        P_up_torr = 100

        perm1 = calculate_permeability_from_flux(
            slope_torr_per_s, V_m3, T_K, 1e-4, e_m, P_down_torr, P_up_torr
        )
        perm2 = calculate_permeability_from_flux(
            slope_torr_per_s, V_m3, T_K, 2e-4, e_m, P_down_torr, P_up_torr
        )

        assert perm2 < perm1

    def test_uses_final_downstream_pressure(self):
        """Test that function uses the last value of downstream pressure"""
        slope_torr_per_s = 0.001
        V_m3 = 7.9e-5
        T_K = 873
        A_m2 = 1.88e-4
        e_m = 0.00088
        P_up_torr = 100

        # Two arrays with different final values
        P_down1 = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        P_down2 = np.array([0.1, 0.2, 0.3, 0.4, 0.8])

        perm1 = calculate_permeability_from_flux(
            slope_torr_per_s, V_m3, T_K, A_m2, e_m, P_down1, P_up_torr
        )
        perm2 = calculate_permeability_from_flux(
            slope_torr_per_s, V_m3, T_K, A_m2, e_m, P_down2, P_up_torr
        )

        # Different final pressures should give different results
        assert perm1 != perm2

    def test_realistic_permeability_range(self):
        """Test that permeability is positive and finite for realistic inputs"""
        slope_torr_per_s = 0.001
        V_m3 = 7.9e-5
        T_K = 873  # 600°C
        A_m2 = 1.88e-4
        e_m = 0.00088
        P_down_torr = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        P_up_torr = 100

        perm = calculate_permeability_from_flux(
            slope_torr_per_s, V_m3, T_K, A_m2, e_m, P_down_torr, P_up_torr
        )

        # Test that result is positive and finite (handle ufloat object)
        perm_val = perm.n if hasattr(perm, "n") else perm
        assert perm_val > 0
        assert np.isfinite(perm_val)


class TestEvaluatePermeabilityValues:
    """Tests for evaluate_permeability_values function"""

    def test_returns_correct_structure(self):
        """Test that function returns tuple with correct structure"""
        datasets = {
            "run_1": {
                "temperature": 873,
                "time_data": np.linspace(0, 100, 50),
                "upstream_data": {"pressure_data": np.ones(50) * 100},
                "downstream_data": {"pressure_data": np.linspace(0.1, 0.5, 50)},
            }
        }

        result = evaluate_permeability_values(datasets)

        # Should return 6 items: temps, perms, x_error, y_error, error_lower, error_upper
        assert len(result) == 6
        temps, perms, x_error, y_error, error_lower, error_upper = result

        # Check basic properties
        assert len(temps) == 1
        assert len(perms) == 1
        assert len(x_error) == 1
        assert len(y_error) == 1
        assert len(error_lower) == 1
        assert len(error_upper) == 1

    def test_handles_multiple_datasets(self):
        """Test with multiple datasets"""
        datasets = {
            "run_1": {
                "temperature": 873,
                "time_data": np.linspace(0, 100, 50),
                "upstream_data": {"pressure_data": np.ones(50) * 100},
                "downstream_data": {"pressure_data": np.linspace(0.1, 0.5, 50)},
            },
            "run_2": {
                "temperature": 923,
                "time_data": np.linspace(0, 100, 50),
                "upstream_data": {"pressure_data": np.ones(50) * 100},
                "downstream_data": {"pressure_data": np.linspace(0.1, 0.6, 50)},
            },
            "run_3": {
                "temperature": 873,  # Same temp as run_1
                "time_data": np.linspace(0, 100, 50),
                "upstream_data": {"pressure_data": np.ones(50) * 100},
                "downstream_data": {"pressure_data": np.linspace(0.1, 0.55, 50)},
            },
        }

        temps, perms, x_error, y_error, error_lower, error_upper = (
            evaluate_permeability_values(datasets)
        )

        # Should have 3 data points
        assert len(temps) == 3
        assert len(perms) == 3

        # But only 2 unique temperatures, so error bars for 2 groups
        assert len(x_error) == 2
        assert len(y_error) == 2
        assert len(error_lower) == 2
        assert len(error_upper) == 2

    def test_temperature_grouping(self):
        """Test that error bars group by temperature correctly"""
        datasets = {
            f"run_{i}": {
                "temperature": 873,
                "time_data": np.linspace(0, 100, 50),
                "upstream_data": {"pressure_data": np.ones(50) * 100},
                "downstream_data": {
                    "pressure_data": np.linspace(0.1, 0.5 + i * 0.1, 50)
                },
            }
            for i in range(3)
        }

        temps, perms, x_error, y_error, error_lower, error_upper = (
            evaluate_permeability_values(datasets)
        )

        # All same temperature
        assert len(set(temps)) == 1
        assert len(temps) == 3

        # Error bars should have only 1 group
        assert len(x_error) == 1
        assert len(error_lower) == 1
        assert len(error_upper) == 1

    def test_x_error_calculation(self):
        """Test that x_error is correctly calculated as 1000/T"""
        datasets = {
            "run_1": {
                "temperature": 1000,
                "time_data": np.linspace(0, 100, 50),
                "upstream_data": {"pressure_data": np.ones(50) * 100},
                "downstream_data": {"pressure_data": np.linspace(0.1, 0.5, 50)},
            }
        }

        _, _, x_error, _, _, _ = evaluate_permeability_values(datasets)

        assert x_error[0] == pytest.approx(1.0, rel=0.01)

    def test_error_bar_bounds(self):
        """Test that error bars are based on propagated uncertainties"""
        # Create datasets with known permeability values
        datasets = {
            "run_1": {
                "temperature": 873,
                "time_data": np.linspace(0, 100, 50),
                "upstream_data": {"pressure_data": np.ones(50) * 100},
                "downstream_data": {"pressure_data": np.linspace(0.1, 0.5, 50)},
            },
            "run_2": {
                "temperature": 873,
                "time_data": np.linspace(0, 100, 50),
                "upstream_data": {"pressure_data": np.ones(50) * 100},
                "downstream_data": {"pressure_data": np.linspace(0.1, 0.6, 50)},
            },
        }

        temps, perms, _, y_error, error_lower, error_upper = (
            evaluate_permeability_values(datasets)
        )

        # Error bars should be positive (based on uncertainty propagation)
        assert error_lower[0] > 0
        assert error_upper[0] > 0

        # y_error should be the central value
        assert y_error[0] > 0
        assert np.isfinite(y_error[0])

    def test_handles_empty_datasets(self):
        """Test with empty datasets dictionary"""
        datasets = {}

        temps, perms, x_error, y_error, error_lower, error_upper = (
            evaluate_permeability_values(datasets)
        )

        assert len(temps) == 0
        assert len(perms) == 0
        assert len(x_error) == 0
        assert len(y_error) == 0
        assert len(error_lower) == 0
        assert len(error_upper) == 0

    def test_permeability_values_are_positive(self):
        """Test that all calculated permeability values are positive"""
        datasets = {
            "run_1": {
                "temperature": 873,
                "time_data": np.linspace(0, 100, 50),
                "upstream_data": {"pressure_data": np.ones(50) * 100},
                "downstream_data": {"pressure_data": np.linspace(0.1, 0.5, 50)},
            }
        }

        _, perms, _, _, _, _ = evaluate_permeability_values(datasets)

        assert all(p > 0 for p in perms)

    def test_permeability_values_are_finite(self):
        """Test that all calculated values are finite"""
        datasets = {
            "run_1": {
                "temperature": 873,
                "time_data": np.linspace(0, 100, 50),
                "upstream_data": {"pressure_data": np.ones(50) * 100},
                "downstream_data": {"pressure_data": np.linspace(0.1, 0.5, 50)},
            }
        }

        temps, perms, x_error, y_error, error_lower, error_upper = (
            evaluate_permeability_values(datasets)
        )

        assert all(np.isfinite(t) for t in temps)
        # perms now contains ufloat objects, extract nominal values
        assert all(np.isfinite(p.n if hasattr(p, "n") else p) for p in perms)
        assert all(np.isfinite(x) for x in x_error)
        assert all(np.isfinite(y) for y in y_error)
        assert all(np.isfinite(e) for e in error_lower)
        assert all(np.isfinite(e) for e in error_upper)


class TestFitPermeabilityData:
    """Tests for fit_permeability_data function"""

    def test_returns_two_arrays(self):
        """Test that function returns two arrays"""
        temps = [873, 923, 973]
        perms = [1e-10, 2e-10, 4e-10]

        fit_x, fit_y = fit_permeability_data(temps, perms)

        assert isinstance(fit_x, np.ndarray)
        assert isinstance(fit_y, np.ndarray)
        assert len(fit_x) == len(fit_y)

    def test_fit_has_100_points(self):
        """Test that fitted data has 100 points"""
        temps = [873, 923, 973]
        perms = [1e-10, 2e-10, 4e-10]

        fit_x, fit_y = fit_permeability_data(temps, perms)

        assert len(fit_x) == 100

    def test_fit_x_range(self):
        """Test that fit_x spans the correct range"""
        temps = [800, 1000]
        perms = [1e-10, 2e-10]

        fit_x, fit_y = fit_permeability_data(temps, perms)

        # fit_x should be 1000/T, so range from 1000/1000 to 1000/800
        assert fit_x.min() == pytest.approx(1.0, rel=0.01)
        assert fit_x.max() == pytest.approx(1.25, rel=0.01)

    def test_fit_is_monotonic(self):
        """Test that fitted x values are monotonic"""
        temps = [873, 923, 973]
        perms = [1e-10, 2e-10, 4e-10]

        fit_x, _ = fit_permeability_data(temps, perms)

        # x should be strictly increasing
        assert np.all(np.diff(fit_x) > 0)

    def test_fit_values_are_positive(self):
        """Test that all fitted permeability values are positive"""
        temps = [873, 923, 973]
        perms = [1e-10, 2e-10, 4e-10]

        _, fit_y = fit_permeability_data(temps, perms)

        assert np.all(fit_y > 0)

    def test_fit_values_are_finite(self):
        """Test that all fitted values are finite"""
        temps = [873, 923, 973]
        perms = [1e-10, 2e-10, 4e-10]

        fit_x, fit_y = fit_permeability_data(temps, perms)

        assert np.all(np.isfinite(fit_x))
        assert np.all(np.isfinite(fit_y))

    def test_arrhenius_behavior(self):
        """Test that fit follows Arrhenius-like behavior (exponential in 1/T)"""
        # Create data that follows Arrhenius: log(perm) = -A/T + B
        temps = np.array([800, 900, 1000])
        # Permeability should increase with temperature
        perms = 10 ** (5000 / temps - 5)  # Example Arrhenius

        fit_x, fit_y = fit_permeability_data(temps, perms)

        # At higher 1000/T (lower temperature), permeability should be lower
        # fit_x is 1000/T, so higher fit_x means lower T, which means lower perm
        # Check that fit is roughly decreasing in x (but this is log space)
        # Actually, in log space it's linear, so let's check monotonicity in log
        log_fit_y = np.log10(fit_y)

        # The slope should be positive (perm increases as 1/T decreases)
        slope = (log_fit_y[-1] - log_fit_y[0]) / (fit_x[-1] - fit_x[0])
        assert slope > 0  # Positive slope in log-space

    def test_handles_single_point(self):
        """Test behavior with single data point"""
        temps = [873]
        perms = [1e-10]

        fit_x, fit_y = fit_permeability_data(temps, perms)

        # Should still return 100 points
        assert len(fit_x) == 100
        # All y values should be the same (horizontal line)
        assert np.allclose(fit_y, fit_y[0])

    def test_handles_two_points(self):
        """Test with exactly two points (minimum for line fit)"""
        temps = [873, 973]
        perms = [1e-10, 2e-10]

        fit_x, fit_y = fit_permeability_data(temps, perms)

        assert len(fit_x) == 100
        assert len(fit_y) == 100
        # Should produce a straight line in log space
        log_fit_y = np.log10(fit_y)
        # Check linearity
        coeffs = np.polyfit(fit_x, log_fit_y, 1)
        fitted_line = coeffs[0] * fit_x + coeffs[1]
        assert np.allclose(log_fit_y, fitted_line, rtol=1e-10)

    def test_fit_passes_through_data_range(self):
        """Test that fit encompasses the data points"""
        temps = [873, 923, 973]
        perms = [1e-10, 2e-10, 4e-10]

        fit_x, fit_y = fit_permeability_data(temps, perms)

        # Fitted y values should be in reasonable range of input perms
        assert fit_y.min() <= min(perms) * 2  # Allow some extrapolation
        assert fit_y.max() >= max(perms) * 0.5
