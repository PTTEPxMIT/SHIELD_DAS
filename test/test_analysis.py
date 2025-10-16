import numpy as np
import pytest

from shield_das.analysis import (
    average_pressure_after_increase,
    calculate_flux_from_sample,
    calculate_permeability_from_flux,
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

        # Test that result is positive and finite
        assert perm > 0
        assert np.isfinite(perm)
