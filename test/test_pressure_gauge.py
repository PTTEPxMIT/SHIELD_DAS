"""Tests for pressure gauge classes in SHIELD DAS.

This module tests the PressureGauge base class and specific gauge implementations:
WGM701_Gauge, CVM211_Gauge, and Baratron626D_Gauge.
"""

import numpy as np
import pytest

from shield_das.pressure_gauge import (
    Baratron626D_Gauge,
    CVM211_Gauge,
    PressureGauge,
    WGM701_Gauge,
)

# =============================================================================
# Tests for PressureGauge base class
# =============================================================================


def test_pressure_gauge_initializes_with_name():
    """Test PressureGauge stores name on initialization."""
    gauge = PressureGauge("Test Gauge", 5, "upstream")
    assert gauge.name == "Test Gauge"


def test_pressure_gauge_initializes_with_ain_channel():
    """Test PressureGauge stores AIN channel on initialization."""
    gauge = PressureGauge("Test Gauge", 5, "upstream")
    assert gauge.ain_channel == 5


def test_pressure_gauge_initializes_with_gauge_location():
    """Test PressureGauge stores gauge location on initialization."""
    gauge = PressureGauge("Test Gauge", 5, "upstream")
    assert gauge.gauge_location == "upstream"


def test_pressure_gauge_initializes_voltage_data_empty():
    """Test PressureGauge initializes with empty voltage_data list."""
    gauge = PressureGauge("Test Gauge", 5, "upstream")
    assert gauge.voltage_data == []


def test_pressure_gauge_gauge_location_accepts_upstream():
    """Test gauge_location setter accepts 'upstream'."""
    gauge = PressureGauge("Test", 1, "upstream")
    assert gauge.gauge_location == "upstream"


def test_pressure_gauge_gauge_location_accepts_downstream():
    """Test gauge_location setter accepts 'downstream'."""
    gauge = PressureGauge("Test", 1, "downstream")
    assert gauge.gauge_location == "downstream"


def test_pressure_gauge_gauge_location_rejects_invalid_value():
    """Test gauge_location setter rejects invalid values."""
    with pytest.raises(ValueError, match="must be 'upstream' or 'downstream'"):
        PressureGauge("Test", 1, "invalid")


def test_pressure_gauge_record_voltage_in_test_mode_appends():
    """Test record_ain_channel_voltage appends to voltage_data in test mode."""
    gauge = PressureGauge("Test", 5, "upstream")
    initial_length = len(gauge.voltage_data)

    gauge.record_ain_channel_voltage(labjack=None)

    assert len(gauge.voltage_data) == initial_length + 1


def test_pressure_gauge_record_voltage_in_test_mode_returns_number():
    """Test record_ain_channel_voltage returns numeric value in test mode."""
    gauge = PressureGauge("Test", 5, "upstream")
    gauge.record_ain_channel_voltage(labjack=None)

    assert isinstance(gauge.voltage_data[-1], int | float)


def test_pressure_gauge_record_voltage_in_test_mode_in_range():
    """Test record_ain_channel_voltage produces values in 0-10V range."""
    gauge = PressureGauge("Test", 5, "upstream")
    gauge.record_ain_channel_voltage(labjack=None)

    voltage = gauge.voltage_data[-1]
    assert 0 <= voltage <= 10


def test_pressure_gauge_multiple_recordings_accumulate():
    """Test multiple voltage recordings accumulate in voltage_data."""
    gauge = PressureGauge("Test", 5, "upstream")

    gauge.record_ain_channel_voltage(labjack=None)
    gauge.record_ain_channel_voltage(labjack=None)
    gauge.record_ain_channel_voltage(labjack=None)

    assert len(gauge.voltage_data) == 3


# =============================================================================
# Tests for WGM701_Gauge
# =============================================================================


def test_wgm701_initializes_with_default_name():
    """Test WGM701_Gauge initializes with default name."""
    gauge = WGM701_Gauge()
    assert gauge.name == "WGM701"


def test_wgm701_initializes_with_default_ain_channel():
    """Test WGM701_Gauge initializes with default AIN channel 10."""
    gauge = WGM701_Gauge()
    assert gauge.ain_channel == 10


def test_wgm701_initializes_with_default_location_downstream():
    """Test WGM701_Gauge initializes with default location downstream."""
    gauge = WGM701_Gauge()
    assert gauge.gauge_location == "downstream"


def test_wgm701_initializes_with_custom_name():
    """Test WGM701_Gauge accepts custom name."""
    gauge = WGM701_Gauge(name="Custom WGM")
    assert gauge.name == "Custom WGM"


def test_wgm701_initializes_with_custom_ain_channel():
    """Test WGM701_Gauge accepts custom AIN channel."""
    gauge = WGM701_Gauge(ain_channel=12)
    assert gauge.ain_channel == 12


def test_wgm701_initializes_with_custom_location():
    """Test WGM701_Gauge accepts custom location."""
    gauge = WGM701_Gauge(gauge_location="upstream")
    assert gauge.gauge_location == "upstream"


def test_wgm701_voltage_to_pressure_converts_correctly():
    """Test WGM701 voltage_to_pressure formula."""
    gauge = WGM701_Gauge()
    voltage = np.array([5.5])

    pressure = gauge.voltage_to_pressure(voltage)

    assert np.isclose(pressure[0], 1.0, atol=0.01)


def test_wgm701_voltage_to_pressure_at_5_volts():
    """Test WGM701 converts 5V to 0.1 Torr."""
    gauge = WGM701_Gauge()
    voltage = np.array([5.0])

    pressure = gauge.voltage_to_pressure(voltage)

    assert np.isclose(pressure[0], 0.1, atol=0.01)


def test_wgm701_voltage_to_pressure_at_6_volts():
    """Test WGM701 converts 6V to 10 Torr."""
    gauge = WGM701_Gauge()
    voltage = np.array([6.0])

    pressure = gauge.voltage_to_pressure(voltage)

    assert np.isclose(pressure[0], 10.0, atol=0.1)


def test_wgm701_voltage_to_pressure_sets_small_values_to_zero():
    """Test WGM701 sets very small pressure values to zero."""
    gauge = WGM701_Gauge()
    voltage = np.array([0.0])

    pressure = gauge.voltage_to_pressure(voltage)

    assert pressure[0] == 0.0


def test_wgm701_voltage_to_pressure_caps_at_760_torr():
    """Test WGM701 caps pressure at 760 Torr."""
    gauge = WGM701_Gauge()
    voltage = np.array([10.0])

    pressure = gauge.voltage_to_pressure(voltage)

    assert pressure[0] <= 760.0


def test_wgm701_voltage_to_pressure_handles_array_input():
    """Test WGM701 voltage_to_pressure handles array input."""
    gauge = WGM701_Gauge()
    voltages = np.array([5.0, 5.5, 6.0])

    pressures = gauge.voltage_to_pressure(voltages)

    assert len(pressures) == 3


def test_wgm701_calculate_error_returns_array():
    """Test WGM701 calculate_error returns numpy array."""
    gauge = WGM701_Gauge()
    pressure = 100.0

    error = gauge.calculate_error(pressure)

    assert isinstance(error, np.ndarray | float)


def test_wgm701_calculate_error_default_is_half_pressure():
    """Test WGM701 default error is 0.5 * pressure for mid-range."""
    gauge = WGM701_Gauge()
    pressure = 100.0

    error = gauge.calculate_error(pressure)

    assert np.isclose(error, 50.0, atol=0.1)


def test_wgm701_calculate_error_low_range_30_percent():
    """Test WGM701 error is 30% for 7.6e-9 < p < 7.6e-3."""
    gauge = WGM701_Gauge()
    pressure = 1e-4

    error = gauge.calculate_error(pressure)

    assert np.isclose(error, 3e-5, atol=1e-6)


def test_wgm701_calculate_error_mid_range_15_percent():
    """Test WGM701 error is 15% for 7.6e-3 < p < 75."""
    gauge = WGM701_Gauge()
    pressure = 10.0

    error = gauge.calculate_error(pressure)

    assert np.isclose(error, 1.5, atol=0.01)


# =============================================================================
# Tests for CVM211_Gauge
# =============================================================================


def test_cvm211_initializes_with_default_name():
    """Test CVM211_Gauge initializes with default name."""
    gauge = CVM211_Gauge()
    assert gauge.name == "CVM211"


def test_cvm211_initializes_with_default_ain_channel():
    """Test CVM211_Gauge initializes with default AIN channel 8."""
    gauge = CVM211_Gauge()
    assert gauge.ain_channel == 8


def test_cvm211_initializes_with_default_location_upstream():
    """Test CVM211_Gauge initializes with default location upstream."""
    gauge = CVM211_Gauge()
    assert gauge.gauge_location == "upstream"


def test_cvm211_initializes_with_custom_name():
    """Test CVM211_Gauge accepts custom name."""
    gauge = CVM211_Gauge(name="Custom CVM")
    assert gauge.name == "Custom CVM"


def test_cvm211_initializes_with_custom_ain_channel():
    """Test CVM211_Gauge accepts custom AIN channel."""
    gauge = CVM211_Gauge(ain_channel=9)
    assert gauge.ain_channel == 9


def test_cvm211_initializes_with_custom_location():
    """Test CVM211_Gauge accepts custom location."""
    gauge = CVM211_Gauge(gauge_location="downstream")
    assert gauge.gauge_location == "downstream"


def test_cvm211_voltage_to_pressure_converts_correctly():
    """Test CVM211 voltage_to_pressure formula."""
    gauge = CVM211_Gauge()
    voltage = np.array([5.0])

    pressure = gauge.voltage_to_pressure(voltage)

    assert np.isclose(pressure[0], 1.0, atol=0.01)


def test_cvm211_voltage_to_pressure_at_4_volts():
    """Test CVM211 converts 4V to 0.1 Torr."""
    gauge = CVM211_Gauge()
    voltage = np.array([4.0])

    pressure = gauge.voltage_to_pressure(voltage)

    assert np.isclose(pressure[0], 0.1, atol=0.01)


def test_cvm211_voltage_to_pressure_at_6_volts():
    """Test CVM211 converts 6V to 10 Torr."""
    gauge = CVM211_Gauge()
    voltage = np.array([6.0])

    pressure = gauge.voltage_to_pressure(voltage)

    assert np.isclose(pressure[0], 10.0, atol=0.1)


def test_cvm211_voltage_to_pressure_sets_small_values_to_zero():
    """Test CVM211 sets very small pressure values to zero."""
    gauge = CVM211_Gauge()
    voltage = np.array([0.0])

    pressure = gauge.voltage_to_pressure(voltage)

    assert pressure[0] == 0.0


def test_cvm211_voltage_to_pressure_caps_at_1000_torr():
    """Test CVM211 caps pressure at 1000 Torr."""
    gauge = CVM211_Gauge()
    voltage = np.array([10.0])

    pressure = gauge.voltage_to_pressure(voltage)

    assert pressure[0] <= 1000.0


def test_cvm211_voltage_to_pressure_handles_array_input():
    """Test CVM211 voltage_to_pressure handles array input."""
    gauge = CVM211_Gauge()
    voltages = np.array([4.0, 5.0, 6.0])

    pressures = gauge.voltage_to_pressure(voltages)

    assert len(pressures) == 3


def test_cvm211_calculate_error_returns_array():
    """Test CVM211 calculate_error returns numpy array."""
    gauge = CVM211_Gauge()
    pressure = 100.0

    error = gauge.calculate_error(pressure)

    assert isinstance(error, np.ndarray | float)


def test_cvm211_calculate_error_default_is_2_5_percent():
    """Test CVM211 default error is 2.5% of pressure."""
    gauge = CVM211_Gauge()
    pressure = 500.0

    error = gauge.calculate_error(pressure)

    assert np.isclose(error, 12.5, atol=0.1)


def test_cvm211_calculate_error_very_low_range_fixed():
    """Test CVM211 error is fixed 0.1e-3 for 1e-4 < p < 1e-3."""
    gauge = CVM211_Gauge()
    pressure = 5e-4

    error = gauge.calculate_error(pressure)

    assert np.isclose(error, 0.1e-3, atol=1e-5)


def test_cvm211_calculate_error_mid_range_10_percent():
    """Test CVM211 error is 10% for 1e-3 < p < 400."""
    gauge = CVM211_Gauge()
    pressure = 10.0

    error = gauge.calculate_error(pressure)

    assert np.isclose(error, 1.0, atol=0.01)


# =============================================================================
# Tests for Baratron626D_Gauge
# =============================================================================


def test_baratron_initializes_with_ain_channel():
    """Test Baratron626D_Gauge stores AIN channel on initialization."""
    gauge = Baratron626D_Gauge(ain_channel=6, full_scale_Torr=1000.0)
    assert gauge.ain_channel == 6


def test_baratron_initializes_with_default_name():
    """Test Baratron626D_Gauge initializes with default name."""
    gauge = Baratron626D_Gauge(ain_channel=6, full_scale_Torr=1000.0)
    assert gauge.name == "Baratron626D"


def test_baratron_initializes_with_default_location_downstream():
    """Test Baratron626D_Gauge initializes with default location downstream."""
    gauge = Baratron626D_Gauge(ain_channel=6, full_scale_Torr=1000.0)
    assert gauge.gauge_location == "downstream"


def test_baratron_initializes_with_custom_name():
    """Test Baratron626D_Gauge accepts custom name."""
    gauge = Baratron626D_Gauge(
        ain_channel=6, name="Custom Baratron", full_scale_Torr=1.0
    )
    assert gauge.name == "Custom Baratron"


def test_baratron_initializes_with_custom_location():
    """Test Baratron626D_Gauge accepts custom location."""
    gauge = Baratron626D_Gauge(
        ain_channel=6, gauge_location="upstream", full_scale_Torr=1.0
    )
    assert gauge.gauge_location == "upstream"


def test_baratron_stores_full_scale_1000():
    """Test Baratron626D_Gauge stores full_scale_Torr of 1000."""
    gauge = Baratron626D_Gauge(ain_channel=6, full_scale_Torr=1000.0)
    assert gauge.full_scale_Torr == 1000.0


def test_baratron_stores_full_scale_1():
    """Test Baratron626D_Gauge stores full_scale_Torr of 1."""
    gauge = Baratron626D_Gauge(ain_channel=6, full_scale_Torr=1.0)
    assert gauge.full_scale_Torr == 1.0


def test_baratron_full_scale_rejects_invalid_value():
    """Test Baratron626D_Gauge rejects invalid full_scale_Torr values."""
    with pytest.raises(ValueError, match="must be either 1 or 1000"):
        Baratron626D_Gauge(ain_channel=6, full_scale_Torr=10.0)


def test_baratron_full_scale_requires_value():
    """Test Baratron626D_Gauge raises error when full_scale_Torr is None."""
    gauge = Baratron626D_Gauge(ain_channel=6, full_scale_Torr=1000.0)
    gauge._full_scale_Torr = None

    with pytest.raises(ValueError, match="must be set"):
        _ = gauge.full_scale_Torr


def test_baratron_voltage_to_pressure_1000_scale():
    """Test Baratron voltage_to_pressure with 1000 Torr scale."""
    gauge = Baratron626D_Gauge(ain_channel=6, full_scale_Torr=1000.0)
    voltage = np.array([5.0])

    pressure = gauge.voltage_to_pressure(voltage)

    assert np.isclose(pressure[0], 500.0, atol=0.1)


def test_baratron_voltage_to_pressure_1_scale():
    """Test Baratron voltage_to_pressure with 1 Torr scale."""
    gauge = Baratron626D_Gauge(ain_channel=6, full_scale_Torr=1.0)
    voltage = np.array([5.0])

    pressure = gauge.voltage_to_pressure(voltage)

    assert np.isclose(pressure[0], 0.5, atol=0.001)


def test_baratron_voltage_to_pressure_1000_sets_small_values_to_zero():
    """Test Baratron with 1000 scale sets values below 0.5 to zero."""
    gauge = Baratron626D_Gauge(ain_channel=6, full_scale_Torr=1000.0)
    voltage = np.array([0.001])

    pressure = gauge.voltage_to_pressure(voltage)

    assert pressure[0] == 0.0


def test_baratron_voltage_to_pressure_1_sets_small_values_to_zero():
    """Test Baratron with 1 scale sets values below 0.0005 to zero."""
    gauge = Baratron626D_Gauge(ain_channel=6, full_scale_Torr=1.0)
    voltage = np.array([0.0001])

    pressure = gauge.voltage_to_pressure(voltage)

    assert pressure[0] == 0.0


def test_baratron_voltage_to_pressure_1000_caps_at_1000():
    """Test Baratron with 1000 scale caps at 1000 Torr."""
    gauge = Baratron626D_Gauge(ain_channel=6, full_scale_Torr=1000.0)
    voltage = np.array([15.0])

    pressure = gauge.voltage_to_pressure(voltage)

    assert pressure[0] == 1000.0


def test_baratron_voltage_to_pressure_1_caps_at_1():
    """Test Baratron with 1 scale caps at 1 Torr."""
    gauge = Baratron626D_Gauge(ain_channel=6, full_scale_Torr=1.0)
    voltage = np.array([15.0])

    pressure = gauge.voltage_to_pressure(voltage)

    assert pressure[0] == 1.0


def test_baratron_voltage_to_pressure_handles_array_input():
    """Test Baratron voltage_to_pressure handles array input."""
    gauge = Baratron626D_Gauge(ain_channel=6, full_scale_Torr=1000.0)
    voltages = np.array([1.0, 5.0, 9.0])

    pressures = gauge.voltage_to_pressure(voltages)

    assert len(pressures) == 3


def test_baratron_calculate_error_returns_array():
    """Test Baratron calculate_error returns numpy array."""
    gauge = Baratron626D_Gauge(ain_channel=6, full_scale_Torr=1000.0)
    pressure = 100.0

    error = gauge.calculate_error(pressure)

    assert isinstance(error, np.ndarray | float)


def test_baratron_calculate_error_default_is_0_5_percent():
    """Test Baratron default error is 0.5% of pressure for low values."""
    gauge = Baratron626D_Gauge(ain_channel=6, full_scale_Torr=1000.0)
    pressure = 0.5

    error = gauge.calculate_error(pressure)

    assert np.isclose(error, 0.0025, atol=0.0001)


def test_baratron_calculate_error_high_range_0_25_percent():
    """Test Baratron error is 0.25% for p > 1."""
    gauge = Baratron626D_Gauge(ain_channel=6, full_scale_Torr=1000.0)
    pressure = 100.0

    error = gauge.calculate_error(pressure)

    assert np.isclose(error, 0.25, atol=0.01)


# =============================================================================
# Parametrized tests for multiple gauge types
# =============================================================================


@pytest.mark.parametrize(
    "gauge_class,default_location",
    [
        (WGM701_Gauge, "downstream"),
        (CVM211_Gauge, "upstream"),
    ],
)
def test_gauge_default_locations(gauge_class, default_location):
    """Test each gauge class has correct default location.

    Args:
        gauge_class: Gauge class to test
        default_location: Expected default location
    """
    gauge = gauge_class()
    assert gauge.gauge_location == default_location


@pytest.mark.parametrize(
    "gauge_class,default_channel",
    [
        (WGM701_Gauge, 10),
        (CVM211_Gauge, 8),
    ],
)
def test_gauge_default_channels(gauge_class, default_channel):
    """Test each gauge class has correct default AIN channel.

    Args:
        gauge_class: Gauge class to test
        default_channel: Expected default AIN channel
    """
    gauge = gauge_class()
    assert gauge.ain_channel == default_channel


@pytest.mark.parametrize(
    "gauge_class,expected_name",
    [
        (WGM701_Gauge, "WGM701"),
        (CVM211_Gauge, "CVM211"),
    ],
)
def test_gauge_default_names(gauge_class, expected_name):
    """Test each gauge class has correct default name.

    Args:
        gauge_class: Gauge class to test
        expected_name: Expected default name
    """
    gauge = gauge_class()
    assert gauge.name == expected_name


@pytest.mark.parametrize(
    "location",
    ["upstream", "downstream"],
)
def test_all_gauges_accept_valid_locations(location):
    """Test all gauge classes accept valid location strings.

    Args:
        location: Location string to test
    """
    wgm = WGM701_Gauge(gauge_location=location)
    cvm = CVM211_Gauge(gauge_location=location)
    baratron = Baratron626D_Gauge(
        ain_channel=6, gauge_location=location, full_scale_Torr=1.0
    )

    assert wgm.gauge_location == location
    assert cvm.gauge_location == location
    assert baratron.gauge_location == location
