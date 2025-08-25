import numpy as np
import numpy.typing as npt


def voltage_to_pressure(voltage: npt.NDArray, full_scale_torr: float) -> npt.NDArray:
    """
    Converts the voltage reading from a Instrutech WGM701 pressure gauge
    to pressure in Torr.

    Args:
        voltage: The voltage reading from the gauge
        full_scale_torr: The full scale of the gauge in Torr (1 or 1000)

    Returns:
        float: The pressure in Torr
    """
    # Convert voltage to pressure in Torr
    pressure = voltage * (full_scale_torr / 10.0)

    # Apply valid range based on full scale
    if full_scale_torr == 1000:
        pressure = np.where(pressure < 0.5, 0, pressure)
        pressure = np.clip(pressure, 0, 1000)
    elif full_scale_torr == 1:
        pressure = np.where(pressure < 0.0005, 0, pressure)
        pressure = np.clip(pressure, 0, 1)

    return pressure


def calculate_error(pressure_value: float) -> float:
    """
    Calculate the error in the pressure reading.

    Args:
        pressure_value: The pressure reading in Torr

    Returns:
        float: The error in the pressure reading
    """

    p = np.asarray(pressure_value, dtype=float)

    # Initialize with default error (0.5% of pressure)
    error = p * 0.005

    # Apply conditions with np.where
    error = np.where(p > 1, p * 0.0025, error)

    return error
