import numpy as np


class Thermocouple:
    """
    Class to handle Type K thermocouple data acquisition and conversion.

    This class reads the thermocouple voltage and converts it to temperature
    using NIST ITS-90 polynomial coefficients.

    Args:
        name: Name of the thermocouple

    Attributes:
        name: Name of the thermocouple
        voltage_data: List to store voltage readings in millivolts
        local_temperature_data: List to store local temperature readings for
            cold junction compensation in degrees Celsius
    """

    name: str

    voltage_data: list[float]
    local_temperature_data: list[float]

    def __init__(
        self,
        name: str = "type K thermocouple",
    ):
        self.name = name

        # Data storage
        self.voltage_data = []
        self.local_temperature_data = []

    def record_ain_channel_voltage(
        self,
        labjack,  # Remove type hint to avoid import issues
        resolution_index: int | None = 8,
        gain_index: int | None = 2,
    ) -> float:
        """
        Read temperature from a Type K thermocouple connected to a LabJack U6 using
        differential input mode.

        This function reads the cold junction temperature from the device's internal
        sensor, reads the differential voltage from the thermocouple input channels,
        applies cold junction compensation, and converts the resulting voltage to
        temperature.

        args:
            labjack: An instance of the LabJack U6 device.
            pos_channel: The positive analog input channel number connected to the
                thermocouple positive lead (default 0).
            gain_index: The LabJack gain setting index to set input voltage range and
                resolution (default 3, ±0.1 V range).

        returns:
            float: The calculated temperature in degrees Celsius.
        """
        if labjack is None:
            rng = np.random.default_rng()
            ain_channel_voltage = rng.uniform(0.1, 0.2)
            local_temperature = rng.uniform(20, 25)
        else:
            ain_channel_voltage = labjack.getAIN(
                positiveChannel=0,
                resolutionIndex=resolution_index,
                gainIndex=gain_index,
                differential=True,
            )
            # convert volts to millivolts
            ain_channel_voltage *= 1000
            ain_channel_voltage *= -1

            # get cold junction temperature in Celsius
            local_temperature = labjack.getTemperature() - 273.15 + 2.5

        self.local_temperature_data.append(local_temperature)
        self.voltage_data.append(ain_channel_voltage)


def evaluate_poly(
    coeffs: list[float] | tuple[float], x: float | np.ndarray
) -> float | np.ndarray:
    """ "
    Evaluate a polynomial at x given the list of coefficients.

    The polynomial is:
        P(x) = a0 + a1*x + a2*x^2 + ... + an*x^n
    where coeffs = [a0, a1, ..., an]

    args:
        coeffs:Polynomial coefficients ordered by ascending power.
        x: The value(s) at which to evaluate the polynomial (scalar or array).

    returns;
        float or ndarray: The evaluated polynomial result(s).
    """
    if not coeffs:
        # Return 0 for empty coefficient list (matches expected behavior)
        return 0.0 if isinstance(x, (int, float)) else np.zeros_like(x, dtype=float)

    return sum(a * x**i for i, a in enumerate(coeffs))


def volts_to_temp_constants(mv: float | np.ndarray) -> tuple[float, ...]:
    """
    Select the appropriate NIST ITS-90 polynomial coefficients for converting
    Type K thermocouple voltage (in millivolts) to temperature (°C).

    The valid voltage range is -5.891 mV to 54.886 mV.

    Note: When mv is an array, this function returns coefficients for a single
    voltage range. For array inputs, use mv_to_temp_c which handles mixed ranges.

    args:
        mv: Thermocouple voltage in millivolts (scalar or array).

    returns:
        tuple of float: Polynomial coefficients for the voltage-to-temperature
        conversion.

    raises:
        ValueError: If the input voltage is out of the valid range.
    """
    # For arrays, use the first element to determine range
    # (mv_to_temp_c handles mixed ranges properly)
    voltage = np.asarray(mv).flatten()[0] if isinstance(mv, np.ndarray) else mv

    # Use a small tolerance for floating-point comparison
    if voltage < -5.892 or voltage > 54.887:
        raise ValueError("Voltage out of valid Type K range (-5.891 to 54.886 mV).")
    if voltage < 0:
        # Range: -5.891 mV to 0 mV
        return (
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
    elif voltage < 20.644:
        # Range: 0 mV to 20.644 mV
        return (
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
    else:
        # Range: 20.644 mV to 54.886 mV
        return (
            -1.318058e2,
            4.830222e1,
            -1.646031e0,
            5.464731e-2,
            -9.650715e-4,
            8.802193e-6,
            -3.11081e-8,
        )


def temp_to_volts_constants(
    temp_c: float | np.ndarray,
) -> tuple[tuple[float, ...], tuple[float, float, float] | None]:
    """
    Select the appropriate NIST ITS-90 polynomial coefficients for converting
    temperature (°C) to Type K thermocouple voltage (in millivolts).

    Valid temperature range is -270°C to 1372°C.

    Note: When temp_c is an array, this function returns coefficients for a
    single temperature range. For array inputs, use temp_c_to_mv which handles
    mixed ranges.

    args:
        temp_c: Temperature in degrees Celsius (scalar or array).

    returns:
        Tuple containing:
            - tuple of float: Polynomial coefficients for
              temperature-to-voltage conversion.
            - tuple of three floats or None: Extended exponential term
              coefficients for temp >= 0°C, else None.

    raises:
        ValueError: If the input temperature is out of the valid range.
    """
    # For arrays, use the first element to determine range
    # (temp_c_to_mv handles mixed ranges properly)
    temperature = (
        np.asarray(temp_c).flatten()[0] if isinstance(temp_c, np.ndarray) else temp_c
    )

    if temperature < -270 or temperature > 1372:
        raise ValueError("Temperature out of valid Type K range (-270 to 1372 C).")
    if temperature < 0:
        # Range: -270 °C to 0 °C
        return (
            0.0e0,
            0.39450128e-1,
            0.236223736e-4,
            -0.328589068e-6,
            -0.499048288e-8,
            -0.675090592e-10,
            -0.574103274e-12,
            -0.310888729e-14,
            -0.104516094e-16,
            -0.198892669e-19,
            -0.163226975e-22,
        ), None
    else:
        # Range: 0 °C to 1372 °C, with extended exponential term
        return (
            -0.176004137e-1,
            0.38921205e-1,
            0.1855877e-4,
            -0.994575929e-7,
            0.318409457e-9,
            -0.560728449e-12,
            0.560750591e-15,
            -0.3202072e-18,
            0.971511472e-22,
            -0.121047213e-25,
        ), (0.1185976e0, -0.1183432e-3, 0.1269686e3)


def temp_c_to_mv(temp_c: float | np.ndarray) -> float | np.ndarray:
    """
    Convert temperature (°C) to Type K thermocouple voltage (mV) using
    NIST ITS-90 polynomial approximations and an exponential correction for
    temperatures ≥ 0 °C.

    args:
        temp_c: Temperature in degrees Celsius (scalar or array).

    returns:
        float or ndarray: Thermocouple voltage in millivolts.

    raises:
        ValueError: If any temperature is out of the valid range (-270 to 1372 C).
    """
    # Handle scalar case directly
    if isinstance(temp_c, (int, float)):
        coeffs, extended = temp_to_volts_constants(temp_c)
        mv = evaluate_poly(coeffs, temp_c)
        if extended:
            a0, a1, a2 = extended
            mv += a0 * np.exp(a1 * (temp_c - a2) ** 2)
        return mv

    # Handle array case
    temp_c = np.asarray(temp_c)
    is_scalar = temp_c.ndim == 0
    temp_c = np.atleast_1d(temp_c)

    # Validate temperature range
    if np.any(temp_c < -270) or np.any(temp_c > 1372):
        raise ValueError("Temperature out of valid Type K range (-270 to 1372 C).")

    # Initialize output array
    mv = np.zeros_like(temp_c, dtype=float)

    # Handle negative temperatures (Range: -270 °C to 0 °C)
    mask_neg = temp_c < 0
    if np.any(mask_neg):
        coeffs_neg, _ = temp_to_volts_constants(-100.0)  # Representative value
        mv[mask_neg] = evaluate_poly(coeffs_neg, temp_c[mask_neg])

    # Handle positive temperatures (Range: 0 °C to 1372 °C)
    mask_pos = temp_c >= 0
    if np.any(mask_pos):
        coeffs_pos, extended = temp_to_volts_constants(100.0)  # Representative value
        mv[mask_pos] = evaluate_poly(coeffs_pos, temp_c[mask_pos])
        if extended:
            a0, a1, a2 = extended
            mv[mask_pos] += a0 * np.exp(a1 * (temp_c[mask_pos] - a2) ** 2)

    return mv.item() if is_scalar else mv


def mv_to_temp_c(mv: float | np.ndarray) -> float | np.ndarray:
    """
    Convert Type K thermocouple voltage (mV) to temperature (°C) using
    NIST ITS-90 polynomial approximations.

    args:
        mv: Thermocouple voltage in millivolts (scalar or array).

    returns:
        float or ndarray: Temperature in degrees Celsius.
    """
    # Handle scalar case directly
    if isinstance(mv, (int, float)):
        coeffs = volts_to_temp_constants(mv)
        return evaluate_poly(coeffs, mv)

    # Handle array case
    mv = np.asarray(mv)
    is_scalar = mv.ndim == 0
    mv = np.atleast_1d(mv)

    # Initialize output array
    temp = np.zeros_like(mv, dtype=float)

    # Range 1: -5.891 mV to 0 mV
    mask_neg = mv < 0
    if np.any(mask_neg):
        coeffs_neg = volts_to_temp_constants(-1.0)  # Representative value
        temp[mask_neg] = evaluate_poly(coeffs_neg, mv[mask_neg])

    # Range 2: 0 mV to 20.644 mV
    mask_mid = (mv >= 0) & (mv < 20.644)
    if np.any(mask_mid):
        coeffs_mid = volts_to_temp_constants(10.0)  # Representative value
        temp[mask_mid] = evaluate_poly(coeffs_mid, mv[mask_mid])

    # Range 3: 20.644 mV to 54.886 mV
    mask_high = mv >= 20.644
    if np.any(mask_high):
        coeffs_high = volts_to_temp_constants(30.0)  # Representative value
        temp[mask_high] = evaluate_poly(coeffs_high, mv[mask_high])

    return temp.item() if is_scalar else temp
