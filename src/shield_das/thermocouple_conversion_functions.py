import numpy as np
import u6


def evaluate_poly(coeffs: list[float] | tuple[float], x: float) -> float:
    """ "
    Evaluate a polynomial at x given the list of coefficients.

    The polynomial is:
        P(x) = a0 + a1*x + a2*x^2 + ... + an*x^n
    where coeffs = [a0, a1, ..., an]

    args:
        coeffs:Polynomial coefficients ordered by ascending power.
        x: The value at which to evaluate the polynomial.

    returns;
        float: The evaluated polynomial result.
    """
    return sum(a * x**i for i, a in enumerate(coeffs))


def volts_to_temp_constants(mv: float) -> tuple[float, ...]:
    """
    Select the appropriate NIST ITS-90 polynomial coefficients for converting
    Type K thermocouple voltage (in millivolts) to temperature (°C).

    The valid voltage range is -5.891 mV to 54.886 mV.

    args:
        mv: Thermocouple voltage in millivolts.

    returns:
        tuple of float: Polynomial coefficients for the voltage-to-temperature conversion.

    raises:
        ValueError: If the input voltage is out of the valid range.
    """
    if mv < -5.891 or mv > 54.886:
        raise ValueError("Voltage out of valid Type K range (-5.891 to 54.886 mV).")
    if mv < 0:
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
    elif mv < 20.644:
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
    temp_c: float,
) -> tuple[tuple[float, ...], tuple[float, float, float] | None]:
    """
    Select the appropriate NIST ITS-90 polynomial coefficients for converting
    temperature (°C) to Type K thermocouple voltage (in millivolts).

    Valid temperature range is -270°C to 1372°C.

    args:
        temp_c: Temperature in degrees Celsius.

    returns:
        Tuple containing:
            - tuple of float: Polynomial coefficients for temperature-to-voltage conversion.
            - tuple of three floats or None: Extended exponential term coefficients for temp >= 0°C, else None.

    raises:
        ValueError: If the input temperature is out of the valid range.
    """
    if temp_c < -270 or temp_c > 1372:
        raise ValueError("Temperature out of valid Type K range (-270 to 1372 °C).")
    if temp_c < 0:
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


def temp_c_to_mv(temp_c: float) -> float:
    """
    Convert temperature (°C) to Type K thermocouple voltage (mV) using
    NIST ITS-90 polynomial approximations and an exponential correction for
    temperatures ≥ 0 °C.

    args:
        temp_c: Temperature in degrees Celsius.

    returns:
        float: Thermocouple voltage in millivolts.
    """
    coeffs, extended = temp_to_volts_constants(temp_c)
    mv = evaluate_poly(coeffs, temp_c)
    if extended:
        a0, a1, a2 = extended
        mv += a0 * np.exp(a1 * (temp_c - a2) ** 2)
    return mv


def mv_to_temp_c(mv: float) -> float:
    """
    Convert Type K thermocouple voltage (mV) to temperature (°C) using
    NIST ITS-90 polynomial approximations.

    args:
        mv: Thermocouple voltage in millivolts.

    returns:
        float: Temperature in degrees Celsius.
    """
    coeffs = volts_to_temp_constants(mv)
    return evaluate_poly(coeffs, mv)


def read_type_k_temp_diff(
    u6_device: u6.U6, pos_channel: int = 0, gain_index: int = 3
) -> float:
    """
    Read temperature from a Type K thermocouple connected to a LabJack U6 using differential input mode.

    This function reads the cold junction temperature from the device's internal sensor,
    reads the differential voltage from the thermocouple input channels,
    applies cold junction compensation, and converts the resulting voltage to temperature.

    args:
        u6_device: An instance of the LabJack U6 device.
        pos_channel: The positive analog input channel number connected to the thermocouple positive lead (default 0).
        gain_index: The LabJack gain setting index to set input voltage range and resolution (default 3, ±0.1 V range).

    returns:
        float: The calculated temperature in degrees Celsius.
    """
    # Read cold junction temperature in Celsius (LabJack returns Kelvin)
    cjt_c = u6_device.getTemperature() - 273.15

    # Read differential thermocouple voltage (volts)
    tc_v = u6_device.getAIN(
        pos_channel, resolutionIndex=12, gainIndex=gain_index, differential=True
    )

    # Convert thermocouple voltage to millivolts
    tc_mv = tc_v * 1000

    # Calculate cold junction compensation voltage (mV)
    cjc_mv = temp_c_to_mv(cjt_c)

    # Total thermocouple voltage including cold junction compensation
    total_mv = tc_mv + cjc_mv

    # Convert total voltage to temperature in Celsius
    return mv_to_temp_c(total_mv)
