import numpy as np
import numpy.typing as npt
import h_transport_materials as htm


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


def calculate_flux_from_sample():
    return None


def calculate_permeability_from_flux(
    slope_torr_per_s: float,
    V_m3: float,
    T_K: float,
    A_m2: float,
    e_m: float,
    P_up_torr: float,
) -> float:
    """
    Calculate permeability from flux method.

    Parameters:
        slope_torr_per_s: Slope of downstream pressure increase in Torr/s.
        V_m3: Volume of downstream chamber in m^3.
        T_K: Temperature in Kelvin.
        A_m2: Cross-sectional area of the sample in m^2.
        e_m: Thickness of the sample in meters.
        P_up_torr: Upstream pressure in Torr.

    Returns:
        Permeability: Permeability in mol/(m·s·Pa^0.5).
    """
    R = 8.314  # J/(mol*K)
    dPdt_Pa_per_s = slope_torr_per_s * 133.3  # convert Torr/s to Pa/s
    J = dPdt_Pa_per_s * V_m3 / (R * T_K * A_m2) * 6.022 * 10**23  # H/(m^2*s)
    permeability = J * e_m / (P_up_torr * 133.3) ** 0.5

    return permeability


def import_htm_data(material: str):
    permeabilities = htm.permeabilities.filter(material=f"{material}").filter(
        isotope="h"
    )

    def arrhenius_law(T, A, E):
        k_B = 8.617333262145e-5  # eV/K
        return A * np.exp(-E / (k_B * T))

    all_x_values = []
    all_y_values = []
    labels = []

    for entry in permeabilities:
        if entry.range is None:
            T_bounds = (300, 1200)
        else:
            T_bounds = (entry.range[0].magnitude, entry.range[-1].magnitude)

        x_values = np.geomspace(T_bounds[0], T_bounds[1], num=100)

        y_values = arrhenius_law(
            x_values, entry.pre_exp.magnitude, entry.act_energy.magnitude
        )
        # obtain labels
        label = f"{entry.author} ({entry.year})"
        label = label.capitalize()

        all_x_values.append(x_values)
        all_y_values.append(y_values)
        labels.append(label)

    return all_x_values, all_y_values, labels
