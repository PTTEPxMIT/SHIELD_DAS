import numpy as np


def average_pressure_after_increase(time, pressure, window=5, slope_threshold=1e-3):
    """
    Detects when pressure stabilizes after a sudden increase and returns
    the average pressure after that time in torr.
    """
    time = np.asarray(time)
    pressure = np.asarray(pressure)

    # Smooth slope estimation
    slopes = np.gradient(pressure, time)
    smooth_slopes = np.convolve(slopes, np.ones(window) / window, mode="same")

    # Find settled point: first time slope is flat after initial 5 seconds
    settled_mask = (np.abs(smooth_slopes) < slope_threshold) & (time > time.min() + 5)
    settled_index = np.argmax(settled_mask) if settled_mask.any() else len(time) // 2

    return np.mean(pressure[settled_index:])


def calculate_flux_from_sample(t_data, P_data):
    """Calculate flux from downstream pressure rise, filtering unreliable gauge data."""
    x, y = np.asarray(t_data), np.asarray(P_data)

    # Filter to reliable gauge range
    valid = (y >= 0.05) & (y <= 0.95)
    x, y = x[valid], y[valid]

    # Linear fit to stable region
    slope, _ = np.polyfit(x, y, 1)

    return slope


def calculate_permeability_from_flux(
    slope_torr_per_s: float,
    V_m3: float,
    T_K: float,
    A_m2: float,
    e_m: float,
    P_down_torr: float,
    P_up_torr: float,
) -> float:
    """Calculates permeability using Takaishi-Sensui method, see 10.1039/tf9635902503
    for more details"""

    TORR_TO_PA = 133.3
    R = 8.314  # J/(mol·K)
    N_A = 6.022e23  # Avogadro's number

    V1_ratio = 0.35  # ratio of V1 to total volume

    V1 = V_m3 * V1_ratio
    V2 = V_m3 * (1 - V1_ratio)
    T1 = T_K
    T2 = 300  # ambient temperature in Kelvin

    A = 1.24 * 56.3 / 10e-5
    B = 8 * 7.7 / 10e-2
    C = 10.6 * 2.73
    d = 0.0155  # diameter of pipe

    P2dot = slope_torr_per_s * TORR_TO_PA
    P2 = P_down_torr[-1] * TORR_TO_PA  # convert Torr to Pa

    # --- helper quantities ---
    num2 = C * (d * P2) ** 0.5 + (T2 / T1) ** 0.5 + A * d**2 * P2**2 + B * d * P2  # #2
    den3 = C * (d * P2) ** 0.5 + A * d**2 * P2**2 + B * d * P2 + 1  # #3

    num1 = (
        B * d * P2dot
        + (C * d * P2dot) / (2 * (d * P2) ** 0.5)
        + 2 * A * d**2 * P2 * P2dot
    )

    # --- assemble dn/dt ---
    n_dot = (
        (V2 * P2dot) / (R * T2)
        + (V1 * P2dot) / (R * T1 * num2)
        + (V1 * P2 * num1) / (R * T1 * num2 * den3)
        - (V1 * P2 * num1) / (R * T1 * num2**2)
    )

    J_TS = n_dot / A_m2 * N_A  # H/(m^2*s)

    Perm_TS = J_TS * e_m / (P_up_torr * TORR_TO_PA) ** 0.5

    return Perm_TS
