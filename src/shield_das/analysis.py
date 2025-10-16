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

    # Create weights that emphasize later points (exponential weighting)
    weights = np.exp(np.linspace(-1, 0, len(x)))

    # Weighted linear fit
    slope, _ = np.polyfit(x, y, 1, w=weights)

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


def evaluate_permeability_values(datasets):
    # Calculate and plot permeability for each dataset
    temps, perms = [], []
    SAMPLE_DIAMETER = 0.0155  # meters
    SAMPLE_AREA = np.pi * (SAMPLE_DIAMETER / 2) ** 2
    SAMPLE_THICKNESS = 0.00088  # meters
    CHAMBER_VOLUME = 7.9e-5  # m³

    for dataset in datasets.values():
        temp = dataset["temperature"]
        time = dataset["time_data"]
        p_up = dataset["upstream_data"]["pressure_data"]
        p_down = dataset["downstream_data"]["pressure_data"]

        # Calculate permeability
        p_avg_up = average_pressure_after_increase(time, p_up)
        flux = calculate_flux_from_sample(time, p_down)
        perm = calculate_permeability_from_flux(
            flux,
            CHAMBER_VOLUME,
            temp,
            SAMPLE_AREA,
            SAMPLE_THICKNESS,
            p_down,
            p_avg_up,
        )

        temps.append(temp)
        perms.append(perm)

    # Group data by temperature to calculate error bars
    from collections import defaultdict

    temp_groups = defaultdict(list)
    for temp, perm in zip(temps, perms):
        temp_groups[temp].append(perm)

    # Calculate error bars for each unique temperature
    unique_temps = []
    error_lower = []
    error_upper = []

    for temp in sorted(temp_groups.keys()):
        perm_values = np.array(temp_groups[temp])
        min_perm = perm_values.min()
        max_perm = perm_values.max()

        unique_temps.append(temp)
        # Error bars: from 10% below min to 10% above max
        # Calculate relative to the center point between min and max
        center = (min_perm + max_perm) / 2
        error_lower.append(center - min_perm * 0.9)
        error_upper.append(max_perm * 1.1 - center)

    # Convert to arrays for plotting
    x_error = 1000 / np.array(unique_temps)
    y_error = [
        (min_perm * 0.9 + max_perm * 1.1) / 2
        for min_perm, max_perm in [
            (
                np.array(temp_groups[temp]).min(),
                np.array(temp_groups[temp]).max(),
            )
            for temp in sorted(temp_groups.keys())
        ]
    ]

    return temps, perms, x_error, y_error, error_lower, error_upper


def fit_permeability_data(temps, perms):
    log_y = np.log10(perms)
    x_all = 1000 / np.array(temps)
    # Linear fit: log10(perm) = m * (1000/T) + c
    coeffs = np.polyfit(x_all, log_y, 1)
    fit_x = np.linspace(x_all.min(), x_all.max(), 100)
    fit_y = 10 ** (coeffs[0] * fit_x + coeffs[1])

    return fit_x, fit_y
