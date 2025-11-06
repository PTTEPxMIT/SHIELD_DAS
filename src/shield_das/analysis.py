import numpy as np
from uncertainties import ufloat


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
):
    """Calculates permeability using Takaishi-Sensui method with uncertainty propagation.

    See 10.1039/tf9635902503 for more details on the Takaishi-Sensui method.

    Uses the uncertainties package to properly propagate measurement uncertainties
    through all calculations, following the approach in mwe.py.

    Returns:
        ufloat: Permeability value with propagated uncertainty (mol/(m·s·Pa^0.5))
    """

    TORR_TO_PA = 133.3
    R = 8.314  # J/(mol·K)
    N_A = 6.022e23  # Avogadro's number

    # Define parameters with uncertainties (matching mwe.py approach)
    # Volume uncertainty: ~12% based on measurement precision
    V_with_unc = ufloat(V_m3, V_m3 * 0.12)

    # Volume ratio uncertainty: significant uncertainty in heated vs ambient volume split
    V1_ratio = ufloat(0.35, 0.1)

    V1 = V_with_unc * V1_ratio
    V2 = V_with_unc * (1 - V1_ratio)
    T1 = T_K
    T2 = 300  # ambient temperature in Kelvin

    # Takaishi-Sensui constants
    A = 1.24 * 56.3 / 10e-5
    B = 8 * 7.7 / 10e-2
    C = 10.6 * 2.73
    d = 0.0155  # diameter of pipe

    P2dot = slope_torr_per_s * TORR_TO_PA

    # Use final downstream pressure (assuming P_down_torr is array-like)
    if hasattr(P_down_torr, "__len__"):
        P2 = P_down_torr[-1] * TORR_TO_PA
    else:
        P2 = P_down_torr * TORR_TO_PA

    # --- helper quantities ---
    num2 = C * (d * P2) ** 0.5 + (T2 / T1) ** 0.5 + A * d**2 * P2**2 + B * d * P2
    den3 = C * (d * P2) ** 0.5 + A * d**2 * P2**2 + B * d * P2 + 1

    num1 = (
        B * d * P2dot
        + (C * d * P2dot) / (2 * (d * P2) ** 0.5)
        + 2 * A * d**2 * P2 * P2dot
    )

    # --- assemble dn/dt with uncertainty propagation ---
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
    """Evaluate permeability values with uncertainty propagation.

    Uses the uncertainties package to track measurement uncertainties through
    all calculations, providing rigorous error bars based on error propagation
    rather than empirical spread.

    Returns:
        temps: List of temperatures (K)
        perms: List of permeability ufloat objects with uncertainties
        x_error: Array of 1000/T values for error bar plotting
        y_error: Array of permeability nominal values for error bar plotting
        error_lower: Array of lower error bar values
        error_upper: Array of upper error bar values
    """
    # Calculate and plot permeability for each dataset
    temps, perms = [], []
    SAMPLE_DIAMETER = 0.0155  # meters
    SAMPLE_AREA = np.pi * (SAMPLE_DIAMETER / 2) ** 2
    CHAMBER_VOLUME = 7.9e-5  # m³

    for dataset in datasets.values():
        temp = dataset["temperature"]
        time = dataset["time_data"]
        p_up = dataset["upstream_data"]["pressure_data"]
        p_down = dataset["downstream_data"]["pressure_data"]

        SAMPLE_THICKNESS = dataset.get("sample_thickness", 0.00088)  # meters

        # Calculate permeability with uncertainty propagation
        p_avg_up = average_pressure_after_increase(time, p_up)
        flux = calculate_flux_from_sample(time, p_down)

        # This now returns a ufloat with uncertainty
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

    # Group data by temperature to combine measurements
    from collections import defaultdict

    temp_groups = defaultdict(list)
    for temp, perm in zip(temps, perms):
        temp_groups[temp].append(perm)

    # Calculate weighted average and combined uncertainties for each temperature
    unique_temps = []
    avg_perms = []
    error_lower = []
    error_upper = []

    for temp in sorted(temp_groups.keys()):
        perm_values = temp_groups[temp]

        if len(perm_values) == 1:
            # Single measurement - use its uncertainty directly
            avg_perm = perm_values[0]
        else:
            # Multiple measurements - combine using weighted average
            # Weight by inverse variance (1/sigma^2)
            vals = np.array([p.n if hasattr(p, "n") else p for p in perm_values])
            stds = np.array([p.s if hasattr(p, "s") else 0 for p in perm_values])

            # Avoid division by zero - if std is 0, use small weight
            weights = np.where(stds > 0, 1.0 / stds**2, 1e-10)

            # Weighted mean
            mean_val = np.sum(weights * vals) / np.sum(weights)

            # Combined uncertainty (standard error of weighted mean)
            mean_std = np.sqrt(1.0 / np.sum(weights))

            avg_perm = ufloat(mean_val, mean_std)

        unique_temps.append(temp)
        avg_perms.append(avg_perm)

        # Extract nominal value and uncertainty for error bars
        if hasattr(avg_perm, "n"):
            error_lower.append(avg_perm.s)  # symmetric error bars
            error_upper.append(avg_perm.s)
        else:
            error_lower.append(0)
            error_upper.append(0)

    # Convert to arrays for plotting
    x_error = 1000 / np.array(unique_temps)
    y_error = np.array([p.n if hasattr(p, "n") else p for p in avg_perms])

    return temps, perms, x_error, y_error, error_lower, error_upper


def fit_permeability_data(temps, perms):
    """Fit Arrhenius equation to permeability data using weighted least squares.

    Weights measurements by inverse of their uncertainty, giving more importance
    to precise measurements. This matches the approach in mwe.py.

    Args:
        temps: List of temperatures (K)
        perms: List of permeability values (can be ufloats with uncertainties)

    Returns:
        fit_x: Array of 1000/T values for plotting
        fit_y: Array of fitted permeability values
    """
    # Convert to numpy arrays and handle ufloat objects
    temps = np.array(temps)

    # Extract nominal values and uncertainties
    if hasattr(perms[0], "n"):
        # ufloat objects - extract nominal and std dev
        perm_vals = np.array([p.n for p in perms])
        perm_stds = np.array([p.s for p in perms])
    else:
        # Regular floats
        perm_vals = np.array(perms)
        perm_stds = np.zeros_like(perm_vals)

    # Log transform for Arrhenius fit
    # For ufloats, we could use unp.log, but we'll do it manually to control weights
    log_perm = np.log10(perm_vals)

    # Propagate uncertainties through log transform: d(log(x))/dx = 1/x
    # So sigma_log(x) = sigma_x / x (for natural log, divide by ln(10) for log10)
    log_perm_stds = perm_stds / (perm_vals * np.log(10))

    # Calculate weights (inverse of variance)
    # Use w = 1/sigma; fallback to 1 when std is 0 or missing
    weights = np.where(
        (log_perm_stds > 0) & np.isfinite(log_perm_stds), 1.0 / log_perm_stds, 1.0
    )

    # Fit in 1/T space: log10(perm) = m * (1000/T) + c
    x_all = 1000 / temps
    coeffs = np.polyfit(x_all, log_perm, 1, w=weights)

    # Generate smooth fit line
    fit_x = np.linspace(x_all.min(), x_all.max(), 100)
    fit_y = 10 ** (coeffs[0] * fit_x + coeffs[1])

    return fit_x, fit_y
