import numpy as np


def average_pressure_after_increase(time, pressure, window=5, slope_threshold=1e-3):
    """
    Detects when the pressure stabilizes after a sudden increase and
    returns the average pressure after that time in torr.

    Parameters
    ----------
    time : array-like
        Time values (seconds).
    pressure : array-like
        Pressure values corresponding to time.
    window : int, optional
        Number of points to use for local slope estimation (default=5).
    slope_threshold : float, optional
        Threshold for determining when slope is "flat" (default=1e-3).

    Returns
    -------
    avg_pressure : float
        Average pressure after the increase.
    t_start : float
        Detected time when pressure flattens.
    """

    time = np.asarray(time)
    pressure = np.asarray(pressure)

    # Estimate slope using central differences
    slopes = np.gradient(pressure, time)

    # Smooth slope with rolling average
    smooth_slopes = np.convolve(slopes, np.ones(window) / window, mode="same")

    # Find first time slope falls below threshold after the jump
    for i in range(len(smooth_slopes)):
        if abs(smooth_slopes[i]) < slope_threshold and time[i] > min(time) + 5:
            # treat this as the "settled" point
            settled_index = i
            break
    else:
        settled_index = int(0.5 * len(time))  # fallback: halfway

    # Compute average after that point
    avg_pressure = np.mean(pressure[settled_index:])
    # t_start = time[settled_index]

    return avg_pressure


def calculate_flux_from_sample(t_data, P_data, tail_frac=0.25, tol=0.08):
    x = np.asarray(t_data)
    y = np.asarray(P_data)

    # Filter data to only include pressure values between 0.03 and 0.985
    valid_mask = (y >= 0.05) & (y <= 0.95)
    x = x[valid_mask]
    y = y[valid_mask]

    if x.size < 5:
        raise ValueError("need at least 5 points to fit an asymptote")

    g = np.gradient(y, x)
    n = len(x)
    tail_start = max(0, int(n * (1 - tail_frac)))
    tail_g = g[tail_start:]

    med = np.median(tail_g)
    if med == 0:
        med = np.mean(tail_g) or 1e-12

    rel_ok = np.abs((tail_g - med) / (med if med != 0 else 1e-12)) <= tol

    use_rel_idx = None
    for i in range(len(rel_ok)):
        if rel_ok[i:].all():
            use_rel_idx = tail_start + i
            break
    if use_rel_idx is None:
        use_rel_idx = tail_start

    Xfit = x[use_rel_idx:]
    Yfit = y[use_rel_idx:]
    slope, intercept = np.polyfit(Xfit, Yfit, 1)

    return slope


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
    J = dPdt_Pa_per_s * V_m3 / (R * T_K * A_m2) * 6.022e23  # H/(m^2*s)
    permeability = J * e_m / ((P_up_torr * 133.3) ** 0.5)

    return permeability
