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


def calculate_flux_from_sample(t_data, P_data, tail_frac=0.25, tol=0.08):
    """Calculate flux from downstream pressure rise, filtering unreliable gauge data."""
    x, y = np.asarray(t_data), np.asarray(P_data)

    # Filter to reliable gauge range
    valid = (y >= 0.05) & (y <= 0.95)
    x, y = x[valid], y[valid]

    if len(x) < 5:
        raise ValueError("Need at least 5 valid points to fit asymptote")

    # Find stable tail region where gradient is constant
    grad = np.gradient(y, x)
    tail_start = max(0, int(len(x) * (1 - tail_frac)))
    tail_grad = grad[tail_start:]

    # Identify where gradient variation is within tolerance
    med = np.median(tail_grad) or np.mean(tail_grad) or 1e-12
    stable = np.abs((tail_grad - med) / med) <= tol

    # Find first index where all subsequent gradients are stable
    for i in range(len(stable)):
        if stable[i:].all():
            fit_start = tail_start + i
            break
    else:
        fit_start = tail_start

    # Linear fit to stable region
    slope, _ = np.polyfit(x[fit_start:], y[fit_start:], 1)
    return slope


def calculate_permeability_from_flux(
    slope_torr_per_s: float,
    V_m3: float,
    T_K: float,
    A_m2: float,
    e_m: float,
    P_up_torr: float,
) -> float:
    """Calculate permeability from flux method."""
    TORR_TO_PA = 133.3
    R = 8.314  # J/(mol·K)
    N_A = 6.022e23  # Avogadro's number

    # Convert flux to H atoms per m²·s
    flux_Pa_per_s = slope_torr_per_s * TORR_TO_PA
    flux_H_atoms = flux_Pa_per_s * V_m3 * N_A / (R * T_K * A_m2)

    # Calculate permeability
    return flux_H_atoms * e_m / (P_up_torr * TORR_TO_PA) ** 0.5
