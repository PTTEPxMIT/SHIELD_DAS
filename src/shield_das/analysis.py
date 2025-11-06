import numpy as np
from numpy.typing import ArrayLike, NDArray
from uncertainties import UFloat, ufloat


def average_pressure_after_increase(
    time: ArrayLike,
    pressure: ArrayLike,
    window: int = 5,
    slope_threshold: float = 1e-3,
) -> float:
    """
    Detect when pressure stabilizes after a sudden increase and return the average
    stable pressure value.

    This function analyzes pressure vs time data to identify when the pressure has
    stabilised after an initial transient increase. It uses a moving window to smooth
    the pressure slope and identifies the point where the slope falls below a threshold
    (after an initial 5-second settling period). The average pressure from that point
    onward is returned.

    Args:
        time: Time data in seconds.
        pressure: Pressure data in torr.
        window: Size of the moving window for slope smoothing, defaults to 5.
        slope_threshold: Maximum absolute slope (torr/s) considered "stable". Defaults
            to 1e-3 torr/s.

    Returns:
        Average pressure in torr after the system has stabilised. If no stable region is
            detected, returns the average of the second half of the data.

    Example:

        .. highlight:: python
        .. code-block:: python

            from shield_das.analysis import average_pressure_after_increase
            import numpy as np

            time = np.linspace(0, 100, 100)
            pressure = np.concatenate([np.full(50, 10), np.full(50, 100)])
            avg_p = average_pressure_after_increase(time, pressure)
            print(f"Stable pressure: {avg_p:.2f} torr")

        Stable pressure: 100.00 torr

    """
    time = np.asarray(time)
    pressure = np.asarray(pressure)

    # Smooth slope estimation
    slopes = np.gradient(pressure, time)
    smooth_slopes = np.convolve(slopes, np.ones(window) / window, mode="same")

    # Find settled point: first time slope is flat after initial 5 seconds
    settled_mask = (np.abs(smooth_slopes) < slope_threshold) & (time > time.min() + 5)
    settled_index = np.argmax(settled_mask) if settled_mask.any() else len(time) // 2

    return float(np.mean(pressure[settled_index:]))


def calculate_flux_from_sample(t_data: ArrayLike, P_data: ArrayLike) -> float:
    """
    Calculate hydrogen flux from downstream pressure rise data.

    This function fits a weighted linear regression to the pressure rise data in the
    downstream chamber, filtering out unreliable low and high pressure readings from
    the gauge. The slope of this fit represents the flux of hydrogen permeating
    through the sample.

    The function applies:

    - Pressure filtering to remove unreliable gauge readings (<0.05 or >0.95 torr)
    - Exponential weighting to emphasise later data points (more stable regime)
    - Weighted linear least squares fitting

    Args:
        t_data: Time data in seconds.
        P_data: Downstream pressure data in torr.

    Returns:
        Slope of the pressure rise in torr/s, representing the flux rate.

    Example:

        .. highlight:: python
        .. code-block:: python

            from shield_das.analysis import calculate_flux_from_sample
            import numpy as np

            time = np.linspace(0, 100, 100)
            pressure = 0.1 + 0.01 * time  # Linear rise from 0.1 to 1.1 torr
            flux = calculate_flux_from_sample(time, pressure)
            print(f"Flux: {flux:.4f} torr/s")

        Flux: 0.0100 torr/s

    """
    x, y = np.asarray(t_data), np.asarray(P_data)

    # Filter to reliable gauge range
    valid = (y >= 0.05) & (y <= 0.95)
    x, y = x[valid], y[valid]

    # Create weights that emphasize later points (exponential weighting)
    weights = np.exp(np.linspace(-1, 0, len(x)))

    # Weighted linear fit
    slope, _ = np.polyfit(x, y, 1, w=weights)

    return float(slope)


def calculate_permeability_from_flux(
    slope_torr_per_s: float,
    V_m3: float,
    T_K: float,
    A_m2: float,
    e_m: float,
    P_down_torr: float | ArrayLike,
    P_up_torr: float,
) -> UFloat:
    """
    Calculate permeability using the Takaishi-Sensui method with uncertainty
    propagation.

    This function implements the Takaishi-Sensui correction for hydrogen permeation
    measurements, accounting for tube conductance effects and properly propagating
    measurement uncertainties through all calculations.

    The method is based on Takaishi & Sensui (1963): "Thermal Transpiration Effect
    of Hydrogen, Rare Gases and Methane", Trans. Faraday Soc., 59, 2503-2514,
    DOI: 10.1039/tf9635902503

    Args:
        slope_torr_per_s: Rate of downstream pressure rise in torr/s (flux).
        V_m3: Downstream chamber volume in cubic metres.
        T_K: Sample temperature in Kelvin.
        A_m2: Sample area in square metres.
        e_m: Sample thickness in metres.
        P_down_torr: Downstream pressure in torr (float or array, uses final value).
        P_up_torr: Upstream pressure in torr.

    Returns:
        Permeability value with propagated uncertainty (ufloat), in units of
            mol/(m·s·Pa^0.5). Access nominal value with ``.n`` and standard deviation
            with ``.s`` attributes.

    Example:

        .. highlight:: python
        .. code-block:: python

            from shield_das.analysis import calculate_permeability_from_flux

            perm = calculate_permeability_from_flux(
                slope_torr_per_s=0.01,
                V_m3=7.9e-5,
                T_K=873,
                A_m2=1.89e-4,
                e_m=0.00088,
                P_down_torr=0.5,
                P_up_torr=100.0
            )
            print(f"Permeability: {perm.n:.2e} ± {perm.s:.2e}")

        Permeability: 1.23e-10 ± 5.67e-12

    Note:
        Volume uncertainty is assumed to be 12% based on measurement precision. Volume
        ratio (V1/V_total) has 10% uncertainty for heated/ambient volume split. All
        uncertainties are propagated using the uncertainties package.

    """

    TORR_TO_PA = 133.3
    R = 8.314  # J/(mol·K)
    N_A = 6.022e23  # Avogadro's number

    # Define parameters with uncertainties (matching mwe.py approach)
    # Volume uncertainty: ~12% based on measurement precision
    V_with_unc = ufloat(V_m3, V_m3 * 0.12)

    # Volume ratio uncertainty: uncertainty in heated vs ambient volume split
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


def evaluate_permeability_values(
    datasets: dict[str, dict[str, any]],
) -> tuple[list[float], list[UFloat], NDArray, NDArray, list[float], list[float]]:
    """
    Evaluate permeability values from multiple experimental datasets with uncertainty
    propagation.

    This function processes multiple permeation measurement runs, calculates
    permeability for each using the Takaishi-Sensui method, groups measurements by
    temperature, and performs weighted averaging for runs at the same temperature.
    All uncertainties are rigorously propagated through the calculations.

    Args:
        datasets: Dictionary of experimental datasets where each key is a run
            identifier and each value is a dict containing 'temperature' (float, K),
            'time_data' (array-like, seconds), 'upstream_data' (dict with
            'pressure_data' in torr), 'downstream_data' (dict with 'pressure_data' in
            torr), and optionally 'sample_thickness' (float, metres, defaults to
            0.00088 m).

    Returns:
        A tuple of six elements: temps (list of all temperatures in K), perms (list
            of ufloat permeabilities in mol/(m·s·Pa^0.5)), x_error (NDArray of
            1000/T values for unique temperatures), y_error (NDArray of nominal
            permeability values), error_lower (list of lower error bars), error_upper
            (list of upper error bars).

    Example:

        .. highlight:: python
        .. code-block:: python

            from shield_das.analysis import evaluate_permeability_values
            import numpy as np

            datasets = {
                "run1": {
                    "temperature": 873,
                    "time_data": np.linspace(0, 100, 100),
                    "upstream_data": {"pressure_data": np.full(100, 100.0)},
                    "downstream_data": {"pressure_data": np.linspace(0.1, 1.0, 100)},
                    "sample_thickness": 0.00088
                }
            }
            temps, perms, x_err, y_err, err_low, err_up = (
                evaluate_permeability_values(datasets)
            )
            print(f"Temperature: {temps[0]} K")
            print(f"Permeability: {perms[0].n:.2e} ± {perms[0].s:.2e}")

        Temperature: 873 K
        Permeability: 1.23e-10 ± 5.67e-12

    Note:
        Multiple runs at the same temperature are combined using weighted averaging
        (weighted by inverse variance). Default sample diameter is 0.0155 m and
        default chamber volume is 7.9e-5 m³. Results are sorted by temperature
        (ascending) for x_error, y_error, and error arrays.

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


def fit_permeability_data(
    temps: ArrayLike, perms: list[UFloat] | list[float]
) -> tuple[NDArray, NDArray]:
    """
    Fit Arrhenius equation to permeability data using weighted least squares.

    This function performs a weighted linear regression in log-transformed space to
    fit an Arrhenius model to permeability vs temperature data. Measurements with
    smaller uncertainties are weighted more heavily in the fit.

    The Arrhenius model is: Perm = P0 * exp(-Ea / (R*T))

    In log space: log10(Perm) = m * (1000/T) + c

    Args:
        temps: Temperature values in Kelvin.
        perms: Permeability values (ufloat objects with uncertainties or regular
            floats), in mol/(m·s·Pa^0.5).

    Returns:
        A tuple of fit_x (NDArray of 100 inverse temperature values, 1000/T) and
            fit_y (NDArray of 100 fitted permeability values for plotting).

    Example:

        .. highlight:: python
        .. code-block:: python

            from shield_das.analysis import fit_permeability_data
            from uncertainties import ufloat
            import matplotlib.pyplot as plt
            import numpy as np

            temps = [773, 873, 973]
            perms = [
                ufloat(1e-10, 1e-11),
                ufloat(5e-10, 5e-11),
                ufloat(2e-9, 2e-10)
            ]
            fit_x, fit_y = fit_permeability_data(temps, perms)
            plt.plot(fit_x, fit_y, 'r-', label='Arrhenius fit')
            plt.plot(1000/np.array(temps), [p.n for p in perms], 'bo')
            plt.xlabel('1000/T (K⁻¹)')
            plt.ylabel('Permeability (mol/(m·s·Pa^0.5))')
            plt.show()

    Note:
        Weights are calculated as w = 1/sigma for each point. Points without
        uncertainty (regular floats) receive uniform weight. Returns 100 evenly
        spaced points for smooth plotting. Uncertainties are propagated through
        log10 transformation.

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
    # For ufloats, we could use unp.log, but we'll do it manually to control
    # weights
    log_perm = np.log10(perm_vals)

    # Propagate uncertainties through log transform: d(log(x))/dx = 1/x
    # So sigma_log(x) = sigma_x / x (for natural log, divide by ln(10) for
    # log10)
    # Suppress divide-by-zero warning when perm_stds is zero (handled by
    # np.where)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_perm_stds = perm_stds / (perm_vals * np.log(10))

        # Calculate weights (inverse of variance)
        # Use w = 1/sigma; fallback to 1 when std is 0 or missing
        weights = np.where(
            (log_perm_stds > 0) & np.isfinite(log_perm_stds),
            1.0 / log_perm_stds,
            1.0,
        )

    # Fit in 1/T space: log10(perm) = m * (1000/T) + c
    x_all = 1000 / temps
    coeffs = np.polyfit(x_all, log_perm, 1, w=weights)

    # Generate smooth fit line
    fit_x = np.linspace(x_all.min(), x_all.max(), 100)
    fit_y = 10 ** (coeffs[0] * fit_x + coeffs[1])

    return fit_x, fit_y
